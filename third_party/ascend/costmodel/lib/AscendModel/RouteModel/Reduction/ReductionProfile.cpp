#include "AscendModel/RouteModel/Reduction/ReductionProfile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <cmath>
#include <system_error>
using namespace mlir::ascend;

namespace {
llvm::Error invalid(llvm::StringRef text) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 text.str().c_str());
}
std::string key(llvm::StringRef family, ReductionMode mode, llvm::StringRef op,
                llvm::StringRef type) {
  return family.str() + "/" +
         (mode == ReductionMode::SIMD ? "simd/" : "simt/") + op.str() + "/" +
         type.str();
}
llvm::Expected<ReductionExpression>
parseExpression(const llvm::json::Value &v, unsigned depth, unsigned &nodes) {
  if (depth > 32 || ++nodes > 1024)
    return invalid("reduction expression is too large");
  const auto *o = v.getAsObject();
  if (!o)
    return invalid("reduction expression must be an object");
  ReductionExpression e;
  if (auto n = o->getNumber("constant")) {
    if (o->size() != 1 || !std::isfinite(*n))
      return invalid("invalid reduction constant");
    e.constant = *n;
    return e;
  }
  if (auto f = o->getString("feature")) {
    if (o->size() != 1 || !llvm::StringSwitch<bool>(*f)
                               .Cases("R", "G", "Q", "L", true)
                               .Cases("W", "rank", "d1", "d2", true)
                               .Default(false))
      return invalid("unknown reduction feature");
    e.kind = ReductionExpression::Kind::Feature;
    e.name = f->str();
    return e;
  }
  auto op = o->getString("op");
  const auto *args = o->getArray("args");
  if (!op || !args || o->size() != 2 || args->empty() || args->size() > 16)
    return invalid("invalid reduction expression operator");
  unsigned arity =
      llvm::StringSwitch<unsigned>(*op)
          .Cases("add", "mul", "max", "min", "and", "or", 0)
          .Cases("sub", "div", "pow", "lt", "le", "gt", 2)
          .Cases("ge", "eq", "ne", 2)
          .Cases("neg", "not", "indicator", "ceil", "sqrt", "log2", 1)
          .Default(99);
  if (arity == 99 || (arity != 0 && args->size() != arity))
    return invalid("unsupported reduction operator or arity");
  e.kind = ReductionExpression::Kind::Operator;
  e.name = op->str();
  for (const auto &a : *args) {
    auto parsed = parseExpression(a, depth + 1, nodes);
    if (!parsed)
      return parsed.takeError();
    e.arguments.push_back(std::move(*parsed));
  }
  if (*op == "pow" &&
      (e.arguments[1].kind != ReductionExpression::Kind::Constant ||
       e.arguments[1].constant < 0 || e.arguments[1].constant > 4))
    return invalid("reduction power requires a fixed exponent in [0,4]");
  return e;
}
} // namespace

static llvm::Expected<llvm::StringMap<ReductionFunction>>
parseFirstAxisFunctions(const llvm::json::Array &functions) {
  llvm::StringMap<ReductionFunction> result;
  if (functions.size() > 1024)
    return invalid("too many first-axis functions");
  for (const auto &value : functions) {
    const auto *f = value.getAsObject();
    if (!f || !f->getString("family") ||
        *f->getString("family") != "first_axis")
      return invalid("unsupported reduction family");
    auto mode = f->getString("mode"), op = f->getString("op"),
         dtype = f->getString("dtype");
    const auto *warps = f->getArray("supported_warps"),
               *segments = f->getArray("segments");
    if (!mode || (*mode != "simd" && *mode != "simt") || !op || !dtype ||
        !llvm::StringSwitch<bool>(*op)
             .Cases("sum", "max", "min", "prod", true)
             .Cases("xor", "and", "or", true)
             .Default(false) ||
        !llvm::StringSwitch<bool>(*dtype)
             .Cases("bool", "f16", "bf16", "f32", true)
             .Cases("i8", "i16", "i32", "i64", true)
             .Cases("u8", "u16", "u32", "u64", true)
             .Default(false) ||
        !warps || warps->empty() || !segments || segments->empty() ||
        segments->size() > 128)
      return invalid("invalid reduction function");
    ReductionFunction function;
    for (const auto &w : *warps) {
      auto n = w.getAsInteger();
      if (!n || *n < 1 || *n > 64 || (*n & (*n - 1)) ||
          std::find(function.supportedWarps.begin(),
                    function.supportedWarps.end(),
                    *n) != function.supportedWarps.end())
        return invalid("invalid reduction warp list");
      function.supportedWarps.push_back(*n);
    }
    for (const auto &s : *segments) {
      const auto *object = s.getAsObject();
      if (!object)
        return invalid("invalid reduction segment");
      auto id = object->getString("id");
      auto lane = object->getInteger("lane_width");
      const auto *cond = object->get("condition"),
                 *expr = object->get("expression");
      if (!id || !lane || *lane <= 0 || !cond || !expr)
        return invalid("incomplete reduction segment");
      unsigned nodes = 0;
      auto c = parseExpression(*cond, 0, nodes);
      if (!c)
        return c.takeError();
      nodes = 0;
      auto e = parseExpression(*expr, 0, nodes);
      if (!e)
        return e.takeError();
      function.segments.push_back(
          {id->str(), *lane, std::move(*c), std::move(*e)});
    }
    if (!result
             .try_emplace(key(*f->getString("family"),
                              *mode == "simd" ? ReductionMode::SIMD
                                              : ReductionMode::SIMT,
                              *op, *dtype),
                          std::move(function))
             .second)
      return invalid("duplicate reduction function");
  }
  return result;
}

bool ReductionCostProfile::isValid() const {
  // Missing calibration is a compatibility state: use legacy shuffle costs.
  return llvm::all_of(parameters, [](const auto &entry) {
    return !entry.second.empty() &&
           llvm::all_of(entry.second,
                        [](double value) { return std::isfinite(value); });
  });
}

llvm::Expected<ReductionCostProfile>
mlir::ascend::parseReductionParameters(const llvm::json::Object &object) {
  ReductionCostProfile profile;
  for (const auto &[key, value] : object) {
    const auto *array = value.getAsArray();
    if (!array || array->empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "reduction parameter '%s' must be a non-empty numeric array",
          key.str().c_str());
    std::vector<double> parameters;
    for (const auto &entry : *array) {
      auto number = entry.getAsNumber();
      if (!number || !std::isfinite(*number))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "reduction parameter '%s' must contain finite numbers",
            key.str().c_str());
      parameters.push_back(*number);
    }
    if (llvm::StringRef(key).starts_with("rn_candidate_") &&
        (parameters.size() != 18 || parameters[0] <= 0 || parameters[1] < 0))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "reduction parameter '%s' requires 18 values, Q > 0 and N0 >= 0",
          key.str().c_str());
    profile.parameters[key] = std::move(parameters);
  }
  return profile;
}

llvm::Expected<ReductionProfile>
ReductionProfile::loadFromFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "cannot read reduction profile '%s'",
                                   path.str().c_str());
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return parsed.takeError();
  const auto *root = parsed->getAsObject();
  if (!root || root->getInteger("schema_version") != 1 ||
      root->getString("unit") != "syscnt_cycle" || !root->getString("target") ||
      root->getString("target")->empty() ||
      !root->getString("profile_version") ||
      root->getString("profile_version")->empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "invalid reduction profile schema, target, version or unit");
  const auto *families = root->getObject("families");
  const auto *tail = families ? families->getObject("tail_axis") : nullptr;
  if (!tail || !tail->getObject("simd") || !tail->getObject("simt"))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "reduction profile requires tail_axis simd and simt parameters");
  auto simdParameters = parseReductionParameters(*tail->getObject("simd"));
  if (!simdParameters)
    return simdParameters.takeError();
  auto simtParameters = parseReductionParameters(*tail->getObject("simt"));
  if (!simtParameters)
    return simtParameters.takeError();
  ReductionProfile profile;
  profile.target = root->getString("target")->str();
  profile.simd = std::move(*simdParameters);
  profile.simt = std::move(*simtParameters);
  if (const auto *value = families->get("first_axis")) {
    const auto *functions = value->getAsArray();
    if (!functions)
      return invalid("first_axis must be an array");
    auto parsedFunctions = parseFirstAxisFunctions(*functions);
    if (!parsedFunctions)
      return parsedFunctions.takeError();
    profile.firstAxisFunctions = std::move(*parsedFunctions);
  }
  auto bytes = buffer.get()->getBuffer();
  auto hash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  profile.contentHash = llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true);
  return profile;
}

const ReductionCostProfile &
ReductionProfile::getModeProfile(ReductionMode mode) const {
  return mode == ReductionMode::SIMD ? simd : simt;
}

const ReductionFunction *
ReductionProfile::findFirstAxis(ReductionMode mode, llvm::StringRef op,
                                llvm::StringRef dtype) const {
  auto it = firstAxisFunctions.find(key("first_axis", mode, op, dtype));
  return it == firstAxisFunctions.end() ? nullptr : &it->second;
}
