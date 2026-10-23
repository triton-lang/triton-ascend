#include "AscendModel/RouteModel/Reduction/ReductionProfile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
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
class ExpressionParser {
  llvm::StringRef text;
  unsigned nodes = 0;

  void space() { text = text.ltrim(); }
  bool consume(llvm::StringRef token) {
    space();
    return text.consume_front(token);
  }
  llvm::Expected<ReductionExpression>
  operation(llvm::StringRef name, std::vector<ReductionExpression> args) {
    unsigned arity =
        llvm::StringSwitch<unsigned>(name)
            .Cases("add", "mul", "max", "min", "and", "or", 0)
            .Cases("sub", "div", "pow", "lt", "le", "gt", 2)
            .Cases("ge", "eq", "ne", 2)
            .Cases("neg", "not", "indicator", "ceil", "sqrt", "log2", 1)
            .Default(99);
    if (arity == 99 || args.empty() || args.size() > 16 ||
        (arity && args.size() != arity))
      return invalid("unsupported reduction operator or arity");
    if (name == "pow" && (args[1].kind != ReductionExpression::Kind::Constant ||
                          args[1].constant < -4 || args[1].constant > 4))
      return invalid("reduction power requires a fixed exponent in [-4,4]");
    ReductionExpression e;
    e.kind = ReductionExpression::Kind::Operator;
    e.name = name.str();
    e.arguments = std::move(args);
    return e;
  }
  llvm::Expected<ReductionExpression> primary(unsigned depth) {
    if (depth > 32 || ++nodes > 1024)
      return invalid("reduction expression is too large");
    space();
    if (consume("(")) {
      auto e = expression(0, depth + 1);
      if (!e)
        return e.takeError();
      if (!consume(")"))
        return invalid("missing reduction closing parenthesis");
      return std::move(*e);
    }
    // Negative literals remain constants, preserving fixed power exponents.
    if (!text.empty() && (llvm::isDigit(text.front()) || text.front() == '.' ||
                          text.front() == '-')) {
      size_t n = text.starts_with("-") ? 1 : 0;
      while (n < text.size() && (llvm::isDigit(text[n]) || text[n] == '.'))
        ++n;
      if (n < text.size() && (text[n] == 'e' || text[n] == 'E')) {
        ++n;
        if (n < text.size() && (text[n] == '+' || text[n] == '-'))
          ++n;
        while (n < text.size() && llvm::isDigit(text[n]))
          ++n;
      }
      double value;
      if (text.take_front(n).getAsDouble(value) || !std::isfinite(value))
        return invalid("invalid reduction numeric literal");
      text = text.drop_front(n);
      ReductionExpression e;
      e.constant = value;
      return e;
    }
    size_t n = 0;
    while (n < text.size() && (llvm::isAlnum(text[n]) || text[n] == '_'))
      ++n;
    if (!n)
      return invalid("expected reduction literal, feature or function");
    auto name = text.take_front(n);
    text = text.drop_front(n);
    if (consume("(")) {
      std::vector<ReductionExpression> args;
      if (!consume(")")) {
        do {
          auto a = expression(0, depth + 1);
          if (!a)
            return a.takeError();
          args.push_back(std::move(*a));
          if (args.size() > 16)
            return invalid("too many reduction arguments");
        } while (consume(","));
        if (!consume(")"))
          return invalid("missing reduction function parenthesis");
      }
      auto op = llvm::StringSwitch<llvm::StringRef>(name)
                    .Case("all_of", "and")
                    .Case("any_of", "or")
                    .Case("logical_not", "not")
                    .Default(name);
      return operation(op, std::move(args));
    }
    if (!llvm::StringSwitch<bool>(name)
             .Cases("R", "G", "Q", "L", true)
             .Cases("W", "rank", "d1", "d2", true)
             .Cases("O", "K", "H", "N", "route_id", true)
             .Cases("R_t", "A_t", "N_local", "N_warp", "N_cross", true)
             .Cases("uK", "wK", "tK", "tO", "tH", "sH", true)
             .Cases("wO", "wH", "S", "S2", "Sr", "Lr", true)
             .Cases("Lm", "Sc", "gLane", "mem", "Npack", true)
             .Cases("localTo", "sqrtN", "one", "bar", true)
             .Cases("Glogu", "hingeS", "hingeS2", true)
             .Cases("packedTraffic", "r16To", "R3", true)
             .Default(false))
      return invalid("unknown reduction feature");
    ReductionExpression e;
    e.kind = ReductionExpression::Kind::Feature;
    e.name = name.str();
    return e;
  }
  llvm::Expected<ReductionExpression> expression(unsigned minimum,
                                                 unsigned depth) {
    auto lhs = primary(depth);
    if (!lhs)
      return lhs.takeError();
    while (true) {
      space();
      llvm::StringRef token, name;
      unsigned precedence = 0;
      for (auto t :
           {"**", "<=", ">=", "==", "!=", "+", "-", "*", "/", "<", ">"}) {
        if (!text.starts_with(t))
          continue;
        token = t;
        name = llvm::StringSwitch<llvm::StringRef>(token)
                   .Case("**", "pow")
                   .Case("+", "add")
                   .Case("-", "sub")
                   .Case("*", "mul")
                   .Case("/", "div")
                   .Case("<", "lt")
                   .Case("<=", "le")
                   .Case(">", "gt")
                   .Case(">=", "ge")
                   .Case("==", "eq")
                   .Case("!=", "ne")
                   .Default("");
        precedence = token == "**"                  ? 4
                     : token == "*" || token == "/" ? 3
                     : token == "+" || token == "-" ? 2
                                                    : 1;
        break;
      }
      if (!precedence || precedence < minimum)
        break;
      if (depth > 32 || ++nodes > 1024)
        return invalid("reduction expression is too large");
      text = text.drop_front(token.size());
      auto rhs = expression(precedence + (token == "**" ? 0 : 1), depth + 1);
      if (!rhs)
        return rhs.takeError();
      std::vector<ReductionExpression> args;
      args.push_back(std::move(*lhs));
      args.push_back(std::move(*rhs));
      lhs = operation(name, std::move(args));
      if (!lhs)
        return lhs.takeError();
    }
    return std::move(*lhs);
  }

  static bool validDepth(const ReductionExpression &e, unsigned depth = 0) {
    return depth <= 32 && llvm::all_of(e.arguments, [depth](const auto &arg) {
             return validDepth(arg, depth + 1);
           });
  }

public:
  explicit ExpressionParser(llvm::StringRef input) : text(input) {}
  llvm::Expected<ReductionExpression> parse() {
    if (text.size() > 16384)
      return invalid("reduction expression text is too long");
    auto e = expression(0, 0);
    if (!e)
      return e.takeError();
    space();
    if (!text.empty())
      return invalid("trailing reduction expression text");
    if (!validDepth(*e))
      return invalid("reduction expression tree is too deep");
    return std::move(*e);
  }
};
llvm::Expected<ReductionExpression>
parseExpression(const llvm::json::Value &v) {
  auto text = v.getAsString();
  if (!text)
    return invalid("reduction expression must be a string");
  return ExpressionParser(*text).parse();
}
} // namespace

namespace {
// Validate metadata and segment structure eagerly, without parsing expressions.
llvm::Expected<ReductionFunction> validateFunction(const llvm::json::Object &f,
                                                   llvm::StringRef family) {
  const auto *warps = f.getArray("supported_warps");
  const auto *segments = f.getArray("segments");
  if (!warps || warps->empty() || !segments || segments->empty() ||
      segments->size() > 128)
    return invalid("invalid reduction function");
  ReductionFunction function;
  if (family == "middle_axis") {
    auto policy = f.getString("layout_policy_id");
    if (!policy || *policy != "row_major_128bit_v1")
      return invalid("unsupported middle-axis layout policy");
    function.layoutPolicy = policy->str();
  }
  for (const auto &w : *warps) {
    auto n = w.getAsInteger();
    if (!n || *n < 1 || *n > 64 || (*n & (*n - 1)) ||
        llvm::is_contained(function.supportedWarps, *n))
      return invalid("invalid reduction warp list");
    function.supportedWarps.push_back(*n);
  }
  for (const auto &s : *segments) {
    const auto *object = s.getAsObject();
    if (!object)
      return invalid("invalid reduction segment");
    auto id = object->getString("id");
    auto lane = object->getInteger("lane_width");
    auto cond = object->getString("condition");
    auto expr = object->getString("expression");
    if (!id || !lane || *lane <= 0 || !cond || !expr)
      return invalid("incomplete reduction segment or non-string expression");
    if (cond->size() > 16384 || expr->size() > 16384)
      return invalid("reduction expression text is too long");
  }
  return function;
}
struct LazyReductionFunction {
  const llvm::json::Object *raw;
  ReductionFunction function;
  std::once_flag once;
  std::string error;
  LazyReductionFunction(const llvm::json::Object *raw,
                        ReductionFunction function)
      : raw(raw), function(std::move(function)) {}
};
} // namespace

// The raw JSON never changes after indexing. Entries reference its leaves, and
// copies of ReductionProfile share the same once-initialized immutable results.
struct ReductionProfile::State {
  llvm::json::Value raw = nullptr;
  std::string path;
  llvm::StringMap<std::unique_ptr<LazyReductionFunction>> functions;
  std::atomic<size_t> parsed{0};

  llvm::Error add(llvm::StringRef family, llvm::StringRef mode,
                  llvm::StringRef op, llvm::StringRef dtype,
                  const llvm::json::Object &leaf) {
    if ((mode != "simd" && mode != "simt") ||
        !llvm::StringSwitch<bool>(op)
             .Cases("sum", "max", "min", "prod", true)
             .Cases("xor", "and", "or", true)
             .Default(false) ||
        !llvm::StringSwitch<bool>(dtype)
             .Cases("bool", "f16", "bf16", "f32", true)
             .Cases("i8", "i16", "i32", "i64", true)
             .Cases("u8", "u16", "u32", "u64", true)
             .Default(false))
      return invalid("invalid reduction mode, op or dtype");
    auto function = validateFunction(leaf, family);
    if (!function)
      return function.takeError();
    auto name =
        key(family, mode == "simd" ? ReductionMode::SIMD : ReductionMode::SIMT,
            op, dtype);
    if (!functions
             .try_emplace(name, std::make_unique<LazyReductionFunction>(
                                    &leaf, std::move(*function)))
             .second)
      return invalid("duplicate reduction function");
    return llvm::Error::success();
  }

  llvm::Error index(const llvm::json::Value &value, llvm::StringRef family) {
    size_t count = 0;
    if (const auto *array = value.getAsArray()) {
      if (array->size() > 1024)
        return invalid("too many reduction functions");
      for (const auto &v : *array) {
        const auto *f = v.getAsObject();
        if (!f || f->getString("family") != family || !f->getString("mode") ||
            !f->getString("op") || !f->getString("dtype"))
          return invalid("invalid reduction function record");
        if (auto error = add(family, *f->getString("mode"), *f->getString("op"),
                             *f->getString("dtype"), *f))
          return error;
      }
      return llvm::Error::success();
    }
    const auto *modes = value.getAsObject();
    if (!modes)
      return invalid(
          "reduction family must be an array or mode/op/dtype object");
    for (const auto &mode : *modes) {
      if (mode.first != "simd" && mode.first != "simt")
        return invalid("unsupported reduction mode");
      const auto *ops = mode.second.getAsObject();
      if (!ops)
        return invalid("reduction mode must contain op objects");
      for (const auto &op : *ops) {
        const auto *dtypes = op.second.getAsObject();
        if (!dtypes)
          return invalid("reduction op must contain dtype objects");
        for (const auto &dtype : *dtypes) {
          const auto *leaf = dtype.second.getAsObject();
          if (!leaf || leaf->get("family") || leaf->get("mode") ||
              leaf->get("op") || leaf->get("dtype"))
            return invalid("reduction leaf must not repeat path fields");
          if (++count > 1024)
            return invalid("too many reduction functions");
          if (auto error =
                  add(family, mode.first, op.first, dtype.first, *leaf))
            return error;
        }
      }
    }
    return llvm::Error::success();
  }

  llvm::Expected<const ReductionFunction *> get(llvm::StringRef name) {
    auto it = functions.find(name);
    if (it == functions.end())
      return nullptr;
    auto &entry = *it->second;
    std::call_once(entry.once, [&] {
      std::vector<ReductionSegment> segments;
      for (const auto &v : *entry.raw->getArray("segments")) {
        const auto &s = *v.getAsObject();
        const auto id = *s.getString("id");
        auto fail = [&](llvm::Error error) {
          entry.error = path + ": " + name.str() + "/" + id.str() + ": " +
                        llvm::toString(std::move(error));
        };
        auto condition = parseExpression(*s.get("condition"));
        if (!condition) {
          fail(condition.takeError());
          return;
        }
        auto expression = parseExpression(*s.get("expression"));
        if (!expression) {
          fail(expression.takeError());
          return;
        }
        segments.push_back({id.str(), *s.getInteger("lane_width"),
                            std::move(*condition), std::move(*expression)});
      }
      entry.function.segments = std::move(segments);
      parsed.fetch_add(1, std::memory_order_relaxed);
    });
    if (!entry.error.empty())
      return invalid(entry.error);
    return &entry.function;
  }
};

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
  profile.state = std::make_shared<State>();
  profile.state->path = path.str();
  profile.state->raw = std::move(*parsed);
  // Reacquire pointers after moving the owning JSON value.
  families = profile.state->raw.getAsObject()->getObject("families");
  for (llvm::StringRef family : {"first_axis", "middle_axis"}) {
    if (const auto *value = families->get(family))
      if (auto error = profile.state->index(*value, family))
        return std::move(error);
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

llvm::Expected<const ReductionFunction *>
ReductionProfile::findFunction(llvm::StringRef family, ReductionMode mode,
                               llvm::StringRef op,
                               llvm::StringRef dtype) const {
  if (!state)
    return nullptr;
  return state->get(key(family, mode, op, dtype));
}

llvm::Expected<const ReductionFunction *>
ReductionProfile::findFirstAxis(ReductionMode mode, llvm::StringRef op,
                                llvm::StringRef dtype) const {
  return findFunction("first_axis", mode, op, dtype);
}

llvm::Expected<const ReductionFunction *>
ReductionProfile::findMiddleAxis(ReductionMode mode, llvm::StringRef op,
                                 llvm::StringRef dtype) const {
  return findFunction("middle_axis", mode, op, dtype);
}

llvm::Error ReductionProfile::validateAllFunctions() const {
  if (state)
    for (const auto &entry : state->functions) {
      auto function = state->get(entry.first());
      if (!function)
        return function.takeError();
    }
  return llvm::Error::success();
}

size_t ReductionProfile::getParsedFunctionCount() const {
  return state ? state->parsed.load(std::memory_order_relaxed) : 0;
}
