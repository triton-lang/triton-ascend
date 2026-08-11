//===- SimdSimtCostModel.cpp - Ascend SIMD/SIMT candidate model ----------===//
//
// The numerical model in this file is the versioned C++ candidate model.  It
// intentionally produces a relative per-program selection score, not an
// end-to-end kernel-time prediction.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Profile/MicrobenchmarkProfile.h"
#include "AscendModel/RouteModel/SimtAnchorAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

constexpr llvm::StringLiteral kAllSimd = "all_simd";
constexpr llvm::StringLiteral kAllSimtOnly = "all_simt_only";
constexpr llvm::StringLiteral kMixedSimdSimt = "mixed_simd_simt";

struct OpProfile {
  double throughput = 0.0;
  double factor = 1.0;
  std::string confidence = "none";
};

struct CoverageProfile {
  double minimumIrregularDensity = 0.0;
  int64_t tinyDotFlopsMax = 0;
  int64_t tinyDotMaxTensorNumel = 0;
  int64_t rowwiseLoopTripSumMax = 0;
  int64_t rowwiseMaskRankSumMax = 0;
  int64_t rowwiseWeightedReductionsMax = 0;
  int64_t rowwiseMaxTensorNumel = 0;
  int64_t rank1WeightedReductionsMax = 0;
  int64_t rank1MaxTensorNumel = 0;
  int64_t triangularLoopCountMax = 0;
  int64_t triangularLoopTripSumMax = 0;
  int64_t triangularMaskRankSumMax = 0;
  int64_t triangularWeightedReductionsMax = 0;
  int64_t triangularMaxTensorNumel = 0;
};

struct StructuralProfile {
  double irregularPerDensity = 0.0;
  double irregularCap = 0.0;
  double tinyDotIrregularPerDensity = 0.0;
  double tinyDotIrregularCap = 0.0;
  double perMaskRank = 0.0;
  double maskCap = 0.0;
  double perWeightedReduction = 0.0;
  double reductionCap = 0.0;
  double perStaticLoopTrip = 0.0;
  double loopCap = 0.0;
  double controlFlow = 0.0;
  double rank1IndirectVectorReduction = 0.0;
  double tinyDot = 0.0;
  int64_t tinyDotFlopsMax = 0;
};

struct MixedSetupFallbackProfile {
  int64_t numWarps = 0;
  double emptySimtSetupCycles = 0.0;
};

struct EventRouteCalibrationProfile {
  double allSimdMultiplier = 1.0;
  double allSimtOnlyMultiplier = 1.0;
  double mixedSimdSimtMultiplier = 1.0;
  bool allSimtOnlyValidated = false;
  bool mixedSimdSimtValidated = false;
  std::string source;
  std::string confidence = "none";
};

struct CandidateProfile {
  std::string profileVersion;
  std::string target;
  std::vector<std::string> compatibleTargets;
  std::string scoreUnit;
  std::string minimumConfidence = "medium";
  std::string contentSha256;
  std::string selectionContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkContentSha256;

  double programIssueScale = 1.0;
  std::string rankingConfidence = "low";
  std::string calibrationSource;
  CoverageProfile coverage;
  StructuralProfile structural;
  llvm::StringMap<EventRouteCalibrationProfile> eventRouteCalibration;

  int64_t simdVectorWidthBits = 2048;
  double simdSetupCycles = 0.0;
  llvm::StringMap<OpProfile> simdOps;
  double simdMte2BytesPerCycle = 0.0;
  double simdMte3BytesPerCycle = 0.0;
  std::string simdMemoryConfidence = "none";
  double simdDotSetupCycles = 0.0;
  double simdDotFlopsPerCycle = 0.0;
  std::string simdDotConfidence = "none";

  int64_t simtWarpSize = 32;
  double simtSetupCycles = 0.0;
  std::string simtSetupConfidence = "none";
  llvm::StringMap<OpProfile> simtOps;
  double simtDotSetupCycles = 0.0;
  double simtDotFlopsPerCycle = 0.0;
  std::string simtDotConfidence = "none";
  double simtPredicateRate = 0.0;
  double simtShuffleRate = 0.0;
  std::string simtShuffleConfidence = "none";
  double simtLoadWarpRate = 0.0;
  double simtStoreWarpRate = 0.0;
  std::string simtMemoryConfidence = "none";
  std::vector<MixedSetupFallbackProfile> mixedSetupFallbacks;
  std::string mixedSetupFallbackConfidence = "none";
};

/// Small fail-fast facade around llvm::json.  It permits a readable profile
/// parser while retaining a single actionable error message.
class ProfileJSONReader {
public:
  const llvm::json::Object *object(const llvm::json::Object &parent,
                                   llvm::StringRef key,
                                   llvm::StringRef context) {
    if (failed())
      return nullptr;
    if (const auto *value = parent.getObject(key))
      return value;
    setError(context + "." + key + " must be an object");
    return nullptr;
  }

  const llvm::json::Array *array(const llvm::json::Object &parent,
                                 llvm::StringRef key, llvm::StringRef context) {
    if (failed())
      return nullptr;
    if (const auto *value = parent.getArray(key))
      return value;
    setError(context + "." + key + " must be an array");
    return nullptr;
  }

  double number(const llvm::json::Object &parent, llvm::StringRef key,
                llvm::StringRef context) {
    if (failed())
      return 0.0;
    if (auto value = parent.getNumber(key))
      return *value;
    setError(context + "." + key + " must be a number");
    return 0.0;
  }

  int64_t integer(const llvm::json::Object &parent, llvm::StringRef key,
                  llvm::StringRef context) {
    if (failed())
      return 0;
    if (auto value = parent.getInteger(key))
      return *value;
    setError(context + "." + key + " must be an integer");
    return 0;
  }

  std::string string(const llvm::json::Object &parent, llvm::StringRef key,
                     llvm::StringRef context) {
    if (failed())
      return {};
    if (auto value = parent.getString(key))
      return value->str();
    setError(context + "." + key + " must be a string");
    return {};
  }

  std::string optionalString(const llvm::json::Object &parent,
                             llvm::StringRef key,
                             llvm::StringRef defaultValue = {}) {
    if (auto value = parent.getString(key))
      return value->str();
    return defaultValue.str();
  }

  double optionalNumber(const llvm::json::Object &parent, llvm::StringRef key,
                        double defaultValue) {
    if (auto value = parent.getNumber(key))
      return *value;
    return defaultValue;
  }

  int64_t optionalInteger(const llvm::json::Object &parent, llvm::StringRef key,
                          int64_t defaultValue) {
    if (auto value = parent.getInteger(key))
      return *value;
    return defaultValue;
  }

  bool failed() const { return !error.empty(); }
  llvm::StringRef getError() const { return error; }

  void setError(const llvm::Twine &message) {
    if (error.empty())
      error = message.str();
  }

private:
  std::string error;
};

static llvm::json::Object toJSON(const llvm::StringMap<int64_t> &values) {
  llvm::json::Object result;
  for (const auto &entry : values)
    result[entry.first()] = entry.second;
  return result;
}

static llvm::json::Object toJSON(const llvm::StringMap<double> &values) {
  llvm::json::Object result;
  for (const auto &entry : values)
    result[entry.first()] = entry.second;
  return result;
}

template <typename SummaryT>
static void initializeWorkMaps(SummaryT &features) {
  for (llvm::StringRef key :
       {"load", "store", "reduce", "scan", "gather", "histogram", "atomic",
        "add", "sub", "mul", "div", "max", "abs", "exp", "log", "cmp", "select",
        "cast", "clamp"}) {
    features.weightedOps[key] = 0;
    features.opElements[key] = 0;
  }
}

static std::string typeToString(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << type;
  os.flush();
  return text;
}

static bool isPointerType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return llvm::StringRef(typeToString(type)).contains("!tt.ptr");
}

static Type getElementType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}

static int64_t parseTypeBitWidth(llvm::StringRef text) {
  for (size_t index = 0; index + 1 < text.size(); ++index) {
    if (text[index] != 'f' && text[index] != 'i')
      continue;
    size_t end = index + 1;
    while (end < text.size() && llvm::isDigit(text[end]))
      ++end;
    if (end == index + 1)
      continue;
    int64_t width = 0;
    if (!text.slice(index + 1, end).getAsInteger(10, width) && width > 0)
      return width;
  }
  return 0;
}

static int64_t getTypeBitWidth(Type type, int64_t defaultWidth = 32) {
  type = getElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  int64_t parsed = parseTypeBitWidth(typeToString(type));
  return parsed > 0 ? parsed : defaultWidth;
}

static bool isMaskTensorType(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return false;
  auto integer = dyn_cast<IntegerType>(tensor.getElementType());
  return integer && integer.getWidth() == 1;
}

static int64_t getStaticNumElements(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return 1;
  int64_t count = 1;
  for (int64_t dim : tensor.getShape()) {
    if (ShapedType::isDynamic(dim) || dim <= 0)
      return 1;
    if (count > std::numeric_limits<int64_t>::max() / dim)
      return std::numeric_limits<int64_t>::max();
    count *= dim;
  }
  return std::max<int64_t>(1, count);
}

static double getStaticTensorBytes(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return 0.0;
  return static_cast<double>(getStaticNumElements(type)) *
         getTypeBitWidth(tensor.getElementType()) / 8.0;
}

static int64_t getOperationElements(Operation *op) {
  int64_t elements = 1;
  for (Type type : op->getOperandTypes())
    elements = std::max(elements, getStaticNumElements(type));
  for (Type type : op->getResultTypes())
    elements = std::max(elements, getStaticNumElements(type));
  return elements;
}

static std::optional<int64_t> getConstantInteger(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getName().getStringRef() != "arith.constant")
    return std::nullopt;
  if (auto integer = definingOp->getAttrOfType<IntegerAttr>("value"))
    return integer.getInt();
  return std::nullopt;
}

static std::optional<int64_t> getKnownStaticLoopTripCount(Operation *op) {
  if (!op || op->getName().getStringRef() != "scf.for" ||
      op->getNumOperands() < 3)
    return std::nullopt;
  auto lower = getConstantInteger(op->getOperand(0));
  auto upper = getConstantInteger(op->getOperand(1));
  auto step = getConstantInteger(op->getOperand(2));
  if (!lower || !upper || !step || *step == 0)
    return std::nullopt;
  int64_t span = *upper - *lower;
  if (span > 0 && *step > 0)
    return std::max<int64_t>(1, (span + *step - 1) / *step);
  if (span < 0 && *step < 0) {
    int64_t positiveSpan = -span;
    int64_t positiveStep = -*step;
    return std::max<int64_t>(1,
                             (positiveSpan + positiveStep - 1) / positiveStep);
  }
  return std::nullopt;
}

static int64_t getModeledLoopTripCount(
    Operation *op,
    const llvm::DenseMap<Operation *, int64_t> &structuralTripEstimates) {
  if (auto knownTripCount = getKnownStaticLoopTripCount(op))
    return *knownTripCount;
  if (auto iterator = structuralTripEstimates.find(op);
      iterator != structuralTripEstimates.end())
    return iterator->second;
  return 1;
}

static int64_t getLoopMultiplier(
    Operation *op,
    const llvm::DenseMap<Operation *, int64_t> &structuralTripEstimates) {
  int64_t multiplier = 1;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() != "scf.for")
      continue;
    int64_t tripCount =
        getModeledLoopTripCount(parent, structuralTripEstimates);
    if (tripCount > 0 &&
        multiplier <= std::numeric_limits<int64_t>::max() / tripCount)
      multiplier *= tripCount;
  }
  return multiplier;
}

static bool isCastOp(llvm::StringRef name) {
  return name.starts_with("arith.ext") || name.starts_with("arith.trunc") ||
         name == "arith.sitofp" || name == "arith.uitofp" ||
         name == "arith.fptosi" || name == "arith.fptoui" ||
         name.starts_with("arith.index_cast");
}

static llvm::StringRef classifyWeightedOp(llvm::StringRef name) {
  if (name == "tt.load")
    return "load";
  if (name == "tt.store")
    return "store";
  if (name == "tt.reduce")
    return "reduce";
  if (name == "tt.scan" || name == "tt.associative_scan")
    return "scan";
  if (name == "tt.gather")
    return "gather";
  if (name == "tt.histogram")
    return "histogram";
  if (name.starts_with("tt.atomic"))
    return "atomic";
  if (name == "arith.addf" || name == "arith.addi")
    return "add";
  if (name == "arith.subf" || name == "arith.subi")
    return "sub";
  if (name == "arith.mulf" || name == "arith.muli")
    return "mul";
  if (name == "arith.divf" || name == "arith.divsi" || name == "arith.divui")
    return "div";
  if (name == "arith.maxnumf" || name == "arith.maxf" ||
      name == "arith.maxsi" || name == "arith.maxui")
    return "max";
  if (name == "math.absf" || name == "math.absi")
    return "abs";
  if (name == "math.exp")
    return "exp";
  if (name == "math.log")
    return "log";
  if (name == "arith.cmpf" || name == "arith.cmpi")
    return "cmp";
  if (name == "arith.select")
    return "select";
  if (isCastOp(name))
    return "cast";
  if (name.starts_with("tt.clamp"))
    return "clamp";
  return {};
}

static void appendUnique(std::vector<std::string> &values,
                         llvm::StringRef value) {
  if (llvm::find(values, value.str()) == values.end())
    values.push_back(value.str());
}

static int confidenceRank(llvm::StringRef confidence) {
  if (confidence == "high")
    return 3;
  if (confidence == "medium")
    return 2;
  if (confidence == "low")
    return 1;
  return 0;
}

static std::string minimumConfidence(llvm::ArrayRef<std::string> values) {
  if (values.empty())
    return "none";
  return *std::min_element(values.begin(), values.end(),
                           [](const std::string &lhs, const std::string &rhs) {
                             return confidenceRank(lhs) < confidenceRank(rhs);
                           });
}

static bool wildcardMatch(llvm::StringRef pattern, llvm::StringRef value) {
  size_t patternIndex = 0;
  size_t valueIndex = 0;
  size_t starIndex = llvm::StringRef::npos;
  size_t retryValueIndex = 0;
  while (valueIndex < value.size()) {
    if (patternIndex < pattern.size() &&
        (pattern[patternIndex] == '?' ||
         pattern[patternIndex] == value[valueIndex])) {
      ++patternIndex;
      ++valueIndex;
      continue;
    }
    if (patternIndex < pattern.size() && pattern[patternIndex] == '*') {
      starIndex = patternIndex++;
      retryValueIndex = valueIndex;
      continue;
    }
    if (starIndex != llvm::StringRef::npos) {
      patternIndex = starIndex + 1;
      valueIndex = ++retryValueIndex;
      continue;
    }
    return false;
  }
  while (patternIndex < pattern.size() && pattern[patternIndex] == '*')
    ++patternIndex;
  return patternIndex == pattern.size();
}

static bool targetMatches(const CandidateProfile &profile,
                          llvm::StringRef actualTarget) {
  if (actualTarget.trim().empty())
    return true;
  std::string actual = actualTarget.trim().lower();
  std::vector<std::string> patterns = profile.compatibleTargets;
  patterns.push_back(profile.target);
  for (std::string pattern : patterns) {
    std::replace(pattern.begin(), pattern.end(), ':', '/');
    llvm::SmallVector<llvm::StringRef> aliases;
    llvm::StringRef(pattern).split(aliases, '/', -1, false);
    for (llvm::StringRef alias : aliases) {
      std::string lower = alias.trim().lower();
      if (!lower.empty() && wildcardMatch(lower, actual))
        return true;
    }
  }
  return false;
}

static double resolveNumberOrMeasurement(
    const llvm::json::Object &object, llvm::StringRef numberKey,
    llvm::StringRef measurementKey, llvm::StringRef expectedUnit,
    const MicrobenchmarkProfile *microbench, ProfileJSONReader &reader,
    llvm::StringRef context, std::string *measurementConfidence = nullptr) {
  if (auto reference = object.getString(measurementKey)) {
    if (!microbench) {
      reader.setError(context + "." + measurementKey +
                      " requires microbenchmark_profile");
      return 0.0;
    }
    llvm::StringRef expectedCycleDomain = "none";
    if (expectedUnit == "system_cycle" ||
        expectedUnit.ends_with("/system_cycle"))
      expectedCycleDomain = "SYS_CNT";
    auto value =
        microbench->requireValue(*reference, expectedUnit, expectedCycleDomain);
    if (!value) {
      reader.setError(llvm::toString(value.takeError()));
      return 0.0;
    }
    if (measurementConfidence) {
      const MicrobenchmarkMeasurement *measurement =
          microbench->getMeasurement(*reference);
      *measurementConfidence = measurement ? measurement->confidence : "none";
    }
    return *value;
  }
  return reader.number(object, numberKey, context);
}

static OpProfile resolveOpProfile(const llvm::json::Object &ops,
                                  llvm::StringRef opName,
                                  llvm::StringRef throughputKey,
                                  llvm::StringRef expectedUnit,
                                  const MicrobenchmarkProfile *microbench,
                                  ProfileJSONReader &reader) {
  OpProfile result;
  const llvm::json::Value *raw = ops.get(opName);
  if (!raw) {
    reader.setError("missing operation profile " + opName);
    return result;
  }
  const auto *op = raw->getAsObject();
  if (!op) {
    reader.setError("operation profile " + opName + " must be an object");
    return result;
  }
  if (auto relative = op->getString("relative_to")) {
    OpProfile base = resolveOpProfile(ops, *relative, throughputKey,
                                      expectedUnit, microbench, reader);
    result.throughput = base.throughput;
    result.factor = reader.optionalNumber(*op, "factor", 1.0);
    result.confidence = reader.optionalString(*op, "confidence", "low");
    return result;
  }
  std::string measuredConfidence = "none";
  result.throughput = resolveNumberOrMeasurement(
      *op, throughputKey, "throughput_measurement", expectedUnit, microbench,
      reader, opName, &measuredConfidence);
  result.factor = reader.optionalNumber(*op, "factor", 1.0);
  result.confidence =
      reader.optionalString(*op, "confidence", measuredConfidence);
  return result;
}

/// Match Python's json.dumps(value, sort_keys=True) representation.  Keeping
/// this stable makes profile_content_sha256 identical across the temporary
/// Python model and this C++ implementation.
static void emitPythonCanonicalJSON(const llvm::json::Value &value,
                                    llvm::raw_ostream &os) {
  if (const auto *object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto &entry : *object)
      keys.push_back(entry.first);
    llvm::sort(keys);
    os << '{';
    bool first = true;
    for (llvm::StringRef key : keys) {
      if (!first)
        os << ", ";
      first = false;
      os << llvm::json::Value(key.str()) << ": ";
      emitPythonCanonicalJSON(*object->get(key), os);
    }
    os << '}';
    return;
  }
  if (const auto *array = value.getAsArray()) {
    os << '[';
    bool first = true;
    for (const llvm::json::Value &element : *array) {
      if (!first)
        os << ", ";
      first = false;
      emitPythonCanonicalJSON(element, os);
    }
    os << ']';
    return;
  }
  os << value;
}

static std::string resolveProfileReference(llvm::StringRef ownerPath,
                                           llvm::StringRef reference) {
  if (llvm::sys::path::is_absolute(reference))
    return reference.str();
  llvm::SmallString<256> resolved(ownerPath);
  llvm::sys::path::remove_filename(resolved);
  llvm::sys::path::append(resolved, reference);
  llvm::sys::path::remove_dots(resolved, true);
  return resolved.str().str();
}

static llvm::Expected<CandidateProfile>
loadCandidateProfile(llvm::StringRef requestedPath) {
  std::string path = requestedPath.empty() ? getDefaultSimdSimtProfilePath()
                                           : requestedPath.str();
  if (path.empty())
    return llvm::createStringError(std::errc::no_such_file_or_directory,
                                   "SIMD/SIMT profile path is empty; set "
                                   "TRITON_ASCEND_SIMD_SIMT_PROFILE");

  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read SIMD/SIMT profile '%s'",
                                   path.c_str());
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse SIMD/SIMT profile '%s': %s",
                                   path.c_str(),
                                   llvm::toString(parsed.takeError()).c_str());
  const auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile root must be an object");
  auto selectionSchemaVersion = root->getInteger("schema_version");
  if (!selectionSchemaVersion ||
      (*selectionSchemaVersion != 1 && *selectionSchemaVersion != 2 &&
       *selectionSchemaVersion != 3 && *selectionSchemaVersion != 4))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile schema_version must be 1, 2, 3, or 4");

  CandidateProfile profile;
  ProfileJSONReader reader;
  std::optional<MicrobenchmarkProfile> microbenchmarkProfile;
  if (auto reference = root->getString("microbenchmark_profile")) {
    std::string resolved = resolveProfileReference(path, *reference);
    auto loaded = MicrobenchmarkProfile::loadFromFile(resolved);
    if (!loaded)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to load shared microbenchmark profile referenced by '%s': %s",
          path.c_str(), llvm::toString(loaded.takeError()).c_str());
    microbenchmarkProfile.emplace(std::move(*loaded));
    profile.microbenchmarkProfileVersion =
        microbenchmarkProfile->getProfileVersion().str();
    profile.microbenchmarkProfileTarget =
        microbenchmarkProfile->getTarget().str();
    profile.microbenchmarkContentSha256 =
        microbenchmarkProfile->getContentSha256().str();
  }
  const MicrobenchmarkProfile *microbench =
      microbenchmarkProfile ? &*microbenchmarkProfile : nullptr;

  profile.profileVersion = reader.string(*root, "profile_version", "profile");
  const bool usesAnchorPartitionProfile =
      profile.profileVersion == "david-v100-simd-simt-20260730-v7" ||
      profile.profileVersion == "david-v100-simd-simt-20260803-v8" ||
      profile.profileVersion == "david-v100-simd-simt-20260804-v9" ||
      profile.profileVersion == "david-v100-simd-simt-20260804-v10" ||
      profile.profileVersion == "david-v100-simd-simt-20260806-v11";
  profile.target = reader.string(*root, "target", "profile");
  if (microbench && llvm::StringRef(profile.target) != microbench->getTarget())
    reader.setError("selection profile target '" + profile.target +
                    "' does not match shared microbenchmark target '" +
                    microbench->getTarget().str() + "'");
  profile.scoreUnit = reader.string(*root, "score_unit", "profile");
  if (const auto *targets =
          reader.array(*root, "compatible_targets", "profile")) {
    for (const llvm::json::Value &target : *targets) {
      if (auto text = target.getAsString())
        profile.compatibleTargets.push_back(text->str());
      else
        reader.setError("profile.compatible_targets entries must be strings");
    }
  }
  if (const auto *policy = reader.object(*root, "policy", "profile"))
    profile.minimumConfidence = reader.optionalString(
        *policy, "minimum_confidence_for_decision", "medium");

  const auto *calibration =
      reader.object(*root, "selection_calibration", "profile");
  if (calibration) {
    profile.programIssueScale = reader.number(
        *calibration, "program_issue_scale", "profile.selection_calibration");
    profile.rankingConfidence =
        reader.optionalString(*calibration, "ranking_confidence", "low");
    profile.calibrationSource =
        reader.optionalString(*calibration, "source", "");

    if (const auto *coverage = reader.object(*calibration, "coverage",
                                             "profile.selection_calibration")) {
      profile.coverage.minimumIrregularDensity =
          reader.number(*coverage, "minimum_irregular_density", "coverage");
      profile.coverage.tinyDotFlopsMax =
          reader.integer(*coverage, "tiny_dot_flops_max", "coverage");
      profile.coverage.tinyDotMaxTensorNumel =
          reader.integer(*coverage, "tiny_dot_max_tensor_numel", "coverage");
      profile.coverage.rowwiseLoopTripSumMax =
          reader.integer(*coverage, "rowwise_loop_trip_sum_max", "coverage");
      profile.coverage.rowwiseMaskRankSumMax =
          reader.integer(*coverage, "rowwise_mask_rank_sum_max", "coverage");
      profile.coverage.rowwiseWeightedReductionsMax = reader.integer(
          *coverage, "rowwise_weighted_reductions_max", "coverage");
      profile.coverage.rowwiseMaxTensorNumel =
          reader.integer(*coverage, "rowwise_max_tensor_numel", "coverage");
      profile.coverage.rank1WeightedReductionsMax = reader.integer(
          *coverage, "rank1_weighted_reductions_max", "coverage");
      profile.coverage.rank1MaxTensorNumel =
          reader.integer(*coverage, "rank1_max_tensor_numel", "coverage");
      // These fields were introduced by v11. Keep bounded defaults only for
      // reading older diagnostic profiles; the v11 JSON schema requires all
      // five values explicitly.
      profile.coverage.triangularLoopCountMax =
          reader.optionalInteger(*coverage, "triangular_loop_count_max", 4);
      profile.coverage.triangularLoopTripSumMax =
          reader.optionalInteger(*coverage, "triangular_loop_trip_sum_max", 56);
      profile.coverage.triangularMaskRankSumMax =
          reader.optionalInteger(*coverage, "triangular_mask_rank_sum_max", 64);
      profile.coverage.triangularWeightedReductionsMax = reader.optionalInteger(
          *coverage, "triangular_weighted_reductions_max", 56);
      profile.coverage.triangularMaxTensorNumel =
          reader.optionalInteger(*coverage, "triangular_max_tensor_numel", 256);
    }

    if (const auto *structural =
            reader.object(*calibration, "simd_structural_penalty_ratio",
                          "profile.selection_calibration")) {
      profile.structural.irregularPerDensity =
          reader.number(*structural, "irregular_per_density", "structural");
      profile.structural.irregularCap =
          reader.number(*structural, "irregular_cap", "structural");
      profile.structural.tinyDotIrregularPerDensity = reader.number(
          *structural, "tiny_dot_irregular_per_density", "structural");
      profile.structural.tinyDotIrregularCap =
          reader.number(*structural, "tiny_dot_irregular_cap", "structural");
      profile.structural.perMaskRank =
          reader.number(*structural, "per_mask_rank", "structural");
      profile.structural.maskCap =
          reader.number(*structural, "mask_cap", "structural");
      profile.structural.perWeightedReduction =
          reader.number(*structural, "per_weighted_reduction", "structural");
      profile.structural.reductionCap =
          reader.number(*structural, "reduction_cap", "structural");
      profile.structural.perStaticLoopTrip =
          reader.number(*structural, "per_static_loop_trip", "structural");
      profile.structural.loopCap =
          reader.number(*structural, "loop_cap", "structural");
      profile.structural.controlFlow =
          reader.number(*structural, "control_flow", "structural");
      profile.structural.rank1IndirectVectorReduction = reader.number(
          *structural, "rank1_indirect_vector_reduction", "structural");
      profile.structural.tinyDot =
          reader.number(*structural, "tiny_dot", "structural");
      profile.structural.tinyDotFlopsMax =
          reader.integer(*structural, "tiny_dot_flops_max", "structural");
    }

    if (calibration->get("event_route_score_multiplier")) {
      const auto *eventCalibration =
          reader.object(*calibration, "event_route_score_multiplier",
                        "profile.selection_calibration");
      if (!eventCalibration)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "invalid event_route_score_multiplier object");
      if (const auto *domains = reader.object(*eventCalibration, "domains",
                                              "event_route_score_multiplier")) {
        for (const auto &entry : *domains) {
          const auto *domain = entry.second.getAsObject();
          if (!domain) {
            reader.setError("event_route_score_multiplier.domains." +
                            entry.first.str() + " must be an object");
            continue;
          }
          EventRouteCalibrationProfile route;
          const std::string context =
              "event_route_score_multiplier.domains." + entry.first.str();
          route.allSimdMultiplier = reader.number(*domain, "all_simd", context);
          route.allSimtOnlyMultiplier =
              reader.number(*domain, "all_simt_only", context);
          route.mixedSimdSimtMultiplier =
              reader.number(*domain, "mixed_simd_simt", context);
          auto allSimtValidated = domain->getBoolean("all_simt_only_validated");
          if (!allSimtValidated) {
            reader.setError(context +
                            ".all_simt_only_validated must be a boolean");
          } else {
            route.allSimtOnlyValidated = *allSimtValidated;
          }
          auto mixedValidated = domain->getBoolean("mixed_simd_simt_validated");
          if (!mixedValidated) {
            reader.setError(context +
                            ".mixed_simd_simt_validated must be a boolean");
          } else {
            route.mixedSimdSimtValidated = *mixedValidated;
          }
          route.source = reader.string(*domain, "source", context);
          route.confidence = reader.string(*domain, "confidence", context);
          profile.eventRouteCalibration[entry.first.str()] = std::move(route);
        }
      }
    }
  }

  const auto *simd = reader.object(*root, "simd", "profile");
  if (simd) {
    if (simd->getString("vector_width_measurement")) {
      profile.simdVectorWidthBits =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
              *simd, "vector_width_bits", "vector_width_measurement", "bit",
              microbench, reader, "simd")));
    } else {
      profile.simdVectorWidthBits =
          reader.integer(*simd, "vector_width_bits", "simd");
    }
    if (const auto *startup =
            reader.object(*simd, "startup_system_cycles", "simd"))
      profile.simdSetupCycles =
          reader.number(*startup, "vector", "simd.startup_system_cycles");
    if (const auto *ops = reader.object(*simd, "ops", "simd")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "predicate.cmp", "predicate.select",
            "convert.cast", "f32.clamp"})
        profile.simdOps[op] = resolveOpProfile(
            *ops, op, "throughput_vector_instructions_per_system_cycle",
            "vector_instruction/system_cycle", microbench, reader);
    }
    if (const auto *memory = reader.object(*simd, "memory", "simd")) {
      profile.simdMte2BytesPerCycle = reader.number(
          *memory, "vector_mte2_bytes_per_system_cycle", "simd.memory");
      profile.simdMte3BytesPerCycle =
          reader.number(*memory, "mte3_bytes_per_system_cycle", "simd.memory");
      profile.simdMemoryConfidence =
          reader.optionalString(*memory, "confidence", "none");
    }
    if (const auto *dot = reader.object(*simd, "dot", "simd")) {
      profile.simdDotSetupCycles =
          reader.number(*dot, "startup_system_cycles", "simd.dot");
      profile.simdDotFlopsPerCycle =
          reader.number(*dot, "flops_per_system_cycle", "simd.dot");
      profile.simdDotConfidence =
          reader.optionalString(*dot, "confidence", "none");
    }
  }

  const auto *simt = reader.object(*root, "simt", "profile");
  if (simt) {
    if (simt->getString("warp_size_measurement")) {
      profile.simtWarpSize =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
              *simt, "warp_size", "warp_size_measurement", "lane", microbench,
              reader, "simt")));
    } else {
      profile.simtWarpSize = reader.integer(*simt, "warp_size", "simt");
    }
    if (const auto *setup =
            reader.object(*simt, "setup_system_cycles", "simt")) {
      profile.simtSetupCycles = resolveNumberOrMeasurement(
          *setup, "empty_launch", "empty_launch_measurement", "system_cycle",
          microbench, reader, "simt.setup_system_cycles",
          &profile.simtSetupConfidence);
    }
    if (const auto *ops = reader.object(*simt, "ops", "simt")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "predicate.cmp", "predicate.select",
            "convert.cast", "f32.clamp"})
        profile.simtOps[op] =
            resolveOpProfile(*ops, op, "throughput_scalar_ops_per_system_cycle",
                             "scalar_op/system_cycle", microbench, reader);
    }
    if (const auto *dot = reader.object(*simt, "dot", "simt")) {
      profile.simtDotSetupCycles =
          reader.number(*dot, "startup_system_cycles", "simt.dot");
      profile.simtDotFlopsPerCycle =
          reader.number(*dot, "flops_per_system_cycle", "simt.dot");
      profile.simtDotConfidence =
          reader.optionalString(*dot, "confidence", "none");
    }
    if (const auto *camodel =
            reader.object(*simt, "camodel_effective", "simt")) {
      if (const auto *rates =
              reader.object(*camodel, "warp_instructions_per_system_cycle",
                            "simt.camodel_effective"))
        profile.simtPredicateRate =
            reader.number(*rates, "predicate", "simt.camodel_effective.rates");
    }
    if (const auto *shuffle = reader.object(*simt, "shuffle", "simt")) {
      std::string measuredConfidence;
      profile.simtShuffleRate = resolveNumberOrMeasurement(
          *shuffle, "warp_instructions_per_system_cycle",
          "throughput_measurement", "warp_instruction/system_cycle", microbench,
          reader, "simt.shuffle", &measuredConfidence);
      profile.simtShuffleConfidence =
          reader.optionalString(*shuffle, "confidence", measuredConfidence);
    }
    if (const auto *memory = reader.object(*simt, "memory", "simt")) {
      std::string loadConfidence;
      std::string storeConfidence;
      profile.simtLoadWarpRate = resolveNumberOrMeasurement(
          *memory, "load_warp_instructions_per_system_cycle",
          "load_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory", &loadConfidence);
      profile.simtStoreWarpRate = resolveNumberOrMeasurement(
          *memory, "store_warp_instructions_per_system_cycle",
          "store_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory", &storeConfidence);
      profile.simtMemoryConfidence = reader.optionalString(
          *memory, "confidence",
          minimumConfidence({loadConfidence, storeConfidence}));
    }
    const llvm::json::Object *mixedSetupFallback = nullptr;
    if (usesAnchorPartitionProfile)
      mixedSetupFallback = reader.object(*simt, "mixed_setup_fallback", "simt");
    else
      mixedSetupFallback = reader.object(*simt, "transition", "simt");
    if (mixedSetupFallback) {
      for (int64_t numWarps : {1, 2, 4, 8, 16, 32}) {
        std::string key = std::to_string(numWarps);
        const auto *entry = mixedSetupFallback->getObject(key);
        if (!entry)
          continue;
        std::string measuredConfidence;
        profile.mixedSetupFallbacks.push_back(
            {numWarps,
             resolveNumberOrMeasurement(
                 *entry, "empty_simt_setup_system_cycles", "measurement",
                 "system_cycle", microbench, reader,
                 "simt.mixed_setup_fallback." + key, &measuredConfidence)});
        if (profile.mixedSetupFallbackConfidence == "none")
          profile.mixedSetupFallbackConfidence = measuredConfidence;
      }
      profile.mixedSetupFallbackConfidence =
          reader.optionalString(*mixedSetupFallback, "confidence",
                                profile.mixedSetupFallbackConfidence);
    }
  }

  if (reader.failed())
    return llvm::createStringError(
        std::errc::invalid_argument, "invalid SIMD/SIMT profile '%s': %s",
        path.c_str(), reader.getError().str().c_str());
  if (profile.profileVersion != "david-v100-simd-simt-20260727-v3" &&
      profile.profileVersion != "david-v100-simd-simt-20260727-v4" &&
      profile.profileVersion != "david-v100-simd-simt-20260728-v5" &&
      profile.profileVersion != "david-v100-simd-simt-20260730-v6" &&
      profile.profileVersion != "david-v100-simd-simt-20260730-v7" &&
      profile.profileVersion != "david-v100-simd-simt-20260803-v8" &&
      profile.profileVersion != "david-v100-simd-simt-20260804-v9" &&
      profile.profileVersion != "david-v100-simd-simt-20260804-v10" &&
      profile.profileVersion != "david-v100-simd-simt-20260806-v11")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "unsupported SIMD/SIMT profile version '%s' "
        "(expected v3, v4, v5, v6, v7, v8, v9, v10, or v11)",
        profile.profileVersion.c_str());
  const bool usesSharedMicrobench =
      profile.profileVersion == "david-v100-simd-simt-20260728-v5" ||
      profile.profileVersion == "david-v100-simd-simt-20260730-v6" ||
      profile.profileVersion == "david-v100-simd-simt-20260730-v7" ||
      profile.profileVersion == "david-v100-simd-simt-20260803-v8" ||
      profile.profileVersion == "david-v100-simd-simt-20260804-v9" ||
      profile.profileVersion == "david-v100-simd-simt-20260804-v10" ||
      profile.profileVersion == "david-v100-simd-simt-20260806-v11";
  if (usesSharedMicrobench && !microbench)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT v5/v6/v7/v8/v9/v10/v11 profile must reference "
        "microbenchmark_profile");
  const bool usesEventCalibrationSchema =
      profile.profileVersion == "david-v100-simd-simt-20260804-v9" ||
      profile.profileVersion == "david-v100-simd-simt-20260804-v10" ||
      profile.profileVersion == "david-v100-simd-simt-20260806-v11";
  if (usesSharedMicrobench &&
      ((!usesAnchorPartitionProfile && *selectionSchemaVersion != 2) ||
       (!usesEventCalibrationSchema && usesAnchorPartitionProfile &&
        *selectionSchemaVersion != 3) ||
       (usesEventCalibrationSchema && *selectionSchemaVersion != 4)))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT v5/v6 requires schema_version 2 and v7 requires "
        "schema_version 3; v9/v10/v11 require schema_version 4");
  if (profile.simdVectorWidthBits <= 0 || profile.simtWarpSize <= 0 ||
      profile.simdMte2BytesPerCycle <= 0.0 ||
      profile.simdMte3BytesPerCycle <= 0.0 || profile.simtLoadWarpRate <= 0.0 ||
      profile.simtStoreWarpRate <= 0.0 || profile.simtShuffleRate <= 0.0 ||
      profile.simtPredicateRate <= 0.0 || profile.mixedSetupFallbacks.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile contains non-positive rates or no mixed setup "
        "fallbacks");
  if (profile.coverage.triangularLoopCountMax <= 0 ||
      profile.coverage.triangularLoopTripSumMax <= 0 ||
      profile.coverage.triangularMaskRankSumMax <= 0 ||
      profile.coverage.triangularWeightedReductionsMax <= 0 ||
      profile.coverage.triangularMaxTensorNumel <= 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile contains non-positive triangular-solve "
        "coverage bounds");
  if (usesEventCalibrationSchema && profile.eventRouteCalibration.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT v9/v10/v11 profile requires event route calibration "
        "domains");
  for (const auto &entry : profile.eventRouteCalibration) {
    const EventRouteCalibrationProfile &route = entry.second;
    if (!std::isfinite(route.allSimdMultiplier) ||
        !std::isfinite(route.allSimtOnlyMultiplier) ||
        !std::isfinite(route.mixedSimdSimtMultiplier) ||
        route.allSimdMultiplier <= 0.0 || route.allSimtOnlyMultiplier <= 0.0 ||
        route.mixedSimdSimtMultiplier <= 0.0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "event route calibration domain '%s' has a non-positive or "
          "non-finite multiplier",
          entry.first().str().c_str());
  }

  std::string canonicalProfile;
  llvm::raw_string_ostream canonicalStream(canonicalProfile);
  emitPythonCanonicalJSON(*parsed, canonicalStream);
  canonicalStream.flush();
  llvm::ArrayRef<uint8_t> byteArray(
      reinterpret_cast<const uint8_t *>(canonicalProfile.data()),
      canonicalProfile.size());
  auto hash = llvm::SHA256::hash(byteArray);
  profile.selectionContentSha256 =
      llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true);

  profile.contentSha256 = profile.selectionContentSha256;
  if (!profile.microbenchmarkContentSha256.empty()) {
    std::string combinedAssets =
        canonicalProfile +
        "\nshared_microbenchmark_sha256=" + profile.microbenchmarkContentSha256;
    llvm::ArrayRef<uint8_t> combinedBytes(
        reinterpret_cast<const uint8_t *>(combinedAssets.data()),
        combinedAssets.size());
    auto combinedHash = llvm::SHA256::hash(combinedBytes);
    profile.contentSha256 =
        llvm::toHex(llvm::ArrayRef<uint8_t>(combinedHash), true);
  }
  return profile;
}

static int64_t mapValue(const llvm::StringMap<int64_t> &values,
                        llvm::StringRef key, int64_t fallback = 0) {
  auto iterator = values.find(key);
  return iterator == values.end() ? fallback : iterator->second;
}

static std::vector<std::pair<llvm::StringRef, int64_t>>
getProfileOpElements(const SimdSimtFeatureSummary &features) {
  const int64_t maxNumel = std::max<int64_t>(1, features.maxTensorNumel);
  auto work = [&](llvm::StringRef elementName, int64_t rawCount) {
    auto iterator = features.opElements.find(elementName);
    if (iterator != features.opElements.end())
      return std::max<int64_t>(0, iterator->second);
    return std::max<int64_t>(0, rawCount) * maxNumel;
  };
  return {
      {"f32.add", work("add", features.addOps)},
      {"f32.sub", work("sub", features.subOps)},
      {"f32.mul", work("mul", features.mulOps)},
      {"f32.div", work("div", features.divOps)},
      {"f32.max", work("max", features.maxOps)},
      {"f32.abs", work("abs", features.absOps)},
      {"f32.exp", work("exp", features.expOps)},
      {"f32.log", work("log", features.logOps)},
      {"predicate.cmp", work("cmp", features.cmpOps)},
      {"predicate.select", work("select", features.selectOps)},
      {"convert.cast", work("cast", features.castOps)},
      {"f32.clamp", work("clamp", features.clampOps)},
  };
}

static std::vector<std::pair<llvm::StringRef, int64_t>>
getProfileOpElements(const SimtAnchorFeatureSummary &features) {
  auto work = [&](llvm::StringRef elementName) {
    auto iterator = features.opElements.find(elementName);
    return iterator == features.opElements.end()
               ? int64_t{0}
               : std::max<int64_t>(0, iterator->second);
  };
  return {
      {"f32.add", work("add")},       {"f32.sub", work("sub")},
      {"f32.mul", work("mul")},       {"f32.div", work("div")},
      {"f32.max", work("max")},       {"f32.abs", work("abs")},
      {"f32.exp", work("exp")},       {"f32.log", work("log")},
      {"predicate.cmp", work("cmp")}, {"predicate.select", work("select")},
      {"convert.cast", work("cast")}, {"f32.clamp", work("clamp")},
  };
}

static std::pair<bool, std::string>
rankingCalibrationCoverage(const SimdSimtFeatureSummary &features,
                           int64_t weightedReductions, int64_t dotFlops,
                           const CandidateProfile &profile,
                           double irregularDensity) {
  if (features.hasDynamicShape)
    return {false, "dynamic_shape"};
  const CoverageProfile &coverage = profile.coverage;
  const int64_t maxNumel = features.maxTensorNumel;
  const int64_t staticLoopTrips = features.staticLoopTripCountSum;
  const int64_t maskRankSum = features.maskRankSum;
  const bool hasTriangularSolve = llvm::is_contained(
      features.simtAnchors.mechanismKinds, "triangular_solve_loop");
  // The triangular solve has intentionally dynamic loop bounds (min(T, 16),
  // min(T, 32), ...), so the generic unknown-trip-count rejection is too
  // coarse. Admit only the independently bounded BT64 anchor shape. Do not
  // borrow masked-rowwise limits: a full tile has four loops and 56 weighted
  // reductions after structural trip modeling.
  if (hasTriangularSolve && features.simtAnchors.count == 1 &&
      features.simtAnchors.staticLoopCount > 0 &&
      features.simtAnchors.staticLoopCount <= coverage.triangularLoopCountMax &&
      features.simtAnchors.staticLoopTripCountSum <=
          coverage.triangularLoopTripSumMax &&
      maxNumel <= coverage.triangularMaxTensorNumel &&
      maskRankSum <= coverage.triangularMaskRankSumMax &&
      weightedReductions <= coverage.triangularWeightedReductionsMax)
    return {true, "triangular_solve_loop"};
  if (features.hasUnknownTripCount)
    return {false, "unknown_loop_trip_count"};
  if (dotFlops > 0 && dotFlops <= coverage.tinyDotFlopsMax &&
      staticLoopTrips == 0 && maxNumel <= coverage.tinyDotMaxTensorNumel &&
      irregularDensity >= coverage.minimumIrregularDensity)
    return {true, "tiny_irregular_dot"};
  if (dotFlops == 0 && features.rank1IndirectVectorReduce &&
      weightedReductions > 0 &&
      weightedReductions <= coverage.rank1WeightedReductionsMax &&
      maxNumel <= coverage.rank1MaxTensorNumel &&
      staticLoopTrips <= coverage.rowwiseLoopTripSumMax)
    return {true, "rank1_indirect_vector_reduction"};
  if (dotFlops == 0 && staticLoopTrips > 0 &&
      staticLoopTrips <= coverage.rowwiseLoopTripSumMax && maskRankSum > 0 &&
      maskRankSum <= coverage.rowwiseMaskRankSumMax && weightedReductions > 0 &&
      weightedReductions <= coverage.rowwiseWeightedReductionsMax &&
      maxNumel <= coverage.rowwiseMaxTensorNumel &&
      irregularDensity >= coverage.minimumIrregularDensity)
    return {true, "masked_rowwise_reduction"};
  return {false, "out_of_calibration_domain"};
}

static SimtApplicabilityResult
evaluateSimtApplicability(const SimdSimtFeatureSummary &features,
                          bool targetSupported) {
  SimtApplicabilityResult result;
  result.targetSupported = targetSupported;
  result.recognizedAnchorCount = features.simtAnchors.recognizedCount;
  result.materializableAnchorCount = features.simtAnchors.count;
  result.mechanisms = features.simtAnchors.mechanismKinds;
  for (const std::string &kind : features.observedMixedKinds)
    appendUnique(result.mechanisms, kind);
  llvm::sort(result.mechanisms);
  result.mechanismDetected =
      result.recognizedAnchorCount > 0 || !result.mechanisms.empty();
  result.materializable =
      targetSupported && result.materializableAnchorCount > 0;
  if (!result.mechanismDetected)
    result.reasons.push_back("no_recognized_simt_mechanism");
  else if (!targetSupported)
    result.reasons.push_back("target_does_not_support_simt_materialization");
  else if (result.materializableAnchorCount == 0)
    result.reasons.push_back("no_materializable_simt_anchor");
  return result;
}

static llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>>
legalCandidates(const SimdSimtCandidateScores &scores, bool allSimdLegal,
                bool allSimtLegal, bool mixedLegal) {
  llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>> candidates;
  if (allSimdLegal)
    candidates.push_back({scores.allSimd, SimdSimtCandidateKind::AllSIMD});
  if (allSimtLegal)
    candidates.push_back(
        {scores.allSimtOnly, SimdSimtCandidateKind::AllSIMTOnly});
  if (mixedLegal)
    candidates.push_back(
        {scores.mixedSimdSimt, SimdSimtCandidateKind::MixedSIMDSIMT});
  llvm::stable_sort(candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  return candidates;
}

static SimdSimtCandidateKind chooseBest(const SimdSimtCandidateScores &scores,
                                        bool allSimdLegal, bool allSimtLegal,
                                        bool mixedLegal) {
  return legalCandidates(scores, allSimdLegal, allSimtLegal, mixedLegal)
      .front()
      .second;
}

static SimdSimtCandidateKind
chooseRunnerUp(const SimdSimtCandidateScores &scores, bool allSimdLegal,
               bool allSimtLegal, bool mixedLegal, SimdSimtCandidateKind best) {
  auto candidates =
      legalCandidates(scores, allSimdLegal, allSimtLegal, mixedLegal);
  return candidates.size() > 1 ? candidates[1].second : best;
}

static void sortAndUnique(std::vector<std::string> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

} // namespace

llvm::StringRef
mlir::ascend::stringifySimdSimtCandidate(SimdSimtCandidateKind candidate) {
  switch (candidate) {
  case SimdSimtCandidateKind::AllSIMD:
    return kAllSimd;
  case SimdSimtCandidateKind::AllSIMTOnly:
    return kAllSimtOnly;
  case SimdSimtCandidateKind::MixedSIMDSIMT:
    return kMixedSimdSimt;
  }
  llvm_unreachable("unknown SIMD/SIMT candidate");
}

double SimdSimtCandidateScores::get(SimdSimtCandidateKind candidate) const {
  switch (candidate) {
  case SimdSimtCandidateKind::AllSIMD:
    return allSimd;
  case SimdSimtCandidateKind::AllSIMTOnly:
    return allSimtOnly;
  case SimdSimtCandidateKind::MixedSIMDSIMT:
    return mixedSimdSimt;
  }
  llvm_unreachable("unknown SIMD/SIMT candidate");
}

llvm::json::Object SimdSimtCandidateScores::toJSON() const {
  llvm::json::Object result;
  result[kAllSimd] = allSimd;
  result[kAllSimtOnly] = allSimtOnly;
  result[kMixedSimdSimt] = mixedSimdSimt;
  return result;
}

static llvm::json::Array toReasonJSON(const std::vector<std::string> &reasons) {
  llvm::json::Array result;
  for (const std::string &reason : reasons)
    result.push_back(reason);
  return result;
}

static llvm::json::Object
toLowerabilityJSON(const CandidateLowerability &lowerability) {
  llvm::json::Object result;
  auto route = [](CandidateLoweringStatus status,
                  const std::vector<std::string> &reasons) {
    llvm::json::Object entry;
    entry["status"] = stringifyCandidateLoweringStatus(status).str();
    entry["reasons"] = toReasonJSON(reasons);
    return entry;
  };
  result[kAllSimd] = route(lowerability.allSimd, lowerability.allSimdReasons);
  result[kAllSimtOnly] =
      route(lowerability.allSimtOnly, lowerability.allSimtOnlyReasons);
  result[kMixedSimdSimt] = route(lowerability.mixed, lowerability.mixedReasons);
  return result;
}

static llvm::json::Object toAtomicFactsJSON(const TensorAtomicFacts &facts) {
  llvm::json::Object result;
  result["update_elements"] = facts.updateElements;
  result["address_rank"] = facts.addressRank;
  result["value_type"] = facts.valueType;
  result["offset_type"] = facts.offsetType;
  result["operation"] = facts.operation;
  result["has_mask"] = facts.hasMask;
  if (facts.staticMaskActiveFraction)
    result["static_mask_active_fraction"] = *facts.staticMaskActiveFraction;
  else
    result["static_mask_active_fraction"] = nullptr;
  result["result_used"] = facts.resultUsed;
  result["address_is_lane_varying"] = facts.addressIsLaneVarying;
  result["address_depends_on_loaded_index"] = facts.addressDependsOnLoadedIndex;
  result["contention"] = facts.contention;
  return result;
}

static llvm::json::Object toHistogramFactsJSON(const HistogramFacts &facts) {
  llvm::json::Object result;
  result["input_elements"] = facts.inputElements;
  result["num_bins"] = facts.numBins;
  result["input_type"] = facts.inputType;
  result["result_type"] = facts.resultType;
  return result;
}

static llvm::json::Object
toPlainCumsumFactsJSON(const PlainCumsumFacts &facts) {
  llvm::json::Object result;
  result["axis_extent"] = facts.axisExtent;
  result["element_type"] = facts.elementType;
  result["reverse"] = facts.reverse;
  return result;
}

llvm::json::Object SimtAnchorFeatureSummary::toJSON() const {
  llvm::json::Object result;
  result["recognized_count"] = recognizedCount;
  result["count"] = count;
  result["materializable_count"] = count;
  result["covered_operation_count"] = coveredOperationCount;
  result["load_ops"] = loadOps;
  result["store_ops"] = storeOps;
  result["reduce_ops"] = reduceOps;
  result["scan_ops"] = scanOps;
  result["gather_ops"] = gatherOps;
  result["dot_ops"] = dotOps;
  result["atomic_ops"] = atomicOps;
  result["histogram_ops"] = histogramOps;
  result["max_tensor_numel"] = maxTensorNumel;
  result["max_element_bits"] = maxElementBits;
  result["mask_rank_sum"] = maskRankSum;
  result["unique_mask_values"] = uniqueMaskValues;
  result["unique_mask_rank_sum"] = uniqueMaskRankSum;
  result["predicate_elements"] = predicateElements;
  result["predicate_lane_evaluations"] = predicateLaneEvaluations;
  result["pointer_tensor_ops"] = pointerTensorOps;
  result["loaded_index_dependent_memory_ops"] = loadedIndexDependentMemoryOps;
  result["lane_dependent_pointer_ops"] = laneDependentPointerOps;
  result["max_reduce_axis_extent"] = maxReduceAxisExtent;
  result["weighted_reduce_axis_elements"] = weightedReduceAxisElements;
  result["shuffle_lane_steps"] = shuffleLaneSteps;
  result["static_loop_count"] = staticLoopCount;
  result["static_loop_trip_count_sum"] = staticLoopTripCountSum;
  result["modeled_dynamic_loop_count"] = modeledDynamicLoopCount;
  result["modeled_dynamic_loop_trip_count_sum"] =
      modeledDynamicLoopTripCountSum;
  result["has_control_flow"] = hasControlFlow;
  result["weighted_ops"] = ::toJSON(weightedOps);
  result["op_elements"] = ::toJSON(opElements);
  result["load_bytes"] = loadBytes;
  result["store_bytes"] = storeBytes;
  result["load_warp_instructions"] = loadWarpInstructions;
  result["store_warp_instructions"] = storeWarpInstructions;
  result["dot_flops"] = dotFlops;
  result["captured_tensor_count"] = capturedTensorCount;
  result["escaping_tensor_count"] = escapingTensorCount;
  result["captured_tensor_bytes"] = capturedTensorBytes;
  result["escaping_tensor_bytes"] = escapingTensorBytes;
  llvm::json::Array mechanisms;
  for (const std::string &kind : mechanismKinds)
    mechanisms.push_back(kind);
  result["mechanism_kinds"] = std::move(mechanisms);
  llvm::json::Array atomicFacts;
  for (const TensorAtomicFacts &facts : tensorAtomics)
    atomicFacts.push_back(toAtomicFactsJSON(facts));
  result["tensor_atomics"] = std::move(atomicFacts);
  llvm::json::Array histogramFacts;
  for (const HistogramFacts &facts : histograms)
    histogramFacts.push_back(toHistogramFactsJSON(facts));
  result["histograms"] = std::move(histogramFacts);
  llvm::json::Array cumsumFacts;
  for (const PlainCumsumFacts &facts : plainCumsums)
    cumsumFacts.push_back(toPlainCumsumFactsJSON(facts));
  result["plain_cumsums"] = std::move(cumsumFacts);
  result["kernel_lowerability"] = toLowerabilityJSON(kernelLowerability);
  return result;
}

llvm::json::Object SimdSimtFeatureSummary::toJSON() const {
  llvm::json::Object result;
  result["load_ops"] = loadOps;
  result["store_ops"] = storeOps;
  result["reduce_ops"] = reduceOps;
  result["scan_ops"] = scanOps;
  result["gather_ops"] = gatherOps;
  result["dot_ops"] = dotOps;
  result["atomic_ops"] = atomicOps;
  result["histogram_ops"] = histogramOps;
  result["broadcast_ops"] = broadcastOps;
  result["expand_dims_ops"] = expandDimsOps;
  result["splat_ops"] = splatOps;
  result["addptr_ops"] = addPtrOps;
  result["arith_ops"] = arithOps;
  result["math_ops"] = mathOps;
  result["add_ops"] = addOps;
  result["sub_ops"] = subOps;
  result["mul_ops"] = mulOps;
  result["div_ops"] = divOps;
  result["max_ops"] = maxOps;
  result["abs_ops"] = absOps;
  result["exp_ops"] = expOps;
  result["log_ops"] = logOps;
  result["cmp_ops"] = cmpOps;
  result["select_ops"] = selectOps;
  result["cast_ops"] = castOps;
  result["clamp_ops"] = clampOps;
  result["scalar_ops"] = scalarOps;
  result["max_tensor_rank"] = maxTensorRank;
  result["max_tensor_numel"] = maxTensorNumel;
  result["max_element_bits"] = maxElementBits;
  result["mask_tensor_ops"] = maskTensorOps;
  result["mask_rank_sum"] = maskRankSum;
  result["unique_mask_values"] = uniqueMaskValues;
  result["unique_mask_rank_sum"] = uniqueMaskRankSum;
  result["predicate_elements"] = predicateElements;
  result["predicate_lane_evaluations"] = predicateLaneEvaluations;
  result["mask_broadcast_ops"] = maskBroadcastOps;
  result["pointer_tensor_ops"] = pointerTensorOps;
  result["pointer_unstructured_dims"] = pointerUnstructuredDims;
  result["loaded_index_dependent_memory_ops"] = loadedIndexDependentMemoryOps;
  result["lane_dependent_pointer_ops"] = laneDependentPointerOps;
  result["row_local_reduce_ops"] = rowLocalReduceOps;
  result["max_reduce_axis_extent"] = maxReduceAxisExtent;
  result["weighted_reduce_axis_elements"] = weightedReduceAxisElements;
  result["shuffle_lane_steps"] = shuffleLaneSteps;
  result["scalar_load_ops"] = scalarLoadOps;
  result["scalar_store_ops"] = scalarStoreOps;
  result["vector_ptr_splat_ops"] = vectorPtrSplatOps;
  result["vector_reduce_to_scalar_ops"] = vectorReduceToScalarOps;
  result["rank1_indirect_vector_reduce"] = rank1IndirectVectorReduce;
  result["weighted_ops"] = ::toJSON(weightedOps);
  result["op_elements"] = ::toJSON(opElements);
  result["load_bytes"] = loadBytes;
  result["store_bytes"] = storeBytes;
  result["load_warp_instructions"] = loadWarpInstructions;
  result["store_warp_instructions"] = storeWarpInstructions;
  result["dot_flops"] = dotFlops;
  result["dot_output_elements"] = dotOutputElements;
  llvm::json::Array dotShapes;
  for (const auto &shape : dotMNK)
    dotShapes.push_back(llvm::json::Array({shape[0], shape[1], shape[2]}));
  result["dot_mnk"] = std::move(dotShapes);
  result["static_loop_count"] = staticLoopCount;
  result["static_loop_trip_count_sum"] = staticLoopTripCountSum;
  result["static_loop_trip_count_max"] = staticLoopTripCountMax;
  result["modeled_dynamic_loop_count"] = modeledDynamicLoopCount;
  result["modeled_dynamic_loop_trip_count_sum"] =
      modeledDynamicLoopTripCountSum;
  result["has_dot"] = hasDot;
  result["has_gather"] = hasGather;
  result["has_atomic"] = hasAtomic;
  result["has_histogram"] = hasHistogram;
  result["has_scan"] = hasScan;
  result["has_explicit_scope"] = hasExplicitScope;
  result["has_control_flow"] = hasControlFlow;
  result["has_dynamic_shape"] = hasDynamicShape;
  result["has_unknown_trip_count"] = hasUnknownTripCount;
  result["simt_anchors"] = simtAnchors.toJSON();
  llvm::json::Array mixedKinds;
  for (const std::string &kind : observedMixedKinds)
    mixedKinds.push_back(kind);
  result["observed_mixed_kinds"] = std::move(mixedKinds);
  result["mixed_required"] = !observedMixedKinds.empty();
  result["mandatory_mixed_enabled"] = false;
  return result;
}

llvm::json::Object SimtApplicabilityResult::toJSON() const {
  llvm::json::Object result;
  result["mechanism_detected"] = mechanismDetected;
  result["target_supported"] = targetSupported;
  result["materializable"] = materializable;
  result["recognized_anchor_count"] = recognizedAnchorCount;
  result["materializable_anchor_count"] = materializableAnchorCount;
  llvm::json::Array mechanismValues;
  for (const std::string &mechanism : mechanisms)
    mechanismValues.push_back(mechanism);
  result["mechanisms"] = std::move(mechanismValues);
  llvm::json::Array reasonValues;
  for (const std::string &reason : reasons)
    reasonValues.push_back(reason);
  result["reasons"] = std::move(reasonValues);
  return result;
}

llvm::json::Object
SimdSimtCostBreakdown::toJSON(const SimdSimtFeatureSummary &features) const {
  llvm::json::Object result;
  llvm::json::Object compute;
  compute["simd"] = simdComputeCycles;
  compute["simt"] = simtComputeCycles;
  compute["simd_dot"] = simdDotCycles;
  compute["simt_dot"] = simtDotCycles;
  result["compute_only"] = std::move(compute);

  llvm::json::Object memory;
  memory["load_bytes"] = features.loadBytes;
  memory["store_bytes"] = features.storeBytes;
  memory["simd_load_system_cycles"] = simdLoadCycles;
  memory["simd_store_system_cycles"] = simdStoreCycles;
  memory["simd_roofline_system_cycles"] = simdMemoryCycles;
  memory["simt_load_warp_instructions"] = features.loadWarpInstructions;
  memory["simt_store_warp_instructions"] = features.storeWarpInstructions;
  memory["simt_load_system_cycles"] = simtLoadCycles;
  memory["simt_store_system_cycles"] = simtStoreCycles;
  memory["simt_serial_memory_system_cycles"] = simtMemoryCycles;
  // Compatibility field retained for existing report consumers. SIMT memory
  // is not roofline-overlapped with SIMT compute by the route model.
  memory["simt_roofline_system_cycles"] = simtMemoryCycles;
  result["memory"] = std::move(memory);

  llvm::json::Object structure;
  structure["irregular_density"] = irregularDensity;
  structure["tiny_dot_underfill"] = tinyDotUnderfill;
  structure["components"] = ::toJSON(structuralComponents);
  structure["penalty_ratio"] = structuralPenaltyRatio;
  structure["simd_structural_penalty_system_cycles"] =
      simdStructuralPenaltyCycles;
  result["structure"] = std::move(structure);

  llvm::json::Object mixed;
  mixed["derived_simd_fraction"] = mixedSimdFraction;
  mixed["cost_source"] = mixedCostSource;
  mixed["setup_fallback_num_warps"] = mixedSetupFallbackNumWarps;
  mixed["mixed_setup_fallback_system_cycles"] = mixedSetupFallbackCycles;
  mixed["standalone_serialized_setup_system_cycles"] =
      standaloneSimtSetupCycles;
  mixed["setup_proxy_delta_system_cycles"] = setupProxyDeltaCycles;
  mixed["directional_transition_system_cycles"] = nullptr;
  mixed["directional_transition_measurement_status"] = "unmeasured";
  llvm::json::Object partition;
  partition["simd_regular_compute_system_cycles"] =
      mixedSimdRegularComputeCycles;
  partition["simd_regular_dot_system_cycles"] = mixedSimdRegularDotCycles;
  partition["simd_regular_memory_system_cycles"] = mixedSimdRegularMemoryCycles;
  partition["simd_regular_payload_system_cycles"] =
      mixedSimdRegularPayloadCycles;
  partition["simt_anchor_compute_system_cycles"] = mixedSimtAnchorComputeCycles;
  partition["simt_anchor_dot_system_cycles"] = mixedSimtAnchorDotCycles;
  partition["simt_anchor_memory_system_cycles"] = mixedSimtAnchorMemoryCycles;
  partition["simt_anchor_shuffle_system_cycles"] = mixedSimtAnchorShuffleCycles;
  partition["simt_anchor_predicate_system_cycles"] =
      mixedSimtAnchorPredicateCycles;
  partition["simt_anchor_payload_system_cycles"] = mixedSimtAnchorPayloadCycles;
  partition["measured_boundary_system_cycles"] = nullptr;
  partition["applied_boundary_fallback_system_cycles"] = mixedBoundaryCycles;
  partition["remaining_simd_structural_penalty_ratio"] =
      mixedRemainingStructuralPenaltyRatio;
  mixed["partition"] = std::move(partition);
  result["mixed"] = std::move(mixed);

  llvm::json::Object execution;
  execution["shuffle_warp_instructions"] = simtShuffleInstructions;
  execution["shuffle_system_cycles"] = simtShuffleCycles;
  execution["predicate_warp_instructions"] = simtPredicateInstructions;
  execution["predicate_system_cycles"] = simtPredicateCycles;
  execution["program_issue_scale"] = programIssueScale;
  execution["simd_setup_system_cycles"] = simdSetupCycles;
  execution["simt_setup_system_cycles"] = simtSetupCycles;
  execution["simd_issue_payload_system_cycles"] = simdIssuePayloadCycles;
  execution["simt_issue_payload_system_cycles"] = simtIssuePayloadCycles;
  execution["simt_issue_aggregation"] = "serial_sum";
  result["simt_execution"] = std::move(execution);

  llvm::json::Object opBreakdown;
  opBreakdown["simd_ops_system_cycles"] = ::toJSON(simdOpSystemCycles);
  opBreakdown["simt_ops_system_cycles"] = ::toJSON(simtOpSystemCycles);
  result["op_breakdown"] = std::move(opBreakdown);
  return result;
}

llvm::json::Object SimdSimtCostReport::toJSON() const {
  llvm::json::Object result;
  result["schema_version"] = schemaVersion;
  result["model"] = model;
  result["profile_version"] = profileVersion;
  result["profile_target"] = profileTarget;
  result["actual_target"] = actualTarget;
  result["target_compatible"] = targetCompatible;
  result["profile_content_sha256"] = profileContentSha256;
  result["selection_profile_content_sha256"] = selectionProfileContentSha256;
  llvm::json::Object sharedEvidence;
  sharedEvidence["profile_version"] = microbenchmarkProfileVersion;
  sharedEvidence["target"] = microbenchmarkProfileTarget;
  sharedEvidence["content_sha256"] = microbenchmarkProfileContentSha256;
  result["shared_microbenchmark_profile"] = std::move(sharedEvidence);
  result["unit"] = scoreUnit;
  result["score_scope"] = scoreScope;
  result["selection_score_valid"] = selectionScoreValid;
  result["absolute_cost_valid"] = absoluteCostValid;
  result["excludes"] = llvm::json::Array({"host_launch", "grid_wave_count"});
  result["candidate_costs_evaluated"] = candidateCostsEvaluated;
  if (candidateCostsEvaluated) {
    result["candidate_costs"] = candidateCosts.toJSON();
    result["candidate_ratios_to_best"] = candidateRatiosToBest.toJSON();
    result["decision_kind"] = stringifySimdSimtCandidate(decision);
    result["runner_up_kind"] = stringifySimdSimtCandidate(runnerUp);
    result["best_score"] = bestScore;
    result["runner_up_score"] = runnerUpScore;
    result["gain_score"] = gainScore;
    result["decision_advantage"] = decisionAdvantage;
    result["required_gain_score"] = requiredGainScore;
  } else {
    result["candidate_costs"] = nullptr;
    result["candidate_ratios_to_best"] = nullptr;
    result["decision_kind"] = nullptr;
    result["runner_up_kind"] = nullptr;
    result["best_score"] = nullptr;
    result["runner_up_score"] = nullptr;
    result["gain_score"] = nullptr;
    result["decision_advantage"] = nullptr;
    result["required_gain_score"] = nullptr;
  }
  llvm::json::Object eventCalibration;
  eventCalibration["applied"] = eventRouteCalibrationApplied;
  eventCalibration["domain"] = eventRouteCalibrationApplied
                                   ? llvm::json::Value(calibrationDomain)
                                   : llvm::json::Value(nullptr);
  eventCalibration["all_simt_only_validated"] = eventAllSimtOnlyValidated;
  eventCalibration["mixed_simd_simt_validated"] = eventMixedSimdSimtValidated;
  eventCalibration["source"] = eventRouteCalibrationSource;
  eventCalibration["confidence"] = eventRouteCalibrationConfidence;
  if (candidateCostsEvaluated) {
    eventCalibration["raw_candidate_costs"] =
        uncalibratedCandidateCosts.toJSON();
    eventCalibration["score_multipliers"] = eventRouteScoreMultipliers.toJSON();
  } else {
    eventCalibration["raw_candidate_costs"] = nullptr;
    eventCalibration["score_multipliers"] = nullptr;
  }
  result["event_route_calibration"] = std::move(eventCalibration);
  llvm::json::Array selectableCandidates;
  if (allSimdCandidateLegal)
    selectableCandidates.push_back(kAllSimd);
  if (allSimtOnlyCandidateLegal)
    selectableCandidates.push_back(kAllSimtOnly);
  if (mixedCandidateLegal)
    selectableCandidates.push_back(kMixedSimdSimt);
  result["selectable_candidates"] = std::move(selectableCandidates);
  result["margin_ratio"] = marginRatio;
  result["ranking_confidence"] = rankingConfidence;
  result["minimum_confidence_for_decision"] = minimumConfidenceForDecision;
  result["absolute_confidence"] = absoluteConfidence;
  result["confidence"] = rankingConfidence;
  result["gate_passed"] = gatePassed;

  llvm::json::Array reasons;
  for (const std::string &reason : gateReasons)
    reasons.push_back(reason);
  result["gate_reasons"] = std::move(reasons);
  llvm::json::Array unsupportedValues;
  for (const std::string &value : unsupported)
    unsupportedValues.push_back(value);
  result["unsupported"] = std::move(unsupportedValues);

  llvm::json::Object structure;
  structure["calibration_covered"] = calibrationCovered;
  structure["calibration_domain"] = calibrationDomain;
  structure["calibration_sample_domain"] = calibrationDomain;
  result["calibration"] = std::move(structure);
  result["applicability"] = applicability.toJSON();

  llvm::json::Object contract;
  contract["version"] = 2;
  contract["enabled"] = false;
  contract["requested"] = !features.observedMixedKinds.empty();
  contract["mandatory_override_suppressed"] =
      !features.observedMixedKinds.empty();
  contract["mandatory"] = false;
  contract["required"] = false;
  contract["target_kind"] = nullptr;
  llvm::json::Array routeKinds;
  for (const std::string &kind : features.observedMixedKinds)
    routeKinds.push_back(kind);
  contract["route_kinds"] = std::move(routeKinds);
  contract["all_simt_only_reference_only"] = false;
  result["mixed_execution_contract"] = std::move(contract);

  llvm::json::Object roles;
  roles[kAllSimd] =
      allSimdCandidateLegal ? "selectable_candidate" : "inapplicable";
  roles[kAllSimtOnly] =
      allSimtOnlyCandidateLegal ? "selectable_candidate" : "inapplicable";
  roles[kMixedSimdSimt] =
      mixedCandidateLegal ? "selectable_candidate" : "inapplicable";
  result["candidate_roles"] = std::move(roles);

  if (candidateCostsEvaluated) {
    llvm::json::Object analytical;
    analytical[kAllSimd] = breakdown.simdAnalyticalCycles;
    analytical[kAllSimtOnly] = breakdown.simtAnalyticalCycles;
    result["analytical_candidate_costs"] = std::move(analytical);

    llvm::json::Object detail = breakdown.toJSON(features);
    for (auto &entry : detail)
      result[entry.first] = std::move(entry.second);
    if (auto *structureObject = result.getObject("structure")) {
      (*structureObject)["calibration_covered"] = calibrationCovered;
      (*structureObject)["calibration_domain"] = calibrationDomain;
      (*structureObject)["calibration_sample_domain"] = calibrationDomain;
    }
  } else {
    result["analytical_candidate_costs"] = nullptr;
    llvm::json::Object structure;
    structure["irregular_density"] = breakdown.irregularDensity;
    structure["calibration_covered"] = calibrationCovered;
    structure["calibration_domain"] = calibrationDomain;
    structure["calibration_sample_domain"] = calibrationDomain;
    result["structure"] = std::move(structure);
  }
  if (includeFeaturesInJSON)
    result["features"] = features.toJSON();
  result["des_feedback_applied"] = llvm::json::Array();
  result["des_feedback_validation_errors"] = llvm::json::Array();
  return result;
}

void SimdSimtCostReport::printJSON(llvm::raw_ostream &os, bool pretty) const {
  llvm::json::Object object = toJSON();
  if (pretty)
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(object)));
  else
    os << llvm::json::Value(std::move(object));
}

std::string mlir::ascend::getDefaultSimdSimtProfilePath() {
  if (const char *environment = std::getenv("TRITON_ASCEND_SIMD_SIMT_PROFILE"))
    if (*environment)
      return environment;
#ifdef TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH
  return TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH;
#else
  return {};
#endif
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module, bool compileOn91095) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan = buildMixedSimtAnchorPlan(module, compileOn91095);
  return analyzeSimdSimtFeatures(module, anchorPlan);
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module,
                                      const SimtAnchorPlan &anchorPlan) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");

  SimdSimtFeatureSummary features;
  initializeWorkMaps(features);
  initializeWorkMaps(features.simtAnchors);
  llvm::DenseSet<Operation *> anchorSet;
  llvm::DenseMap<Operation *, int64_t> structuralTripEstimates;
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable)
      continue;
    if (anchor.scopeOperations.empty()) {
      if (anchor.operation)
        anchorSet.insert(anchor.operation);
    } else {
      for (Operation *scopeOperation : anchor.scopeOperations)
        if (scopeOperation)
          anchorSet.insert(scopeOperation);
    }
    if (anchor.kind == SimtAnchorKind::TriangularSolveLoop) {
      // A recognized solve_tril block has a fixed 16x16 state and starts its
      // recurrence at row 2: a full block performs 16 - 2 = 14 iterations.
      // The TTIR upper bound is min(runtime_remaining, block_end), so it is
      // not a compile-time constant even though the full-tile estimate is
      // structurally known.  Do not apply this fallback to generic loops.
      for (Operation *scopeOperation : anchor.scopeOperations)
        if (scopeOperation &&
            scopeOperation->getName().getStringRef() == "scf.for")
          structuralTripEstimates[scopeOperation] = 14;
    }
  }
  features.simtAnchors.recognizedCount = anchorPlan.anchors.size();
  features.simtAnchors.count = anchorPlan.materializableCount();
  features.simtAnchors.kernelLowerability = anchorPlan.kernelLowerability;
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    std::string kind = stringifySimtAnchorKind(anchor.kind).str();
    appendUnique(features.simtAnchors.mechanismKinds, kind);
    appendUnique(features.observedMixedKinds, kind);
    if (const auto *facts = std::get_if<TensorAtomicFacts>(&anchor.facts))
      features.simtAnchors.tensorAtomics.push_back(*facts);
    else if (const auto *facts = std::get_if<HistogramFacts>(&anchor.facts))
      features.simtAnchors.histograms.push_back(*facts);
    else if (const auto *facts = std::get_if<PlainCumsumFacts>(&anchor.facts))
      features.simtAnchors.plainCumsums.push_back(*facts);
  }

  auto isInAnchor = [&](Operation *op) {
    for (Operation *current = op; current; current = current->getParentOp())
      if (anchorSet.contains(current))
        return true;
    return false;
  };

  llvm::DenseSet<Value> capturedTensors;
  llvm::DenseSet<Value> escapingTensors;
  llvm::DenseSet<Value> uniqueMasks;
  llvm::DenseSet<Value> anchorUniqueMasks;
  auto isValueDefinedInAnchor = [&](Value value) {
    if (Operation *definingOp = value.getDefiningOp())
      return isInAnchor(definingOp);
    auto argument = dyn_cast<BlockArgument>(value);
    Operation *parent = argument ? argument.getOwner()->getParentOp() : nullptr;
    return parent && isInAnchor(parent);
  };
  module.walk([&](Operation *op) {
    if (!isInAnchor(op))
      return;
    for (Value operand : op->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()) ||
          isValueDefinedInAnchor(operand) ||
          !capturedTensors.insert(operand).second)
        continue;
      ++features.simtAnchors.capturedTensorCount;
      features.simtAnchors.capturedTensorBytes +=
          getStaticTensorBytes(operand.getType());
    }
    for (Value result : op->getResults()) {
      if (!isa<RankedTensorType>(result.getType()) || result.use_empty())
        continue;
      bool escapes = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        return !isInAnchor(use.getOwner());
      });
      if (!escapes || !escapingTensors.insert(result).second)
        continue;
      ++features.simtAnchors.escapingTensorCount;
      features.simtAnchors.escapingTensorBytes +=
          getStaticTensorBytes(result.getType());
    }
  });
  auto updateTypeStats = [&](Type type, bool inAnchor) {
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      if (!tensor.hasStaticShape())
        features.hasDynamicShape = true;
      features.maxElementBits =
          std::max(features.maxElementBits, getTypeBitWidth(type));
      features.maxTensorRank =
          std::max<int64_t>(features.maxTensorRank, tensor.getRank());
      features.maxTensorNumel =
          std::max(features.maxTensorNumel, getStaticNumElements(type));
      if (inAnchor) {
        features.simtAnchors.maxElementBits = std::max(
            features.simtAnchors.maxElementBits, getTypeBitWidth(type));
        features.simtAnchors.maxTensorNumel = std::max(
            features.simtAnchors.maxTensorNumel, getStaticNumElements(type));
      }
    }
  };

  module.walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    const int64_t elements = getOperationElements(op);
    const int64_t loopMultiplier =
        getLoopMultiplier(op, structuralTripEstimates);
    const bool inAnchor = isInAnchor(op);
    if (inAnchor)
      ++features.simtAnchors.coveredOperationCount;

    for (Type type : op->getOperandTypes())
      updateTypeStats(type, inAnchor);
    for (Type type : op->getResultTypes())
      updateTypeStats(type, inAnchor);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          updateTypeStats(argument.getType(), inAnchor);

    if (name.starts_with("arith."))
      ++features.arithOps;
    if (name.starts_with("math."))
      ++features.mathOps;
    if (name.starts_with("scf.") || name.starts_with("cf.")) {
      features.hasControlFlow = true;
      if (inAnchor)
        features.simtAnchors.hasControlFlow = true;
    }
    if (name == "scope.scope")
      features.hasExplicitScope = true;

    auto incrementRaw = [&](int64_t &counter, int64_t &anchorCounter) {
      ++counter;
      if (inAnchor)
        ++anchorCounter;
    };
    if (name == "tt.load")
      incrementRaw(features.loadOps, features.simtAnchors.loadOps);
    else if (name == "tt.store")
      incrementRaw(features.storeOps, features.simtAnchors.storeOps);
    else if (name == "tt.reduce")
      incrementRaw(features.reduceOps, features.simtAnchors.reduceOps);
    else if (name == "tt.scan" || name == "tt.associative_scan")
      incrementRaw(features.scanOps, features.simtAnchors.scanOps);
    else if (name == "tt.gather")
      incrementRaw(features.gatherOps, features.simtAnchors.gatherOps);
    else if (name == "tt.dot")
      incrementRaw(features.dotOps, features.simtAnchors.dotOps);
    else if (name.starts_with("tt.atomic"))
      incrementRaw(features.atomicOps, features.simtAnchors.atomicOps);
    else if (name == "tt.histogram")
      incrementRaw(features.histogramOps, features.simtAnchors.histogramOps);
    else if (name == "tt.broadcast")
      ++features.broadcastOps;
    else if (name == "tt.expand_dims")
      ++features.expandDimsOps;
    else if (name == "tt.splat")
      ++features.splatOps;
    else if (name == "tt.addptr")
      ++features.addPtrOps;

    if (name == "arith.addf" || name == "arith.addi")
      ++features.addOps;
    else if (name == "arith.subf" || name == "arith.subi")
      ++features.subOps;
    else if (name == "arith.mulf" || name == "arith.muli")
      ++features.mulOps;
    else if (name == "arith.divf" || name == "arith.divsi" ||
             name == "arith.divui")
      ++features.divOps;
    else if (name == "arith.maxnumf" || name == "arith.maxf" ||
             name == "arith.maxsi" || name == "arith.maxui")
      ++features.maxOps;
    else if (name == "math.absf" || name == "math.absi")
      ++features.absOps;
    else if (name == "math.exp")
      ++features.expOps;
    else if (name == "math.log")
      ++features.logOps;
    else if (name == "arith.cmpf" || name == "arith.cmpi")
      ++features.cmpOps;
    else if (name == "arith.select")
      ++features.selectOps;
    else if (isCastOp(name))
      ++features.castOps;
    else if (name.starts_with("tt.clamp"))
      ++features.clampOps;

    llvm::StringRef weightedKind = classifyWeightedOp(name);
    if (!weightedKind.empty()) {
      int64_t weightedElements = elements;
      if (name == "tt.histogram" && op->getNumOperands() > 0)
        if (auto input =
                dyn_cast<RankedTensorType>(op->getOperand(0).getType()))
          if (input.hasStaticShape())
            weightedElements = getStaticNumElements(input);
      features.weightedOps[weightedKind] += loopMultiplier;
      features.opElements[weightedKind] += weightedElements * loopMultiplier;
      if (inAnchor) {
        features.simtAnchors.weightedOps[weightedKind] += loopMultiplier;
        features.simtAnchors.opElements[weightedKind] +=
            weightedElements * loopMultiplier;
      }
    }

    if (name == "scf.for") {
      auto knownTripCount = getKnownStaticLoopTripCount(op);
      if (!knownTripCount)
        features.hasUnknownTripCount = true;
      int64_t tripCount = getModeledLoopTripCount(op, structuralTripEstimates);
      const bool usedStructuralEstimate =
          !knownTripCount && structuralTripEstimates.contains(op);
      ++features.staticLoopCount;
      features.staticLoopTripCountSum += tripCount;
      features.staticLoopTripCountMax =
          std::max(features.staticLoopTripCountMax, tripCount);
      if (usedStructuralEstimate) {
        ++features.modeledDynamicLoopCount;
        features.modeledDynamicLoopTripCountSum += tripCount;
      }
      if (inAnchor) {
        ++features.simtAnchors.staticLoopCount;
        features.simtAnchors.staticLoopTripCountSum += tripCount;
        if (usedStructuralEstimate) {
          ++features.simtAnchors.modeledDynamicLoopCount;
          features.simtAnchors.modeledDynamicLoopTripCountSum += tripCount;
        }
      }
    }

    auto dataTypeAndElements = [&](bool load) -> std::pair<Type, int64_t> {
      if (load && op->getNumResults() > 0)
        return {op->getResult(0).getType(),
                getStaticNumElements(op->getResult(0).getType())};
      if (!load && op->getNumOperands() > 1)
        return {op->getOperand(1).getType(),
                getStaticNumElements(op->getOperand(1).getType())};
      if (op->getNumOperands() > 0)
        return {op->getOperand(0).getType(), elements};
      return {Type(), elements};
    };
    if (name == "tt.load" || name == "tt.store") {
      bool load = name == "tt.load";
      auto [dataType, dataElements] = dataTypeAndElements(load);
      int64_t bitWidth = dataType ? getTypeBitWidth(dataType) : 32;
      double bytes =
          static_cast<double>(dataElements) * loopMultiplier * bitWidth / 8.0;
      int64_t warpInstructions =
          static_cast<int64_t>(std::ceil(dataElements / 32.0)) * loopMultiplier;
      if (load) {
        features.loadBytes += bytes;
        features.loadWarpInstructions += warpInstructions;
        if (inAnchor) {
          features.simtAnchors.loadBytes += bytes;
          features.simtAnchors.loadWarpInstructions += warpInstructions;
        }
      } else {
        features.storeBytes += bytes;
        features.storeWarpInstructions += warpInstructions;
        if (inAnchor) {
          features.simtAnchors.storeBytes += bytes;
          features.simtAnchors.storeWarpInstructions += warpInstructions;
        }
      }
    }

    if (name == "tt.dot" && op->getNumOperands() >= 2) {
      auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
      if (lhs && rhs && lhs.getRank() >= 2 && rhs.getRank() >= 2) {
        int64_t m = lhs.getShape()[lhs.getRank() - 2];
        int64_t k = lhs.getShape()[lhs.getRank() - 1];
        int64_t n = rhs.getShape()[rhs.getRank() - 1];
        if (m > 0 && n > 0 && k > 0) {
          features.dotFlops += 2 * m * n * k * loopMultiplier;
          features.dotOutputElements += m * n * loopMultiplier;
          features.dotMNK.push_back({m, n, k});
          if (inAnchor)
            features.simtAnchors.dotFlops += 2 * m * n * k * loopMultiplier;
        }
      }
    }

    std::vector<int64_t> rankedResultAndOperandRanks;
    bool hasRankedInput = false;
    bool hasRankedResult = false;
    for (Type type : op->getOperandTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        rankedResultAndOperandRanks.push_back(tensor.getRank());
        hasRankedInput = true;
      }
    for (Type type : op->getResultTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        rankedResultAndOperandRanks.push_back(tensor.getRank());
        hasRankedResult = true;
      }
    if (name == "tt.reduce") {
      if (rankedResultAndOperandRanks.size() > 1) {
        auto [minimum, maximum] =
            std::minmax_element(rankedResultAndOperandRanks.begin(),
                                rankedResultAndOperandRanks.end());
        if (*maximum > *minimum)
          ++features.rowLocalReduceOps;
      }
      if (hasRankedInput && !hasRankedResult)
        ++features.vectorReduceToScalarOps;
      if (op->getNumOperands() > 0) {
        auto source = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        auto axis = op->getAttrOfType<IntegerAttr>("axis");
        if (source && source.hasStaticShape() && axis) {
          int64_t axisValue = axis.getInt();
          if (axisValue < 0)
            axisValue += source.getRank();
          if (axisValue >= 0 && axisValue < source.getRank()) {
            int64_t extent = source.getShape()[axisValue];
            if (extent > 0) {
              features.maxReduceAxisExtent =
                  std::max(features.maxReduceAxisExtent, extent);
              features.weightedReduceAxisElements += extent * loopMultiplier;
              const int64_t shuffleLevels = static_cast<int64_t>(
                  std::ceil(std::log2(static_cast<double>(extent))));
              const int64_t inputElements = getStaticNumElements(source);
              const int64_t shuffleLaneSteps =
                  inputElements * shuffleLevels * loopMultiplier;
              features.shuffleLaneSteps += shuffleLaneSteps;
              if (inAnchor) {
                features.simtAnchors.maxReduceAxisExtent =
                    std::max(features.simtAnchors.maxReduceAxisExtent, extent);
                features.simtAnchors.weightedReduceAxisElements +=
                    extent * loopMultiplier;
                features.simtAnchors.shuffleLaneSteps += shuffleLaneSteps;
              }
            }
          }
        }
      }
    }

    if ((name == "tt.scan" || name == "tt.associative_scan") &&
        op->getNumOperands() > 0) {
      auto source = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (source && source.hasStaticShape() && axis) {
        int64_t axisValue = axis.getInt();
        if (axisValue < 0)
          axisValue += source.getRank();
        if (axisValue >= 0 && axisValue < source.getRank()) {
          int64_t extent = source.getShape()[axisValue];
          if (extent > 0) {
            const int64_t shuffleLevels = static_cast<int64_t>(
                std::ceil(std::log2(static_cast<double>(extent))));
            const int64_t laneSteps =
                getStaticNumElements(source) * shuffleLevels * loopMultiplier;
            features.shuffleLaneSteps += laneSteps;
            if (inAnchor)
              features.simtAnchors.shuffleLaneSteps += laneSteps;
          }
        }
      }
    }

    std::vector<int64_t> maskRanks;
    for (Type type : op->getOperandTypes())
      if (isMaskTensorType(type))
        maskRanks.push_back(cast<RankedTensorType>(type).getRank());
    for (Type type : op->getResultTypes())
      if (isMaskTensorType(type))
        maskRanks.push_back(cast<RankedTensorType>(type).getRank());
    if (!maskRanks.empty()) {
      ++features.maskTensorOps;
      for (int64_t rank : maskRanks) {
        features.maskRankSum += rank;
        if (inAnchor)
          features.simtAnchors.maskRankSum += rank;
      }
      auto addPredicateLaneEvaluations = [&](Type type) {
        if (!isMaskTensorType(type))
          return;
        const int64_t laneEvaluations =
            getStaticNumElements(type) * loopMultiplier;
        features.predicateLaneEvaluations += laneEvaluations;
        if (inAnchor)
          features.simtAnchors.predicateLaneEvaluations += laneEvaluations;
      };
      for (Type type : op->getOperandTypes())
        addPredicateLaneEvaluations(type);
      for (Type type : op->getResultTypes())
        addPredicateLaneEvaluations(type);
      if (name == "tt.broadcast" || name == "tt.expand_dims")
        ++features.maskBroadcastOps;
    }

    auto recordUniqueMask = [&](Value value) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.getElementType().isInteger(1))
        return;
      if (uniqueMasks.insert(value).second) {
        ++features.uniqueMaskValues;
        features.uniqueMaskRankSum += type.getRank();
        const int64_t elements = getStaticNumElements(type);
        features.predicateElements += elements;
      }
      if (inAnchor && anchorUniqueMasks.insert(value).second) {
        ++features.simtAnchors.uniqueMaskValues;
        features.simtAnchors.uniqueMaskRankSum += type.getRank();
        const int64_t elements = getStaticNumElements(type);
        features.simtAnchors.predicateElements += elements;
      }
    };
    for (Value operand : op->getOperands())
      recordUniqueMask(operand);
    for (Value result : op->getResults())
      recordUniqueMask(result);

    bool isPointerOperation =
        name == "tt.addptr" || name == "tt.load" || name == "tt.store";
    if (isLoadedIndexDependentMemoryOp(op)) {
      ++features.loadedIndexDependentMemoryOps;
      if (inAnchor)
        ++features.simtAnchors.loadedIndexDependentMemoryOps;
    }
    if (isPointerOperation) {
      std::set<std::string> uniqueShapes;
      auto collectShape = [&](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor)
          return;
        std::string key;
        llvm::raw_string_ostream os(key);
        os << tensor.getRank();
        for (int64_t dim : tensor.getShape())
          os << 'x' << dim;
        os.flush();
        uniqueShapes.insert(std::move(key));
      };
      for (Type type : op->getOperandTypes())
        collectShape(type);
      for (Type type : op->getResultTypes())
        collectShape(type);
      int64_t maxPointerRank = 0;
      for (const std::string &shape : uniqueShapes) {
        llvm::StringRef shapeRef(shape);
        int64_t rank = 0;
        (void)shapeRef.take_front(shapeRef.find('x')).getAsInteger(10, rank);
        ++features.pointerTensorOps;
        if (inAnchor)
          ++features.simtAnchors.pointerTensorOps;
        maxPointerRank = std::max(maxPointerRank, rank);
        if (rank > 1)
          features.pointerUnstructuredDims += rank;
      }
      if (maxPointerRank > 1) {
        ++features.laneDependentPointerOps;
        if (inAnchor)
          ++features.simtAnchors.laneDependentPointerOps;
      }
    }

    bool anyRankedType = llvm::any_of(op->getOperandTypes(), [](Type type) {
      return isa<RankedTensorType>(type);
    });
    anyRankedType |= llvm::any_of(op->getResultTypes(), [](Type type) {
      return isa<RankedTensorType>(type);
    });
    if (name == "tt.load" && !anyRankedType && op->getNumOperands() > 0 &&
        isPointerType(op->getOperand(0).getType()))
      ++features.scalarLoadOps;
    if (name == "tt.store" && !anyRankedType && op->getNumOperands() > 0 &&
        isPointerType(op->getOperand(0).getType()))
      ++features.scalarStoreOps;
    if (name == "tt.splat" && op->getNumOperands() > 0 &&
        op->getNumResults() > 0 && isPointerType(op->getOperand(0).getType()) &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      ++features.vectorPtrSplatOps;
  });

  features.scalarOps = features.addOps + features.subOps + features.mulOps +
                       features.divOps + features.maxOps + features.absOps +
                       features.expOps + features.logOps + features.cmpOps +
                       features.selectOps + features.castOps +
                       features.clampOps;
  features.hasDot = features.dotOps > 0;
  features.hasGather = features.gatherOps > 0;
  features.hasAtomic = features.atomicOps > 0;
  features.hasHistogram = features.histogramOps > 0;
  features.hasScan = features.scanOps > 0;
  features.rank1IndirectVectorReduce =
      features.maxTensorRank == 1 && features.reduceOps > 0 &&
      features.vectorReduceToScalarOps > 0 && features.vectorPtrSplatOps > 0 &&
      features.scalarLoadOps >= 2;

  return features;
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::estimateSimdSimtCandidates(
    const SimdSimtFeatureSummary &features,
    const SimdSimtCostModelOptions &options) {
  if (!std::isfinite(options.marginRatio) || options.marginRatio < 0.0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT marginRatio must be finite and non-negative");
  auto profileOrError = loadCandidateProfile(options.profilePath);
  if (!profileOrError)
    return profileOrError.takeError();
  CandidateProfile profile = std::move(*profileOrError);

  SimdSimtCostReport report;
  report.profileVersion = profile.profileVersion;
  report.profileTarget = profile.target;
  report.actualTarget = options.actualTarget;
  report.profileContentSha256 = profile.contentSha256;
  report.selectionProfileContentSha256 = profile.selectionContentSha256;
  report.microbenchmarkProfileVersion = profile.microbenchmarkProfileVersion;
  report.microbenchmarkProfileTarget = profile.microbenchmarkProfileTarget;
  report.microbenchmarkProfileContentSha256 =
      profile.microbenchmarkContentSha256;
  report.scoreUnit = profile.scoreUnit;
  report.minimumConfidenceForDecision = profile.minimumConfidence;
  report.targetCompatible = targetMatches(profile, options.actualTarget);
  report.features = features;
  report.applicability =
      evaluateSimtApplicability(features, options.compileOn91095);
  report.allSimdCandidateLegal =
      features.simtAnchors.kernelLowerability.allSimd ==
      CandidateLoweringStatus::Native;
  report.allSimtOnlyCandidateLegal =
      options.compileOn91095 && !features.hasExplicitScope &&
      features.simtAnchors.kernelLowerability.allSimtOnly ==
          CandidateLoweringStatus::Native;
  report.mixedCandidateLegal = !features.hasExplicitScope &&
                               report.applicability.materializable &&
                               features.simtAnchors.kernelLowerability.mixed ==
                                   CandidateLoweringStatus::Native;
  report.includeFeaturesInJSON = options.includeFeaturesInJSON;
  report.marginRatio = options.marginRatio;

  // Coverage is a validity check over extracted features, not a cost term.
  // Evaluate it before all resource, structural, and transition scoring so
  // production auto selection can reject out-of-domain kernels cheaply.
  const int64_t weightedReductions =
      mapValue(features.weightedOps, "reduce", features.reduceOps);
  const int64_t dotFlops = features.dotFlops;
  const int64_t pointerOps = std::max<int64_t>(1, features.pointerTensorOps);
  report.breakdown.irregularDensity = std::min(
      1.0, static_cast<double>(features.laneDependentPointerOps) / pointerOps);
  auto [covered, domain] =
      rankingCalibrationCoverage(features, weightedReductions, dotFlops,
                                 profile, report.breakdown.irregularDensity);
  report.calibrationCovered = covered;
  report.calibrationDomain = std::move(domain);
  report.selectionScoreValid = covered;

  const EventRouteCalibrationProfile *eventRouteCalibration = nullptr;
  if (report.calibrationCovered) {
    auto calibration =
        profile.eventRouteCalibration.find(report.calibrationDomain);
    if (calibration != profile.eventRouteCalibration.end()) {
      eventRouteCalibration = &calibration->second;
      report.eventRouteCalibrationApplied = true;
      report.eventAllSimtOnlyValidated =
          eventRouteCalibration->allSimtOnlyValidated;
      report.eventMixedSimdSimtValidated =
          eventRouteCalibration->mixedSimdSimtValidated;
      report.eventRouteCalibrationSource = eventRouteCalibration->source;
      report.eventRouteCalibrationConfidence =
          eventRouteCalibration->confidence;
      report.eventRouteScoreMultipliers = {
          eventRouteCalibration->allSimdMultiplier,
          eventRouteCalibration->allSimtOnlyMultiplier,
          eventRouteCalibration->mixedSimdSimtMultiplier};

      // BackendConditional is deliberately not globally selectable.  A
      // bounded Event domain may promote it only after the whole-kernel pure
      // SIMT path passed correctness on the target represented by this
      // versioned profile.  Unsupported/AliasesMixed statuses never promote.
      if (options.compileOn91095 && !features.hasExplicitScope &&
          eventRouteCalibration->allSimtOnlyValidated &&
          features.simtAnchors.kernelLowerability.allSimtOnly ==
              CandidateLoweringStatus::BackendConditional)
        report.allSimtOnlyCandidateLegal = true;
      if (!eventRouteCalibration->mixedSimdSimtValidated)
        report.mixedCandidateLegal = false;
    }
  }

  if (!report.calibrationCovered && !options.scoreOutsideCalibrationCoverage) {
    if (!report.targetCompatible)
      report.gateReasons.push_back("target_incompatible");
    report.gateReasons.push_back("selection_score_invalid");
    report.gatePassed = false;
    return report;
  }

  const int64_t numWarps =
      std::max<int64_t>(1, static_cast<int64_t>(options.numWarps));
  const int64_t maxNumel = std::max<int64_t>(1, features.maxTensorNumel);
  const int64_t elementBits =
      features.maxElementBits > 0
          ? std::max<int64_t>(8, features.maxElementBits)
          : 32;
  const int64_t vectorWidth =
      std::max<int64_t>(1, profile.simdVectorWidthBits / elementBits);
  std::vector<std::string> resourceConfidence;
  if (report.allSimtOnlyCandidateLegal)
    resourceConfidence.push_back(profile.simtSetupConfidence);
  if (report.mixedCandidateLegal)
    resourceConfidence.push_back(profile.mixedSetupFallbackConfidence);
  if (eventRouteCalibration)
    resourceConfidence.push_back(eventRouteCalibration->confidence);

  llvm::StringMap<int64_t> rawCountByKind;
  rawCountByKind["gather"] = features.gatherOps;
  rawCountByKind["histogram"] = features.histogramOps;
  rawCountByKind["atomic"] = features.atomicOps;
  for (llvm::StringRef kind : {"gather", "histogram", "atomic"}) {
    int64_t coreWork = mapValue(features.opElements, kind,
                                mapValue(rawCountByKind, kind) * maxNumel);
    if (coreWork > 0)
      report.unsupported.push_back((kind + "_core_cost_uncalibrated").str());
  }

  int64_t classifiedScalarOps =
      features.addOps + features.subOps + features.mulOps + features.divOps +
      features.maxOps + features.absOps + features.expOps + features.logOps +
      features.cmpOps + features.selectOps + features.castOps +
      features.clampOps;
  int64_t unclassifiedScalarOps =
      std::max<int64_t>(0, features.scalarOps - classifiedScalarOps);
  if (unclassifiedScalarOps)
    report.unsupported.push_back(std::to_string(unclassifiedScalarOps) +
                                 " unclassified arithmetic ops");

  for (const auto &[opName, elements] : getProfileOpElements(features)) {
    if (elements <= 0)
      continue;
    auto simdIterator = profile.simdOps.find(opName);
    auto simtIterator = profile.simtOps.find(opName);
    if (simdIterator == profile.simdOps.end() ||
        simtIterator == profile.simtOps.end()) {
      report.unsupported.push_back(opName.str());
      continue;
    }
    const OpProfile &simd = simdIterator->second;
    const OpProfile &simt = simtIterator->second;
    if (simd.throughput <= 0.0 || simt.throughput <= 0.0) {
      report.unsupported.push_back(opName.str());
      continue;
    }
    double simdCycles = std::ceil(static_cast<double>(elements) / vectorWidth) /
                        simd.throughput * simd.factor;
    double simtCycles =
        static_cast<double>(elements) / simt.throughput * simt.factor;
    report.breakdown.simdOpSystemCycles[opName] = simdCycles;
    report.breakdown.simtOpSystemCycles[opName] = simtCycles;
    report.breakdown.simdComputeCycles += simdCycles;
    report.breakdown.simtComputeCycles += simtCycles;
    resourceConfidence.push_back(simd.confidence);
    resourceConfidence.push_back(simt.confidence);
  }

  for (const auto &[opName, elements] :
       getProfileOpElements(features.simtAnchors)) {
    if (elements <= 0)
      continue;
    auto simdIterator = profile.simdOps.find(opName);
    auto simtIterator = profile.simtOps.find(opName);
    if (simdIterator == profile.simdOps.end() ||
        simtIterator == profile.simtOps.end())
      continue;
    const OpProfile &simd = simdIterator->second;
    const OpProfile &simt = simtIterator->second;
    if (simd.throughput <= 0.0 || simt.throughput <= 0.0)
      continue;
    report.breakdown.mixedSimdRegularComputeCycles -=
        std::ceil(static_cast<double>(elements) / vectorWidth) /
        simd.throughput * simd.factor;
    report.breakdown.mixedSimtAnchorComputeCycles +=
        static_cast<double>(elements) / simt.throughput * simt.factor;
  }
  report.breakdown.mixedSimdRegularComputeCycles +=
      report.breakdown.simdComputeCycles;
  report.breakdown.mixedSimdRegularComputeCycles =
      std::max(0.0, report.breakdown.mixedSimdRegularComputeCycles);

  report.breakdown.simdLoadCycles =
      features.loadBytes / profile.simdMte2BytesPerCycle;
  report.breakdown.simdStoreCycles =
      features.storeBytes / profile.simdMte3BytesPerCycle;
  report.breakdown.simdMemoryCycles = std::max(
      report.breakdown.simdLoadCycles, report.breakdown.simdStoreCycles);
  const double mixedSimdRegularLoadCycles =
      std::max(0.0, features.loadBytes - features.simtAnchors.loadBytes) /
      profile.simdMte2BytesPerCycle;
  const double mixedSimdRegularStoreCycles =
      std::max(0.0, features.storeBytes - features.simtAnchors.storeBytes) /
      profile.simdMte3BytesPerCycle;
  report.breakdown.mixedSimdRegularMemoryCycles =
      std::max(mixedSimdRegularLoadCycles, mixedSimdRegularStoreCycles);
  if (features.loadBytes != 0.0 || features.storeBytes != 0.0)
    resourceConfidence.push_back(profile.simdMemoryConfidence);

  const int64_t loadWarpInstructions =
      features.loadWarpInstructions != 0
          ? features.loadWarpInstructions
          : features.loadOps *
                static_cast<int64_t>(std::ceil(static_cast<double>(maxNumel) /
                                               profile.simtWarpSize));
  const int64_t storeWarpInstructions =
      features.storeWarpInstructions != 0
          ? features.storeWarpInstructions
          : features.storeOps *
                static_cast<int64_t>(std::ceil(static_cast<double>(maxNumel) /
                                               profile.simtWarpSize));
  report.features.loadWarpInstructions = loadWarpInstructions;
  report.features.storeWarpInstructions = storeWarpInstructions;
  report.breakdown.simtLoadCycles =
      loadWarpInstructions / profile.simtLoadWarpRate;
  report.breakdown.simtStoreCycles =
      storeWarpInstructions / profile.simtStoreWarpRate;
  report.breakdown.simtMemoryCycles =
      report.breakdown.simtLoadCycles + report.breakdown.simtStoreCycles;
  report.breakdown.mixedSimtAnchorMemoryCycles =
      features.simtAnchors.loadWarpInstructions / profile.simtLoadWarpRate +
      features.simtAnchors.storeWarpInstructions / profile.simtStoreWarpRate;
  if (loadWarpInstructions != 0 || storeWarpInstructions != 0)
    resourceConfidence.push_back(profile.simtMemoryConfidence);

  const int64_t weightedScans =
      mapValue(features.weightedOps, "scan", features.scanOps);
  if (weightedScans)
    report.unsupported.push_back("scan_template_ranking_uncalibrated");
  const int64_t shuffleLevels = static_cast<int64_t>(
      std::ceil(std::log2(static_cast<double>(profile.simtWarpSize))));
  report.breakdown.simtShuffleInstructions =
      features.shuffleLaneSteps > 0
          ? std::ceil(static_cast<double>(features.shuffleLaneSteps) /
                      profile.simtWarpSize)
          : static_cast<double>(weightedReductions + weightedScans) *
                std::ceil(static_cast<double>(maxNumel) /
                          profile.simtWarpSize) *
                shuffleLevels;
  report.breakdown.simtShuffleCycles =
      report.breakdown.simtShuffleInstructions / profile.simtShuffleRate;
  const int64_t anchorWeightedReductions =
      mapValue(features.simtAnchors.weightedOps, "reduce",
               features.simtAnchors.reduceOps);
  const int64_t anchorWeightedScans = mapValue(
      features.simtAnchors.weightedOps, "scan", features.simtAnchors.scanOps);
  const int64_t anchorMaxNumel =
      std::max<int64_t>(1, features.simtAnchors.maxTensorNumel);
  const double anchorShuffleInstructions =
      features.simtAnchors.shuffleLaneSteps > 0
          ? std::ceil(
                static_cast<double>(features.simtAnchors.shuffleLaneSteps) /
                profile.simtWarpSize)
          : static_cast<double>(anchorWeightedReductions +
                                anchorWeightedScans) *
                std::ceil(static_cast<double>(anchorMaxNumel) /
                          profile.simtWarpSize) *
                shuffleLevels;
  report.breakdown.mixedSimtAnchorShuffleCycles =
      anchorShuffleInstructions / profile.simtShuffleRate;
  if (report.breakdown.simtShuffleInstructions != 0.0)
    resourceConfidence.push_back(profile.simtShuffleConfidence);

  report.breakdown.simtPredicateInstructions =
      features.predicateLaneEvaluations > 0
          ? std::ceil(static_cast<double>(features.predicateLaneEvaluations) /
                      profile.simtWarpSize)
          : static_cast<double>(features.maskRankSum) *
                std::ceil(static_cast<double>(maxNumel) / profile.simtWarpSize);
  report.breakdown.simtPredicateCycles =
      report.breakdown.simtPredicateInstructions / profile.simtPredicateRate;
  const double anchorPredicateInstructions =
      features.simtAnchors.predicateLaneEvaluations > 0
          ? std::ceil(static_cast<double>(
                          features.simtAnchors.predicateLaneEvaluations) /
                      profile.simtWarpSize)
          : static_cast<double>(features.simtAnchors.maskRankSum) *
                std::ceil(static_cast<double>(anchorMaxNumel) /
                          profile.simtWarpSize);
  report.breakdown.mixedSimtAnchorPredicateCycles =
      anchorPredicateInstructions / profile.simtPredicateRate;

  if (dotFlops) {
    report.breakdown.simdDotCycles =
        profile.simdDotSetupCycles +
        static_cast<double>(dotFlops) / profile.simdDotFlopsPerCycle;
    report.breakdown.simtDotCycles =
        profile.simtDotSetupCycles +
        static_cast<double>(dotFlops) / profile.simtDotFlopsPerCycle;
    resourceConfidence.push_back(profile.simdDotConfidence);
    resourceConfidence.push_back(profile.simtDotConfidence);
  }
  const int64_t regularDotFlops =
      std::max<int64_t>(0, dotFlops - features.simtAnchors.dotFlops);
  if (regularDotFlops)
    report.breakdown.mixedSimdRegularDotCycles =
        profile.simdDotSetupCycles +
        static_cast<double>(regularDotFlops) / profile.simdDotFlopsPerCycle;
  if (features.simtAnchors.dotFlops)
    report.breakdown.mixedSimtAnchorDotCycles =
        profile.simtDotSetupCycles +
        static_cast<double>(features.simtAnchors.dotFlops) /
            profile.simtDotFlopsPerCycle;

  report.breakdown.simdSetupCycles = profile.simdSetupCycles;
  report.breakdown.simtSetupCycles = profile.simtSetupCycles;
  report.breakdown.simdIssuePayloadCycles = std::max(
      report.breakdown.simdComputeCycles + report.breakdown.simdDotCycles,
      report.breakdown.simdMemoryCycles);
  // The current SIMT lowering emits a dependency-ordered warp instruction
  // stream. Loads feed compute, compute feeds shuffle/reduction, and stores
  // consume the result. There is no measured overlap contract that would
  // justify a roofline max(compute, memory), so charge the serial path.
  report.breakdown.simtIssuePayloadCycles =
      report.breakdown.simtComputeCycles + report.breakdown.simtShuffleCycles +
      report.breakdown.simtDotCycles + report.breakdown.simtMemoryCycles +
      report.breakdown.simtPredicateCycles;
  report.breakdown.programIssueScale = profile.programIssueScale;
  report.breakdown.simdAnalyticalCycles =
      profile.simdSetupCycles +
      report.breakdown.simdIssuePayloadCycles * profile.programIssueScale;
  report.breakdown.simtAnalyticalCycles =
      profile.simtSetupCycles +
      report.breakdown.simtIssuePayloadCycles * profile.programIssueScale;

  const bool tinyDot =
      dotFlops > 0 && dotFlops <= profile.structural.tinyDotFlopsMax;
  report.breakdown.tinyDotUnderfill =
      tinyDot ? std::max(0.0, 1.0 - static_cast<double>(dotFlops) /
                                        profile.structural.tinyDotFlopsMax)
              : 0.0;
  const double irregularPerDensity =
      tinyDot ? profile.structural.tinyDotIrregularPerDensity
              : profile.structural.irregularPerDensity;
  const double irregularCap = tinyDot ? profile.structural.tinyDotIrregularCap
                                      : profile.structural.irregularCap;
  report.breakdown.structuralComponents["irregular_addressing"] = std::min(
      irregularCap, report.breakdown.irregularDensity * irregularPerDensity);
  report.breakdown.structuralComponents["mask_materialization"] =
      std::min(profile.structural.maskCap,
               features.maskRankSum * profile.structural.perMaskRank);
  report.breakdown.structuralComponents["reduction_lowering"] =
      std::min(profile.structural.reductionCap,
               weightedReductions * profile.structural.perWeightedReduction);
  report.breakdown.structuralComponents["static_loop_control"] = std::min(
      profile.structural.loopCap,
      features.staticLoopTripCountSum * profile.structural.perStaticLoopTrip);
  report.breakdown.structuralComponents["control_flow"] =
      features.hasControlFlow ? profile.structural.controlFlow : 0.0;
  report.breakdown.structuralComponents["tiny_dot_startup"] =
      tinyDot ? profile.structural.tinyDot * report.breakdown.tinyDotUnderfill
              : 0.0;
  report.breakdown.structuralComponents["rank1_indirect_vector_reduction"] =
      features.rank1IndirectVectorReduce
          ? profile.structural.rank1IndirectVectorReduction
          : 0.0;
  for (const auto &component : report.breakdown.structuralComponents)
    report.breakdown.structuralPenaltyRatio += component.second;
  // Candidate costs must remain independent. Structural terms describe work
  // omitted by the SIMD roofline, so charge them against A_SIMD itself;
  // changing SIMT throughput/setup must never change the all-SIMD score.
  report.breakdown.simdStructuralPenaltyCycles =
      report.breakdown.simdAnalyticalCycles *
      report.breakdown.structuralPenaltyRatio;
  report.candidateCosts.allSimd = report.breakdown.simdAnalyticalCycles +
                                  report.breakdown.simdStructuralPenaltyCycles;
  report.candidateCosts.allSimtOnly = report.breakdown.simtAnalyticalCycles;

  const MixedSetupFallbackProfile *nearestSetupFallback = nullptr;
  for (const MixedSetupFallbackProfile &fallback : profile.mixedSetupFallbacks)
    if (!nearestSetupFallback ||
        std::abs(fallback.numWarps - numWarps) <
            std::abs(nearestSetupFallback->numWarps - numWarps))
      nearestSetupFallback = &fallback;
  if (!nearestSetupFallback)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile has no mixed setup "
                                   "fallback");
  report.breakdown.mixedSetupFallbackNumWarps = nearestSetupFallback->numWarps;
  report.breakdown.standaloneSimtSetupCycles = profile.simtSetupCycles;
  report.breakdown.mixedSetupFallbackCycles =
      nearestSetupFallback->emptySimtSetupCycles;
  report.breakdown.setupProxyDeltaCycles =
      std::max(0.0, report.breakdown.mixedSetupFallbackCycles -
                        report.breakdown.standaloneSimtSetupCycles);

  // A mixed route is not a convex blend of two whole-kernel costs. Charge
  // exact materializable anchors at SIMT rates and remaining operations at
  // SIMD rates. The regular SIMD phase keeps its measured roofline model;
  // the SIMT anchor is a serial load/compute/shuffle/store instruction path.
  report.breakdown.mixedSimdRegularPayloadCycles =
      std::max(report.breakdown.mixedSimdRegularComputeCycles +
                   report.breakdown.mixedSimdRegularDotCycles,
               report.breakdown.mixedSimdRegularMemoryCycles);
  report.breakdown.mixedSimtAnchorPayloadCycles =
      report.breakdown.mixedSimtAnchorComputeCycles +
      report.breakdown.mixedSimtAnchorDotCycles +
      report.breakdown.mixedSimtAnchorShuffleCycles +
      report.breakdown.mixedSimtAnchorMemoryCycles +
      report.breakdown.mixedSimtAnchorPredicateCycles;

  const int64_t remainingPointerOps = std::max<int64_t>(
      0, features.pointerTensorOps - features.simtAnchors.pointerTensorOps);
  const int64_t remainingLaneDependentPointerOps =
      std::max<int64_t>(0, features.laneDependentPointerOps -
                               features.simtAnchors.laneDependentPointerOps);
  const double remainingIrregularDensity =
      remainingPointerOps > 0
          ? std::min(1.0,
                     static_cast<double>(remainingLaneDependentPointerOps) /
                         remainingPointerOps)
          : 0.0;
  const int64_t remainingMaskRank = std::max<int64_t>(
      0, features.maskRankSum - features.simtAnchors.maskRankSum);
  const int64_t remainingWeightedReductions =
      std::max<int64_t>(0, weightedReductions - anchorWeightedReductions);
  const int64_t remainingLoopTrips =
      std::max<int64_t>(0, features.staticLoopTripCountSum -
                               features.simtAnchors.staticLoopTripCountSum);
  const bool remainingControlFlow =
      features.hasControlFlow && !features.simtAnchors.hasControlFlow;
  const bool remainingRank1Reduction =
      features.rank1IndirectVectorReduce && remainingWeightedReductions > 0;
  const bool remainingTinyDot = regularDotFlops > 0 && tinyDot;

  double remainingStructuralPenalty = 0.0;
  remainingStructuralPenalty +=
      std::min(irregularCap, remainingIrregularDensity * irregularPerDensity);
  remainingStructuralPenalty +=
      std::min(profile.structural.maskCap,
               remainingMaskRank * profile.structural.perMaskRank);
  remainingStructuralPenalty += std::min(
      profile.structural.reductionCap,
      remainingWeightedReductions * profile.structural.perWeightedReduction);
  remainingStructuralPenalty +=
      std::min(profile.structural.loopCap,
               remainingLoopTrips * profile.structural.perStaticLoopTrip);
  if (remainingControlFlow)
    remainingStructuralPenalty += profile.structural.controlFlow;
  if (remainingRank1Reduction)
    remainingStructuralPenalty +=
        profile.structural.rank1IndirectVectorReduction;
  if (remainingTinyDot)
    remainingStructuralPenalty +=
        profile.structural.tinyDot * report.breakdown.tinyDotUnderfill;
  report.breakdown.mixedRemainingStructuralPenaltyRatio =
      remainingStructuralPenalty;

  double totalPartitionWork = features.loadBytes + features.storeBytes +
                              static_cast<double>(features.dotFlops);
  for (const auto &entry : features.opElements)
    totalPartitionWork += std::max<int64_t>(0, entry.second);
  double anchorPartitionWork =
      features.simtAnchors.loadBytes + features.simtAnchors.storeBytes +
      static_cast<double>(features.simtAnchors.dotFlops);
  for (const auto &entry : features.simtAnchors.opElements)
    anchorPartitionWork += std::max<int64_t>(0, entry.second);
  report.breakdown.mixedSimdFraction =
      totalPartitionWork > 0.0
          ? std::clamp(1.0 - anchorPartitionWork / totalPartitionWork, 0.0, 1.0)
          : 0.0;

  if (features.simtAnchors.count > 0) {
    const double regularPayloadWithResidual =
        report.breakdown.mixedSimdRegularPayloadCycles *
        (1.0 + remainingStructuralPenalty);
    report.candidateCosts.mixedSimdSimt =
        report.breakdown.mixedSetupFallbackCycles +
        profile.programIssueScale *
            (regularPayloadWithResidual +
             report.breakdown.mixedSimtAnchorPayloadCycles) +
        report.breakdown.mixedBoundaryCycles;
    report.breakdown.mixedCostSource =
        "materializable_anchor_resource_partition";
  } else {
    report.candidateCosts.mixedSimdSimt =
        std::max(report.candidateCosts.allSimd,
                 report.candidateCosts.allSimtOnly) +
        report.breakdown.mixedSetupFallbackCycles;
    report.breakdown.mixedCostSource =
        "inapplicable_without_materializable_anchor";
  }

  report.uncalibratedCandidateCosts = report.candidateCosts;
  if (eventRouteCalibration) {
    // The analytical/structural formula supplies the feature-sensitive base.
    // A bounded, versioned Event multiplier corrects its route-relative
    // residual for the admitted domain.  Keeping the raw score in the report
    // makes the empirical correction explicit instead of hiding it in a
    // structural penalty or a Python heuristic.
    report.candidateCosts.allSimd *= eventRouteCalibration->allSimdMultiplier;
    report.candidateCosts.allSimtOnly *=
        eventRouteCalibration->allSimtOnlyMultiplier;
    report.candidateCosts.mixedSimdSimt *=
        eventRouteCalibration->mixedSimdSimtMultiplier;
  }

  report.candidateCostsEvaluated = true;
  sortAndUnique(report.unsupported);
  const unsigned legalCandidateCount =
      static_cast<unsigned>(report.allSimdCandidateLegal) +
      static_cast<unsigned>(report.allSimtOnlyCandidateLegal) +
      static_cast<unsigned>(report.mixedCandidateLegal);
  if (legalCandidateCount == 0)
    return llvm::createStringError(
        std::errc::not_supported,
        "SIMD/SIMT route model found no independently lowerable candidate");
  report.decision =
      chooseBest(report.candidateCosts, report.allSimdCandidateLegal,
                 report.allSimtOnlyCandidateLegal, report.mixedCandidateLegal);
  report.runnerUp =
      chooseRunnerUp(report.candidateCosts, report.allSimdCandidateLegal,
                     report.allSimtOnlyCandidateLegal,
                     report.mixedCandidateLegal, report.decision);
  report.bestScore = report.candidateCosts.get(report.decision);
  report.runnerUpScore = report.candidateCosts.get(report.runnerUp);
  if (legalCandidateCount <= 1) {
    report.decisionAdvantage = 0.0;
  } else if (report.allSimdCandidateLegal) {
    report.decisionAdvantage =
        report.decision == SimdSimtCandidateKind::AllSIMD
            ? report.runnerUpScore - report.bestScore
            : report.candidateCosts.allSimd - report.bestScore;
  } else {
    report.decisionAdvantage = report.runnerUpScore - report.bestScore;
  }
  report.gainScore = report.decisionAdvantage;
  const double gainBaseline = report.allSimdCandidateLegal
                                  ? report.candidateCosts.allSimd
                                  : report.runnerUpScore;
  report.requiredGainScore =
      legalCandidateCount > 1
          ? std::max(64.0, gainBaseline * options.marginRatio)
          : 0.0;
  double ratioDenominator =
      std::max(1.0e-9, std::min({report.candidateCosts.allSimd,
                                 report.candidateCosts.allSimtOnly,
                                 report.candidateCosts.mixedSimdSimt}));
  report.candidateRatiosToBest = {
      report.candidateCosts.allSimd / ratioDenominator,
      report.candidateCosts.allSimtOnly / ratioDenominator,
      report.candidateCosts.mixedSimdSimt / ratioDenominator};

  report.absoluteConfidence = minimumConfidence(resourceConfidence);
  if (!report.unsupported.empty()) {
    report.rankingConfidence = "none";
  } else if (report.breakdown.structuralPenaltyRatio > 0.0 && covered) {
    report.rankingConfidence = minimumConfidence(
        {report.absoluteConfidence, profile.rankingConfidence});
  } else {
    report.rankingConfidence = report.absoluteConfidence;
  }
  if (!report.targetCompatible)
    report.rankingConfidence = "none";

  if (!report.targetCompatible)
    report.gateReasons.push_back("target_incompatible");
  if (!report.selectionScoreValid)
    report.gateReasons.push_back("selection_score_invalid");
  if (!report.unsupported.empty())
    report.gateReasons.push_back("unsupported_cost_terms");
  if (confidenceRank(report.rankingConfidence) <
      confidenceRank(report.minimumConfidenceForDecision))
    report.gateReasons.push_back("ranking_confidence_" +
                                 report.rankingConfidence + "_below_" +
                                 report.minimumConfidenceForDecision);
  if (legalCandidateCount > 1 &&
      report.decision != SimdSimtCandidateKind::AllSIMD &&
      !(report.decisionAdvantage > report.requiredGainScore))
    report.gateReasons.push_back("decision_advantage_not_above_required_gain");
  report.gatePassed = report.gateReasons.empty();
  return report;
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimdSimtCostModelOptions &options) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan =
      buildMixedSimtAnchorPlan(module, options.compileOn91095);
  return analyzeSimdSimtCandidates(module, anchorPlan, options);
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimtAnchorPlan &anchorPlan,
    const SimdSimtCostModelOptions &options) {
  auto features = analyzeSimdSimtFeatures(module, anchorPlan);
  if (!features)
    return features.takeError();
  return estimateSimdSimtCandidates(*features, options);
}
