#include "AscendModel/RouteModel/Reduction/ReductionProfile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <cmath>
#include <system_error>
using namespace mlir::ascend;

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
