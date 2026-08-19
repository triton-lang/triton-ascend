//===- MicrobenchmarkProfile.cpp - Shared cost-model evidence -------------===//

#include "AscendModel/Profile/MicrobenchmarkProfile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static std::string sha256(llvm::StringRef content) {
  llvm::ArrayRef<uint8_t> bytes(
      reinterpret_cast<const uint8_t *>(content.data()), content.size());
  auto digest = llvm::SHA256::hash(bytes);
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

static llvm::Error invalidProfile(llvm::Twine message) {
  return llvm::createStringError(std::errc::invalid_argument,
                                 "invalid microbenchmark profile: %s",
                                 message.str().c_str());
}

} // namespace

llvm::Expected<MicrobenchmarkProfile>
MicrobenchmarkProfile::loadFromFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read microbenchmark profile '%s'",
                                   path.str().c_str());

  llvm::StringRef content = buffer.get()->getBuffer();
  auto parsed = llvm::json::parse(content);
  if (!parsed)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to parse microbenchmark profile '%s': %s", path.str().c_str(),
        llvm::toString(parsed.takeError()).c_str());
  return loadFromJSON(*parsed, content);
}

llvm::Expected<MicrobenchmarkProfile>
MicrobenchmarkProfile::loadFromJSON(const llvm::json::Value &json,
                                    llvm::StringRef contentForHash) {
  const auto *root = json.getAsObject();
  if (!root)
    return invalidProfile("root must be an object");

  auto schemaVersion = root->getInteger("schema_version");
  if (!schemaVersion || *schemaVersion != 1)
    return invalidProfile("schema_version must be 1");

  auto version = root->getString("profile_version");
  auto targetName = root->getString("target");
  const auto *rawMeasurements = root->getObject("measurements");
  if (!version || version->empty())
    return invalidProfile("profile_version must be a non-empty string");
  if (!targetName || targetName->empty())
    return invalidProfile("target must be a non-empty string");
  if (!rawMeasurements || rawMeasurements->empty())
    return invalidProfile("measurements must be a non-empty object");

  MicrobenchmarkProfile result;
  result.profileVersion = version->str();
  result.target = targetName->str();

  for (const auto &entry : *rawMeasurements) {
    llvm::StringRef key = entry.first;
    const auto *object = entry.second.getAsObject();
    if (!object)
      return invalidProfile("measurement '" + key + "' must be an object");

    auto value = object->getNumber("value");
    auto unit = object->getString("unit");
    auto cycleDomain = object->getString("cycle_domain");
    auto scope = object->getString("scope");
    auto sourceKind = object->getString("source_kind");
    auto source = object->getString("source");
    auto confidence = object->getString("confidence");
    if (!value || !std::isfinite(*value))
      return invalidProfile("measurement '" + key +
                            "' must have a finite numeric value");
    if (!unit || unit->empty())
      return invalidProfile("measurement '" + key + "' must have a unit");
    if (!cycleDomain || cycleDomain->empty())
      return invalidProfile("measurement '" + key +
                            "' must have a cycle_domain");
    if (!scope || scope->empty())
      return invalidProfile("measurement '" + key + "' must have a scope");
    if (!sourceKind || sourceKind->empty())
      return invalidProfile("measurement '" + key +
                            "' must have a source_kind");
    if (!source || source->empty())
      return invalidProfile("measurement '" + key + "' must have a source");
    if (!confidence || confidence->empty())
      return invalidProfile("measurement '" + key + "' must have a confidence");
    if (*confidence != "high" && *confidence != "medium" &&
        *confidence != "low")
      return invalidProfile("measurement '" + key +
                            "' confidence must be high, medium, or low");
    if (*cycleDomain != "none" && *cycleDomain != "wall_clock" &&
        *cycleDomain != "SYS_CNT" && *cycleDomain != "device_compute")
      return invalidProfile("measurement '" + key +
                            "' has an unsupported cycle_domain");

    MicrobenchmarkMeasurement measurement;
    measurement.value = *value;
    measurement.unit = unit->str();
    measurement.cycleDomain = cycleDomain->str();
    measurement.scope = scope->str();
    measurement.sourceKind = sourceKind->str();
    measurement.source = source->str();
    measurement.confidence = confidence->str();
    result.measurements[key] = std::move(measurement);
  }

  if (!contentForHash.empty()) {
    result.contentSha256 = sha256(contentForHash);
  } else {
    std::string serialized;
    llvm::raw_string_ostream stream(serialized);
    stream << json;
    stream.flush();
    result.contentSha256 = sha256(serialized);
  }
  return result;
}

const MicrobenchmarkMeasurement *
MicrobenchmarkProfile::getMeasurement(llvm::StringRef key) const {
  auto iterator = measurements.find(key);
  return iterator == measurements.end() ? nullptr : &iterator->second;
}

llvm::Expected<double> MicrobenchmarkProfile::requireValue(
    llvm::StringRef key, std::optional<llvm::StringRef> expectedUnit,
    std::optional<llvm::StringRef> expectedCycleDomain) const {
  const MicrobenchmarkMeasurement *measurement = getMeasurement(key);
  if (!measurement)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "microbenchmark profile is missing measurement '%s'",
        key.str().c_str());
  if (expectedUnit && measurement->unit != *expectedUnit)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "microbenchmark measurement '%s' has unit '%s', expected '%s'",
        key.str().c_str(), measurement->unit.c_str(),
        expectedUnit->str().c_str());
  if (expectedCycleDomain && measurement->cycleDomain != *expectedCycleDomain)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "microbenchmark measurement '%s' has cycle_domain '%s', expected '%s'",
        key.str().c_str(), measurement->cycleDomain.c_str(),
        expectedCycleDomain->str().c_str());
  return measurement->value;
}

llvm::Expected<double> MicrobenchmarkProfile::convertRatePerCycle(
    double rate, llvm::StringRef sourceFrequencyKey,
    llvm::StringRef destinationFrequencyKey) const {
  auto sourceMHz = requireValue(sourceFrequencyKey, "MHz", "wall_clock");
  if (!sourceMHz)
    return sourceMHz.takeError();
  auto destinationMHz =
      requireValue(destinationFrequencyKey, "MHz", "wall_clock");
  if (!destinationMHz)
    return destinationMHz.takeError();
  if (*sourceMHz <= 0.0 || *destinationMHz <= 0.0)
    return invalidProfile("cycle-domain frequencies must be positive");
  return rate * *sourceMHz / *destinationMHz;
}
