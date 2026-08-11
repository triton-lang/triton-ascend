//===- MicrobenchmarkProfile.h - Shared cost-model evidence -----*- C++ -*-===//
//
// This file defines the versioned, model-neutral measurement catalog shared
// by the absolute autotune cost model and the SIMD/SIMT route selector.
//
// Measurements describe observed hardware behaviour.  Model-specific
// calibration (for example structural penalties, coverage gates, or pipeline
// overlap) deliberately does not belong in this profile.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_PROFILE_MICROBENCHMARKPROFILE_H
#define ASCENDMODEL_PROFILE_MICROBENCHMARKPROFILE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>

namespace mlir {
namespace ascend {

struct MicrobenchmarkMeasurement {
  double value = 0.0;
  std::string unit;
  std::string cycleDomain;
  std::string scope;
  std::string sourceKind;
  std::string source;
  std::string confidence;
};

/// A model-neutral catalog of target facts and microbenchmark measurements.
///
/// Every scalar carries its physical unit, cycle domain, measurement scope,
/// provenance, and confidence.  Consumers may use different subsets and
/// different formulas, but must not reinterpret a value in another unit or
/// scope silently.
class MicrobenchmarkProfile {
public:
  static llvm::Expected<MicrobenchmarkProfile>
  loadFromFile(llvm::StringRef path);

  static llvm::Expected<MicrobenchmarkProfile>
  loadFromJSON(const llvm::json::Value &json,
               llvm::StringRef contentForHash = {});

  llvm::StringRef getProfileVersion() const { return profileVersion; }
  llvm::StringRef getTarget() const { return target; }
  llvm::StringRef getContentSha256() const { return contentSha256; }

  const MicrobenchmarkMeasurement *getMeasurement(llvm::StringRef key) const;

  llvm::Expected<double> requireValue(
      llvm::StringRef key,
      std::optional<llvm::StringRef> expectedUnit = std::nullopt,
      std::optional<llvm::StringRef> expectedCycleDomain = std::nullopt) const;

  /// Convert a rate expressed per source-domain cycle into a rate expressed
  /// per destination-domain cycle.
  llvm::Expected<double>
  convertRatePerCycle(double rate, llvm::StringRef sourceFrequencyKey,
                      llvm::StringRef destinationFrequencyKey) const;

private:
  std::string profileVersion;
  std::string target;
  std::string contentSha256;
  llvm::StringMap<MicrobenchmarkMeasurement> measurements;
};

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_PROFILE_MICROBENCHMARKPROFILE_H
