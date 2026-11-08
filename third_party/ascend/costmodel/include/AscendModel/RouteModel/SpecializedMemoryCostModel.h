//===- SpecializedMemoryCostModel.h - Public structured GM model --*- C++
//-*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef ASCENDMODEL_ROUTEMODEL_SPECIALIZEDMEMORYCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_SPECIALIZEDMEMORYCOSTMODEL_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::ascend {

enum class SpecializedMemoryFamily { Segments, Paged };
enum class SpecializedMemoryMode { SIMD, SIMT };
enum class SpecializedMemoryOperation { Load, Store };
enum class SpecializedMemoryMask {
  None,
  Prefix,
  Rectangular,
  WholeRows,
  WholePages
};
enum class SpecializedMemoryVisibility { Constexpr, Host, DeviceHidden };

/// Facts proved from the target pointer/mask SSA and explicitly provided Host
/// launch scalars. Unknown facts remain nullopt. Index/page-table contents,
/// addresses, seeds, physical footprints and sampled mask ratios are not
/// inputs.
///
/// This model predicts one complete measured memory fragment in microseconds,
/// including address/index preparation and the formal fragment's sink/envelope.
/// It is not an additive single-instruction or arbitrary production-stage cost.
struct SpecializedMemoryGeometry {
  SpecializedMemoryFamily family = SpecializedMemoryFamily::Segments;
  SpecializedMemoryMode mode = SpecializedMemoryMode::SIMD;
  SpecializedMemoryOperation operation = SpecializedMemoryOperation::Load;
  SpecializedMemoryMask mask = SpecializedMemoryMask::None;
  SpecializedMemoryVisibility visibility =
      SpecializedMemoryVisibility::Constexpr;
  std::vector<int64_t> shape;
  std::string pattern; // D: loaded symbolic index; S: regular coordinate.
  std::string payloadType = "float32";
  int64_t storageBytes = 4;
  int64_t warps = 1;
  int64_t outerFactor = 1;

  /// Positive last-axis step, literal or explicitly held by Host. Device-only
  /// values must remain unknown, even when available to an offline harness.
  std::optional<int64_t> regularStep;
  /// Constant coefficients proved by affine target-pointer analysis. The vector
  /// has one entry per axis; entries for the other axis kind may be nullopt.
  /// A Host value does not turn an unproved constant SSA coefficient into one.
  std::vector<std::optional<int64_t>> regularAxisCoefficients;
  std::vector<std::optional<int64_t>> discreteAxisCoefficients;

  /// Bound of the proved structured predicate, literal or explicitly supplied
  /// by Host. It is independent of whether the regular stride is device-only.
  std::optional<int64_t> maskBound;
  std::optional<int64_t> pageSize;
  std::optional<int64_t> logicalOrigin;
  /// Distinguish a literal origin from a Host-provided origin. The aligned
  /// page feature requires the literal proof used by the frozen extractor.
  bool logicalOriginIsLiteral = false;

  /// The caller must own the complete boundary described above. A stage-only
  /// caller receives an explicit unsupported result instead of a guessed scale.
  bool completeFragmentBoundary = false;
  /// TTIR-proved direct fragment: three pointer arguments, one rank1 i32/i64
  /// index Load, one DS payload access and its contiguous source/sink; no other
  /// payload arithmetic or memory operations. This is a boundary, not a kernel
  /// ID.
  bool directThreePointerEnvelope = false;
};

struct SpecializedMemoryEstimate {
  bool supported = false;
  double microseconds = 0.0;
  const char *reason = "uninitialized";
  const char *modelSHA256 = nullptr;
};

/// Frozen 950PR F1 profile, <=60 parameters per mode/operation. No fitting,
/// device access, dynamic loading or profile discovery occurs in this API.
SpecializedMemoryEstimate
estimateSpecializedMemory(const SpecializedMemoryGeometry &geometry);

const char *getSpecializedMemoryModelSHA256();
unsigned
getSpecializedMemoryParameterCount(SpecializedMemoryMode mode,
                                   SpecializedMemoryOperation operation);

} // namespace mlir::ascend
#endif
