//===- StridedShortCostModel.h - StridedShort direct access model -------===//
//
// Closed-form completion costs for StridedShort direct load/store accesses.
// The shape of every formula lives in StridedShortCostModel.cpp; the numbers
// live in the profile.  Nothing here reads an operator or kernel name: an
// access is priced from its static geometry alone.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ASCEND_MODEL_ROUTEMODEL_STRIDEDSHORTCOSTMODEL_H
#define MLIR_ASCEND_MODEL_ROUTEMODEL_STRIDEDSHORTCOSTMODEL_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace mlir::ascend {

/// Fitted numbers of one slot.  The shape lives in StridedShortCostModel.cpp.
struct StridedShortCompletionFit {
  double intercept = 0.0;
  std::array<double, 8> coefficient{};
  std::array<double, 2> structural{};
  std::string evidence;
  bool isValid() const;
};

/// One slot per calibrated (mode, direction, stride regime).
enum class StridedShortFormula {
  SimdLoadStride2R1,
  SimdLoadStrideGT2R1Dense,
  SimdLoadStrideGT2R1Sparse, // inner stride > 32 B: own slot, shorter shape
  SimdLoadStride2R2,
  SimdLoadStrideGT2R2,
  SimdStoreR1,
  SimdStoreR2,
  SimdStoreMaskedStrideGT2, // masked: destination read-back, own shape
  SimdStoreMaskedStride2,   // masked at stride 2: block move + extract_slice
  SimtLoadR1,
  SimtLoadR2,
  SimtStoreR1,
  SimtStoreR2,
};

/// Address-set statistics of one access, derived from its static geometry.
/// Together with direction and mode these are the only inputs to selection.
struct StridedAddressStats {
  int64_t elements = 0;
  int64_t elementBytes = 0;
  int64_t innerStrideBytes = 0;
  int64_t rowPitchBytes = 0;
  int64_t rows = 1;
  int64_t cols = 0;
  int64_t spanBytes = 0;
  std::map<int64_t, int64_t> maxPerBlock; // max elements sharing a B-byte block
  std::map<int64_t, int64_t>
      sumSquaresPerBlock;                    // sum of cnt^2 over B-byte blocks
  std::map<int64_t, int64_t> distinctBlocks; // distinct B-byte blocks touched
  std::map<int64_t, int64_t> rowBlocksSum;   // same, counted per row and summed
  std::map<int64_t, int64_t> sharedBlocks; // B-byte blocks shared by >= 2 rows
  std::map<int64_t, int64_t> maxPerRowBlock; // max one row's elements per block
  double meanWarpLines = 0.0;                // 128 B lines per 32-lane wave
  int64_t maxConc1k = 0;      // max 128 B lines in one 1 KB block
  int64_t maxRowsPerKilo = 0; // max rows meeting in one 1 KB block
};

/// The one and only string form of a slot: the profile JSON key.
llvm::StringRef stringifyStridedShortFormula(StridedShortFormula formula);

/// Select the slot for one access.  Pure function of geometry, statistics and
/// mode; each slot's domain filter lives in the branch that returns it, so no
/// second predicate can disagree.  nullopt = no slot owns it, use the fallback.
std::optional<StridedShortFormula>
selectStridedShortFormula(const StageMemoryAccess &access,
                          const StridedAddressStats &stats, bool simd);

/// Evaluate one slot at one access.  nullopt = the fit is unusable, so the
/// caller must leave the bytes to the analytical fallback.
std::optional<double> evaluateStridedShortCompletionFormula(
    StridedShortFormula formula, const StridedShortCompletionFit &fit,
    const StridedAddressStats &stats, int64_t warps);

} // namespace mlir::ascend

#endif // MLIR_ASCEND_MODEL_ROUTEMODEL_STRIDEDSHORTCOSTMODEL_H
