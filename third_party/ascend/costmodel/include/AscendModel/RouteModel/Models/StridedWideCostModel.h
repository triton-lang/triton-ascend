//===- StridedWideCostModel.h - StridedWide direct access model
//----------------===//
//
// Closed-form completion costs for StridedWide direct load/store accesses
// (rank-2, exactly one unit-stride axis, contiguous run >= 32 B, inner /
// row-major orientation).  Unmasked accesses are priced directly; a masked
// access is admitted only when the SIMD path and the mask is dynamically
// all-true (no statically-provable partial coverage), because the measured
// masked/unmasked ratio for whole-row masks is 0.994-0.998.  The shape of every
// formula lives in StridedWideCostModel.cpp; the numbers live in the profile
// under stage_resources.<mode>.memory.strided_wide_completion.  Nothing here
// reads an operator or kernel name: an access is priced from its static
// geometry alone, and only inside the swept domain -- outside it the access
// keeps the fallback mechanisms.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ASCEND_MODEL_ROUTEMODEL_STRIDEDWIDECOSTMODEL_H
#define MLIR_ASCEND_MODEL_ROUTEMODEL_STRIDEDWIDECOSTMODEL_H

#include "AscendModel/RouteModel/Models/StridedShortCostModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::ascend {

/// Fitted numbers of one slot: a plain linear model
///   T = intercept + sum_i coefficient[i] * term_i
/// whose term order is documented per case in StridedWideCostModel.cpp and
/// mirrored by the profile's "terms" array.  All numbers are SYS_CNT
/// nanoseconds, which are system cycles at the profile's 1 GHz counter.
struct SwideCompletionFit {
  static constexpr size_t kMaxTerms = 32;
  double intercept = 0.0;
  std::array<double, kMaxTerms> coefficient{};
  /// SuperBlock factor correction, shared by the SIMT slots of one rows
  /// segment (load/store and masked/unmasked share the same a,b):
  ///   scale(F) = 1 / (1 + a * x + b * x^2),  x = 1/F - 1
  /// so scale(1) is exactly 1 and the existing F1 coefficients stay intact.
  /// The correction is applied to the per-program cost; applySuperBlock still
  /// multiplies the body by F once, reconstructing the measured block time
  /// (the F programs overlap instead of serializing).
  std::array<double, 2> factorScaleCoefficient{};
  std::string evidence;
  bool isValid() const;
  double factorScale(int64_t factor) const;
};

/// One slot per calibrated (mode, direction, rows/W segment).
enum class SwideFormula {
  SimdLoadSwideRowsLe4,
  SimdLoadSwideRowsGt4,
  SimdStoreSwideRowsLe2,
  SimdStoreSwideRowsGt2,
  SimtLoadSwideRows2,
  SimtLoadSwideRows8,
  SimtLoadSwideRows64Wle2,
  SimtLoadSwideRows64W38,
  SimtLoadSwideRows64Wgt8,
  SimtStoreSwideRows2,
  SimtStoreSwideRows8,
  SimtStoreSwideRows64Wle2,
  SimtStoreSwideRows64W38,
  SimtStoreSwideRows64Wgt8,
  // Masked SIMT slots: own coefficients per segment (the masked relation is
  // not a smooth offset of the unmasked one).  SIMD masked shares the slots
  // above because the measured masked/unmasked ratio is 1.000 there.
  SimtLoadSwideMaskedRows2,
  SimtLoadSwideMaskedRows8,
  SimtLoadSwideMaskedRows64Wle2,
  SimtLoadSwideMaskedRows64W38,
  SimtLoadSwideMaskedRows64Wgt8,
  SimtLoadSwideMaskedRows16Wle2,
  SimtLoadSwideMaskedRows16W38,
  SimtLoadSwideMaskedRows16Wgt8,
  SimtStoreSwideMaskedRows2,
  SimtStoreSwideMaskedRows8,
  SimtStoreSwideMaskedRows64Wle2,
  SimtStoreSwideMaskedRows64W38,
  SimtStoreSwideMaskedRows64Wgt8,
  SimtStoreSwideMaskedRows16Wle2,
  SimtStoreSwideMaskedRows16W38,
  SimtStoreSwideMaskedRows16Wgt8,
};

/// The one and only string form of a slot: the profile JSON key.
llvm::StringRef stringifySwideFormula(SwideFormula formula);

/// The byte windows whose address-set statistics the fits read.
llvm::ArrayRef<int64_t> swideStatWindows();

/// Select the slot for one access.  Pure function of geometry, statistics and
/// mode; returns nullopt outside the swept domain so the access keeps the
/// layout-memory curves / analytical fallback.
std::optional<SwideFormula> selectSwideFormula(const StageMemoryAccess &access,
                                               const StridedAddressStats &stats,
                                               bool simd, int64_t warps);

/// Evaluate one slot at one access (system cycles for ONE operation;
/// the caller scales by the access count).  nullopt = the fit is unusable.
std::optional<double> evaluateSwideFormula(SwideFormula formula,
                                           const SwideCompletionFit &fit,
                                           const StridedAddressStats &stats,
                                           int64_t warps);

} // namespace mlir::ascend

#endif // MLIR_ASCEND_MODEL_ROUTEMODEL_STRIDEDWIDECOSTMODEL_H
