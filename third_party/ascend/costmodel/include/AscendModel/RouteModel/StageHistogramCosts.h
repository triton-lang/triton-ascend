//===- StageHistogramCosts.h - Calibrated histogram cost table -*- C++ -*-===//
//
// `tt.histogram` spends its time inside the lowering template, so TTIR resource
// counting cannot price it; the op is billed from a calibrated coefficient
// table carried by the device profile instead.  A row is a bag of named
// coefficients, so a new coefficient or row normally needs only the profile to
// change.  The one thing C++ still spells out is which rows must exist: a
// missing row has no safe default and must fail loudly.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGEHISTOGRAMCOSTS_H
#define ASCENDMODEL_ROUTEMODEL_STAGEHISTOGRAMCOSTS_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace llvm {
namespace json {
class Object;
} // namespace json
} // namespace llvm

namespace mlir::ascend {

// Defined in StageCostModels.h; only referenced here.
struct LogicalStage;

/// Structure constants of the histogram lowering that the calibrated rows
/// mirror; they select which row a shape is billed against.
struct HistogramStructure {
  int64_t chunkLanes = 0;     ///< Lanes per dhistv2 instruction.
  int64_t threadsPerWarp = 0; ///< Lanes per SIMT warp (simt_only split).
  int64_t maxSegmentsU16 = 0;
  int64_t maxSegmentsU32 = 0;
  int64_t selfAmortisedSegments = 0; ///< Segments self-amortised by one pass.
  int64_t chunksPerSegment = 0;      ///< Chunks each further segment supplies.
  int64_t flushChunks = 0;           ///< Chunks a dhistv2 accumulator may hold.
  double atomicWindowBytes = 0.0;    ///< Per-thread simt_only working window.
};

/// One calibrated `tt.histogram` row: the coefficient bag the profile row
/// literally is.  A row carries only the coefficients its formula uses, and the
/// evaluator composes them as
///   chunk         C = intercept + per_chunk*chunks
///   chunk_knee    C = intercept + per_chunk*min(chunks, knee_chunks)
///                       + per_chunk_beyond_knee*max(chunks - knee_chunks, 0)
///   chunk_segment C = intercept + per_chunk_per_segment*chunks*segments
///                       + per_segment*segments
///   simt_template C = intercept + (scan_per_elem + count_per_elem)*n
///   simt_only     C = intercept + per_elem*n + per_iter*iter
///                       + per_bin*bpt_read + per_thread*threads
///                       + per_kink*max(0, n - atomic_window_bytes/width_bytes)
struct HistogramRateRow {
  llvm::StringMap<double> coefficients;

  bool has(llvm::StringRef key) const { return coefficients.count(key) != 0; }
  double get(llvm::StringRef key) const {
    const auto it = coefficients.find(key);
    return it == coefficients.end() ? 0.0 : it->second;
  }
};

/// Row names each group must define.  This is the only place C++ names a row,
/// the fail-fast contract that keeps a profile from silently pricing a shape at
/// zero.  Extending a group means editing the profile rows and the list here.
inline constexpr llvm::StringLiteral kHistogramSimdTemplateKeys[] = {
    "u8",
    "u16_bins_lt_256",
    "u16_bins_256_unmasked",
    "u16_bins_256_masked",
    "u16_bins_gt_256",
    "u32_bins_lt_256_unmasked",
    "u32_bins_lt_256_masked",
    "u32_bins_256_unmasked",
    "u32_bins_256_masked",
    "u32_bins_gt_256",
    "u32_small_bins_park",
    "u32_small_bins_pred_mask",
    "u64_bins_lt_256_unmasked",
    "u64_bins_lt_256_masked",
    "u64_bins_256_unmasked",
    "u64_bins_256_masked"};
inline constexpr llvm::StringLiteral kHistogramSimtTemplateKeys[] = {
    "u8", "u16", "u32"};
inline constexpr llvm::StringLiteral kHistogramSimtOnlyKeys[] = {"u8", "u16",
                                                                 "u32", "u64"};

/// Calibrated `tt.histogram` rows, grouped and keyed exactly like the profile
/// `histogram` section, so the model looks a row up from the derived shape.
struct StageHistogramRates {
  HistogramStructure structure;
  llvm::StringMap<HistogramRateRow> simdTemplate;
  llvm::StringMap<HistogramRateRow> simtTemplate;
  llvm::StringMap<HistogramRateRow> simtOnly;

  bool isValid() const;
};

/// Fill `rates` from the profile `histogram` section.  Every row named above is
/// required: there is no compiled-in fallback, so a profile that omits a row is
/// reported instead of silently pricing the op at zero.
llvm::Error readHistogramRates(const llvm::json::Object &section,
                               StageHistogramRates &rates);

/// Calibrated cost of one `tt.histogram` of the given shape, in system cycles
/// per call: derive the shape key, select the row, evaluate its coefficients.
double estimateHistogramCycles(int64_t inputElements, int64_t numBins,
                               unsigned widthBytes, StageMode mode,
                               bool unmasked, int64_t numWarps,
                               const StageHistogramRates &rates);

/// Sum `estimateHistogramCycles` over every statically shaped `tt.histogram`
/// owned by `stage`, billing each op once per Stage iteration.  Returns whether
/// any such op was found, so the caller keeps its generic estimate if none.
bool estimateHistogramStageCost(const LogicalStage &stage, StageMode mode,
                                double &perCallTotal, int64_t numWarps,
                                const StageHistogramRates &rates);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGEHISTOGRAMCOSTS_H
