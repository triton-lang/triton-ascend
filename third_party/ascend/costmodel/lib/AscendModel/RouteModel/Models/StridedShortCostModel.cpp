//===- StridedShortCostModel.cpp - StridedShort direct access model ---===//
//
// Shape of every closed-form strided direct completion cost.  The numbers
// come from the profile; nothing here reads an operator or kernel name.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/Models/StridedShortCostModel.h"

#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace mlir;

namespace mlir::ascend {

// Address-set statistics of one access.

bool StridedShortCompletionFit::isValid() const {
  if (evidence.empty())
    return false;
  if (!std::isfinite(intercept))
    return false;
  for (double value : coefficient)
    if (!std::isfinite(value))
      return false;
  for (double value : structural)
    if (!std::isfinite(value))
      return false;
  return true;
}

llvm::StringRef stringifyStridedShortFormula(StridedShortFormula formula) {
  switch (formula) {
  case StridedShortFormula::SimdLoadStride2R1:
    return "simd_load_stride2_r1";
  case StridedShortFormula::SimdLoadStrideGT2R1Dense:
    return "simd_load_stride_gt2_r1_dense";
  case StridedShortFormula::SimdLoadStrideGT2R1Sparse:
    return "simd_load_stride_gt2_r1_sparse";
  case StridedShortFormula::SimdLoadStride2R2:
    return "simd_load_stride2_r2";
  case StridedShortFormula::SimdLoadStrideGT2R2:
    return "simd_load_stride_gt2_r2";
  case StridedShortFormula::SimdStoreR1:
    return "simd_store_r1";
  case StridedShortFormula::SimdStoreR2:
    return "simd_store_r2";
  case StridedShortFormula::SimdStoreMaskedStrideGT2:
    return "simd_store_masked_stride_gt2";
  case StridedShortFormula::SimdStoreMaskedStride2:
    return "simd_store_masked_stride2";
  case StridedShortFormula::SimtLoadR1:
    return "simt_load_r1";
  case StridedShortFormula::SimtLoadR2:
    return "simt_load_r2";
  case StridedShortFormula::SimtStoreR1:
    return "simt_store_r1";
  case StridedShortFormula::SimtStoreR2:
    return "simt_store_r2";
  }
  llvm_unreachable("unhandled strided direct formula");
}

std::optional<StridedShortFormula>
selectStridedShortFormula(const StageMemoryAccess &access,
                          const StridedAddressStats &stats, bool simd) {
  // Common domain precondition.  Every filter below is copied from the
  // corresponding PHYS report's scope list: a point outside its fitted grid
  // must fall through to the analytical fallback rather than be extrapolated,
  // because the fits are least squares over a bounded sweep, not physical laws
  // that hold outside it.
  if (access.elementBits % 8 != 0)
    return std::nullopt;
  const double eb = static_cast<double>(stats.elementBytes);
  const double isb = static_cast<double>(stats.innerStrideBytes);
  const double n = static_cast<double>(stats.elements);
  const double R = static_cast<double>(stats.rows);
  const double C = static_cast<double>(stats.cols);
  if (eb <= 0.0 || isb <= 0.0 || n <= 0.0)
    return std::nullopt;

  const bool rankOne = access.effectiveRank() <= 1;
  if (access.store) {
    if (simd) {
      // A dynamic mask is NOT a multiplier on the unmasked slots: it inserts a
      // destination read-back, a select and a whole-tile write-back, so the
      // masked stores own two calibrated slots of their own, one per lowering.
      // The unmasked fits were swept without any mask and must not price a
      // masked access (the measured ratio is 1.42-1.45 for a provable-prefix
      // mask, and far larger for a mask that cuts a row).
      if (access.masked) {
        // Fitted domain (README 3.4): eb <= 4 B and an element stride of
        // 2..128, swept over rank one and rank two, 482 points in total.
        if (eb > 4.0 || isb > 128.0 * eb)
          return std::nullopt;
        // The rank-two sweep starts at eight columns; a narrower tile was
        // never measured, so it keeps the analytical fallback.
        if (!rankOne && C < 8.0)
          return std::nullopt;
        const double stride = isb / eb;
        if (stride >= 3.0)
          return StridedShortFormula::SimdStoreMaskedStrideGT2;
        if (stride >= 2.0)
          return StridedShortFormula::SimdStoreMaskedStride2;
        // A stride of one element is contiguous, not strided.
        return std::nullopt;
      }
      // The unmasked slots also exclude ISB > 128 (each element on its own
      // cache line is a scatter, not a tile store) and ISB < 2*eb (that regime
      // belongs to the wide/contiguous path, which these formulas do not
      // describe).
      if (rankOne) {
        if (eb <= 4.0 && isb <= 128.0 && isb >= 2.0 * eb)
          return StridedShortFormula::SimdStoreR1;
        return std::nullopt;
      }
      if (eb <= 4.0 && C * eb >= 32.0 && isb >= 2.0 * eb)
        return StridedShortFormula::SimdStoreR2;
      return std::nullopt;
    }
    if (rankOne) {
      if (eb <= 8.0)
        return StridedShortFormula::SimtStoreR1;
      return std::nullopt;
    }
    if (eb <= 8.0 && R >= 2.0 && C * eb >= 32.0 && C >= 8.0 && n >= 64.0)
      return StridedShortFormula::SimtStoreR2;
    return std::nullopt;
  }

  if (!simd) {
    if (rankOne) {
      if (eb <= 4.0 && isb <= 128.0)
        return StridedShortFormula::SimtLoadR1;
      return std::nullopt;
    }
    if (eb <= 4.0 && R >= 2.0)
      return StridedShortFormula::SimtLoadR2;
    return std::nullopt;
  }

  // A SIMD rank-one load has two distinct lowerings: a block move when the
  // inner stride is exactly two elements wide, and an element gather
  // otherwise.  The wide slots were swept at ISB == 2*eb only, with eb <= 4.
  if (rankOne) {
    if (eb <= 4.0 && isb == 2.0 * eb)
      return StridedShortFormula::SimdLoadStride2R1;
    if (n >= 32.0 && isb <= 128.0) {
      // Past 32 B of inner stride every element owns its 64 B block, so the
      // dense fit's crowding terms carry no information and one six-term fit
      // over both regimes is biased in each.  The split is a property of the
      // access geometry, never of a caller.
      if (isb > 32.0)
        return StridedShortFormula::SimdLoadStrideGT2R1Sparse;
      return StridedShortFormula::SimdLoadStrideGT2R1Dense;
    }
    return std::nullopt;
  }
  if (eb <= 4.0 && R >= 2.0 && C >= 4.0 && isb == 2.0 * eb)
    return StridedShortFormula::SimdLoadStride2R2;
  // footprint/payload bounds how far the R rows drift apart.
  const double pitchBytes = static_cast<double>(stats.rowPitchBytes);
  const double footprint = (R - 1.0) * pitchBytes + (C - 1.0) * isb + eb;
  const double payload = n * eb;
  if (R >= 2.0 && payload > 0.0 && footprint / payload <= 32.0)
    return StridedShortFormula::SimdLoadStrideGT2R2;
  return std::nullopt;
}

// Formula evaluation.

/// Evaluate one strided direct completion formula.
///
/// Formula -> coefficient layout (values come from the profile):
///   SimdLoadStride2R1    T = i + c0*span - c1*max(0, span - s0)      [s0=knee]
///   SimdLoadStride2R2 A  T = i + c0*R + c1*rmax1k + c2*R*[rb<32]     [src>=32]
///   SimdLoadStride2R2 B  T = c3 + c4*n + c5*nblk64                   [src<32]
///   SimdLoadStrideGT2R1Dense  T = i + c0*n + c1*(n*eb) + c2*mx64 + c3*mx512
///                         + c4*sq1024
///   SimdLoadStrideGT2R2  T = i + c0*bs*L64 + c1*(1-bs)*n*wrow64
///                         + c2*(1-bs)*w1024 + c3*bs*L256 + c4*bs*multi256
///                         + c5*(n*n*eb)
///   SimdStoreR1       T = i + c0*nbytes + c1*mx128 + c2*S64 + c3*S512
///   SimdStoreR2       T = i + c0*Q4 + c1*n*Q4 + c2*max(0, n - s0)
///                                                          [s0=cmd floor]
///   SimdStoreMaskedStrideGT2     T = i + c0*nbytes + c1*mx128 + c2*S64 +
///   c3*S512
///                               + c4*S1024
///   SimdStoreMaskedStride2  same terms as StrideGT2, stride == 2 numbers
///   SimtLoadR1        T = max(c0*W, c1*W*iters*mwl,
///                             i + c2*min(mwl,8) + c3*min(mwl,4)
///                               + c4*iters + c5*L)
///   SimtLoadR2        T = max(c0*W, c1*W*iters*mwl,
///                             i + c2*conc1k + c3*iters + c4*min(mwl,4))
///   SimtStoreR1       T = max(c0*W, i + c1*iters,
///                             L128*(c2*[eb=1] + c3*[eb=2]
///                                   + c4*[eb=4] + c5*[eb=8]))
///   SimtStoreR2       T = max(L128*(c0*[eb<4] + c1*[eb=4] + c2*[eb=8]),
///                             n*min(R,32)*(c3*[eb<4] + c4*[eb=4] + c5*[eb=8]),
///                             c6*max(0, iters - s0), s1)
///
/// The switch deliberately has no default label: -Wswitch then proves that
/// every enumerator is priced, so a new slot cannot silently fall back to "no
/// formula" the way a chain of string comparisons silently fell through.
std::optional<double> evaluateStridedShortCompletionFormula(
    StridedShortFormula formula, const StridedShortCompletionFit &fit,
    const StridedAddressStats &stats, int64_t warps) {
  if (!fit.isValid())
    return std::nullopt;
  const double n = static_cast<double>(stats.elements);
  const double eb = static_cast<double>(stats.elementBytes);
  const double isb = static_cast<double>(stats.innerStrideBytes);
  const double R = static_cast<double>(stats.rows);
  const double C = static_cast<double>(stats.cols);
  const double iters = static_cast<double>(llvm::divideCeil(
      stats.elements, int64_t(32) * std::max<int64_t>(1, warps)));
  const double W = static_cast<double>(std::max<int64_t>(1, warps));
  auto maxBlock = [&](int64_t window) {
    auto it = stats.maxPerBlock.find(window);
    return it == stats.maxPerBlock.end() ? 0.0
                                         : static_cast<double>(it->second);
  };
  auto sumSquares = [&](int64_t window) {
    auto it = stats.sumSquaresPerBlock.find(window);
    return it == stats.sumSquaresPerBlock.end()
               ? 0.0
               : static_cast<double>(it->second);
  };
  auto distinct = [&](int64_t window) {
    auto it = stats.distinctBlocks.find(window);
    return it == stats.distinctBlocks.end() ? 0.0
                                            : static_cast<double>(it->second);
  };
  auto shared = [&](int64_t window) {
    auto it = stats.sharedBlocks.find(window);
    return it == stats.sharedBlocks.end() ? 0.0
                                          : static_cast<double>(it->second);
  };
  auto maxRowBlock = [&](int64_t window) {
    auto it = stats.maxPerRowBlock.find(window);
    return it == stats.maxPerRowBlock.end() ? 0.0
                                            : static_cast<double>(it->second);
  };
  const std::array<double, 8> &c = fit.coefficient;
  const std::array<double, 2> &s = fit.structural;

  switch (formula) {
  case StridedShortFormula::SimdLoadStride2R1: {
    const double span = static_cast<double>(stats.spanBytes);
    return fit.intercept + c[0] * span - c[1] * std::max(0.0, span - s[0]);
  }
  case StridedShortFormula::SimdLoadStride2R2: {
    const double src = C * isb;
    const double rowBytes = (C - 1.0) * isb + eb;
    // Branch A's crowding term is the count of ROWS sharing a 1 KB block
    // (rmax1k), not the count of 128 B lines.  The two differ whenever a row
    // spans more than one line.
    if (src >= 32.0)
      return fit.intercept + c[0] * R +
             c[1] * static_cast<double>(stats.maxRowsPerKilo) +
             c[2] * R * (rowBytes < 32.0 ? 1.0 : 0.0);
    return c[3] + c[4] * n + c[5] * distinct(64);
  }
  case StridedShortFormula::SimdLoadStrideGT2R1Dense:
    return fit.intercept + c[0] * n + c[1] * (n * eb) + c[2] * maxBlock(64) +
           c[3] * maxBlock(512) + c[4] * sumSquares(1024);
  case StridedShortFormula::SimdLoadStrideGT2R1Sparse:
    // Four terms: at this stride the 512 B and 1 KB windows follow from the
    // geometry, so the extra terms would only add collinear freedom.
    return fit.intercept + c[0] * n + c[1] * (n * eb) + c[2] * maxBlock(64);
  case StridedShortFormula::SimdLoadStrideGT2R2: {
    const double bs = (C * eb < 32.0) ? 1.0 : 0.0;
    return fit.intercept + c[0] * bs * distinct(64) +
           c[1] * (1.0 - bs) * n * maxRowBlock(64) +
           c[2] * (1.0 - bs) * maxBlock(1024) + c[3] * bs * distinct(256) +
           c[4] * bs * shared(256) + c[5] * (n * n * eb);
  }
  case StridedShortFormula::SimdStoreR1:
    return fit.intercept + c[0] * (n * eb) + c[1] * maxBlock(128) +
           c[2] * sumSquares(64) + c[3] * sumSquares(512);
  case StridedShortFormula::SimdStoreR2: {
    if (isb <= 0.0)
      return std::nullopt;
    const double q4 = std::min(C, s[0] / isb);
    return fit.intercept + c[0] * q4 + c[1] * n * q4 +
           c[2] * std::max(0.0, n - s[1]);
  }
  case StridedShortFormula::SimdStoreMaskedStrideGT2:
  case StridedShortFormula::SimdStoreMaskedStride2:
    // Both masked variants share the unmasked store's four terms and add the
    // 1 KB aggregate band, which only becomes observable once the mask forces
    // the destination to be read back over the whole tile.
    return fit.intercept + c[0] * (n * eb) + c[1] * maxBlock(128) +
           c[2] * sumSquares(64) + c[3] * sumSquares(512) +
           c[4] * sumSquares(1024);
  case StridedShortFormula::SimtLoadR1:
    return std::max({c[0] * W, c[1] * W * iters * stats.meanWarpLines,
                     fit.intercept + c[2] * std::min(stats.meanWarpLines, 8.0) +
                         c[3] * std::min(stats.meanWarpLines, 4.0) +
                         c[4] * iters + c[5] * distinct(128)});
  case StridedShortFormula::SimtLoadR2:
    return std::max(
        {c[0] * W, c[1] * W * iters * stats.meanWarpLines,
         fit.intercept + c[2] * static_cast<double>(stats.maxConc1k) +
             c[3] * iters + c[4] * std::min(stats.meanWarpLines, 4.0)});
  case StridedShortFormula::SimtStoreR1: {
    const double lineWeight = eb == 1.0   ? c[2]
                              : eb == 2.0 ? c[3]
                              : eb == 4.0 ? c[4]
                              : eb == 8.0 ? c[5]
                                          : 0.0;
    // One formula shape only: the fitted monotone warp term.  A quadratic
    // U-shape in log2(W) was implemented, measured against the eight-case
    // suite and rejected -- enabling it moved this slot's only exercising
    // example from +17.2% to +20.8%, i.e. away from the measurement.  The
    const double wTerm = c[0] * W;
    return std::max(
        {wTerm, fit.intercept + c[1] * iters, distinct(128) * lineWeight});
  }
  case StridedShortFormula::SimtStoreR2: {
    const double low = eb < 4.0 ? 1.0 : 0.0;
    const double mid = eb == 4.0 ? 1.0 : 0.0;
    const double high = eb == 8.0 ? 1.0 : 0.0;
    const double lineBranch =
        distinct(128) * (c[0] * low + c[1] * mid + c[2] * high);
    const double rowBranch =
        n * std::min(R, 32.0) * (c[3] * low + c[4] * mid + c[5] * high);
    const double iterBranch = c[6] * std::max(0.0, iters - s[0]);
    return std::max({lineBranch, rowBranch, iterBranch, s[1]});
  }
  }
  llvm_unreachable("unhandled strided direct formula");
}

} // namespace mlir::ascend
