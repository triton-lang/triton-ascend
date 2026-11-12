//===- StridedWideCostModel.cpp - StridedWide direct access model
//--------------===//
//
// Shape of every StridedWide closed-form completion cost.  The numbers come
// from the profile (stage_resources.<mode>.memory.strided_wide_completion);
// nothing here reads an operator or kernel name.
//
// Calibration provenance: profiles/microbench/data_provider/strided_ldst/
// swide_phys/ (WIDE2/LOADV1 sweeps; slot in-sample MAPE 4.00-9.64%).  The
// fits are least squares over a bounded sweep, not physical laws that hold
// outside it, so selectSwideFormula() bounds every access to the measured
// domain and anything else keeps the layout-memory curves / analytical
// fallback.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/Models/StridedWideCostModel.h"

#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>

using namespace mlir;

namespace mlir::ascend {

bool SwideCompletionFit::isValid() const {
  if (evidence.empty())
    return false;
  if (!std::isfinite(intercept))
    return false;
  for (double value : coefficient)
    if (!std::isfinite(value))
      return false;
  for (double value : factorScaleCoefficient)
    if (!std::isfinite(value))
      return false;
  return true;
}

double SwideCompletionFit::factorScale(int64_t factor) const {
  if (factor <= 1 ||
      (factorScaleCoefficient[0] == 0.0 && factorScaleCoefficient[1] == 0.0))
    return 1.0;
  const double x = 1.0 / static_cast<double>(factor) - 1.0;
  const double denominator =
      1.0 + factorScaleCoefficient[0] * x + factorScaleCoefficient[1] * x * x;
  // Rational form: no exponential in the compile-time cost model, exactly
  // matches the fitted F=2/F=4 anchors, and stays positive for F=8/16.
  return std::clamp(1.0 / std::max(denominator, 1.0e-9), 0.02, 1.5);
}

llvm::StringRef stringifySwideFormula(SwideFormula formula) {
  switch (formula) {
  case SwideFormula::SimdLoadSwideRowsLe4:
    return "simd_load_swide_rows_le4";
  case SwideFormula::SimdLoadSwideRowsGt4:
    return "simd_load_swide_rows_gt4";
  case SwideFormula::SimdStoreSwideRowsLe2:
    return "simd_store_swide_rows_le2";
  case SwideFormula::SimdStoreSwideRowsGt2:
    return "simd_store_swide_rows_gt2";
  case SwideFormula::SimtLoadSwideRows2:
    return "simt_load_swide_rows2";
  case SwideFormula::SimtLoadSwideRows8:
    return "simt_load_swide_rows8";
  case SwideFormula::SimtLoadSwideRows64Wle2:
    return "simt_load_swide_rows64_wle2";
  case SwideFormula::SimtLoadSwideRows64W38:
    return "simt_load_swide_rows64_w3_8";
  case SwideFormula::SimtLoadSwideRows64Wgt8:
    return "simt_load_swide_rows64_wgt8";
  case SwideFormula::SimtStoreSwideRows2:
    return "simt_store_swide_rows2";
  case SwideFormula::SimtStoreSwideRows8:
    return "simt_store_swide_rows8";
  case SwideFormula::SimtStoreSwideRows64Wle2:
    return "simt_store_swide_rows64_wle2";
  case SwideFormula::SimtStoreSwideRows64W38:
    return "simt_store_swide_rows64_w3_8";
  case SwideFormula::SimtStoreSwideRows64Wgt8:
    return "simt_store_swide_rows64_wgt8";
  case SwideFormula::SimtLoadSwideMaskedRows2:
    return "simt_load_swide_masked_rows2";
  case SwideFormula::SimtLoadSwideMaskedRows8:
    return "simt_load_swide_masked_rows8";
  case SwideFormula::SimtLoadSwideMaskedRows64Wle2:
    return "simt_load_swide_masked_rows64_wle2";
  case SwideFormula::SimtLoadSwideMaskedRows64W38:
    return "simt_load_swide_masked_rows64_w3_8";
  case SwideFormula::SimtLoadSwideMaskedRows64Wgt8:
    return "simt_load_swide_masked_rows64_wgt8";
  case SwideFormula::SimtLoadSwideMaskedRows16Wle2:
    return "simt_load_swide_masked_rows16_wle2";
  case SwideFormula::SimtLoadSwideMaskedRows16W38:
    return "simt_load_swide_masked_rows16_w3_8";
  case SwideFormula::SimtLoadSwideMaskedRows16Wgt8:
    return "simt_load_swide_masked_rows16_wgt8";
  case SwideFormula::SimtStoreSwideMaskedRows2:
    return "simt_store_swide_masked_rows2";
  case SwideFormula::SimtStoreSwideMaskedRows8:
    return "simt_store_swide_masked_rows8";
  case SwideFormula::SimtStoreSwideMaskedRows64Wle2:
    return "simt_store_swide_masked_rows64_wle2";
  case SwideFormula::SimtStoreSwideMaskedRows64W38:
    return "simt_store_swide_masked_rows64_w3_8";
  case SwideFormula::SimtStoreSwideMaskedRows64Wgt8:
    return "simt_store_swide_masked_rows64_wgt8";
  case SwideFormula::SimtStoreSwideMaskedRows16Wle2:
    return "simt_store_swide_masked_rows16_wle2";
  case SwideFormula::SimtStoreSwideMaskedRows16W38:
    return "simt_store_swide_masked_rows16_w3_8";
  case SwideFormula::SimtStoreSwideMaskedRows16Wgt8:
    return "simt_store_swide_masked_rows16_wgt8";
  }
  llvm_unreachable("unhandled strided-wide formula");
}

llvm::ArrayRef<int64_t> swideStatWindows() {
  static constexpr int64_t kWindows[] = {32,  64,   128,  256,
                                         512, 1024, 2048, 4096};
  return kWindows;
}

namespace {

double blockStat(const StridedAddressStats &stats, int64_t window) {
  auto it = stats.maxPerBlock.find(window);
  return it == stats.maxPerBlock.end() ? 0.0 : static_cast<double>(it->second);
}

/// The address-set / geometry features the fits read.  Field names mirror the
/// profile's per-slot "terms" arrays one-to-one; swideFeatures() defines each
/// value exactly as the calibration's featurizer did.
struct SwideFeatures {
  double n, W, run, gap, span, duty;
  double duty2, mis32, iters, q, h, h_div_w;
  double h_pos, elem_per_lane, bytes_per_lane, elem_per_warp, bytes_per_warp,
      lines_per_seg;
  double total_lines, elem_per_line, rows_x_duty, rows2, inv_n, sqrt_n;
  double minW8, minW16, maxW4, maxW8, maxW16, n2W;
  double cseg, cseg_mod32, cseg_x_invW, L32, L64, L128;
  double L256, L512, L1024, L2048, L4096, mx32;
  double mx64, mx128, mx256, mx512, mx1024, mx2048;
  double mx4096, S32, S64, S128, S256, S512;
  double S2048, S4096, W2, minW2, inv_sectors, is32;
  double logW, log_n, iters_x_W, run_x_W, span_div_run, cseg_x_minW4;
  double bytes_per_segline, r16_x_segW;
};

SwideFeatures swideFeatures(const StridedAddressStats &stats, int64_t warps) {
  const double n = static_cast<double>(stats.elements);
  const double eb = static_cast<double>(stats.elementBytes);
  const double W = static_cast<double>(std::max<int64_t>(1, warps));
  const double rows = static_cast<double>(stats.rows);
  const double run = static_cast<double>(stats.cols) * eb;
  const double pitch = static_cast<double>(stats.rowPitchBytes);
  const double gap = pitch - run;
  const double nbytes = n * eb;
  const double duty = run / pitch;
  auto distinct = [&stats](int64_t window) {
    auto it = stats.distinctBlocks.find(window);
    return it == stats.distinctBlocks.end() ? 0.0
                                            : static_cast<double>(it->second);
  };
  auto sumSquares = [&stats](int64_t window) {
    auto it = stats.sumSquaresPerBlock.find(window);
    return it == stats.sumSquaresPerBlock.end()
               ? 0.0
               : static_cast<double>(it->second);
  };

  SwideFeatures f;
  f.n = n;
  f.W = W;
  f.run = run;
  f.gap = gap;
  f.span = (rows - 1.0) * pitch + run;
  f.duty = duty;
  f.duty2 = duty * duty;
  f.mis32 = (static_cast<int64_t>(pitch) % 32) != 0 ? 1.0 : 0.0;
  const double wave = 32.0 * W;
  f.iters = static_cast<double>(llvm::divideCeil(
      stats.elements, int64_t(32) * std::max<int64_t>(1, warps)));
  f.q = n / wave;
  f.h = std::max(n - wave, 0.0);
  f.h_div_w = f.h / W;
  f.h_pos = f.h > 0.0 ? 1.0 : 0.0;
  f.elem_per_lane = n / wave;
  f.bytes_per_lane = nbytes / wave;
  f.elem_per_warp = n / W;
  f.bytes_per_warp = nbytes / W;
  const double linesPerSeg = static_cast<double>(
      llvm::divideCeil(stats.cols * stats.elementBytes, 128));
  f.lines_per_seg = linesPerSeg;
  f.total_lines = rows * linesPerSeg;
  f.elem_per_line = n / f.total_lines;
  f.rows_x_duty = rows * duty;
  f.rows2 = rows * rows;
  f.inv_n = 1.0 / n;
  f.sqrt_n = std::sqrt(n);
  f.minW8 = std::min(W, 8.0);
  f.minW16 = std::min(W, 16.0);
  f.maxW4 = std::max(0.0, W - 4.0);
  f.maxW8 = std::max(0.0, W - 8.0);
  f.maxW16 = std::max(0.0, W - 16.0);
  f.n2W = n * n / W;
  // Elements per contiguous segment and its relation to the 32 SIMT lanes;
  // whether a warp instruction covers one segment or crosses several is what
  // the calibration measured as the zigzag in W.
  f.cseg = static_cast<double>(stats.cols);
  f.cseg_mod32 = static_cast<double>(stats.cols % 32);
  f.cseg_x_invW = f.cseg / W;
  // The 128 B line count follows the calibration's featurizer, which uses the
  // coarse ceil(payload/128) rather than the address-set statistic that the
  // wider windows keep.
  f.L32 = distinct(32);
  f.L64 = distinct(64);
  f.L128 = std::ceil(nbytes / 128.0);
  f.L256 = distinct(256);
  f.L512 = distinct(512);
  f.L1024 = distinct(1024);
  f.L2048 = distinct(2048);
  f.L4096 = distinct(4096);
  f.mx32 = blockStat(stats, 32);
  f.mx64 = blockStat(stats, 64);
  f.mx128 = blockStat(stats, 128);
  f.mx256 = blockStat(stats, 256);
  f.mx512 = blockStat(stats, 512);
  f.mx1024 = blockStat(stats, 1024);
  f.mx2048 = blockStat(stats, 2048);
  f.mx4096 = blockStat(stats, 4096);
  f.S32 = sumSquares(32);
  f.S64 = sumSquares(64);
  f.S128 = sumSquares(128);
  f.S256 = sumSquares(256);
  f.S512 = sumSquares(512);
  f.S2048 = sumSquares(2048);
  f.S4096 = sumSquares(4096);
  // Derived quantities used by the masked SIMT slots.  Definitions mirror the
  // calibration featurizer one-for-one (refit_simt.enrich / feats_v2.build).
  f.W2 = W * W;
  f.minW2 = std::min(W, 2.0);
  const double sectors = std::ceil(run / 32.0);
  f.inv_sectors = sectors > 0.0 ? 1.0 / sectors : 0.0;
  f.is32 = run == 32.0 ? 1.0 : 0.0;
  f.logW = std::log(W);
  f.log_n = std::log(n);
  f.iters_x_W = f.iters * W;
  f.run_x_W = run * W;
  f.span_div_run = f.span / run;
  f.cseg_x_minW4 = f.cseg * std::min(W, 4.0);
  const double linesPerSegLocal = std::ceil(run / 128.0);
  f.bytes_per_segline = linesPerSegLocal > 0.0 ? run / linesPerSegLocal : 0.0;
  f.r16_x_segW = std::max(rows - 16.0, 0.0) * std::ceil(rows / W);
  return f;
}

} // namespace

std::optional<SwideFormula> selectSwideFormula(const StageMemoryAccess &access,
                                               const StridedAddressStats &stats,
                                               bool simd, int64_t warps) {
  // Common domain precondition, copied from the calibration sweep's scope:
  // rank-2, exactly one unit-stride (inner) axis, contiguous run 32..2048 B,
  // a real gap on the row pitch, unmasked, clean static address provenance.
  // A point outside this grid must fall through to the fallback mechanisms:
  // the fits are least squares over a bounded sweep and must not extrapolate.
  // Masked accesses: only a dynamically all-true mask is priced (no
  // statically provable partial coverage).  A partial mask cuts inside the
  // contiguous run and pays sub-line partial writes the fits never saw.
  // SIMD masked shares the unmasked slots: the paired store sweep measured
  // median 1.000 (p10/p90 0.999/1.004, 99.2% within 5%) and the load joint
  // fit costs 0.14 pp.  SIMT masked uses its own slots below, because the
  // SIMT masked relation is not a smooth offset (p90 1.29, max 2.26).
  if (access.masked && access.activeElementRatio < 1.0 - 1.0e-9)
    return std::nullopt;
  if (access.addressDependsOnLoadedIndex ||
      access.addressHasNonAffineTensorComputation ||
      access.addressHasTensorControlFlowSelection)
    return std::nullopt;
  if (access.effectiveRank() != 2 || access.shape.size() != 2 ||
      access.strides.size() != 2)
    return std::nullopt;
  if (!access.hasStaticLayout() || access.elementBits % 8 != 0)
    return std::nullopt;
  // Inner/row-major only: the unit-stride axis is the last one and carries the
  // contiguous run.  The column-major twin was never measured (it is absent
  // from every profile fingerprint and from real kernels), so it keeps the
  // fallback rather than being priced by the wrong orientation.
  if (!access.strides.back() || *access.strides.back() != 1)
    return std::nullopt;
  if (access.shape.back() <= 1)
    return std::nullopt;
  const int64_t eb = access.elementBits / 8;
  if (eb != 1 && eb != 2 && eb != 4)
    return std::nullopt;
  const int64_t run = access.shape.back() * eb;
  if (run < 32 || run > 2048)
    return std::nullopt;
  const int64_t rows = access.shape[0];
  const int64_t cols = access.shape[1];
  if (rows < 2 || cols <= 0)
    return std::nullopt;
  if (!access.strides[0] || *access.strides[0] <= 0)
    return std::nullopt;
  const int64_t pitch = *access.strides[0] * eb;
  const int64_t gap = pitch - run;
  // Interior interpolation inside the measured endpoints.  The swept gap
  // interval is 2..8192 B for every family except the rows=16 sweeps, which
  // were extended to a 16 KiB row pitch (gap 16320 B) for the solve_tril h64
  // geometry; there the interval is 2..16384 B for run <= 128 B (the extended
  // sweeps' run range).  Outside the endpoints the fallback owns the access.
  int64_t maximumGap = 8192;
  if (rows == 16 && run <= 128 && (simd || access.masked))
    maximumGap = 16384;
  if (gap < 2 || gap > maximumGap)
    return std::nullopt;
  if ((rows - 1) * pitch + run > 512 * 1024)
    return std::nullopt;
  if (access.effectiveRank() != 2 ||
      access.layoutClass() != StageMemoryAccess::LayoutClass::StridedWide)
    return std::nullopt;
  if (stats.cols <= 0 || stats.elementBytes != eb || stats.rows != rows)
    return std::nullopt;
  if (simd) {
    // The SIMD sweeps cover rows 2..256 with the delivered K thresholds;
    // features are linear in the geometry, so in-grid interpolation applies.
    if (rows > 256)
      return std::nullopt;
    if (access.store)
      return rows <= 2 ? SwideFormula::SimdStoreSwideRowsLe2
                       : SwideFormula::SimdStoreSwideRowsGt2;
    return rows <= 4 ? SwideFormula::SimdLoadSwideRowsLe4
                     : SwideFormula::SimdLoadSwideRowsGt4;
  }

  // The SIMT sweeps rotated warps at {1, 2, 4, 8, 16, 32, 64}.  The load
  // sweep topped out at a 128 B run; wider SIMT loads keep the fallback.
  if (warps < 1 || warps > 64)
    return std::nullopt;
  if (!access.store && run > 128)
    return std::nullopt;
  if (access.masked) {
    // Masked SIMT slots: rows {2, 8, 64} across all warps from the masked
    // WIDE2 sweep, plus rows=16 from the supplementary sweep (run 32..128 B).
    const bool store = access.store;
    if (rows == 2)
      return store ? SwideFormula::SimtStoreSwideMaskedRows2
                   : SwideFormula::SimtLoadSwideMaskedRows2;
    if (rows == 8)
      return store ? SwideFormula::SimtStoreSwideMaskedRows8
                   : SwideFormula::SimtLoadSwideMaskedRows8;
    if (rows == 16) {
      if (run > 128)
        return std::nullopt;
      if (warps <= 2)
        return store ? SwideFormula::SimtStoreSwideMaskedRows16Wle2
                     : SwideFormula::SimtLoadSwideMaskedRows16Wle2;
      if (warps <= 8)
        return store ? SwideFormula::SimtStoreSwideMaskedRows16W38
                     : SwideFormula::SimtLoadSwideMaskedRows16W38;
      return store ? SwideFormula::SimtStoreSwideMaskedRows16Wgt8
                   : SwideFormula::SimtLoadSwideMaskedRows16Wgt8;
    }
    if (rows == 64) {
      if (warps <= 2)
        return store ? SwideFormula::SimtStoreSwideMaskedRows64Wle2
                     : SwideFormula::SimtLoadSwideMaskedRows64Wle2;
      if (warps <= 8)
        return store ? SwideFormula::SimtStoreSwideMaskedRows64W38
                     : SwideFormula::SimtLoadSwideMaskedRows64W38;
      return store ? SwideFormula::SimtStoreSwideMaskedRows64Wgt8
                   : SwideFormula::SimtLoadSwideMaskedRows64Wgt8;
    }
    return std::nullopt;
  }
  // Unmasked SIMT: the sweeps rotated rows only at {2, 8, 64}; the segment
  // rule below is the calibration's own split, so an off-grid rows value has
  // no segment to belong to.
  if (rows != 2 && rows != 8 && rows != 64)
    return std::nullopt;
  if (access.store) {
    if (rows == 2)
      return SwideFormula::SimtStoreSwideRows2;
    if (rows == 8)
      return SwideFormula::SimtStoreSwideRows8;
    if (warps <= 2)
      return SwideFormula::SimtStoreSwideRows64Wle2;
    if (warps <= 8)
      return SwideFormula::SimtStoreSwideRows64W38;
    return SwideFormula::SimtStoreSwideRows64Wgt8;
  }
  if (rows == 2)
    return SwideFormula::SimtLoadSwideRows2;
  if (rows == 8)
    return SwideFormula::SimtLoadSwideRows8;
  if (warps <= 2)
    return SwideFormula::SimtLoadSwideRows64Wle2;
  if (warps <= 8)
    return SwideFormula::SimtLoadSwideRows64W38;
  return SwideFormula::SimtLoadSwideRows64Wgt8;
}

// Formula evaluation.
//
// Formula -> coefficient layout (values come from the profile; the per-slot
// "terms" array in the profile documents the same order):
//   SimdLoadSwideRowsLe4   T = i + c0*n + c1*rows_x_duty + c2*duty2 + c3*S128
//                              + c4*lines_per_seg + c5*run + c6*mx128
//                              + c7*sqrt_n + c8*inv_rows + c9*mx512
//                              + c10*S4096 + c11*L1024 + c12*total_lines
//                              + c13*inv_n + c14*gap
//   SimdLoadSwideRowsGt4   T = i + c0*n + c1*rows_x_duty + c2*total_lines
//                              + c3*S256 + c4*duty + c5*sqrt_n + c6*sqrt_rows
//                              + c7*mx128 + c8*inv_rows + c9*S4096 + c10*cseg
//                              + c11*mis32 + c12*S128 + c13*duty2 + c14*L512
//                              + c15*L1024
//   SimdStoreSwideRowsLe2  T = i + c0*duty + c1*mis32 + c2*rows_x_duty
//                              + c3*inv_n + c4*gap + c5*S2048 + c6*S4096
//                              + c7*L4096 + c8*mx512 + c9*mx128
//                              + c10*elem_per_line + c11*sqrt_n + c12*mx256
//   SimdStoreSwideRowsGt2  T = i + c0*rows_x_duty + c1*duty2 + c2*total_lines
//                              + c3*mis32 + c4*L128 + c5*mx4096 + c6*rows2
//                              + c7*mx1024 + c8*run + c9*duty + c10*cseg
//                              + c11*mx512 + c12*mx2048 + c13*mx256
//                              + c14*elem_per_line + c15*mx128
//   SimtLoadSwideRows2     T = i + c0*maxW16 + c1*maxW4 + c2*cseg_x_invW
//                              + c3*cseg_mod32 + c4*minW8 + c5*iters
//                              + c6*h_pos
//   SimtLoadSwideRows8     T = i + c0*L256 + c1*maxW16 + c2*L2048 + c3*h_div_w
//                              + c4*L512 + c5*L4096 + c6*L128 + c7*S4096
//                              + c8*cseg + c9*S2048 + c10*maxW8
//                              + c11*bytes_per_lane + c12*h
//   SimtLoadSwideRows64Wle2 T = i + c0*n2W + c1*mx64 + c2*S64 + c3*h
//                              + c4*cseg_mod32
//   SimtLoadSwideRows64W38 T = i + c0*cseg_x_invW + c1*bytes_per_lane
//                              + c2*S128 + c3*cseg_mod32 + c4*n2W
//                              + c5*elem_per_lane + c6*elem_per_warp
//                              + c7*h_div_w + c8*iters + c9*bytes_per_warp
//                              + c10*q + c11*W + c12*S64 + c13*span
//   SimtLoadSwideRows64Wgt8 T = i + c0*n2W + c1*h_pos + c2*L2048 + c3*mx256
//                              + c4*S2048 + c5*W + c6*cseg + c7*mx1024
//   SimtStoreSwideRows2    T = i + c0*maxW8 + c1*S4096 + c2*h_pos + c3*n2W
//                              + c4*maxW16 + c5*minW16
//   SimtStoreSwideRows8    T = i + c0*maxW8 + c1*bytes_per_lane + c2*mis32
//                              + c3*maxW16 + c4*minW16 + c5*h_div_w
//                              + c6*mx128 + c7*mx64 + c8*mx256
//   SimtStoreSwideRows64Wle2 T = i + c0*bytes_per_lane + c1*L128 + c2*n2W
//                              + c3*L256 + c4*duty2 + c5*W + c6*mis32
//                              + c7*S128 + c8*mx128 + c9*cseg
//   SimtStoreSwideRows64W38 T = i + c0*L128 + c1*W + c2*L256 + c3*mis32
//                              + c4*L512 + c5*bytes_per_lane + c6*duty2
//                              + c7*duty + c8*nbytes
//   SimtStoreSwideRows64Wgt8 T = i + c0*duty2 + c1*L256 + c2*maxW16
//                              + c3*bytes_per_lane + c4*mis32 + c5*L128
//                              + c6*L2048 + c7*duty + c8*invW
//
// The switch deliberately has no default label: -Wswitch then proves that
// every enumerator is priced.
std::optional<double> evaluateSwideFormula(SwideFormula formula,
                                           const SwideCompletionFit &fit,
                                           const StridedAddressStats &stats,
                                           int64_t warps) {
  if (!fit.isValid())
    return std::nullopt;
  const SwideFeatures f = swideFeatures(stats, warps);
  const std::array<double, SwideCompletionFit::kMaxTerms> &c = fit.coefficient;

  switch (formula) {
  case SwideFormula::SimdLoadSwideRowsLe4: // terms: n, rows_x_duty, duty2,
                                           // S128, lines_per_seg, run
    return fit.intercept + c[0] * f.n + c[1] * f.rows_x_duty + c[2] * f.duty2 +
           c[3] * f.S128 + c[4] * f.lines_per_seg + c[5] * f.run;
  case SwideFormula::SimdLoadSwideRowsGt4: // terms: n, rows_x_duty,
                                           // total_lines, S256, duty, sqrt_n
    return fit.intercept + c[0] * f.n + c[1] * f.rows_x_duty +
           c[2] * f.total_lines + c[3] * f.S256 + c[4] * f.duty +
           c[5] * f.sqrt_n;
  case SwideFormula::SimdStoreSwideRowsLe2: // terms: duty, mis32, rows_x_duty,
                                            // inv_n, gap, S2048, S4096, L4096,
                                            // mx512, mx128, elem_per_line,
                                            // sqrt_n, mx256
    return fit.intercept + c[0] * f.duty + c[1] * f.mis32 +
           c[2] * f.rows_x_duty + c[3] * f.inv_n + c[4] * f.gap +
           c[5] * f.S2048 + c[6] * f.S4096 + c[7] * f.L4096 + c[8] * f.mx512 +
           c[9] * f.mx128 + c[10] * f.elem_per_line + c[11] * f.sqrt_n +
           c[12] * f.mx256;
  case SwideFormula::
      SimdStoreSwideRowsGt2: // terms: rows_x_duty, duty2, total_lines, mis32,
                             // L128, mx4096, rows2, mx1024, run, duty, cseg,
                             // mx512, mx2048, mx256, elem_per_line, mx128
    return fit.intercept + c[0] * f.rows_x_duty + c[1] * f.duty2 +
           c[2] * f.total_lines + c[3] * f.mis32 + c[4] * f.L128 +
           c[5] * f.mx4096 + c[6] * f.rows2 + c[7] * f.mx1024 + c[8] * f.run +
           c[9] * f.duty + c[10] * f.cseg + c[11] * f.mx512 + c[12] * f.mx2048 +
           c[13] * f.mx256 + c[14] * f.elem_per_line + c[15] * f.mx128;
  case SwideFormula::SimtLoadSwideRows2: // terms: maxW16, maxW4, cseg_x_invW,
                                         // cseg_mod32, minW8, h_div_w, h_pos
    return fit.intercept + c[0] * f.maxW16 + c[1] * f.maxW4 +
           c[2] * f.cseg_x_invW + c[3] * f.cseg_mod32 + c[4] * f.minW8 +
           c[5] * f.h_div_w + c[6] * f.h_pos;
  case SwideFormula::SimtLoadSwideRows8: // terms: L256, maxW16, L2048, h_div_w,
                                         // L512, L4096, L128, S4096, cseg,
                                         // S2048, maxW8, bytes_per_lane, h
    return fit.intercept + c[0] * f.L256 + c[1] * f.maxW16 + c[2] * f.L2048 +
           c[3] * f.h_div_w + c[4] * f.L512 + c[5] * f.L4096 + c[6] * f.L128 +
           c[7] * f.S4096 + c[8] * f.cseg + c[9] * f.S2048 + c[10] * f.maxW8 +
           c[11] * f.bytes_per_lane + c[12] * f.h;
  case SwideFormula::SimtLoadSwideRows64Wle2: // terms: n2W, mx64, S64, h,
                                              // cseg_mod32
    return fit.intercept + c[0] * f.n2W + c[1] * f.mx64 + c[2] * f.S64 +
           c[3] * f.h + c[4] * f.cseg_mod32;
  case SwideFormula::
      SimtLoadSwideRows64W38: // terms: cseg_x_invW, bytes_per_lane, S128,
                              // cseg_mod32, n2W, elem_per_lane, elem_per_warp,
                              // h_div_w, iters, bytes_per_warp, q, W, S64, span
    return fit.intercept + c[0] * f.cseg_x_invW + c[1] * f.bytes_per_lane +
           c[2] * f.S128 + c[3] * f.cseg_mod32 + c[4] * f.n2W +
           c[5] * f.elem_per_lane + c[6] * f.elem_per_warp + c[7] * f.h_div_w +
           c[8] * f.iters + c[9] * f.bytes_per_warp + c[10] * f.q +
           c[11] * f.W + c[12] * f.S64 + c[13] * f.span;
  case SwideFormula::SimtLoadSwideRows64Wgt8: // terms: n2W, h_pos, L2048,
                                              // mx256, S2048, W, cseg, mx1024
    return fit.intercept + c[0] * f.n2W + c[1] * f.h_pos + c[2] * f.L2048 +
           c[3] * f.mx256 + c[4] * f.S2048 + c[5] * f.W + c[6] * f.cseg +
           c[7] * f.mx1024;
  case SwideFormula::SimtStoreSwideRows2: // terms: maxW8, S4096, h_pos, n2W,
                                          // maxW16, minW16
    return fit.intercept + c[0] * f.maxW8 + c[1] * f.S4096 + c[2] * f.h_pos +
           c[3] * f.n2W + c[4] * f.maxW16 + c[5] * f.minW16;
  case SwideFormula::SimtStoreSwideRows8: // terms: maxW8, bytes_per_lane,
                                          // mis32, maxW16, minW16, h_div_w,
                                          // mx128, mx64
    return fit.intercept + c[0] * f.maxW8 + c[1] * f.bytes_per_lane +
           c[2] * f.mis32 + c[3] * f.maxW16 + c[4] * f.minW16 +
           c[5] * f.h_div_w + c[6] * f.mx128 + c[7] * f.mx64;
  case SwideFormula::SimtStoreSwideRows64Wle2: // terms: bytes_per_lane, L128,
                                               // n2W, L256, duty2, r16_x_segW,
                                               // mis32, S128
    return fit.intercept + c[0] * f.bytes_per_lane + c[1] * f.L128 +
           c[2] * f.n2W + c[3] * f.L256 + c[4] * f.duty2 + c[5] * f.r16_x_segW +
           c[6] * f.mis32 + c[7] * f.S128;
  case SwideFormula::SimtStoreSwideRows64W38: // terms: L128, r16_x_segW, L256,
                                              // mis32, L512, bytes_per_lane,
                                              // duty2, duty
    return fit.intercept + c[0] * f.L128 + c[1] * f.r16_x_segW + c[2] * f.L256 +
           c[3] * f.mis32 + c[4] * f.L512 + c[5] * f.bytes_per_lane +
           c[6] * f.duty2 + c[7] * f.duty;
  case SwideFormula::SimtStoreSwideRows64Wgt8: // terms: duty2, L256, W,
                                               // bytes_per_lane, mis32, L128,
                                               // L2048, duty
    return fit.intercept + c[0] * f.duty2 + c[1] * f.L256 + c[2] * f.W +
           c[3] * f.bytes_per_lane + c[4] * f.mis32 + c[5] * f.L128 +
           c[6] * f.L2048 + c[7] * f.duty;
  case SwideFormula::SimtLoadSwideMaskedRows16W38: // terms: h_div_w, L256,
                                                   // mx2048, mx128, S64, S4096,
                                                   // gap
    return fit.intercept + c[0] * f.h_div_w + c[1] * f.L256 + c[2] * f.mx2048 +
           c[3] * f.mx128 + c[4] * f.S64 + c[5] * f.S4096 + c[6] * f.gap;
  case SwideFormula::SimtLoadSwideMaskedRows16Wgt8: // terms: iters_x_W,
                                                    // cseg_mod32, mx32, mx128,
                                                    // S256, inv_sectors
    return fit.intercept + c[0] * f.iters_x_W + c[1] * f.cseg_mod32 +
           c[2] * f.mx32 + c[3] * f.mx128 + c[4] * f.S256 +
           c[5] * f.inv_sectors;
  case SwideFormula::
      SimtLoadSwideMaskedRows16Wle2: // terms: cseg_x_invW, inv_n, S4096, W2,
                                     // span, L256, L2048, L1024, run_x_W
    return fit.intercept + c[0] * f.cseg_x_invW + c[1] * f.inv_n +
           c[2] * f.S4096 + c[3] * f.W2 + c[4] * f.span + c[5] * f.L256 +
           c[6] * f.L2048 + c[7] * f.L1024 + c[8] * f.run_x_W;
  case SwideFormula::SimtLoadSwideMaskedRows2: // terms: W2, S32, cseg_mod32,
                                               // n2W, bytes_per_lane
    return fit.intercept + c[0] * f.W2 + c[1] * f.S32 + c[2] * f.cseg_mod32 +
           c[3] * f.n2W + c[4] * f.bytes_per_lane;
  case SwideFormula::SimtLoadSwideMaskedRows64W38: // terms: cseg_x_invW, L32,
                                                   // cseg_mod32, log_n
    return fit.intercept + c[0] * f.cseg_x_invW + c[1] * f.L32 +
           c[2] * f.cseg_mod32 + c[3] * f.log_n;
  case SwideFormula::SimtLoadSwideMaskedRows64Wgt8: // terms: n2W, W2, h_pos,
                                                    // inv_n, cseg_x_invW, S32
    return fit.intercept + c[0] * f.n2W + c[1] * f.W2 + c[2] * f.h_pos +
           c[3] * f.inv_n + c[4] * f.cseg_x_invW + c[5] * f.S32;
  case SwideFormula::SimtLoadSwideMaskedRows64Wle2: // terms: n2W, cseg_x_invW,
                                                    // h, W, bytes_per_lane,
                                                    // run_x_W, cseg_x_minW4,
                                                    // L32
    return fit.intercept + c[0] * f.n2W + c[1] * f.cseg_x_invW + c[2] * f.h +
           c[3] * f.W + c[4] * f.bytes_per_lane + c[5] * f.run_x_W +
           c[6] * f.cseg_x_minW4 + c[7] * f.L32;
  case SwideFormula::SimtLoadSwideMaskedRows8: // terms: L256, maxW16, mx2048,
                                               // minW2, span_div_run, n2W,
                                               // L2048, L512, S256, cseg_mod32
    return fit.intercept + c[0] * f.L256 + c[1] * f.maxW16 + c[2] * f.mx2048 +
           c[3] * f.minW2 + c[4] * f.span_div_run + c[5] * f.n2W +
           c[6] * f.L2048 + c[7] * f.L512 + c[8] * f.S256 + c[9] * f.cseg_mod32;
  case SwideFormula::SimtStoreSwideMaskedRows16W38: // terms: L64, mis32, L32,
                                                    // mx128, inv_n, S128
    return fit.intercept + c[0] * f.L64 + c[1] * f.mis32 + c[2] * f.L32 +
           c[3] * f.mx128 + c[4] * f.inv_n + c[5] * f.S128;
  case SwideFormula::SimtStoreSwideMaskedRows16Wgt8: // terms: iters_x_W,
                                                     // cseg_mod32, W2, S256,
                                                     // inv_n
    return fit.intercept + c[0] * f.iters_x_W + c[1] * f.cseg_mod32 +
           c[2] * f.W2 + c[3] * f.S256 + c[4] * f.inv_n;
  case SwideFormula::SimtStoreSwideMaskedRows16Wle2: // terms: bytes_per_lane,
                                                     // mis32, L256, duty2
    return fit.intercept + c[0] * f.bytes_per_lane + c[1] * f.mis32 +
           c[2] * f.L256 + c[3] * f.duty2;
  case SwideFormula::SimtStoreSwideMaskedRows2: // terms: maxW8, h, W2, h_pos,
                                                // lines_per_seg,
                                                // bytes_per_segline
    return fit.intercept + c[0] * f.maxW8 + c[1] * f.h + c[2] * f.W2 +
           c[3] * f.h_pos + c[4] * f.lines_per_seg + c[5] * f.bytes_per_segline;
  case SwideFormula::SimtStoreSwideMaskedRows64W38: // terms: L32, duty2, L256,
                                                    // logW, bytes_per_lane,
                                                    // duty, L2048, rows_x_duty
    return fit.intercept + c[0] * f.L32 + c[1] * f.duty2 + c[2] * f.L256 +
           c[3] * f.logW + c[4] * f.bytes_per_lane + c[5] * f.duty +
           c[6] * f.L2048 + c[7] * f.rows_x_duty;
  case SwideFormula::SimtStoreSwideMaskedRows64Wgt8: // terms: iters_x_W, mis32,
                                                     // mx128, bytes_per_lane,
                                                     // W2, S512, S32, S2048
    return fit.intercept + c[0] * f.iters_x_W + c[1] * f.mis32 +
           c[2] * f.mx128 + c[3] * f.bytes_per_lane + c[4] * f.W2 +
           c[5] * f.S512 + c[6] * f.S32 + c[7] * f.S2048;
  case SwideFormula::SimtStoreSwideMaskedRows64Wle2: // terms: L256, mis32,
                                                     // duty2, run_x_W, L2048,
                                                     // duty, is32, L64
    return fit.intercept + c[0] * f.L256 + c[1] * f.mis32 + c[2] * f.duty2 +
           c[3] * f.run_x_W + c[4] * f.L2048 + c[5] * f.duty + c[6] * f.is32 +
           c[7] * f.L64;
  case SwideFormula::SimtStoreSwideMaskedRows8: // terms: maxW8, bytes_per_lane,
                                                // W2, S256, cseg_x_invW,
                                                // cseg_mod32, maxW4
    return fit.intercept + c[0] * f.maxW8 + c[1] * f.bytes_per_lane +
           c[2] * f.W2 + c[3] * f.S256 + c[4] * f.cseg_x_invW +
           c[5] * f.cseg_mod32 + c[6] * f.maxW4;
  }
  llvm_unreachable("unhandled strided-wide formula");
}

} // namespace mlir::ascend
