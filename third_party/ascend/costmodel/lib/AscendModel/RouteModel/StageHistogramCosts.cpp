//===- StageHistogramCosts.cpp - Calibrated histogram cost model ---------===//
//
// Reads and validates the coefficient table declared in StageHistogramCosts.h,
// then evaluates one row per tt.histogram shape.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/StageHistogramCosts.h"
#include "AscendModel/RouteModel/StageCostModels.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <system_error>

namespace mlir::ascend {
namespace {

// Every diagnostic names the section by its path in the profile.
constexpr llvm::StringLiteral kSectionPath = "profile.histogram";

// Reading this one section is kept local: the first JSON error wins and every
// diagnostic names its path in the profile.
void setError(std::string &error, const llvm::Twine &message) {
  if (error.empty())
    error = message.str();
}

const llvm::json::Object *readObject(const llvm::json::Object &parent,
                                     llvm::StringRef key,
                                     llvm::StringRef context,
                                     std::string &error) {
  if (!error.empty())
    return nullptr;
  if (const auto *value = parent.getObject(key))
    return value;
  setError(error, context + "." + key + " must be an object");
  return nullptr;
}

int64_t readInteger(const llvm::json::Object &parent, llvm::StringRef key,
                    llvm::StringRef context, std::string &error) {
  if (!error.empty())
    return 0;
  if (auto value = parent.getInteger(key))
    return *value;
  setError(error, context + "." + key + " must be an integer");
  return 0;
}

double readNumber(const llvm::json::Object &parent, llvm::StringRef key,
                  llvm::StringRef context, std::string &error) {
  if (!error.empty())
    return 0.0;
  if (auto value = parent.getNumber(key))
    return *value;
  setError(error, context + "." + key + " must be a number");
  return 0.0;
}

/// Every structure constant the row gates read, named as in the profile.
void readStructure(const llvm::json::Object &structure, const std::string &path,
                   HistogramStructure &out, std::string &error) {
  out.chunkLanes = readInteger(structure, "chunk_lanes", path, error);
  out.threadsPerWarp = readInteger(structure, "threads_per_warp", path, error);
  out.maxSegmentsU16 = readInteger(structure, "max_segments_u16", path, error);
  out.maxSegmentsU32 = readInteger(structure, "max_segments_u32", path, error);
  out.selfAmortisedSegments =
      readInteger(structure, "self_amortised_segments", path, error);
  out.chunksPerSegment =
      readInteger(structure, "chunks_per_segment", path, error);
  out.flushChunks = readInteger(structure, "flush_chunks", path, error);
  out.atomicWindowBytes =
      readNumber(structure, "atomic_window_bytes", path, error);
}

/// Read one row verbatim: every numeric field becomes a coefficient.  Unknown
/// coefficients are kept, and rowValid() still rejects an incomplete formula.
HistogramRateRow readRow(const llvm::json::Object &object,
                         const std::string &path, std::string &error) {
  HistogramRateRow row;
  for (const auto &field : object) {
    auto value = field.second.getAsNumber();
    if (!value) {
      setError(error, path + "." + field.first.str() + " must be a number");
      return row;
    }
    row.coefficients[field.first.str()] = *value;
  }
  return row;
}

/// Read one `<name>: { rows: {...} }` group, keyed by the profile's own row
/// names.
void readGroup(const llvm::json::Object &section, llvm::StringRef name,
               const std::string &prefix,
               llvm::StringMap<HistogramRateRow> &out, std::string &error) {
  const auto *group = readObject(section, name, prefix, error);
  if (!group)
    return;
  const std::string groupPath = prefix + "." + name.str();
  const auto *rows = readObject(*group, "rows", groupPath, error);
  if (!rows)
    return;
  const std::string rowsPath = groupPath + ".rows";
  for (const auto &entry : *rows) {
    const auto *object = entry.second.getAsObject();
    if (!object) {
      setError(error,
               rowsPath + "." + entry.first.str() + " must be an object");
      continue;
    }
    out[entry.first.str()] =
        readRow(*object, rowsPath + "." + entry.first.str(), error);
  }
}

/// A required coefficient must be present, finite, and strictly positive.
bool positive(const HistogramRateRow &row, llvm::StringRef key) {
  return row.has(key) && std::isfinite(row.get(key)) && row.get(key) > 0.0;
}

/// A row is valid when the formula inferred from its coefficients is complete.
/// Intercepts only have to be finite (the narrow atomic rows are fitted with a
/// negative one), every other coefficient must be positive, and per_kink may be
/// zero.
bool rowValid(const HistogramRateRow &row) {
  if (!row.has("intercept") || !std::isfinite(row.get("intercept")))
    return false;
  if (row.has("knee_chunks") || row.has("per_chunk_beyond_knee"))
    return positive(row, "per_chunk") &&
           positive(row, "per_chunk_beyond_knee") &&
           positive(row, "knee_chunks");
  if (row.has("per_chunk_per_segment") || row.has("per_segment"))
    return positive(row, "per_chunk_per_segment") &&
           positive(row, "per_segment");
  if (row.has("per_iter") || row.has("per_bin") || row.has("per_thread") ||
      row.has("per_kink"))
    return positive(row, "per_elem") && positive(row, "per_iter") &&
           positive(row, "per_bin") && positive(row, "per_thread") &&
           row.has("per_kink") && row.get("per_kink") >= 0.0;
  if (row.has("scan_per_elem") || row.has("count_per_elem"))
    return positive(row, "scan_per_elem") && positive(row, "count_per_elem");
  return positive(row, "per_chunk");
}

/// Every contracted row name must be present and valid.
bool groupValid(const llvm::StringMap<HistogramRateRow> &rows,
                llvm::ArrayRef<llvm::StringLiteral> keys) {
  for (llvm::StringLiteral key : keys) {
    const auto it = rows.find(key);
    if (it == rows.end() || !rowValid(it->second))
      return false;
  }
  return true;
}

// The helpers below mirror the dispatch in the lowering template
// (SIMTHistogram1D.cpp) between a dhistv2 SIMD fast path and a SIMT atomicAdd
// fallback, derive the shape key of the selected row, and evaluate it.

/// Smallest power of two >= value (value >= 1).
double nextPow2(double value) {
  double result = 1.0;
  while (result < value)
    result *= 2.0;
  return result;
}

/// bptRead: bins are zero-inited and read back padded to max(bins, threads).
double histogramPaddedReadback(int64_t numBins, int64_t numWarps,
                               const HistogramStructure &structure) {
  const double threads = static_cast<double>(structure.threadsPerWarp *
                                             std::max<int64_t>(1, numWarps));
  const double paddedBins =
      nextPow2(std::max(static_cast<double>(numBins), threads));
  return std::max(1.0, paddedBins / threads);
}

/// Number of chunks a segmented pass re-reads the input for.
int64_t histogramSegmentCount(int64_t numBins,
                              const HistogramStructure &structure) {
  return (numBins + structure.chunkLanes - 1) / structure.chunkLanes;
}

/// dhistv2 eligibility, mirroring the template gates: u8 is always eligible;
/// u16/u32 need the segment limit and re-read amortisation; u64 only 256 bins.
bool dhistv2Eligible(int64_t inputElements, int64_t numBins,
                     unsigned widthBytes, const HistogramStructure &structure) {
  if (widthBytes == 8)
    return numBins > 0 && numBins <= structure.chunkLanes;
  if (widthBytes > 8)
    return false;
  if (widthBytes == 1)
    return true;
  const int64_t segments = histogramSegmentCount(numBins, structure);
  const int64_t maxSegments =
      widthBytes == 2 ? structure.maxSegmentsU16 : structure.maxSegmentsU32;
  if (segments > maxSegments)
    return false;
  if (segments <= structure.selfAmortisedSegments)
    return true;
  return inputElements / structure.chunkLanes >=
         structure.chunksPerSegment * segments;
}

/// Shape variables every histogram formula is a linear combination of.  A row
/// simply omits the coefficients of the variables its shape never uses.
struct HistogramShapeVars {
  double n = 0.0;
  double chunks = 0.0;
  double segments = 1.0;
  double iter = 0.0;
  double threads = 0.0;
  double bptRead = 0.0;
  /// max(0, n - atomicWindowElements), already clamped.
  double kink = 0.0;
};

/// Evaluate one row: intercept + sum(coefficient * variable).  The knee,
/// segment, and per-element alternatives are selected by which coefficients the
/// row carries, so a new linear combination needs no evaluator change.
double evaluateHistogramRow(const HistogramRateRow &row,
                            const HistogramShapeVars &vars) {
  const bool hasKnee =
      row.has("knee_chunks") || row.has("per_chunk_beyond_knee");
  const double knee = row.get("knee_chunks");
  const double chunks = hasKnee ? std::min(vars.chunks, knee) : vars.chunks;
  const double beyondKnee = hasKnee ? std::max(vars.chunks - knee, 0.0) : 0.0;
  const double perElem = row.get("per_elem") + row.get("scan_per_elem") +
                         row.get("count_per_elem");
  return row.get("intercept") + row.get("per_chunk") * chunks +
         row.get("per_chunk_beyond_knee") * beyondKnee +
         row.get("per_chunk_per_segment") * vars.chunks * vars.segments +
         row.get("per_segment") * vars.segments + perElem * vars.n +
         row.get("per_iter") * vars.iter + row.get("per_bin") * vars.bptRead +
         row.get("per_thread") * vars.threads + row.get("per_kink") * vars.kink;
}

/// Look a row up by the key its shape derived; a profile that passed isValid()
/// always has it.
const HistogramRateRow &
histogramRow(const llvm::StringMap<HistogramRateRow> &group,
             llvm::StringRef key) {
  static const HistogramRateRow empty;
  const auto it = group.find(key);
  return it == group.end() ? empty : it->second;
}

/// dtype part of a simd_template / simt_only row key.
llvm::StringRef histogramDtypeKey(unsigned widthBytes) {
  switch (widthBytes) {
  case 1:
    return "u8";
  case 2:
    return "u16";
  case 4:
    return "u32";
  default:
    return "u64";
  }
}

/// Key of the simd_template row a dhistv2 shape selects.  Mirrors the template
/// dispatch: u8 has one row; u64 only the <= chunkLanes shapes; a multi-segment
/// shape is keyed `bins_gt_256`; u32 has the small-bins fast entry; the u16
/// generic clamp entry is mask-free and shared.
std::string histogramSimdRowKey(int64_t inputElements, int64_t numBins,
                                unsigned widthBytes, bool unmasked,
                                const HistogramStructure &structure) {
  if (widthBytes == 1)
    return "u8";
  const std::string dtype = histogramDtypeKey(widthBytes).str();
  // u64 is admitted only for the <= chunkLanes shapes, and those share their
  // suffix rule with the u16/u32 generic shapes: whether the bins fit a single
  // 256-lane segment, and whether a mask is present.
  const std::string singleSegmentKey =
      dtype + (numBins < structure.chunkLanes ? "_bins_lt_256" : "_bins_256") +
      (unmasked ? "_unmasked" : "_masked");
  if (widthBytes == 8)
    return singleSegmentKey;
  if (numBins > structure.chunkLanes)
    return dtype + "_bins_gt_256";
  // Unmasked u32 in range and n <= flushChunks*chunkLanes routes to the
  // small-bins entry: park the spare bin (bins < 256) or merge predicates
  // (bins == 256); outside the gate it falls through to the generic entry.
  if (widthBytes == 4 && unmasked &&
      inputElements % structure.chunkLanes == 0 &&
      inputElements <= structure.flushChunks * structure.chunkLanes)
    return numBins < structure.chunkLanes ? "u32_small_bins_park"
                                          : "u32_small_bins_pred_mask";
  // The mask is free on the u16 generic clamp entry, so both shapes share it.
  if (widthBytes == 2 && numBins < structure.chunkLanes)
    return "u16_bins_lt_256";
  return singleSegmentKey;
}

} // namespace

llvm::Error readHistogramRates(const llvm::json::Object &section,
                               StageHistogramRates &rates) {
  std::string error;
  const std::string prefix = kSectionPath.str();
  if (const auto *structure = readObject(section, "structure", prefix, error))
    readStructure(*structure, prefix + ".structure", rates.structure, error);

  // Rows are keyed by the shape that selects them, so each group is read
  // verbatim and the model derives the key it needs at cost time.
  readGroup(section, "simd_template", prefix, rates.simdTemplate, error);
  readGroup(section, "simt_template", prefix, rates.simtTemplate, error);
  readGroup(section, "simt_only", prefix, rates.simtOnly, error);

  if (!error.empty())
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument), error);
  return llvm::Error::success();
}

bool StageHistogramRates::isValid() const {
  const HistogramStructure &s = structure;
  if (s.chunkLanes <= 0 || s.threadsPerWarp <= 0 || s.maxSegmentsU16 <= 0 ||
      s.maxSegmentsU32 <= 0 || s.selfAmortisedSegments <= 0 ||
      s.chunksPerSegment <= 0 || s.flushChunks <= 0 ||
      !std::isfinite(s.atomicWindowBytes) || s.atomicWindowBytes <= 0.0)
    return false;
  return groupValid(simdTemplate, kHistogramSimdTemplateKeys) &&
         groupValid(simtTemplate, kHistogramSimtTemplateKeys) &&
         groupValid(simtOnly, kHistogramSimtOnlyKeys);
}

double estimateHistogramCycles(int64_t inputElements, int64_t numBins,
                               unsigned widthBytes, StageMode mode,
                               bool unmasked, int64_t numWarps,
                               const StageHistogramRates &rates) {
  const HistogramStructure &structure = rates.structure;
  const double n = static_cast<double>(inputElements);

  HistogramShapeVars vars;
  vars.n = n;
  vars.chunks = std::ceil(n / static_cast<double>(structure.chunkLanes));
  vars.threads = static_cast<double>(structure.threadsPerWarp *
                                     std::max<int64_t>(1, numWarps));
  vars.bptRead = histogramPaddedReadback(numBins, numWarps, structure);

  // simt_only compile mode, lowered by HistogramOpToLLVM.cpp instead of the
  // template: one range predicate + atomicAdd per element, so no n_counted
  // term.  Masked calls reuse the unmasked rows.
  if (mode == StageMode::SIMT) {
    vars.iter = std::ceil(n / vars.threads);
    const double windowElements =
        structure.atomicWindowBytes / static_cast<double>(widthBytes);
    vars.kink = std::max(0.0, n - windowElements);
    return evaluateHistogramRow(
        histogramRow(rates.simtOnly, histogramDtypeKey(widthBytes)), vars);
  }
  // Template SIMT atomicAdd fallback (dhistv2 gate failed).  n_counted is
  // statically unknown and charged at its worst case n; wider widths reuse the
  // u32 row.
  if (!dhistv2Eligible(inputElements, numBins, widthBytes, structure)) {
    const llvm::StringRef key =
        widthBytes <= 1 ? "u8" : (widthBytes == 2 ? "u16" : "u32");
    return evaluateHistogramRow(histogramRow(rates.simtTemplate, key), vars);
  }
  vars.segments =
      static_cast<double>(histogramSegmentCount(numBins, structure));
  return evaluateHistogramRow(
      histogramRow(rates.simdTemplate,
                   histogramSimdRowKey(inputElements, numBins, widthBytes,
                                       unmasked, structure)),
      vars);
}

bool estimateHistogramStageCost(const LogicalStage &stage, StageMode mode,
                                double &perCallTotal, int64_t numWarps,
                                const StageHistogramRates &rates) {
  perCallTotal = 0.0;
  bool found = false;
  for (Operation *root : stage.operations) {
    if (!root)
      continue;
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "tt.histogram")
        return;
      auto input = op->getNumOperands() > 0
                       ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                       : RankedTensorType();
      auto result = op->getNumResults() > 0
                        ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                        : RankedTensorType();
      if (!input || !result || !input.hasStaticShape() ||
          !result.hasStaticShape() || result.getRank() != 1)
        return;
      const int64_t bits = input.getElementTypeBitWidth();
      const int64_t elements = input.getNumElements();
      const int64_t bins = result.getShape()[0];
      if (elements <= 0 || bins <= 0 || bits == 0 || bits % 8 != 0 || bits > 64)
        return;
      perCallTotal += estimateHistogramCycles(
          elements, bins, static_cast<unsigned>(bits / 8), mode,
          /*unmasked=*/op->getNumOperands() == 1, numWarps, rates);
      found = true;
    });
  }
  return found;
}

} // namespace mlir::ascend
