#include "AscendModel/RouteModel/PrefixScan/PrefixScanFormulaModel.h"
#include "AscendModel/RouteModel/StageCostModels.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <system_error>
#include <tuple>

using namespace mlir::ascend;

namespace {
const std::set<std::string> featureNames = {"one",
                                            "E",
                                            "C",
                                            "logE",
                                            "logW",
                                            "invW",
                                            "f16",
                                            "B",
                                            "chain",
                                            "vecwork",
                                            "payload",
                                            "tileR",
                                            "paddedwork",
                                            "doubling",
                                            "treework",
                                            "q",
                                            "H",
                                            "axisK",
                                            "local",
                                            "warp",
                                            "merge",
                                            "combined",
                                            "shared",
                                            "updates",
                                            "N",
                                            "Ba",
                                            "P",
                                            "blocks",
                                            "rounds",
                                            "live_words",
                                            "partial_values",
                                            "pressure24",
                                            "pressure48",
                                            "pressure127",
                                            "pressure2_8",
                                            "pressure2_16",
                                            "pressure2_24",
                                            "pressure2_48",
                                            "pressure2_64",
                                            "pressure2_96",
                                            "partial_pressure8",
                                            "partial_pressure16",
                                            "partial_pressure2_cta",
                                            "scan_spill_warp_work",
                                            "scan_spill_subwarp_work",
                                            "direct_group_work",
                                            "direct_narrow_group_work",
                                            "direct_half_row_groups"};
const std::set<std::string> branches = {"short",
                                        "long",
                                        "tree",
                                        "chain",
                                        "direct",
                                        "direct_full_groups",
                                        "transpose",
                                        "product_multi",
                                        "scalar_or_generic",
                                        "warp_local",
                                        "cross_warp",
                                        "multi_element",
                                        "multi",
                                        "layout_any",
                                        "layout_warp_local",
                                        "layout_cross_warp",
                                        "layout_cross_warp_full_lane",
                                        "layout_cross_warp_subwarp",
                                        "layout_multi_block"};
double ceilDiv(double a, double b) { return std::ceil(a / b); }
using Features = std::map<std::string, double>;
// Integer helpers used by bitwise work reconstruction.
int64_t ceilDivInt(int64_t a, int64_t b) { return a / b + (a % b != 0); }
int64_t product(const std::vector<int64_t> &v) {
  return std::accumulate(v.begin(), v.end(), int64_t(1),
                         std::multiplies<int64_t>());
}
bool powerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}
// Feature allowlist for the OR/AND/XOR formula asset.
const std::set<std::string> bitwiseFeatureNames = {
    "one",
    "scanLength",
    "elements",
    "sequences",
    "combines",
    "pre",
    "post",
    "rank",
    "axis",
    "is_or",
    "is_xor",
    "post_power_of_two",
    "dim0",
    "dim1",
    "dim2",
    "sequence_update_pairs",
    "sequence_update_quads",
    "compact_buffer_work",
    "far_axis_updates",
    "beyond4",
    "beyond6",
    "scalar_input_loads",
    "scalar_previous_loads",
    "scalar_output_stores",
    "scan_instruction_work",
    "warps",
    "warp_log_work",
    "axis_lanes",
    "axis_warps",
    "axis_blocks",
    "axis_elements_per_thread",
    "values_per_thread",
    "independent_blocks",
    "axis_element_stride",
    "axis_thread_stride",
    "axis_block_stride",
    "inner_parallel_elements",
    "inner_parallel_lanes",
    "inner_parallel_warps",
    "inner_parallel_blocks",
    "live_local_combines",
    "issued_shuffle_steps",
    "data_shuffle_steps",
    "live_shared_loads",
    "source_shared_stores",
    "reverse_flip_steps",
    "live_local_issue_groups",
    "data_shuffle_issue_groups",
    "dead_shuffle_issue_groups",
    "live_shared_word_traffic",
    "projected_data_lop_words",
    "projected_data_select_words",
    "duplicate_warp_values",
    "dag_binary_depth",
    "dag_shuffle_depth",
    "dag_shared_load_depth",
    "dag_select_depth",
    "dag_shared_store_depth",
    "dag_barrier_depth",
    "dag_unit_depth"};

/// Dependency depths for SIMT OR/AND/XOR before native instruction fusion.
class ScanDependencies {
  using Depth = std::array<int64_t, 7>;
  std::vector<Depth> depths;
  std::map<std::tuple<int, std::vector<int>, std::string>, int> intern;

  int node(int kind, std::vector<int> parents = {}, std::string tag = {},
           bool sideEffect = false) {
    auto key = std::make_tuple(kind, parents, tag);
    if (!sideEffect) {
      auto it = intern.find(key);
      if (it != intern.end())
        return it->second;
    }
    Depth d{};
    for (int p : parents)
      for (unsigned i = 0; i < d.size(); ++i)
        d[i] = std::max(d[i], depths[p][i]);
    if (kind >= 0) {
      ++d[kind];
      ++d[6];
    }
    int id = depths.size();
    depths.push_back(d);
    intern[key] = id;
    return id;
  }
  int combine(int a, int b) { return a < 0 ? b : node(0, {a, b}); }
  // condition: positive=true, negative=false, zero=runtime predicate.
  int select(int condition, int a, int b, std::string tag) {
    if (condition > 0)
      return a;
    if (condition < 0)
      return b;
    return a == b ? a : node(3, {a, b}, std::move(tag));
  }
  int shuffle(int value, std::string kind, int64_t offset) {
    return node(1, {value}, kind + std::to_string(offset), true);
  }
  void flip(std::vector<int> &src) {
    std::reverse(src.begin(), src.end());
    for (int &value : src)
      for (int64_t offset : {16, 8, 4, 2, 1})
        value = shuffle(value, "xor", offset);
  }

public:
  Depth run(int64_t a, int64_t lanes, int64_t warps, int64_t blocks,
            int64_t values, int64_t elementStride, int64_t parallelElements,
            int64_t blockStride, int64_t allocatedWarps, int64_t allocatedLanes,
            bool reverse) {
    std::vector<int> src;
    for (int64_t i = 0; i < values; ++i)
      src.push_back(node(-1, {}, std::to_string(i)));
    if (reverse)
      flip(src);
    std::map<int64_t, int> accumulators;
    for (int64_t i = 0; i < values; ++i) {
      int64_t index =
          i % elementStride + (i / elementStride / a) * elementStride;
      auto it = accumulators.find(index);
      src[i] = combine(it == accumulators.end() ? -1 : it->second, src[i]);
      accumulators[index] = src[i];
    }
    std::vector<int64_t> last;
    for (int64_t i = 0; i < values; ++i)
      if ((i / elementStride) % a == a - 1)
        last.push_back(i);
    for (int64_t i : last) {
      int acc = src[i];
      for (int64_t offset : {1, 2, 4, 8, 16}) {
        if (offset > lanes / 2)
          break;
        int received = shuffle(acc, "up", offset);
        acc = select(0, combine(received, acc), acc,
                     "lane_ge" + std::to_string(offset));
      }
      src[i] = acc;
    }
    int barrier = -1;
    if (warps > 1) {
      std::vector<int> stores;
      for (int64_t i : last)
        stores.push_back(node(4, {src[i]}));
      barrier = node(5, stores);
    }
    accumulators.clear();
    std::map<int64_t, int> masked;
    for (size_t chunk = 0; chunk < last.size(); ++chunk) {
      int64_t i = last[chunk], block = chunk / parallelElements;
      int64_t parallelBlock =
          block % blockStride + (block / blockStride / blocks) * blockStride;
      int64_t id = chunk % parallelElements + parallelBlock * parallelElements;
      int64_t axisBlock = (block / blockStride) % blocks;
      int acc = accumulators.count(id) ? accumulators[id] : -1;
      if (warps > 1) {
        int mask = masked.count(id) ? masked[id] : -1;
        for (int64_t w = 0; w < warps; ++w) {
          int partial = node(2, {barrier},
                             std::to_string(chunk) + ":" + std::to_string(w));
          if (acc < 0)
            acc = mask = partial;
          else {
            acc = combine(acc, partial);
            mask = select(w + 1 >= allocatedWarps ? -1 : 0, acc, mask,
                          "warp_ge" + std::to_string(w + 1));
          }
        }
        int temp = combine(mask, src[i]);
        if (axisBlock == 0)
          temp = select(0, temp, src[i], "warp_not_first");
        src[i] = temp;
        int carried = select(allocatedLanes == 1 ? -1 : 0,
                             shuffle(temp, "up", 1), mask, "lane_not_first");
        for (int64_t offset = 1; offset < a; ++offset) {
          int64_t j = i - offset * elementStride;
          int updated = combine(carried, src[j]);
          if (axisBlock == 0)
            updated =
                select(allocatedLanes == 1 && allocatedWarps == 1 ? -1 : 0,
                       updated, src[j], "not_first_thread");
          src[j] = updated;
        }
        accumulators[id] = acc;
        // The next axis block starts with the complete preceding block sum.
        masked[id] = acc;
      } else if (values > 1) {
        if (axisBlock == 0)
          acc = src[i];
        else
          src[i] = combine(acc, src[i]);
        int carried = src[i];
        if (lanes > 1) {
          carried = select(allocatedLanes == 1 ? 1 : 0, acc,
                           shuffle(src[i], "up", 1), "lane_eq0");
          if (blocks > 1)
            acc = shuffle(src[i], "idx", 0);
        } else if (blocks > 1)
          acc = src[i];
        for (int64_t offset = 1; offset < a; ++offset) {
          int64_t j = i - offset * elementStride;
          int updated = combine(carried, src[j]);
          if (axisBlock == 0)
            updated = select(allocatedLanes == 1 && allocatedWarps == 1 ? 1 : 0,
                             src[j], updated, "first_thread");
          src[j] = updated;
        }
        accumulators[id] = acc;
      }
    }
    if (reverse)
      flip(src);
    Depth result{};
    for (int p : src)
      for (unsigned i = 0; i < result.size(); ++i)
        result[i] = std::max(result[i], depths[p][i]);
    return result;
  }
};

// Shared layout projection for arithmetic and bitwise SIMT scan features.
// The legacy projection limits vector elements to the innermost dimension.
struct ScanLayout {
  std::vector<int64_t> sizePerThread, threadsPerWarp, warpsPerCTA, order;
};
ScanLayout projectScanLayout(const std::vector<int64_t> &shape, int64_t bits,
                             int64_t warps, bool rowMajor) {
  size_t rank = shape.size();
  ScanLayout layout;
  auto &s = layout.sizePerThread;
  auto &t = layout.threadsPerWarp;
  auto &v = layout.warpsPerCTA;
  s.assign(rank, 1);
  t.resize(rank);
  v.resize(rank);
  double n = std::accumulate(shape.begin(), shape.end(), 1.0,
                             std::multiplies<double>());
  double vector = std::min(std::floor(128.0 / bits),
                           std::max(std::floor(n / (32.0 * warps)), 1.0));
  if (rowMajor) {
    for (size_t i = rank; i-- > 0;) {
      s[i] = std::min(vector, double(shape[i]));
      vector = std::floor(vector / s[i]);
    }
  } else
    s.back() = std::min(vector, double(shape.back()));
  double remaining = 32;
  for (size_t i = rank; i-- > 0;) {
    t[i] = i == 0
               ? remaining
               : std::min(remaining,
                          std::max(std::floor(double(shape[i]) / s[i]), 1.0));
    remaining = std::floor(remaining / t[i]);
  }
  remaining = warps;
  for (size_t i = rank; i-- > 0;) {
    v[i] = i == 0
               ? remaining
               : std::min(remaining,
                          std::max(std::floor(double(shape[i]) / (s[i] * t[i])),
                                   1.0));
    remaining = std::floor(remaining / v[i]);
    layout.order.push_back(i);
  }
  return layout;
}

// Shared coefficient-weighted evaluation for all table-driven scan formulas.
std::optional<PrefixScanEstimate>
evaluateScanFormula(const PrefixScanFormula &formula,
                    const Features &features) {
  double cycles = 0;
  for (const auto &[name, coefficient] : formula.terms) {
    auto found = features.find(name);
    if (found == features.end())
      return std::nullopt;
    cycles += coefficient * found->second;
  }
  if (!std::isfinite(cycles) || cycles <= 0)
    return std::nullopt;
  return PrefixScanEstimate{formula.id, cycles};
}
// Bitwise only: reconstruct scalar or SIMT work from the scan geometry.
std::optional<Features>
bitwiseScanFeatures(const StageWorkload::PrefixScanKey &scan, unsigned numWarps,
                    bool simd) {
  const auto &shape = std::get<10>(scan);
  const int64_t axis = std::get<8>(scan), elementBits = std::get<9>(scan);
  const bool reverse = std::get<4>(scan);
  const auto &combine = std::get<11>(scan);
  const std::string operation = combine == "arith.ori"    ? "or"
                                : combine == "arith.andi" ? "and"
                                : combine == "arith.xori" ? "xor"
                                                          : "";
  if (shape.empty() || axis < 0 || axis >= int64_t(shape.size()) ||
      (elementBits != 8 && elementBits != 16 && elementBits != 32 &&
       elementBits != 64) ||
      (operation != "or" && operation != "and" && operation != "xor"))
    return std::nullopt;
  int64_t n = 1, pre = 1, post = 1;
  for (size_t d = 0; d < shape.size(); ++d) {
    if (shape[d] <= 0 || n > std::numeric_limits<int64_t>::max() / shape[d])
      return std::nullopt;
    n *= shape[d];
    if (int64_t(d) < axis)
      pre *= shape[d];
    if (int64_t(d) > axis)
      post *= shape[d];
  }
  int64_t e = shape[axis], c = pre * post;
  if (e != std::get<0>(scan) || c != std::get<1>(scan))
    return std::nullopt;
  // Reuse rank-two/three formulas by merging dimensions around the scan axis.
  auto modelShape = shape;
  int64_t modelAxis = axis;
  if (shape.size() > 3) {
    if (axis == 0) {
      modelShape = {e, post};
      modelAxis = 0;
    } else if (axis == int64_t(shape.size()) - 1) {
      modelShape = {pre, e};
      modelAxis = 1;
    } else {
      modelShape = {pre, e, post};
      modelAxis = 1;
    }
  }
  Features f;
  f["one"] = 1;
  f["scanLength"] = e;
  f["elements"] = n;
  f["sequences"] = c;
  f["combines"] = n - c;
  f["pre"] = pre;
  f["post"] = post;
  f["rank"] = modelShape.size();
  f["axis"] = modelAxis;
  f["is_or"] = operation == "or";
  f["is_xor"] = operation == "xor";
  f["post_power_of_two"] = powerOfTwo(post);
  for (size_t d = 0; d < 3; ++d)
    f["dim" + std::to_string(d)] = d < modelShape.size() ? modelShape[d] : 1;
  if (simd) {
    int64_t bytes = elementBits / 8;
    f["sequence_update_pairs"] = c * ceilDivInt(e - 1, 2);
    f["sequence_update_quads"] = c * ceilDivInt(e - 1, 4);
    f["compact_buffer_work"] = n <= 32 / bytes ? n : 0;
    f["far_axis_updates"] =
        c * std::max<int64_t>(0, e - 1 - 2048 / post / bytes);
    f["beyond4"] = std::max<int64_t>(e - 4, 0);
    f["beyond6"] = std::max<int64_t>(e - 6, 0);
    f["scalar_input_loads"] = n;
    f["scalar_previous_loads"] = n - c;
    f["scalar_output_stores"] = n;
    f["scan_instruction_work"] = 3.0 * n - c;
    return f;
  }
  if (!powerOfTwo(numWarps) ||
      llvm::any_of(shape, [](int64_t size) { return !powerOfTwo(size); }))
    return std::nullopt;
  const auto layout = projectScanLayout(shape, elementBits, numWarps, true);
  size_t rank = shape.size();
  if (layout.sizePerThread.size() != rank ||
      layout.threadsPerWarp.size() != rank ||
      layout.warpsPerCTA.size() != rank || layout.order.size() != rank)
    return std::nullopt;
  std::set<int64_t> seen;
  int64_t allocatedLanes = 1, allocatedWarps = 1;
  std::vector<int64_t> contig(rank), lanes(rank), warps(rank), blocks(rank);
  for (size_t d = 0; d < rank; ++d) {
    if (!powerOfTwo(layout.sizePerThread[d]) ||
        !powerOfTwo(layout.threadsPerWarp[d]) ||
        !powerOfTwo(layout.warpsPerCTA[d]) || layout.order[d] < 0 ||
        layout.order[d] >= int64_t(rank) ||
        !seen.insert(layout.order[d]).second ||
        layout.threadsPerWarp[d] > 32 / allocatedLanes ||
        layout.warpsPerCTA[d] > numWarps / allocatedWarps)
      return std::nullopt;
    allocatedLanes *= layout.threadsPerWarp[d];
    allocatedWarps *= layout.warpsPerCTA[d];
    contig[d] = std::min(layout.sizePerThread[d], shape[d]);
    lanes[d] =
        std::min(layout.threadsPerWarp[d], ceilDivInt(shape[d], contig[d]));
    warps[d] = std::min(layout.warpsPerCTA[d],
                        ceilDivInt(ceilDivInt(shape[d], contig[d]), lanes[d]));
    blocks[d] = ceilDivInt(
        ceilDivInt(ceilDivInt(shape[d], contig[d]), lanes[d]), warps[d]);
  }
  if (allocatedLanes != 32 || allocatedWarps != numWarps)
    return std::nullopt;
  int64_t a = contig[axis], l = lanes[axis], k = warps[axis];
  int64_t b = blocks[axis], u = product(contig) * product(blocks);
  int64_t chunks = u / a, p = chunks / b,
          words = std::max<int64_t>(1, elementBits / 32);
  int64_t elementStride = 1, threadStride = 1, blockStride = 1;
  int64_t parallelElements = 1, inner = -1;
  bool beforeAxis = true;
  for (int64_t d : layout.order) {
    if (d == axis)
      beforeAxis = false;
    else {
      if (inner < 0)
        inner = d;
      parallelElements *= contig[d];
      if (beforeAxis) {
        elementStride *= contig[d];
        threadStride *= layout.threadsPerWarp[d];
        blockStride *= blocks[d];
      }
    }
  }
  int64_t sourceLocal = u - chunks;
  if (k > 1)
    sourceLocal += u + p * (b * k - 1);
  else if (l == 1)
    sourceLocal += p * (b - 1) * a;
  else if (u > 1)
    sourceLocal += u - chunks + p * (b - 1);
  int64_t loads = k > 1 ? p * (b * k - 1) : 0;
  int64_t stores = k > 1 ? chunks : 0;
  int64_t local = k > 1 ? (u - chunks) + u + p * std::max<int64_t>(0, b * k - 2)
                        : sourceLocal;
  int64_t postUp = k > 1 || (u > 1 && l > 1) ? chunks : 0;
  int64_t postIdx = k == 1 && u > 1 && l > 1 && b > 1 ? chunks : 0;
  double issued = (chunks * std::log2(double(l)) + postUp + postIdx) * words;
  int64_t dataUp = a > 1 && l > 1 && (k > 1 || u > 1) ? chunks : 0;
  int64_t dataIdx = k == 1 && l > 1 && b > 1 ? p * (b - 1) : 0;
  double shuffle = (chunks * std::log2(double(l)) + dataUp + dataIdx) * words;
  int64_t fusion = k == 1 && l > 1 ? p * (b - 1) : 0;
  int64_t pairFusion =
      k == 2 && a == 1 && b > 1 && (l > 1 || operation == "and" || reverse) ? p
                                                                            : 0;
  double lop =
      words * (local + chunks * std::log2(double(l)) - fusion - pairFusion);
  double selects = words * (chunks * std::log2(double(l)) + p * b * (k - 1) +
                            (a > 1 && l > 1 ? chunks : 0) +
                            (k > 1 && a > 1 && l > 1 ? p : 0));
  if (elementBits == 32 && k == 1 && l == 2 && b == 1 && operation != "and")
    selects -= chunks * (1 + (a > 1));
  f["warps"] = numWarps;
  f["warp_log_work"] = numWarps * std::log2(double(numWarps));
  f["axis_lanes"] = l;
  f["axis_warps"] = k;
  f["axis_blocks"] = b;
  f["axis_elements_per_thread"] = a;
  f["values_per_thread"] = u;
  f["independent_blocks"] = p;
  f["axis_element_stride"] = elementStride;
  f["axis_thread_stride"] = threadStride;
  f["axis_block_stride"] = blockStride;
  f["inner_parallel_elements"] = inner < 0 ? 1 : contig[inner];
  f["inner_parallel_lanes"] = inner < 0 ? 1 : lanes[inner];
  f["inner_parallel_warps"] = inner < 0 ? 1 : warps[inner];
  f["inner_parallel_blocks"] = inner < 0 ? 1 : blocks[inner];
  f["live_local_combines"] = local;
  f["issued_shuffle_steps"] = issued;
  f["data_shuffle_steps"] = shuffle;
  f["live_shared_loads"] = loads;
  f["source_shared_stores"] = stores;
  f["reverse_flip_steps"] = 10 * u * words * reverse;
  f["live_local_issue_groups"] = std::ceil(numWarps * double(local) / 4);
  f["data_shuffle_issue_groups"] = std::ceil(numWarps * shuffle / 4);
  f["dead_shuffle_issue_groups"] = std::ceil(numWarps * (issued - shuffle) / 4);
  f["live_shared_word_traffic"] =
      numWarps * double(loads) * product(lanes) / l * words;
  f["projected_data_lop_words"] = lop;
  f["projected_data_select_words"] = selects;
  f["duplicate_warp_values"] = (numWarps - product(warps)) * u;
  f["scan_instruction_work"] = local + issued + loads + stores;
  auto depth = ScanDependencies().run(
      a, l, k, b, u, elementStride, parallelElements, blockStride,
      layout.warpsPerCTA[axis], layout.threadsPerWarp[axis], reverse);
  const char *names[] = {"dag_binary_depth",       "dag_shuffle_depth",
                         "dag_shared_load_depth",  "dag_select_depth",
                         "dag_shared_store_depth", "dag_barrier_depth",
                         "dag_unit_depth"};
  for (unsigned i = 0; i < depth.size(); ++i)
    f[names[i]] = depth[i];
  return f;
}

// Bitwise only: select a piecewise fit and use the common formula evaluator.
std::optional<PrefixScanEstimate>
estimateBitwiseScan(const StageWorkload::PrefixScanKey &scan, unsigned warps,
                    bool simd, const std::vector<PrefixScanFormula> &formulas) {
  auto features = bitwiseScanFeatures(scan, warps, simd);
  if (!features)
    return std::nullopt;
  // S1 is scalar SIMD; T1/T2 use one/multiple axis warps in SIMT.
  const std::string branch = simd                              ? "S1"
                             : features->at("axis_warps") == 1 ? "T1"
                                                               : "T2";
  auto matches = [&](const PrefixScanFormulaDomain &domain) {
    return domain.operation == std::get<11>(scan) &&
           domain.dataType == std::get<6>(scan) && domain.branch == branch;
  };
  auto priority = [&](const PrefixScanFormulaDomain &domain) {
    return !domain.reverse ? 1 : *domain.reverse == std::get<4>(scan) ? 2 : 0;
  };
  // Select the direction before applying leaf conditions, just as for a tree.
  int best = -1;
  for (const auto &formula : formulas)
    if (formula.projection == "source_work")
      for (const auto &domain : formula.domains)
        if (matches(domain))
          best = std::max(best, priority(domain));
  if (best < 0)
    return std::nullopt;
  std::optional<PrefixScanEstimate> result;
  for (const auto &formula : formulas) {
    if (formula.projection != "source_work")
      continue;
    bool matched = llvm::any_of(formula.domains, [&](const auto &domain) {
      return matches(domain) && priority(domain) == best &&
             llvm::all_of(domain.conditions, [&](const auto &condition) {
               auto found = features->find(condition.feature);
               return found != features->end() &&
                      ((found->second <= condition.threshold) ==
                       condition.lessEqual);
             });
    });
    if (!matched)
      continue;
    if (result)
      return std::nullopt;
    result = evaluateScanFormula(formula, *features);
    if (!result)
      return std::nullopt;
  }
  return result;
}
} // namespace

llvm::Expected<std::vector<PrefixScanFormula>>
mlir::ascend::parsePrefixScanFormulas(const llvm::json::Value &profile) {
  auto invalid = [] {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invalid prefixscan formula asset");
  };
  const auto *root = profile.getAsObject();
  const auto *formulas = profile.getAsArray();
  // OR/AND/XOR assets hold shared coefficients and mode-specific trees.
  if (root) {
    formulas = root->getArray("formulas");
    const auto *dispatch = root->getArray("dispatch");
    if (root->getInteger("schema_version") != 1 ||
        root->getString("target") != "Ascend950PR" ||
        root->getString("cycle_unit") != "system_cycle" || !formulas ||
        formulas->empty() || !dispatch || dispatch->empty())
      return invalid();
  }
  if (!formulas)
    return invalid();
  std::vector<PrefixScanFormula> result;
  std::map<std::string, size_t> ids;
  for (const auto &entry : *formulas) {
    const auto *object = entry.getAsObject();
    if (!object)
      return invalid();
    auto id = object->getString("id");
    const auto *terms = object->getObject("terms");
    const auto *domains = object->getArray("domains");
    if (!id || id->empty() || !terms || terms->empty() ||
        (!root && (!domains || domains->empty())) || ids.count(id->str()))
      return invalid();
    PrefixScanFormula formula;
    formula.id = id->str();
    if (root)
      formula.projection = "source_work";
    else if (auto projection = object->getString("projection")) {
      if (*projection != "legacy" && *projection != "row_major")
        return invalid();
      formula.projection = projection->str();
    } else if (object->get("projection"))
      return invalid();
    const auto &allowedFeatures = root ? bitwiseFeatureNames : featureNames;
    for (const auto &term : *terms) {
      auto value = term.second.getAsNumber();
      if (!value || !std::isfinite(*value) ||
          !allowedFeatures.count(term.first.str()) ||
          (root && term.first != "one" && *value < 0))
        return invalid();
      formula.terms[term.first.str()] = *value;
    }
    if (root && formula.terms["scan_instruction_work"] <= 0)
      return invalid();
    if (!root) {
      // Expand compact alternatives once at load time; matching stays scalar.
      llvm::json::Array expanded;
      for (const auto &value : *domains) {
        const auto *domain = value.getAsObject();
        if (!domain)
          return invalid();
        llvm::json::Array alternatives;
        alternatives.push_back(llvm::json::Object(*domain));
        for (const char *field : {"operation", "data_type", "warps"}) {
          const auto *values = domain->getArray(field);
          if (!values)
            continue;
          if (values->empty())
            return invalid();
          llvm::json::Array next;
          for (const auto &alternative : alternatives)
            for (const auto &item : *values) {
              llvm::json::Object copy(*alternative.getAsObject());
              copy[field] = item;
              next.push_back(std::move(copy));
            }
          alternatives = std::move(next);
        }
        for (auto &alternative : alternatives)
          expanded.push_back(std::move(alternative));
      }
      for (const auto &value : expanded) {
        auto *domain = value.getAsObject();
        if (!domain)
          return invalid();
        auto op = domain->getString("operation");
        auto type = domain->getString("data_type");
        auto branch = domain->getString("branch");
        auto rank = domain->getInteger("rank"),
             axis = domain->getInteger("axis");
        auto warps = domain->getInteger("warps");
        if (!op || !type || !branch || !rank || !axis || !warps || *rank < 1 ||
            *axis < 0 || *axis >= *rank || *warps < 0 || *warps > 64 ||
            !branches.count(branch->str()) ||
            (branch->starts_with("layout_") &&
             formula.projection != "row_major") ||
            (*op != "cumprod" && *op != "cummax" && *op != "cummin") ||
            (*type != "f16" && *type != "f32" && *type != "i32" &&
             *type != "u32" && *type != "i64" && *type != "u64" &&
             *type != "integer32" && *type != "integer64"))
          return invalid();
        formula.domains.push_back(
            {op->str(), type->str(), branch->str(), *rank, *axis, *warps});
      }
    }
    ids[formula.id] = result.size();
    result.push_back(std::move(formula));
  }
  if (!root)
    return result;
  std::set<
      std::tuple<bool, int64_t, std::optional<bool>, std::string, std::string>>
      roots;
  for (const auto &entry : *root->getArray("dispatch")) {
    auto *object = entry.getAsObject();
    if (!object)
      return invalid();
    auto mode = object->getString("mode");
    auto bits = object->getInteger("element_bits");
    auto reverse = object->getBoolean("reverse");
    auto algorithm = object->getString("algorithm");
    auto group = object->getString("operation_group");
    auto *tree = object->get("tree");
    if (!mode || (*mode != "simd" && *mode != "simt") || !bits ||
        (*bits != 8 && *bits != 16 && *bits != 32 && *bits != 64) ||
        (object->get("reverse") && !reverse) || !algorithm || !group || !tree)
      return invalid();
    bool modeSimd = *mode == "simd";
    if ((modeSimd && (*algorithm != "S1" || *group != "all_bitwise")) ||
        (!modeSimd && ((*algorithm != "T1" && *algorithm != "T2") ||
                       (*group != "and" && *group != "or_xor"))) ||
        !roots.emplace(modeSimd, *bits, reverse, algorithm->str(), group->str())
             .second)
      return invalid();
    std::vector<PrefixScanFormulaCondition> conditions;
    std::function<bool(const llvm::json::Value &, unsigned)> expand;
    expand = [&](const llvm::json::Value &value, unsigned depth) {
      if (depth > 128)
        return false;
      const auto *node = value.getAsObject();
      // Leaves store formula IDs directly; older object leaves remain valid.
      auto id = value.getAsString();
      if (!id && node)
        id = node->getString("formula");
      if (id) {
        auto found = ids.find(id->str());
        if (found == ids.end() || (node && node->size() != 1))
          return false;
        for (const char *operation :
             {"arith.ori", "arith.andi", "arith.xori"}) {
          if ((*group == "and" && llvm::StringRef(operation) != "arith.andi") ||
              (*group == "or_xor" &&
               llvm::StringRef(operation) == "arith.andi"))
            continue;
          PrefixScanFormulaDomain domain;
          domain.operation = operation;
          domain.dataType = "i" + std::to_string(*bits);
          domain.branch = algorithm->str();
          domain.reverse = reverse;
          domain.conditions = conditions;
          result[found->second].domains.push_back(std::move(domain));
        }
        return true;
      }
      if (!node)
        return false;
      auto feature = node->getString("feature");
      auto threshold = node->getNumber("threshold");
      auto *left = node->get("left"), *right = node->get("right");
      if (!feature || !bitwiseFeatureNames.count(feature->str()) ||
          !threshold || !std::isfinite(*threshold) || !left || !right)
        return false;
      conditions.push_back({feature->str(), *threshold, true});
      if (!expand(*left, depth + 1))
        return false;
      conditions.back().lessEqual = false;
      if (!expand(*right, depth + 1))
        return false;
      conditions.pop_back();
      return true;
    };
    if (!expand(*tree, 0))
      return invalid();
  }
  llvm::erase_if(result,
                 [](const auto &formula) { return formula.domains.empty(); });
  return result;
}

std::optional<PrefixScanEstimate> mlir::ascend::estimatePrefixScanFormula(
    const StageWorkload::PrefixScanKey &scan, unsigned warps, bool simd,
    const std::vector<PrefixScanFormula> &formulas) {
  if (formulas.empty() || std::get<3>(scan) != 1 || !std::get<12>(scan))
    return std::nullopt;
  const auto &combineOperation = std::get<11>(scan);
  // OR/AND/XOR use numeric splits within the selected bitwise algorithm.
  if (combineOperation == "arith.ori" || combineOperation == "arith.andi" ||
      combineOperation == "arith.xori")
    return estimateBitwiseScan(scan, warps, simd, formulas);
  if (!simd && (warps == 0 || warps > 64))
    return std::nullopt;
  auto shape = std::get<10>(scan);
  int64_t axis = std::get<8>(scan);
  const int64_t bits = std::get<9>(scan);
  if (shape.empty() || axis < 0 || axis >= int64_t(shape.size()) || bits <= 0 ||
      bits > 128 ||
      std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d <= 0; }))
    return std::nullopt;
  const double E = std::get<0>(scan), C = std::get<1>(scan);
  if (E <= 1 || C < 1 || double(shape[axis]) != E)
    return std::nullopt;
  // Use the same aggregate per-AIV geometry as cumsum, keeping post fixed.
  const double post = std::get<2>(scan);
  const double elements = std::accumulate(shape.begin(), shape.end(), 1.0,
                                          std::multiplies<double>());
  if (simd) {
    if (elements / E != C)
      return std::nullopt;
  } else if (shape.size() == 1) {
    if (C != 1)
      return std::nullopt;
  } else if (elements / E != C) {
    // Preserve the existing aggregate per-AIV geometry. Resize one non-axis
    // dimension while retaining the remaining shape factors.
    const size_t adjusted = axis == 0 ? 1 : 0;
    double fixed = 1;
    for (size_t i = 0; i < shape.size(); ++i)
      if (i != adjusted && i != size_t(axis))
        fixed *= shape[i];
    const double extent = C / fixed;
    if (!std::isfinite(extent) || extent < 1 || std::floor(extent) != extent ||
        extent >= double(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    shape[adjusted] = int64_t(extent);
  }
  // Higher ranks use the agreed canonical scan geometry. Keep measured rank
  // two/three shapes intact so their within-axis layout factors are preserved.
  const bool foldedHigherRank = shape.size() > 3;
  if (foldedHigherRank) {
    double pre = 1, after = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i < size_t(axis))
        pre *= shape[i];
      if (i > size_t(axis))
        after *= shape[i];
    }
    if (!std::isfinite(pre) || !std::isfinite(after) ||
        pre >= double(std::numeric_limits<int64_t>::max()) ||
        after >= double(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    if (axis == 0) {
      shape = {int64_t(E), int64_t(after)};
      axis = 0;
    } else if (axis == int64_t(shape.size()) - 1) {
      shape = {int64_t(pre), int64_t(E)};
      axis = 1;
    } else {
      shape = {int64_t(pre), int64_t(E), int64_t(after)};
      axis = 1;
    }
  }
  const std::string &combine = std::get<11>(scan);
  std::string operation, type;
  const bool floating = combine == "arith.mulf" || combine == "arith.maxnumf" ||
                        combine == "arith.minnumf";
  // Calibration used tl.maximum/minimum with PropagateNan.NONE.
  // Propagating-NaN maximumf/minimumf are distinct and must fall back.
  if (combine == "arith.mulf" || combine == "arith.muli")
    operation = "cumprod";
  else if (combine == "arith.maxnumf" || combine == "arith.maxsi" ||
           combine == "arith.maxui")
    operation = "cummax";
  else if (combine == "arith.minnumf" || combine == "arith.minsi" ||
           combine == "arith.minui")
    operation = "cummin";
  else
    return std::nullopt;
  if (floating) {
    // bf16/f16 have equal storage width but are not the same measured dtype.
    type = std::get<6>(scan);
    if (type != "f16" && type != "f32")
      return std::nullopt;
  } else if (combine == "arith.muli")
    type = "integer" + std::to_string(bits);
  else
    type = std::string(combine == "arith.maxui" || combine == "arith.minui"
                           ? "u"
                           : "i") +
           std::to_string(bits);
  const double W = warps, N = E * C;
  double pre = 1;
  for (int64_t i = 0; i < axis; ++i)
    pre *= shape[i];
  std::map<std::string, double> f = {{"one", 1},
                                     {"E", E},
                                     {"C", C},
                                     {"logE", std::log2(E)},
                                     {"f16", type == "f16"}};
  f["N"] = N;
  std::map<std::string, double> dense;
  std::string layoutBranch;
  if (simd) {
    f["B"] = ceilDiv(E, 64);
    f["chain"] = C * (E - 1);
    f["vecwork"] = (E - 1) * ceilDiv(C, 64);
    f["payload"] = 4 * N;
    f["tileR"] = ceilDiv(pre, 4) * ceilDiv(post, 64) * (E - 1);
    const double groups = std::floor(pre / 4);
    f["direct_group_work"] = groups * ceilDiv(post, 64) * (E - 1);
    // Quarter-register rows in a retained outer group loop require additional
    // vector-store ordering in the calibrated lowering. Wider rows do not
    // share this cost, even when their element count is larger.
    f["direct_narrow_group_work"] =
        groups > 1 && post < 32 ? f["direct_group_work"] : 0;
    f["direct_half_row_groups"] =
        groups > 1 && post >= 32 && post < 64 ? groups : 0;
    f["paddedwork"] = (E - 1) * ceilDiv(8 * ceilDiv(C, 8), 64);
    f["doubling"] = 0;
    f["treework"] = 0;
    for (double step = 1; step < E; step *= 2) {
      f["doubling"] += std::max(0.0, std::floor(E / 64) - ceilDiv(step, 64));
      f["treework"] +=
          ceilDiv(C, 64) * (std::floor(E / (2 * step)) * step +
                            std::max(0.0, std::fmod(E, 2 * step) - step));
    }
  } else {
    auto projectLayout = [&](bool rowMajor,
                             std::map<std::string, double> &values) {
      double A, L, K, Ba, P = 1, Bp = 1;
      if (shape.size() == 1) {
        A = std::min(std::floor(128.0 / bits),
                     std::max(std::floor(E / (32 * W)), 1.0));
        L = std::min(32.0, std::max(std::floor(E / A), 1.0));
        K = std::min(W, std::max(std::floor(E / (A * L)), 1.0));
      } else {
        const auto layout = projectScanLayout(shape, bits, warps, rowMajor);
        const auto &s = layout.sizePerThread;
        const auto &t = layout.threadsPerWarp;
        const auto &v = layout.warpsPerCTA;
        A = s[axis];
        L = std::min(double(t[axis]), std::max(std::floor(E / A), 1.0));
        K = std::min(double(v[axis]), std::max(std::floor(E / (A * L)), 1.0));
        for (size_t i = 0; i < shape.size(); ++i)
          if (i != size_t(axis)) {
            P *= s[i];
            Bp *= ceilDiv(shape[i], s[i] * t[i] * v[i]);
          }
      }
      Ba = ceilDiv(E, A * L * K);
      const double H = Ba * Bp * P;
      values["logW"] = std::log2(W);
      values["invW"] = 1 / W;
      values["q"] = E / (32 * W);
      values["H"] = H;
      values["axisK"] = K;
      values["axis_lanes"] = L;
      values["local"] = H * (A - 1);
      values["warp"] = H * std::log2(L);
      values["merge"] = P * Bp * (K * Ba - 1);
      values["shared"] = K > 1 ? H * K : 0;
      values["updates"] = H * A;
      values["combined"] = values["local"] + values["warp"] + values["merge"];
      values["Ba"] = Ba;
      values["P"] = P;
      values["blocks"] = Ba * Bp;
      values["rounds"] = std::ceil(std::log2(K));
      const double words = std::ceil(double(bits) / 32);
      values["live_words"] = (H * A + P * Bp) * words;
      values["partial_values"] = H * K * words;
      // Soft pressure proxy beyond one CTA-sized set of scan partials.
      const double partialExcess =
          std::max(0.0, values["partial_values"] - 32.0 * W);
      values["partial_pressure2_cta"] = partialExcess * partialExcess;
      values["pressure127"] = std::max(0.0, values["live_words"] - 127.0);
      // Approximate live input/output words against the per-thread budget.
      const double registerWords = 32 * W <= 256    ? 127
                                   : 32 * W <= 512  ? 64
                                   : 32 * W <= 1024 ? 32
                                                    : 16;
      const double valueExcess =
          std::max(0.0, 2.0 * H * A * words - registerWords);
      values["scan_spill_warp_work"] = valueExcess * std::log2(L);
      values["scan_spill_subwarp_work"] =
          valueExcess * std::max(0.0, 32.0 / L - 4.0);
      for (int threshold : {8, 16, 24, 32, 48, 64, 96}) {
        double excess = std::max(0.0, values["live_words"] - threshold);
        values["pressure" + std::to_string(threshold)] = excess;
        values["pressure2_" + std::to_string(threshold)] = excess * excess;
      }
      for (int reserve : {8, 16})
        values["partial_pressure" + std::to_string(reserve)] = std::max(
            0.0, values["partial_values"] -
                     std::max(1.0, std::min(128.0, 1024 / W) - reserve));
      if (rowMajor)
        layoutBranch = K > 1    ? "layout_cross_warp"
                       : Ba > 1 ? "layout_multi_block"
                                : "layout_warp_local";
      // Full-lane pricing is enabled only by a matching calibrated domain.
      if (rowMajor && K > 1 && L == 32) {
        const bool fullLaneAsset =
            llvm::any_of(formulas, [&](const auto &formula) {
              return llvm::any_of(formula.domains, [&](const auto &d) {
                return d.branch == "layout_cross_warp_full_lane" &&
                       d.operation == operation && d.dataType == type &&
                       d.rank == int64_t(shape.size()) && d.axis == axis &&
                       (d.warps == 0 || d.warps == warps);
              });
            });
        if (fullLaneAsset)
          layoutBranch = "layout_cross_warp_full_lane";
      }
    };
    dense = f;
    projectLayout(false, f);
    projectLayout(true, dense);
  }
  std::string branch;
  if (simd) {
    if (type == "i64" || type == "u64" || type == "integer64" ||
        (type == "u32" && operation != "cumprod"))
      branch = "scalar_or_generic";
    else if (C == 1)
      branch = E <= 128 ? "short" : "long";
    else if (operation != "cumprod" &&
             (axis == 0 || axis == int64_t(shape.size()) - 1))
      branch = E >= 8 && ceilDiv(C, 64) <= 2 ? "tree" : "chain";
    else if (operation != "cumprod")
      branch = post >= 16 ? "direct" : "transpose";
    else
      branch = "product_multi";
  } else if (shape.size() == 1) {
    branch = E <= 32                ? "warp_local"
             : W > 1 && E <= 32 * W ? "cross_warp"
                                    : "multi_element";
  } else
    branch = "multi";
  // A lane-specific calibrated domain takes precedence over a coarse domain.
  if (!simd && branch == "multi" && dense["axisK"] > 1 &&
      dense["axis_lanes"] < 32) {
    const bool subwarpAsset = llvm::any_of(formulas, [&](const auto &formula) {
      return llvm::any_of(formula.domains, [&](const auto &d) {
        return d.branch == "layout_cross_warp_subwarp" &&
               d.operation == operation && d.dataType == type &&
               d.rank == int64_t(shape.size()) && d.axis == axis &&
               (d.warps == 0 || d.warps == warps);
      });
    });
    if (subwarpAsset)
      branch = layoutBranch = "layout_cross_warp_subwarp";
  }
  // A qualified asset may refine the direct path with complete groups of
  // four rows. Preserve older assets and remainder-row
  // paths; the refined domain takes explicit priority, never JSON order.
  if (simd && branch == "direct" && pre >= 4 && std::fmod(pre, 4) == 0) {
    const bool hasFullGroups = llvm::any_of(formulas, [&](const auto &formula) {
      return llvm::any_of(formula.domains, [&](const auto &d) {
        return d.operation == operation && d.dataType == type &&
               d.rank == int64_t(shape.size()) && d.axis == axis &&
               d.branch == "direct_full_groups" &&
               (d.warps == 0 || d.warps == warps);
      });
    });
    if (hasFullGroups)
      branch = "direct_full_groups";
  }
  auto matchesFeatures = [&](const PrefixScanFormulaDomain &d) {
    return d.operation == operation && d.dataType == type &&
           (d.branch == branch || (!simd && (d.branch == layoutBranch ||
                                             d.branch == "layout_any"))) &&
           (d.warps == 0 || d.warps == warps);
  };
  const bool directDomain =
      std::any_of(formulas.begin(), formulas.end(), [&](const auto &formula) {
        return std::any_of(
            formula.domains.begin(), formula.domains.end(), [&](const auto &d) {
              return matchesFeatures(d) && d.rank == int64_t(shape.size()) &&
                     d.axis == axis;
            });
      });
  std::optional<PrefixScanEstimate> result;
  for (const auto &formula : formulas) {
    bool matches = false;
    for (const auto &d : formula.domains) {
      const bool canonical = d.rank == int64_t(shape.size()) && d.axis == axis;
      // Some head/tail layout regimes were measured only with rank three.
      // A folded high-rank edge uses that family only if no rank-two domain
      // exists, so selection is independent of asset ordering.
      const bool edgeFamily = foldedHigherRank && !directDomain &&
                              shape.size() == 2 && d.rank == 3 &&
                              d.axis == (axis == 0 ? 0 : 2);
      matches |= matchesFeatures(d) && (canonical || edgeFamily);
    }
    if (!matches)
      continue;
    const auto &values = !simd && formula.projection == "row_major" ? dense : f;
    // Mode-incompatible assets are rejected without evaluating absent features.
    if (std::any_of(
            formula.terms.begin(), formula.terms.end(),
            [&](const auto &term) { return !values.count(term.first); }) ||
        (simd && std::any_of(formula.domains.begin(), formula.domains.end(),
                             [](const auto &d) { return d.warps != 0; })))
      return std::nullopt;
    // Conflicting IDs cannot be selected by JSON order or measurement labels.
    if (result)
      return std::nullopt;
    result = evaluateScanFormula(formula, values);
    if (!result)
      return std::nullopt;
  }
  return result;
}

static PrefixScanEstimate
simtPrefixScan1dSegmentCost(const CalibratedPrefixScanProfile &simt,
                            int64_t numWarps, double n) {
  const auto &segments = simt.prefixScan1dWarpSegments;
  auto it = segments.lower_bound(
      static_cast<unsigned>(std::max<int64_t>(1, numWarps)));
  if (it == segments.end())
    it = std::prev(segments.end());
  const auto &segment = it->second;
  const std::string id = "cumsum_simt_1d_w" + std::to_string(it->first);
  const double warps = static_cast<double>(it->first);
  const double threads = 32.0 * warps;
  if (n <= 32.0)
    return {id + "_local", segment.cLocal};
  // warpsPerCTA[axis] = ceil(N/32), capped by the kernel warp count.
  const double chainWarps = std::min(warps, std::ceil(n / 32.0));
  // Cross-warp Sklansky merge.  Both gates mirror canUseMultiWarpSklansky:
  // one element per thread (N <= 32*w) and at most warpSize axis warps
  // (k <= 32, i.e. N <= 1024).  A kernel with more warps than that
  // (num_warps = 64) drops to the general path already at N = 2048 although
  // every thread still owns one element, so pricing it as Sklansky would
  // under-predict it.  ceil(log2 k) is the trip count of the Sklansky
  // `for (h = 1; h < k; h <<= 1)` loop, so the pricing tracks the real round
  // count instead of a linear proxy.
  if (n <= threads && chainWarps <= 32.0) {
    const double rounds = std::ceil(std::log2(std::max(1.0, chainWarps)));
    return {id + "_cross_warp", segment.cFixed + segment.cRound * rounds};
  }
  // General path.  The per-element serial slope is charged from N = 0; the
  // 32*w anchor of the (N - 32*w) form is folded into cGen, so cGen is not
  // directly comparable with the cross-warp segment constant.
  return {id + "_general", segment.cGen + segment.serialRate * n};
}

// SIMT cumsum segmented fit for multidimensional scans. The generic scan
// lowering (AddPartialReduce) distributes the scan across warps with a serial
// UB load chain whose cost is additive in the total element count N, plus a
// per-thread-element hinge that prices register pressure beyond r0.  E is the
// aggregate scan extent (exact for a single multi-dim scan per Stage) and
// determines k = warpsPerCTA[axis], the UB chain length.  Extents below one
// warp (E < 32) take the separate linear regime described inline below.
static PrefixScanEstimate
simtPrefixScan2dSegmentCost(const CalibratedPrefixScanProfile &simt,
                            int64_t numWarps, double extentSum, double n) {
  const auto &segments = simt.prefixScan2dWarpSegments;
  auto it = segments.lower_bound(
      static_cast<unsigned>(std::max<int64_t>(1, numWarps)));
  if (it == segments.end())
    it = std::prev(segments.end());
  const auto &segment = it->second;
  const std::string id = "cumsum_simt_multi_w" + std::to_string(it->first);
  const double extent = std::max(1.0, extentSum);
  // Sub-warp extent: the warp owning the axis holds fewer than 32 valid lanes,
  // so the axis warp never needs a cross-warp chain (k = 1) and the cost
  // collapses onto a single linear-in-N regime.  Layout-coalesced small scans
  // (e.g. a 16-element cumsum promoted to an (8,16,16) scan) land here.
  if (extent < 32.0 && (segment.aSubwarp > 0.0 || segment.bSubwarp > 0.0))
    return {id + "_subwarp", segment.aSubwarp + segment.bSubwarp * n};
  const double threads = 32.0 * static_cast<double>(it->first);
  // warpsPerCTA[axis]: warps holding unique data along the scan axis.
  const double chainWarps =
      std::min(static_cast<double>(it->first), std::ceil(extent / 32.0));
  const double chainLoads = n * std::max(1.0, chainWarps) / threads;
  const double hinge = std::max(0.0, n / threads - segment.r0);
  return {id + "_general", segment.a + segment.b * n + segment.c * chainLoads +
                               segment.d * hinge};
}

// Return the complete cost of one scan. A miss leaves all original resource
// and special-stage accounting to the caller.
std::optional<PrefixScanEstimate>
mlir::ascend::estimateFittedScan(const StageWorkload::PrefixScanKey &shape,
                                 const StageModeProfile &profile, bool simd,
                                 bool allowTableFormulas) {
  if (std::get<0>(shape) <= 1 || std::get<1>(shape) <= 0 ||
      !std::get<12>(shape))
    return std::nullopt;
  const auto &combine = std::get<11>(shape);
  if (combine != "arith.addf" && combine != "arith.addi") {
    if (allowTableFormulas && std::get<3>(shape) == 1)
      return estimatePrefixScanFormula(
          shape,
          profile.prefixScanWarpCount > 0 ? profile.prefixScanWarpCount
                                          : profile.activeWarpCount,
          simd, profile.calibratedPrefixScan.formulas);
    return std::nullopt;
  }
  const auto &calibration = profile.calibratedPrefixScan;
  if (!calibration.enabled || std::get<7>(shape) <= 0)
    return std::nullopt;
  const int64_t extent = std::get<0>(shape);
  const int64_t independentSequences = std::get<1>(shape);
  const double n = static_cast<double>(extent) * independentSequences;
  double completion = 0.0;
  std::string formulaId;
  if (simd && independentSequences == 1) {
    formulaId = extent <= 64     ? "cumsum_simd_1d_small"
                : extent <= 1024 ? "cumsum_simd_1d_mid"
                                 : "cumsum_simd_1d_long";
    if (extent <= 64)
      completion =
          calibration.prefixScanStartupCycles1dSmall +
          extent * calibration.prefixScanDependencyFactor1dSmall / 64.0;
    else
      completion =
          calibration.prefixScanStartupCycles1d +
          extent * std::log2(static_cast<double>(extent)) *
              (extent <= 1024
                   ? calibration.prefixScanDependencyFactor1dMidWork
                   : calibration.prefixScanDependencyFactor1dTiledWork) /
              64.0;
  } else if (simd) {
    const int64_t elementBits = std::get<9>(shape);
    const double elementBytes = elementBits >= 8 ? elementBits / 8 : 0;
    const double vectorElements =
        elementBytes > 0.0 ? 256.0 / elementBytes : 64.0;
    const double steps =
        extent * std::ceil(independentSequences / vectorElements);
    formulaId = steps <= 256 ? "cumsum_simd_multi" : "cumsum_simd_multi_tail";
    completion = calibration.prefixScanStartupCycles +
                 (steps * calibration.prefixScanDependencyFactor +
                  std::max(0.0, steps - 256.0) *
                      calibration.prefixScanDependencyFactorMultiTail) /
                     64.0;
    if (std::get<8>(shape) != 0 && elementBytes > 0.0) {
      const bool rank2 = std::get<7>(shape) == 2;
      const double rate =
          rank2 ? calibration.prefixScanTransposeAr2raBytesPerCycle
                : calibration.prefixScanTransposeDim01BytesPerCycle;
      if (rate > 0.0) {
        formulaId += rank2 ? "_transpose_ar2ra" : "_transpose_dim01";
        completion +=
            (rank2 ? calibration.prefixScanTransposeAr2raFixedCycles
                   : calibration.prefixScanTransposeDim01FixedCycles) +
            2.0 * n * elementBytes / rate;
      }
    }
  } else if (independentSequences == 1 &&
             !calibration.prefixScan1dWarpSegments.empty()) {
    auto result =
        simtPrefixScan1dSegmentCost(calibration, profile.activeWarpCount, n);
    formulaId = std::move(result.formulaId);
    completion = result.cycles;
  } else if (independentSequences > 1 &&
             !calibration.prefixScan2dWarpSegments.empty()) {
    auto result = simtPrefixScan2dSegmentCost(
        calibration, profile.activeWarpCount, extent, n);
    formulaId = std::move(result.formulaId);
    completion = result.cycles;
  }
  if (!std::isfinite(completion) || completion <= 0.0)
    return std::nullopt;
  return PrefixScanEstimate{std::move(formulaId), completion};
}

bool CalibratedPrefixScanProfile::isValid() const {
  auto valid = [](std::initializer_list<double> values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
      return std::isfinite(value) && value >= 0.0;
    });
  };
  if (!valid({prefixScanDependencyFactor, prefixScanDependencyFactor1d,
              prefixScanDependencyFactor1dSmall,
              prefixScanDependencyFactor1dMidWork,
              prefixScanDependencyFactor1dTiledWork,
              prefixScanDependencyFactorMultiTail,
              prefixScanTransposeDim01FixedCycles,
              prefixScanTransposeDim01BytesPerCycle,
              prefixScanTransposeAr2raFixedCycles,
              prefixScanTransposeAr2raBytesPerCycle, prefixScanStartupCycles,
              prefixScanStartupCycles1d, prefixScanStartupCycles1dSmall}))
    return false;
  for (const auto &[warps, s] : prefixScan1dWarpSegments)
    if (warps == 0 ||
        !valid({s.cLocal, s.cFixed, s.cRound, s.cGen, s.serialRate}))
      return false;
  for (const auto &[warps, s] : prefixScan2dWarpSegments)
    if (warps == 0 ||
        !valid({s.a, s.b, s.c, s.d, s.r0, s.aSubwarp, s.bSubwarp}))
      return false;
  return true;
}
