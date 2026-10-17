// Indirect memory calibration and domain guards.
#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::ascend;

namespace {
constexpr const char *factorModel = "unmasked_factor_candidate_20261017";
constexpr const char *prefixMaskModel = "prefix_mask_rank_factor_20261017";
constexpr const char *randomMaskModel = "random_mask_rank_factor_20261017";

#include "IndirectMaskModel.inc"

// Only element-order-preserving reshapes from the measured flat templates.
static Value flatMaskSource(Value value) {
  while (auto reshape = value.getDefiningOp<triton::ReshapeOp>()) {
    if (reshape.getAllowReorder())
      return {};
    value = reshape.getSrc();
  }
  return value;
}

static bool splatZero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                        : DenseElementsAttr();
  if (!dense || !dense.isSplat())
    return false;
  return isa<FloatType>(dense.getElementType())
             ? dense.getSplatValue<APFloat>().isZero()
             : dense.getSplatValue<APInt>().isZero();
}

static std::optional<double>
maskedIncrement(const LogicalStage &stage, const HardwareProfile &hardware,
                const StageImplementation &implementation, bool store) {
  const auto &selector =
      store ? hardware.simtIndirectStoreModel : hardware.simtIndirectLoadModel;
  const bool randomPrior = selector == randomMaskModel;
  if (selector != prefixMaskModel && !randomPrior)
    return std::nullopt;
  const auto &work = stage.workload;
  const auto &pattern = work.addressPatterns.front();
  const int64_t w = hardware.baseSimtWarpCount;
  const int64_t f = implementation.superblockFactor;
  if (hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMT || implementation.localScope ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      (f != 1 && f != 2 && f != 4) || (w != 1 && w != 4) ||
      !pattern.dependsOnLoadedValue || pattern.stageId != stage.id ||
      pattern.memoryOp != (store ? "tt.store" : "tt.load") ||
      stage.features.hasAtomicMemory || work.privateStackLoadInstructions ||
      work.privateStackStoreInstructions ||
      (store && work.hasProvenIndirectStoreReuse) ||
      (store ? work.indirectLoadBytes != 0 : work.indirectStoreBytes != 0))
    return std::nullopt;
  llvm::SmallPtrSet<Operation *, 4> targets;
  for (Operation *owned : stage.operations)
    owned->walk([&](Operation *op) {
      if (op->getName().getStringRef() == (store ? "tt.store" : "tt.load"))
        targets.insert(op);
    });
  if (targets.size() != 1)
    return std::nullopt;
  Operation *op = *targets.begin();
  // Load requires explicit zero other; store requires one mask.
  if (op->getNumOperands() != 3 || (!store && op->getNumResults() != 1))
    return std::nullopt;
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    if (load.getCache() != triton::CacheModifier::NONE ||
        load.getEvict() != triton::EvictionPolicy::NORMAL ||
        load.getIsVolatile())
      return std::nullopt;
  } else if (auto write = dyn_cast<triton::StoreOp>(op)) {
    if (write.getCache() != triton::CacheModifier::NONE ||
        write.getEvict() != triton::EvictionPolicy::NORMAL)
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  auto type = dyn_cast<RankedTensorType>(store ? op->getOperand(1).getType()
                                               : op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 5 || pattern.axes.size() != size_t(type.getRank()))
    return std::nullopt;
  const auto element = type.getElementType();
  const int group = element.isInteger(8)    ? 0
                    : element.isInteger(16) ? 1
                    : element.isF32()       ? 2
                    : element.isInteger(64) ? 3
                                            : -1;
  const int64_t e = type.getNumElements(), rank = type.getRank();
  const int64_t q = e / (32 * w), fi = f == 4 ? 2 : f == 2 ? 1 : 0;
  if (group < 0 || e % (32 * w) || (q != 1 && q != 2 && q != 4) ||
      (store ? work.indirectStoreBytes : work.indirectLoadBytes) !=
          e * element.getIntOrFloatBitWidth() / 8)
    return std::nullopt;
  // Exact union of rank_matrix and refine_matrix, not the Cartesian hull.
  const int64_t rotation = rank + group + fi;
  if (w != (rotation % 2 ? 4 : 1) && q != (1LL << (rotation % 3)))
    return std::nullopt;
  for (int64_t axis = 0; axis < rank; ++axis) {
    const int64_t expected =
        axis == (w == 1 ? rank - 1 : 0) ? e / (1LL << (rank - 1)) : 2;
    if (expected < 2 || type.getDimSize(axis) != expected ||
        pattern.axes[axis].extent != expected)
      return std::nullopt;
  }
  auto address = op->getOperand(0).getDefiningOp<triton::AddPtrOp>();
  if (!address || !address.getPtr().getDefiningOp<triton::SplatOp>())
    return std::nullopt;
  Value offset = flatMaskSource(address.getOffset());
  auto index =
      offset ? offset.getDefiningOp<triton::LoadOp>() : triton::LoadOp();
  auto indexType = offset ? dyn_cast<RankedTensorType>(offset.getType())
                          : RankedTensorType();
  if (!index || index.getMask() || !indexType || indexType.getRank() != 1 ||
      indexType.getNumElements() != e ||
      !indexType.getElementType().isInteger(32))
    return std::nullopt;
  Value mask = op->getOperand(store ? 2 : 1);
  if (mask.getType() != type.clone(IntegerType::get(op->getContext(), 1)))
    return std::nullopt;
  mask = flatMaskSource(mask);
  auto cmp = mask ? mask.getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
  if (!cmp)
    return std::nullopt;
  bool random = false;
  const int64_t k = w == 1 ? e / 2 : e * 3 / 4;
  if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    auto range = cmp.getLhs().getDefiningOp<triton::MakeRangeOp>();
    auto limit = cmp.getRhs().getDefiningOp<arith::ConstantOp>();
    auto values = limit ? dyn_cast<DenseIntElementsAttr>(limit.getValue())
                        : DenseIntElementsAttr();
    if (!range || range.getStart() != 0 || range.getEnd() != e || !values ||
        !values.isSplat() || !values.getElementType().isInteger(32) ||
        values.getSplatValue<APInt>().getSExtValue() != k)
      return std::nullopt;
  } else {
    // An explicit profile prior, not a proof of runtime density. This is only
    // the original shuffled byte-mask family: W1 half, W4 three quarters.
    if (!randomPrior || cmp.getPredicate() != arith::CmpIPredicate::ne ||
        !splatZero(cmp.getRhs()))
      return std::nullopt;
    Value source = cmp.getLhs();
    if (auto extend = source.getDefiningOp<arith::ExtUIOp>())
      source = extend.getIn();
    auto loaded = source.getDefiningOp<triton::LoadOp>();
    auto maskType = dyn_cast<RankedTensorType>(source.getType());
    if (!loaded || loaded.getMask() || !maskType || maskType.getRank() != 1 ||
        maskType.getNumElements() != e ||
        !maskType.getElementType().isInteger(8))
      return std::nullopt;
    random = true;
  }
  if (!store && !splatZero(op->getOperand(2)))
    return std::nullopt;
  // Existing applySuperBlock accounts for F once. Auxiliary work stays priced.
  return maskGroupCycles(store, random, group, e, w, f, rank,
                         type.getShape().back(), k) /
         f;
}

// Experimental one-AIV F-program increments. Known W16/small-E outliers
// are retained intentionally. Rank, type and geometry coverage are not relaxed.
// Load includes address/layout changes versus an index-passthrough baseline;
// store uses fresh unfilled targets and a shared address/value-sink baseline.
static std::optional<double>
factorIncrement(const LogicalStage &stage, const HardwareProfile &hardware,
                const StageImplementation &implementation, bool store) {
  const auto &work = stage.workload;
  const auto &p = work.addressPatterns.front();
  const int64_t w = hardware.baseSimtWarpCount;
  const int64_t f = implementation.superblockFactor;
  if (hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMT || implementation.localScope ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      (f != 1 && f != 2 && f != 4) || (w != 1 && w != 4 && w != 16) ||
      p.axes.size() != 1 || !p.dependsOnLoadedValue || p.stageId != stage.id ||
      p.memoryOp != (store ? "tt.store" : "tt.load") ||
      !work.predicateCounts.empty() || stage.features.activeLaneRatio != 1 ||
      stage.features.hasAtomicMemory || work.privateStackLoadInstructions ||
      work.privateStackStoreInstructions ||
      (store ? work.indirectLoadBytes != 0 : work.indirectStoreBytes != 0))
    return std::nullopt;
  llvm::SmallPtrSet<Operation *, 4> targets;
  for (Operation *owned : stage.operations)
    owned->walk([&](Operation *op) {
      if (op->getName().getStringRef() == (store ? "tt.store" : "tt.load"))
        targets.insert(op);
    });
  if (targets.size() != 1)
    return std::nullopt;
  Operation *op = *targets.begin();
  if (op->getNumOperands() != (store ? 2u : 1u) ||
      (!store && op->getNumResults() != 1))
    return std::nullopt;
  llvm::SmallVector<Value> pending{op->getOperand(0)};
  llvm::SmallPtrSet<Operation *, 16> visited;
  bool loadedIndex = false;
  while (!pending.empty()) {
    Operation *producer = pending.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (producer->getName().getStringRef() == "tt.load") {
      if (producer->getNumResults() != 1)
        return std::nullopt;
      auto indexType =
          dyn_cast<RankedTensorType>(producer->getResult(0).getType());
      if (!indexType || !indexType.getElementType().isInteger(32))
        return std::nullopt;
      loadedIndex = true;
      continue;
    }
    pending.append(producer->operand_begin(), producer->operand_end());
  }
  if (!loadedIndex)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(store ? op->getOperand(1).getType()
                                               : op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 1)
    return std::nullopt;
  const auto element = type.getElementType();
  const int group = element.isInteger(8)    ? 0
                    : element.isInteger(16) ? 1
                    : element.isF32()       ? 2
                    : element.isInteger(64) ? 3
                                            : -1;
  const int64_t e = type.getNumElements();
  if (group < 0 || p.axes.front().extent != e ||
      (store ? work.indirectStoreBytes : work.indirectLoadBytes) !=
          e * element.getIntOrFloatBitWidth() / 8 ||
      (e != 8 && e != 16 && e != 32 && e != 64 && e != 128 && e != 256 &&
       e != 512) ||
      (e == 16 && w != 16) || (e == 32 && w == 4) || (e == 128 && w == 1) ||
      ((e == 64 || e == 256 || e == 512) && w == 16))
    return std::nullopt;
  static constexpr double load[4][11] = {
      {114.06223568631589, 2.19786636783679, -1.3764893947282235,
       -.2650569381840509, 4.5195261359854095, -2.947018852455173,
       1.9148720707208333, -79.53265002228892, 169.77441385415742},
      {128.77054635393904, 1.3649882385214698, -.18912283986571862,
       -.49858876697185, 4.675581837359436, .8772283214588263,
       -.7275675429844192, -115.81232279422217, 95.8950323552327},
      {120.07090406650597, 1.5500745996624559, -.43870878244183104,
       -.18808420588616206, 4.537873383737035, 1.640631714734442,
       .15541731034924547, -106.38488394854417, 91.2560441788196},
      {116.94785273549063, 1.9765804199364312, -.28377577279558625,
       -.24761005509333786, 4.415935669816099, .8325433060321422,
       .08183562516063783, -119.76469633579902, 110.4385779080016}};
  static constexpr double stores[4][11] = {
      {116.62634372635547, 6.111117617348127, -.8143304842483025,
       -.43846744058688536, -1.5395704974200097, -4.53793989093101,
       -3.2228424603743955},
      {124.19945311030615, 5.904956752148043, -.6508286986542869,
       -.06135585492914368, -1.669597000307736, -10.024709617521689,
       -3.1503815285785413, -86.90874840747217, 55.935253213898434,
       6.888336827997633, .001302804707999514},
      {119.45683288107666, 2.5359135035732305, -.5797639385984518,
       -1.1189187029747005, .2571346944963013, -6.812191310593307,
       -2.300401406184211},
      {135.61356141589593, 1.8676621751827858, .6123327974117674,
       -1.0686789769499396, .2490036303670708, -1.9460217818211718,
       -.9072101216953034}};
  const double h = std::max<double>(e - 32 * w, 0), l = std::log2(w);
  const double j = w * f >= 64 ? 1 : 0;
  const double features[] = {1.,
                             double(e),
                             std::max<double>(e - 32, 0),
                             h,
                             h / w,
                             l,
                             double(f - 1),
                             j,
                             j * std::min(e / 32., 1.),
                             l * (f - 1),
                             h * (f - 1) / w};
  const double *coefficients = store ? stores[group] : load[group];
  double value = 0;
  for (unsigned i = 0; i < 11; ++i)
    value += coefficients[i] * features[i];
  // mapWorkload resources are per logical program. applySuperBlock multiplies
  // these exactly once by F; the fitted label already covers the complete
  // group.
  return std::max(0.0, value) / f;
}

// Effective incremental latency of one indirect load (payload - matched ALU),
// not a whole-Stage cost. The random-address prior is intentional: no runtime
// indices or final ISA facts are inputs. Unsupported domains use legacy rates.
static std::optional<double>
calibratedSimdIndirectLoad(const LogicalStage &stage,
                           const HardwareProfile &hardware,
                           const StageImplementation &implementation) {
  const auto &work = stage.workload;
  const bool dtypeRankModel =
      hardware.simdIndirectLoadModel == "random_dtype_matched_ab_20261008";
  if ((!dtypeRankModel &&
       hardware.simdIndirectLoadModel != "random_f32_matched_ab_20261007") ||
      hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMD ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      (!dtypeRankModel && stage.operations.size() != 1) ||
      work.addressPatterns.size() != 1 ||
      (work.privateStackLoadInstructions +
       work.privateStackStoreInstructions) != 0 ||
      !work.predicateCounts.empty() || stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousLoadBytes != 0 || work.indirectStoreBytes != 0)
    return std::nullopt;
  // Real Stage partitioning owns the payload's splat/addptr/reshape producers
  // too. Require one load, not one total operation; those independent helpers
  // retain their normal resource charges. Old profiles keep their old guard.
  Operation *op = nullptr;
  for (Operation *candidate : stage.operations) {
    if (candidate->getName().getStringRef() != "tt.load")
      continue;
    if (op)
      return std::nullopt;
    op = candidate;
  }
  if (!op || op->getNumResults() != 1 || op->getNumOperands() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (dtypeRankModel) {
    if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
        type.getRank() > 5)
      return std::nullopt;
    Type element = type.getElementType();
    const bool supportedInteger =
        element.isInteger(8) || element.isInteger(16) ||
        element.isInteger(32) || element.isInteger(64);
    if (!supportedInteger && !isa<Float16Type, BFloat16Type, Float32Type,
                                  Float8E4M3FNType, Float8E5M2Type>(element))
      return std::nullopt;
    // Triton bool loads are normalized to i8 before this analysis. A raw i1
    // load is not covered: its packed TTIR byte count differs from that ABI.
    const int64_t elements = type.getNumElements();
    const int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
    const auto &pattern = work.addressPatterns.front();
    if (elements < 4 || elements > 2048 || (elements & (elements - 1)) ||
        work.indirectLoadBytes != elementBytes * elements ||
        work.indirectLoadTransactions <= 0 || pattern.memoryOp != "tt.load" ||
        pattern.stageId != stage.id || !pattern.dependsOnLoadedValue ||
        pattern.axes.size() != static_cast<size_t>(type.getRank()))
      return std::nullopt;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      const auto &summary = pattern.axes[axis];
      const int64_t extent = type.getDimSize(axis);
      if (extent < 2 || (extent & (extent - 1)) || summary.extent != extent)
        return std::nullopt;
      // Reshape can erase per-axis loaded provenance without erasing the
      // whole pointer's loaded-index dependency. The measured inner-index
      // templates produce "opaque" here; do not reject them as direct loads.
      const bool opaque = summary.regularity == "opaque_loaded" ||
                          summary.regularity == "opaque";
      const bool fixedOuter = axis + 1 < type.getRank() &&
                              summary.regularity == "fixed_stride" &&
                              summary.knownStride > 0;
      if (!opaque && !fixedOuter)
        return std::nullopt;
    }
    if (type.getRank() == 2 &&
        (elements < 32 || type.getDimSize(1) < 4 || type.getDimSize(1) > 128))
      return std::nullopt;
    // Effective matched-address load increment in SYS_CNT/system_cycles.
    // The tiny 32-bit path has measured unrolling/grouping differences; it is
    // not a universal hardware latency threshold. No SIMD warp division.
    const bool small32 =
        elementBytes == 4 && type.getRank() == 1 && elements <= 16;
    return elements * (small32 ? 49.57621548794853 : 81.94869549595556);
  }
  // Keep the previously versioned FP32 model reproducible for old profiles.
  if (!type || !type.hasStaticShape() ||
      (type.getRank() != 1 && type.getRank() != 2) ||
      !type.getElementType().isF32())
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  const int64_t elements = type.getNumElements();
  if (pattern.memoryOp != "tt.load" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()) ||
      pattern.axes.back().regularity != "opaque_loaded" || elements < 8 ||
      elements > 512 || (elements & (elements - 1)) ||
      work.indirectLoadBytes != 4 * elements ||
      work.indirectLoadTransactions <= 0)
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (pattern.axes[axis].extent != type.getDimSize(axis))
      return std::nullopt;
  if (type.getRank() == 2) {
    const auto &outer = pattern.axes.front();
    const int64_t columns = type.getDimSize(1);
    // Rank-two validation currently supports the narrow-column domain only.
    // Wider columns have measured counterexamples; keep their legacy fallback.
    if (elements < 32 || (columns != 4 && columns != 8) ||
        (outer.regularity != "opaque_loaded" &&
         !(outer.regularity == "fixed_stride" && outer.knownStride == 4096)))
      return std::nullopt;
  }
  // Effective load increment, NOT the old 61.72 + 96.52*N whole-loop fit.
  // The address-preserving baseline has a small extra ALU bias; random-only
  // input validation and baseline-sensitivity results accompany calibration.
  return 83.56748010753823 * elements;
}

static std::optional<double>
calibratedSimtDtypeIndirectLoad(const LogicalStage &stage,
                                const HardwareProfile &hardware,
                                const StageImplementation &implementation) {
  const auto &work = stage.workload;
  if (hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      work.addressPatterns.size() != 1 ||
      (work.privateStackLoadInstructions +
       work.privateStackStoreInstructions) != 0 ||
      !work.predicateCounts.empty() || stage.features.hasAtomicMemory ||
      stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousLoadBytes != 0 || work.indirectStoreBytes != 0)
    return std::nullopt;
  // Partitioned Stages also own address/shape producers. Their resources stay
  // independent; the fit replaces only the single target indirect load.
  // Explicit SIMT regions may be represented by their owning scope op. Walk
  // owned regions too; counting only top-level ops misses real Triton loads.
  llvm::SmallPtrSet<Operation *, 4> loads;
  for (Operation *owned : stage.operations)
    owned->walk([&](Operation *candidate) {
      if (candidate->getName().getStringRef() == "tt.load")
        loads.insert(candidate);
    });
  if (loads.size() != 1)
    return std::nullopt;
  Operation *op = *loads.begin();
  if (op->getNumResults() != 1 || op->getNumOperands() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 5)
    return std::nullopt;
  Type element = type.getElementType();
  const bool supportedInteger = element.isInteger(8) || element.isInteger(16) ||
                                element.isInteger(32) || element.isInteger(64);
  if (!supportedInteger &&
      !isa<Float16Type, BFloat16Type, Float32Type, Float64Type,
           Float8E4M3FNType, Float8E5M2Type>(element))
    return std::nullopt;
  // Raw i1 is not a one-byte TTIR load; normalized bool loads use i8.
  const int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
  const auto &pattern = work.addressPatterns.front();
  if (pattern.memoryOp != "tt.load" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()))
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    const auto &summary = pattern.axes[axis];
    const int64_t extent = type.getDimSize(axis);
    const bool opaque =
        summary.regularity == "opaque_loaded" || summary.regularity == "opaque";
    const bool fixedOuter = axis + 1 < type.getRank() &&
                            summary.regularity == "fixed_stride" &&
                            summary.knownStride > 0;
    if (extent < 2 || (extent & (extent - 1)) || summary.extent != extent ||
        (!opaque && !fixedOuter))
      return std::nullopt;
  }
  const int64_t warps = hardware.baseSimtWarpCount;
  if (warps < 1 || warps > 64 || (warps & (warps - 1)))
    return std::nullopt;
  const double elements = type.getNumElements();
  const double q = elements / (32.0 * warps);
  const int64_t columns = type.getRank() == 1 ? 32 : type.getShape().back();
  if ((q != 1 && q != 2 && q != 4) || columns < 2 || columns > 128 ||
      work.indirectLoadBytes != elementBytes * elements ||
      work.indirectLoadTransactions <= 0)
    return std::nullopt;
  const double h = std::max(4 * q / columns - 2, 0.0);
  // C=2,q=4 has H=6, beyond the calibrated H<=2 range and with measured
  // counterexamples. Do not silently extrapolate this baseline correction.
  if (h > 2)
    return std::nullopt;
  const double d = std::max(std::log2(32.0 / columns), 0.0);
  struct Coefficients {
    double fixed, elements, small, flat, rowBaseline, narrowBaseline;
  };
  // Frozen pure-random, four-storage-width calibration. INT32/UINT32/FP32
  // share the 4-byte row; dtype names do not select historical coefficients.
  static constexpr Coefficients coefficients[] = {
      {7.617911783542814, 1.5780971099377332, 92.18952965944916,
       0.19387737354356369, 0.0, 0.049471659398567715},
      {12.456944150614481, 1.6732132663368477, 95.13859585355112,
       0.11214618741568584, 5.634318857353958, 0.026588037490546692},
      {50.71936539506111, 1.693396365504775, 60.38681262899275,
       0.09265123974451413, 82.91193957489877, 0.026814982781878355},
      {96.09572807125711, 1.937402636239715, 24.962585867214496,
       0.20099293498091206, 41.8669299352394, 0.03855485734655757}};
  const auto &c = coefficients[elementBytes == 1   ? 0
                               : elementBytes == 2 ? 1
                               : elementBytes == 4 ? 2
                                                   : 3];
  // Effective A/B increment in SYS_CNT/system_cycles, already including W
  // warp parallelism. H and E*D compensate baseline mismatch, not GM latency.
  return c.fixed + c.elements * elements + c.small * (elements <= 128) +
         c.flat * elements * (type.getRank() == 1) + c.rowBaseline * h +
         c.narrowBaseline * elements * d;
}

static std::optional<double>
calibratedIndirectLoad(const LogicalStage &stage,
                       const HardwareProfile &hardware,
                       const StageImplementation &implementation) {
  const auto &work = stage.workload;
  if (implementation.mode == StageMode::SIMD)
    return calibratedSimdIndirectLoad(stage, hardware, implementation);
  if (hardware.simtIndirectLoadModel == "random_dtype_six_term_20261008")
    return calibratedSimtDtypeIndirectLoad(stage, hardware, implementation);
  // Explicitly versioned compatibility model, not an INT32 exception inside
  // the four-width model. Old profiles retain their coefficients and guards.
  if (hardware.simtIndirectLoadModel != "random_i32_six_term_20261007" ||
      hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      stage.operations.size() != 1 || work.addressPatterns.size() != 1 ||
      (work.privateStackLoadInstructions +
       work.privateStackStoreInstructions) != 0 ||
      !work.predicateCounts.empty() || stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousLoadBytes != 0 || work.indirectStoreBytes != 0)
    return std::nullopt;
  Operation *op = stage.operations.front();
  if (op->getName().getStringRef() != "tt.load" || op->getNumResults() != 1 ||
      op->getNumOperands() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isInteger(32) ||
      (type.getRank() != 1 && type.getRank() != 2))
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  if (pattern.memoryOp != "tt.load" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()))
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (pattern.axes[axis].extent != type.getDimSize(axis))
      return std::nullopt;
  if (pattern.axes.back().regularity != "opaque_loaded")
    return std::nullopt;
  if (type.getRank() == 2 &&
      pattern.axes.front().regularity != "opaque_loaded" &&
      !(pattern.axes.front().regularity == "fixed_stride" &&
        pattern.axes.front().knownStride == 4096))
    return std::nullopt;
  const int64_t warps = hardware.baseSimtWarpCount;
  if (warps < 1 || warps > 64 || (warps & (warps - 1)))
    return std::nullopt;
  const double elements = type.getNumElements();
  const double q = elements / (32.0 * warps);
  const int64_t columns = type.getRank() == 1 ? 32 : type.getDimSize(1);
  if ((q != 1 && q != 2 && q != 4) ||
      (type.getRank() == 2 && type.getDimSize(0) < 2) || columns < 4 ||
      columns > 128 || (columns & (columns - 1)) ||
      work.indirectLoadBytes != 4 * elements ||
      work.indirectLoadTransactions <= 0)
    return std::nullopt;
  const double h = std::max(4 * q / columns - 2, 0.0);
  const double d = std::max(std::log2(32.0 / columns), 0.0);
  // SYS_CNT ticks are already CostModel system_cycles (no clock conversion).
  return 62.930686069640686 + 1.5502588407166296 * elements +
         141.51064147799983 * h + 47.159408679724294 * (elements <= 128) +
         0.05304437217224805 * elements * d +
         0.19338028618547126 * elements * (type.getRank() == 1);
}

// Random, collision-free addresses; no target fill. Coefficients price the
// measured A/B increment in SYS_CNT ticks, not a worst-case latency bound.
static std::optional<double>
randomDtypeIndirectStore(const LogicalStage &stage,
                         const HardwareProfile &hardware, bool simd,
                         bool reuse) {
  const auto &work = stage.workload;
  if (reuse && !work.hasProvenIndirectStoreReuse)
    return std::nullopt;
  // Real partitioning also owns the store's address producers. Require one
  // target store, not one total operation; helpers keep their normal charges.
  Operation *op = nullptr;
  for (Operation *candidate : stage.operations) {
    if (candidate->getName().getStringRef() != "tt.store")
      continue;
    if (op)
      return std::nullopt;
    op = candidate;
  }
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 0)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 8)
    return std::nullopt;
  Type element = type.getElementType();
  if (!isa<IntegerType>(element) && !element.isF16() && !element.isBF16() &&
      !element.isF32() && !(element.isF64() && !simd) &&
      !isa<Float8E4M3FNType, Float8E5M2Type>(element))
    return std::nullopt;
  unsigned bits = element.getIntOrFloatBitWidth();
  if (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  const int64_t elements = type.getNumElements();
  const int64_t warps = hardware.baseSimtWarpCount;
  if (elements < 8 || elements > (simd ? 4096 : 16384) ||
      (elements & (elements - 1)) || warps < 1 || warps > 64 ||
      (warps & (warps - 1)) || (simd && warps != 1) ||
      (!simd &&
       (elements > 512 * warps || (warps > 1 && elements < 32 * warps))) ||
      pattern.memoryOp != "tt.store" || pattern.stageId != stage.id ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()) ||
      work.indirectStoreBytes != std::max(1u, bits / 8) * elements ||
      work.indirectStoreTransactions <= 0)
    return std::nullopt;
  // Every axis must itself prove loaded provenance. This is stronger than
  // the legacy whole-pointer dependency predicate, whose rank cap excludes
  // ranks 6-8; do not use that predicate to reject proven high-rank axes.
  // The calibrated prior excludes broadcast/structured-prefix addresses.
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    int64_t extent = type.getDimSize(axis);
    if (extent < 1 || (extent & (extent - 1)) ||
        pattern.axes[axis].extent != extent ||
        pattern.axes[axis].regularity != "opaque_loaded")
      return std::nullopt;
  }
  const unsigned group = bits <= 16 ? 0 : (bits == 32 ? 1 : 2);
  if (simd) {
    const double first[] = {152.8758408085307, 162.87111975882544,
                            161.07674811788297};
    const double warm[] = {68.58965840703893, 69.21236829277788,
                           64.45492494749357};
    return (reuse ? 0.0 : 125.41805018354371) +
           (reuse ? warm[group] : first[group]) * elements;
  }
  const double firstSlope[] = {4.664669494263474, 1.3935292896269613,
                               2.0019961511928295};
  const double warmSlope[] = {4.4638287665056, 1.1754145133075695,
                              2.076648173015399};
  const double firstWarp[] = {-25.526112727120946, -19.191779247532597,
                              -35.604138981973364};
  const double warmWarp[] = {-9.072748388478542, -20.88472709201174,
                             -38.4225252759396};
  // q is a fitted scheduling correction, not a negative hardware latency.
  return (reuse ? 70.0956884939018 : 382.1012717872757) +
         (reuse ? warmSlope[group] : firstSlope[group]) * elements +
         (reuse ? 0.8774560851959086 : 0.6797627278581438) *
             std::max<int64_t>(elements - (reuse ? 128 : 256), 0) +
         (reuse ? warmWarp[group] : firstWarp[group]) *
             (elements / (32.0 * warps));
}

// Effective write plus required synchronization increment, not a whole Stage.
// Address randomness and memory history are profile priors: neither is proven
// by pointer analysis. In particular, the no-fill first fit has >20% outliers.
static std::optional<double>
calibratedIndirectStore(const LogicalStage &stage,
                        const HardwareProfile &hardware,
                        const StageImplementation &implementation) {
  const auto &work = stage.workload;
  const bool simd = implementation.mode == StageMode::SIMD;
  if (hardware.target != "Ascend950PR/dav-c310" ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      work.addressPatterns.size() != 1 ||
      (work.privateStackLoadInstructions +
       work.privateStackStoreInstructions) != 0 ||
      !work.predicateCounts.empty() || stage.features.hasAtomicMemory ||
      stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousStoreBytes != 0 || work.indirectLoadBytes != 0)
    return std::nullopt;
  const auto &model =
      simd ? hardware.simdIndirectStoreModel : hardware.simtIndirectStoreModel;
  if ((simd || implementation.mode == StageMode::SIMT) &&
      (model == "random_store_no_fill_first_20261008" ||
       model == "random_store_no_fill_reuse_20261008"))
    return randomDtypeIndirectStore(
        stage, hardware, simd, model == "random_store_no_fill_reuse_20261008");
  // Preserve the historical FP32 model's original admission boundary.
  if (stage.operations.size() != 1)
    return std::nullopt;
  if (simd) {
    if ((hardware.simdIndirectStoreModel !=
             "random_f32_store_no_fill_first_20261007" &&
         hardware.simdIndirectStoreModel !=
             "random_f32_store_no_fill_reuse_20261007") ||
        hardware.baseSimtWarpCount != 1)
      return std::nullopt;
  } else if (implementation.mode != StageMode::SIMT ||
             hardware.simtIndirectStoreModel !=
                 "random_f32_store_fill_ab_20261007") {
    return std::nullopt;
  }
  Operation *op = stage.operations.front();
  if (op->getName().getStringRef() != "tt.store" || op->getNumOperands() != 2 ||
      op->getNumResults() != 0)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      (type.getRank() != 1 && type.getRank() != 2))
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  const int64_t elements = type.getNumElements();
  if (pattern.memoryOp != "tt.store" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()) ||
      pattern.axes.back().regularity != "opaque_loaded" ||
      work.indirectStoreBytes != 4 * elements ||
      work.indirectStoreTransactions <= 0)
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (pattern.axes[axis].extent != type.getDimSize(axis))
      return std::nullopt;
  const bool rankTwo = type.getRank() == 2;
  const int64_t columns = rankTwo ? type.getDimSize(1) : 32;
  const bool inner =
      rankTwo && pattern.axes.front().regularity == "fixed_stride";
  if (rankTwo &&
      (type.getDimSize(0) < 2 ||
       (pattern.axes.front().regularity != "opaque_loaded" && !inner) ||
       (inner && pattern.axes.front().knownStride != (simd ? 32768 : 4096))))
    return std::nullopt;
  if (simd) {
    // Exactly the measured shape domain. Row/column loaded and per-element
    // loaded templates share this slope; unsupported expressions fall back.
    if ((!rankTwo &&
         (elements < 8 || elements > 512 || (elements & (elements - 1)))) ||
        (rankTwo && ((elements != 32 && elements != 128 && elements != 512) ||
                     (columns != 4 && columns != 16 && columns != 64))))
      return std::nullopt;
    if (hardware.simdIndirectStoreModel ==
            "random_f32_store_no_fill_reuse_20261007" &&
        !work.hasProvenIndirectStoreReuse)
      return std::nullopt;
    return elements * (hardware.simdIndirectStoreModel ==
                               "random_f32_store_no_fill_reuse_20261007"
                           ? 65.26772542192847
                           : 150.13769870695648);
  }
  const int64_t warps = hardware.baseSimtWarpCount;
  if (warps < 1 || warps > 64 || (warps & (warps - 1)))
    return std::nullopt;
  const double q = elements / (32.0 * warps);
  if ((q != 1 && q != 2 && q != 4) ||
      (rankTwo && columns != 8 && columns != 32))
    return std::nullopt;
  if (elements >= 512)
    return 1.9039194506414692 * elements;
  const double logW = std::log2(static_cast<double>(warps));
  return 157.465336398077 + 2.046486341690834 * elements -
         13.874579575172818 * std::sqrt(static_cast<double>(elements)) +
         (-0.8540920211063882 - 0.4681840422126738 * q) * logW +
         0.5780871675460912 * elements / columns - 15.490435818576909 * inner;
}

} // namespace

StageMemoryCost IndirectGatherMemoryCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &implementation) const {
  // The calibration used one AIV and the original unsliced TTIR tensor.
  // CV partitioned implementations retain the generic resource model.
  if (implementation.aivSubblocks != 1 ||
      stage.workload.addressPatterns.size() != 1)
    return {};
  const auto &axes = stage.workload.addressPatterns.front().axes;
  if (axes.empty() || !llvm::all_of(axes, [](const AddressAxisSummary &axis) {
        return axis.offsetAxisInfo == "unstructured";
      }))
    return {};
  const auto candidate = [](const std::string &name) {
    return name == factorModel || name == prefixMaskModel ||
           name == randomMaskModel;
  };
  if (implementation.mode == StageMode::SIMT &&
      (candidate(profile.simtIndirectLoadModel) ||
       candidate(profile.simtIndirectStoreModel))) {
    int64_t elements = 1;
    for (const auto &axis : axes) {
      if (axis.extent <= 0 || elements > INT64_MAX / axis.extent)
        return {};
      elements *= axis.extent;
    }
    if (elements / 32.0 < profile.baseSimtWarpCount)
      return {}; // q<1 uses #2528 resources; q=1 remains eligible.
  }
  if (implementation.mode == StageMode::SIMT &&
      (profile.simtIndirectLoadModel == prefixMaskModel ||
       profile.simtIndirectLoadModel == randomMaskModel ||
       profile.simtIndirectStoreModel == prefixMaskModel ||
       profile.simtIndirectStoreModel == randomMaskModel)) {
    HardwareProfile unmasked = profile;
    const bool load = profile.simtIndirectLoadModel == prefixMaskModel ||
                      profile.simtIndirectLoadModel == randomMaskModel;
    const bool store = profile.simtIndirectStoreModel == prefixMaskModel ||
                       profile.simtIndirectStoreModel == randomMaskModel;
    if (load)
      unmasked.simtIndirectLoadModel = factorModel;
    if (store)
      unmasked.simtIndirectStoreModel = factorModel;
    auto result = cost(stage, unmasked, implementation);
    if (load)
      if (auto fitted =
              maskedIncrement(stage, profile, implementation, false)) {
        result.load = fitted;
        result.loadPricing = profile.simtIndirectLoadModel == randomMaskModel
                                 ? randomMaskModel
                                 : prefixMaskModel;
      }
    if (store)
      if (auto fitted = maskedIncrement(stage, profile, implementation, true)) {
        result.store = fitted;
        result.storePricing = profile.simtIndirectStoreModel == randomMaskModel
                                  ? randomMaskModel
                                  : prefixMaskModel;
      }
    return result;
  }
  if (implementation.mode == StageMode::SIMT &&
      (profile.simtIndirectLoadModel == factorModel ||
       profile.simtIndirectStoreModel == factorModel)) {
    HardwareProfile legacy = profile;
    const bool load = profile.simtIndirectLoadModel == factorModel;
    const bool store = profile.simtIndirectStoreModel == factorModel;
    if (load)
      legacy.simtIndirectLoadModel = "random_dtype_six_term_20261008";
    if (store)
      legacy.simtIndirectStoreModel = "random_store_no_fill_first_20261008";
    auto result = cost(stage, legacy, implementation);
    if (load) {
      result.loadPricing = "random_dtype_six_term_20261008";
      if (auto fitted =
              factorIncrement(stage, profile, implementation, false)) {
        result.load = fitted;
        result.loadPricing = factorModel;
      }
    }
    if (store) {
      result.storePricing = "random_store_no_fill_first_20261008";
      if (auto fitted = factorIncrement(stage, profile, implementation, true)) {
        result.store = fitted;
        result.storePricing = factorModel;
      }
    }
    if (!result.load)
      result.loadPricing = nullptr;
    if (!result.store)
      result.storePricing = nullptr;
    return result;
  }
  return {calibratedIndirectLoad(stage, profile, implementation),
          calibratedIndirectStore(stage, profile, implementation)};
}
