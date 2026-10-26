// Indirect memory calibration and domain guards.
#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

using namespace mlir;
using namespace mlir::ascend;

llvm::Expected<std::shared_ptr<const IndirectMemoryParameters>>
IndirectMemoryParameters::load(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "cannot read indirect parameters '%s'",
                                   path.str().c_str());
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return parsed.takeError();
  auto *root = parsed->getAsObject();
  auto *values = root ? root->getObject("parameters") : nullptr;
  if (!values || root->getInteger("schema_version") != 1 ||
      root->getString("unit") != "system_cycle")
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invalid indirect parameter schema: %s",
                                   path.str().c_str());
  auto result = std::make_shared<IndirectMemoryParameters>();
  const std::pair<llvm::StringLiteral, size_t> expected[] = {
      {"mask_basis", 16},
      {"mask_coefficients", 272},
      {"completeLoad8Beta", 95},
      {"completeLoad8Version", 1},
      {"completeLoad16Beta", 41},
      {"completeLoad16Version", 1},
      {"completeLoad16Nodes", 448},
      {"completeLoad16Roots", 16},
      {"completeLoad32Beta", 41},
      {"completeLoad32Version", 1},
      {"completeLoad32Nodes", 4216},
      {"completeLoad32Roots", 64},
      {"completeLoad64Beta", 41},
      {"completeLoad64Version", 1},
      {"completeLoad64Nodes", 4616},
      {"completeLoad64Roots", 64},
      {"completeStore8Beta", 41},
      {"completeStore8Version", 1},
      {"completeStore16Beta", 19},
      {"completeStore16Version", 1},
      {"completeStore32Beta", 26},
      {"completeStore32Version", 1},
      {"completeStore64Beta", 41},
      {"completeStore64Version", 1},
      {"simd_mask_load", 24},
      {"simd_mask_prefix", 24},
      {"simd_mask_positive", 20},
      {"simd_mask_zero", 8},
      {"simd_mask_tail", 12},
      {"simd_mask_b", 10},
      {"native_chain_load", 25},
      {"native_chain_write", 37},
      {"simd_unmasked_load", 2},
      {"simd_unmasked_store", 7},
  };
  auto invalid = [&](llvm::StringRef key) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invalid indirect parameter '%s' in '%s'",
                                   key.str().c_str(), path.str().c_str());
  };
  if (values->size() != std::size(expected))
    return invalid("parameters");
  for (auto [key, count] : expected) {
    auto *array = values->getArray(key);
    if (!array || array->size() != count)
      return invalid(key);
    auto &out = result->arrays[key];
    for (const auto &value : *array) {
      auto number = value.getAsNumber();
      if (!number || !std::isfinite(*number))
        return invalid(key);
      out.push_back(*number);
    }
  }
  for (double basis : result->get("mask_basis"))
    if (basis != 0 && basis != 3)
      return invalid("mask_basis");
  const int versions[] = {6, 3, 3, 3, 3, 1, 2, 3};
  for (int i = 0; i < 8; ++i) {
    std::string name = std::string(i < 4 ? "completeLoad" : "completeStore") +
                       std::to_string(8 << (i % 4));
    if (result->get(name + "Version")[0] != versions[i])
      return invalid(name + "Version");
    if (i == 0 || i >= 4)
      continue;
    auto nodes = result->get(name + "Nodes"),
         roots = result->get(name + "Roots");
    auto integer = [](double n) { return n == std::floor(n); };
    for (double r : roots)
      if (!integer(r) || r < 0 || r >= nodes.size() / 4)
        return invalid(name + "Roots");
    for (size_t n = 0; n < nodes.size() / 4; ++n) {
      double feature = nodes[4 * n];
      if (!integer(feature) || feature < -1 || feature >= 18)
        return invalid(name + "Nodes");
      if (feature == -1)
        continue;
      for (int child : {2, 3}) {
        double next = nodes[4 * n + child];
        // Forward-only indices prove traversal termination as well as bounds.
        if (!integer(next) || next <= n || next >= nodes.size() / 4)
          return invalid(name + "Nodes");
      }
    }
  }
  result->contentSha256 =
      llvm::toHex(llvm::SHA256::hash(
                      llvm::arrayRefFromStringRef(buffer.get()->getBuffer())),
                  true);
  return std::shared_ptr<const IndirectMemoryParameters>(std::move(result));
}

namespace {
constexpr const char *completeModel = "unmasked_complete_factor_20261020";
constexpr const char *prefixMaskModel = "prefix_mask_rank_factor_20261017";
constexpr const char *randomMaskModel = "random_mask_rank_factor_20261017";

#include "IndirectMemoryModels.inc"
constexpr const char *simdMaskModel = "simd_mask_envelope_20261019";

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

static bool defaultCache(Operation *op) {
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return load.getCache() == triton::CacheModifier::NONE &&
           load.getEvict() == triton::EvictionPolicy::NORMAL &&
           !load.getIsVolatile();
  auto store = dyn_cast<triton::StoreOp>(op);
  return store && store.getCache() == triton::CacheModifier::NONE &&
         store.getEvict() == triton::EvictionPolicy::NORMAL;
}

enum class MemoryWalk { Recursive, TopLevel, Regionless };
static Operation *memoryTarget(const LogicalStage &stage, bool store,
                               MemoryWalk walk = MemoryWalk::Recursive) {
  llvm::SmallPtrSet<Operation *, 4> targets;
  auto collect = [&](Operation *op) {
    if (op->getName().getStringRef() == (store ? "tt.store" : "tt.load"))
      targets.insert(op);
  };
  for (Operation *op : stage.operations) {
    if (walk != MemoryWalk::Recursive) {
      if (walk == MemoryWalk::Regionless && op->getNumRegions())
        return nullptr;
      collect(op);
    } else
      op->walk(collect);
  }
  return targets.size() == 1 ? *targets.begin() : nullptr;
}

struct MaskInfo {
  arith::CmpIOp compare;
  int64_t active = 0;
  bool random = false, exclusive = true;
};

// Shared predicate recognition; ownership and density accounting stay separate.
static std::optional<MaskInfo> parseMask(Value mask, int64_t elements) {
  MaskInfo info;
  info.exclusive = mask.hasOneUse();
  while (auto reshape = mask.getDefiningOp<triton::ReshapeOp>()) {
    if (reshape.getAllowReorder())
      return std::nullopt;
    mask = reshape.getSrc();
    info.exclusive &= mask.hasOneUse();
  }
  auto cmp = mask.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;
  info.compare = cmp;
  if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    Value lhs = flatMaskSource(cmp.getLhs());
    auto range =
        lhs ? lhs.getDefiningOp<triton::MakeRangeOp>() : triton::MakeRangeOp();
    auto limit = cmp.getRhs().getDefiningOp<arith::ConstantOp>();
    auto dense = limit ? dyn_cast<DenseIntElementsAttr>(limit.getValue())
                       : DenseIntElementsAttr();
    if (!range || range.getStart() != 0 || range.getEnd() != elements ||
        !dense || !dense.isSplat())
      return std::nullopt;
    info.active = std::clamp(dense.getSplatValue<APInt>().getSExtValue(),
                             int64_t(0), elements);
  } else {
    if (cmp.getPredicate() != arith::CmpIPredicate::ne ||
        !splatZero(cmp.getRhs()))
      return std::nullopt;
    Value source = cmp.getLhs();
    if (auto ext = source.getDefiningOp<arith::ExtUIOp>())
      source = ext.getIn();
    source = flatMaskSource(source);
    auto input =
        source ? source.getDefiningOp<triton::LoadOp>() : triton::LoadOp();
    auto type = source ? dyn_cast<RankedTensorType>(source.getType())
                       : RankedTensorType();
    if (!input || input.getMask() || !type ||
        type.getNumElements() != elements ||
        !type.getElementType().isInteger(8))
      return std::nullopt;
    info.random = true;
  }
  return info;
}

static StageMemoryCost simdMaskedEnvelope(const LogicalStage &stage,
                                          const HardwareProfile &hardware,
                                          const StageImplementation &impl) {
  StageMemoryCost result;
  const auto &work = stage.workload;
  const auto &p = work.addressPatterns.front();
  const bool store = p.memoryOp == "tt.store";
  if (hardware.target != "Ascend950PR/dav-c310" || impl.localScope ||
      impl.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      stage.features.hasLoop || stage.features.hasAtomicMemory ||
      work.privateStackLoadInstructions || work.privateStackStoreInstructions ||
      work.hasProvenIndirectStoreReuse || p.stageId != stage.id ||
      !p.dependsOnLoadedValue || (!store && p.memoryOp != "tt.load") ||
      (store ? hardware.simdIndirectStoreModel
             : hardware.simdIndirectLoadModel) != simdMaskModel)
    return result;
  Operation *target = memoryTarget(stage, store, MemoryWalk::Regionless);
  if (!target || target->getNumOperands() != 3 || !defaultCache(target))
    return result;
  if (!store && !splatZero(target->getOperand(2)))
    return result;
  auto type = dyn_cast<RankedTensorType>(
      store ? target->getOperand(1).getType() : target->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      !type.getElementType().isIntOrFloat() || type.getElementType().isF64())
    return result;
  const int64_t e = type.getNumElements(), rank = type.getRank();
  const int bits = type.getElementType().getIntOrFloatBitWidth();
  if ((bits != 8 && bits != 16 && bits != 32 && bits != 64) || e <= 0 ||
      llvm::any_of(type.getShape(), [](int64_t n) { return n <= 0; }))
    return result;
  if ((store ? work.indirectLoadBytes : work.indirectStoreBytes) != 0 ||
      (store ? work.indirectStoreBytes : work.indirectLoadBytes) !=
          e * (bits / 8))
    return result;
  // Completion summaries have no operation identity. Require a unique match
  // before allowing target-level removal; never erase a second access by kind.
  unsigned matchedAccesses = 0;
  for (const auto &access : work.memoryAccesses) {
    if (!access.addressDependsOnLoadedIndex || access.store != store)
      continue;
    if (!access.masked || access.elementBits != bits || access.count != 1 ||
        !llvm::equal(access.shape, type.getShape()) || ++matchedAccesses > 1)
      return result;
  }
  auto ptr = target->getOperand(0).getDefiningOp<triton::AddPtrOp>();
  if (!ptr || !ptr.getPtr().getDefiningOp<triton::SplatOp>())
    return result;
  Value index = flatMaskSource(ptr.getOffset());
  auto loaded =
      index ? index.getDefiningOp<triton::LoadOp>() : triton::LoadOp();
  if (!loaded || loaded.getMask())
    return result;
  auto indexType = dyn_cast<RankedTensorType>(index.getType());
  if (!indexType || indexType.getNumElements() != e ||
      (!indexType.getElementType().isInteger(32) &&
       !indexType.getElementType().isInteger(64)))
    return result;
  auto mask = parseMask(target->getOperand(store ? 2 : 1), e);
  if (!mask)
    return result;
  auto cmp = mask->compare;
  const bool random = mask->random;
  const int64_t k = mask->active;
  if (store || (!random && rank == 1)) {
    if (!mask->exclusive)
      return result;
    // Cross-stage predicates retain their own charge. Only remove a predicate
    // resource when this stage owns the unique covered operation.
    if (llvm::is_contained(stage.operations, cmp.getOperation())) {
      auto count = work.predicateCounts.find(e);
      if (count == work.predicateCounts.end() || count->second != 1)
        return result;
      result.coveredPredicate = cmp;
    }
  }
  const unsigned g = bits == 8 ? 0 : bits == 16 ? 1 : bits == 32 ? 2 : 3;
  double low = simdMaskCycles(*hardware.indirectMemoryParameters, store, random,
                              g, e, k, rank),
         high = low;
  if (random) {
    // For K>0 these frozen models are affine in K (Load is independent of K).
    // Include the discontinuous zero branch and both positive endpoints.
    for (int64_t active : {int64_t(1), e}) {
      double value = simdMaskCycles(*hardware.indirectMemoryParameters, store,
                                    true, g, e, active, rank);
      low = std::min(low, value);
      high = std::max(high, value);
    }
    result.densityRange = std::make_pair(low, high);
  }
  (store ? result.store : result.load) = low;
  (store ? result.storePricing : result.loadPricing) = simdMaskModel;
  result.nativeSimdEnvelope = true;
  result.coveredMemory = target;
  return result;
}

// Shared SIMT admission and target geometry. Masked and unmasked formulas
// retain distinct boundaries; neither removes external helper charges.
static std::optional<double>
simtIncrement(const LogicalStage &stage, const HardwareProfile &hardware,
              const StageImplementation &implementation, bool store,
              bool allowMasked, const char *&pricing) {
  const auto &work = stage.workload;
  const auto &pattern = work.addressPatterns.front();
  const int64_t w = hardware.baseSimtWarpCount;
  const int64_t f = implementation.superblockFactor;
  if (hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMT ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      f <= 0 || w <= 0 || w > 64 / f || !pattern.dependsOnLoadedValue ||
      pattern.stageId != stage.id ||
      pattern.memoryOp != (store ? "tt.store" : "tt.load") ||
      stage.features.hasAtomicMemory || work.privateStackLoadInstructions ||
      work.privateStackStoreInstructions ||
      (store && work.hasProvenIndirectStoreReuse) ||
      (store ? work.indirectLoadBytes != 0 : work.indirectStoreBytes != 0))
    return std::nullopt;
  Operation *op = memoryTarget(stage, store);
  if (!op || !defaultCache(op) || (!store && op->getNumResults() != 1))
    return std::nullopt;
  const bool masked = op->getNumOperands() == 3;
  if (masked ? !allowMasked : op->getNumOperands() != (store ? 2u : 1u))
    return std::nullopt;
  if (!masked && (stage.features.hasLoop || !work.predicateCounts.empty() ||
                  stage.features.activeLaneRatio != 1))
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(store ? op->getOperand(1).getType()
                                               : op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      pattern.axes.size() != size_t(type.getRank()))
    return std::nullopt;
  Type element = type.getElementType();
  if (!element.isIntOrFloat() ||
      (!masked && !isa<IntegerType, Float16Type, BFloat16Type, Float32Type,
                       Float64Type, Float8E4M3FNType, Float8E5M2Type>(element)))
    return std::nullopt;
  const int bits = element.getIntOrFloatBitWidth();
  const int group = bits == 8    ? 0
                    : bits == 16 ? 1
                    : bits == 32 ? 2
                    : bits == 64 ? 3
                                 : -1;
  const int64_t e = type.getNumElements(), rank = type.getRank();
  if (group < 0 || e <= 0 ||
      (store ? work.indirectStoreBytes : work.indirectLoadBytes) !=
          e * bits / 8)
    return std::nullopt;
  for (int64_t axis = 0; axis < rank; ++axis)
    if (type.getDimSize(axis) <= 0 ||
        (!masked && pattern.axes[axis].extent != type.getDimSize(axis)))
      return std::nullopt;
  auto address = op->getOperand(0).getDefiningOp<triton::AddPtrOp>();
  if (!address || !address.getPtr().getDefiningOp<triton::SplatOp>())
    return std::nullopt;
  Value offset = flatMaskSource(address.getOffset());
  auto index =
      offset ? offset.getDefiningOp<triton::LoadOp>() : triton::LoadOp();
  auto indexType = offset ? dyn_cast<RankedTensorType>(offset.getType())
                          : RankedTensorType();
  if (!index || index.getMask() || !indexType ||
      (!masked && indexType.getRank() != 1) ||
      indexType.getNumElements() != e ||
      (!indexType.getElementType().isInteger(32) &&
       !indexType.getElementType().isInteger(64)))
    return std::nullopt;
  if (masked) {
    Value predicate = op->getOperand(store ? 2 : 1);
    if (predicate.getType() !=
            type.clone(IntegerType::get(op->getContext(), 1)) ||
        (!store && !splatZero(op->getOperand(2))))
      return std::nullopt;
    auto mask = parseMask(predicate, e);
    if (!mask)
      return std::nullopt;
    pricing = mask->random ? randomMaskModel : prefixMaskModel;
    return maskGroupCycles(*hardware.indirectMemoryParameters, store,
                           mask->random, group, e, w, f, rank,
                           type.getShape().back(), mask->active) /
           f;
  }
  pricing = completeModel;
  std::vector<double> shape(type.getShape().begin(), type.getShape().end());
  // applySuperBlock restores F exactly once, including for local scopes.
  return completeGroupCycles(*hardware.indirectMemoryParameters, store, group,
                             e, w, f, shape, isa<FloatType>(element),
                             element.isBF16(),
                             isa<Float8E4M3FNType, Float8E5M2Type>(element)) /
         f;
}

// Effective incremental latency of one indirect load (payload - matched ALU),
// not a whole-Stage cost. The random-address prior is intentional: no runtime
// indices or final ISA facts are inputs. Unsupported domains use legacy rates.
static std::optional<double>
calibratedSimdIndirectLoad(const LogicalStage &stage,
                           const HardwareProfile &hardware,
                           const StageImplementation &implementation) {
  const auto &work = stage.workload;
  if ((hardware.simdIndirectLoadModel != "random_dtype_matched_ab_20261008" &&
       hardware.simdIndirectLoadModel != simdMaskModel) ||
      hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMD ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      work.addressPatterns.size() != 1 ||
      (work.privateStackLoadInstructions +
       work.privateStackStoreInstructions) != 0 ||
      !work.predicateCounts.empty() || stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousLoadBytes != 0 || work.indirectStoreBytes != 0)
    return std::nullopt;
  // Real Stage partitioning owns the payload's splat/addptr/reshape producers
  // too. Require one load, not one total operation; those independent helpers
  // retain their normal resource charges.
  Operation *op = memoryTarget(stage, false, MemoryWalk::TopLevel);
  if (!op || op->getNumResults() != 1 || op->getNumOperands() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  {
    if (!type || !type.hasStaticShape() || type.getRank() < 1)
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
    if (elements <= 0 || work.indirectLoadBytes != elementBytes * elements ||
        work.indirectLoadTransactions <= 0 || pattern.memoryOp != "tt.load" ||
        pattern.stageId != stage.id || !pattern.dependsOnLoadedValue ||
        pattern.axes.size() != static_cast<size_t>(type.getRank()))
      return std::nullopt;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      const auto &summary = pattern.axes[axis];
      const int64_t extent = type.getDimSize(axis);
      if (extent <= 0 || summary.extent != extent)
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
    // Effective matched-address load increment in SYS_CNT/system_cycles.
    // The tiny 32-bit path has measured unrolling/grouping differences; it is
    // not a universal hardware latency threshold. No SIMD warp division.
    const bool small32 =
        elementBytes == 4 && type.getRank() == 1 && elements <= 16;
    return elements * (hardware.indirectMemoryParameters->get(
                          "simd_unmasked_load")[small32 ? 0 : 1]);
  }
}

// Random, collision-free addresses; no target fill. Coefficients price the
// measured A/B increment in SYS_CNT ticks, not a worst-case latency bound.
static std::optional<double>
randomDtypeIndirectStore(const LogicalStage &stage,
                         const IndirectMemoryParameters &parameters,
                         bool reuse) {
  const auto &work = stage.workload;
  if (reuse && !work.hasProvenIndirectStoreReuse)
    return std::nullopt;
  // Real partitioning also owns the store's address producers. Require one
  // target store, not one total operation; helpers keep their normal charges.
  Operation *op = memoryTarget(stage, true, MemoryWalk::TopLevel);
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 0)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1)
    return std::nullopt;
  Type element = type.getElementType();
  if (!isa<IntegerType>(element) && !element.isF16() && !element.isBF16() &&
      !element.isF32() && !isa<Float8E4M3FNType, Float8E5M2Type>(element))
    return std::nullopt;
  unsigned bits = element.getIntOrFloatBitWidth();
  if (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  const int64_t elements = type.getNumElements();
  // num_warps configures SIMT candidates, not this SIMD implementation.
  if (elements <= 0 || pattern.memoryOp != "tt.store" ||
      pattern.stageId != stage.id ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()) ||
      work.indirectStoreBytes != std::max(1u, bits / 8) * elements ||
      work.indirectStoreTransactions <= 0)
    return std::nullopt;
  // The entry point requires every pointer-analysis axis to be unstructured.
  // Auxiliary opaque provenance (e.g. reshape) must not veto that fallback.
  // This classification does not prove random or collision-free addresses;
  // those remain the calibrated profile's input prior.
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    int64_t extent = type.getDimSize(axis);
    if (extent < 1 || pattern.axes[axis].extent != extent ||
        (pattern.axes[axis].regularity != "opaque_loaded" &&
         pattern.axes[axis].regularity != "opaque"))
      return std::nullopt;
  }
  const unsigned group = bits <= 16 ? 0 : (bits == 32 ? 1 : 2);
  auto coefficients = parameters.get("simd_unmasked_store");
  return (reuse ? 0.0 : coefficients[0]) +
         coefficients[(reuse ? 4 : 1) + group] * elements;
}

// Effective write plus required synchronization increment, not a whole Stage.
// Address randomness and memory history are profile priors: neither is proven
// by pointer analysis. In particular, the no-fill first fit has >20% outliers.
static std::optional<double>
calibratedIndirectStore(const LogicalStage &stage,
                        const HardwareProfile &hardware,
                        const StageImplementation &implementation) {
  const auto &work = stage.workload;

  if (implementation.mode != StageMode::SIMD ||
      hardware.target != "Ascend950PR/dav-c310" ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      work.addressPatterns.size() != 1 ||
      (work.privateStackLoadInstructions +
       work.privateStackStoreInstructions) != 0 ||
      !work.predicateCounts.empty() || stage.features.hasAtomicMemory ||
      stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousStoreBytes != 0 || work.indirectLoadBytes != 0)
    return std::nullopt;
  const auto &model = hardware.simdIndirectStoreModel;
  if (model != "random_store_no_fill_first_20261008" &&
      model != "random_store_no_fill_reuse_20261008" && model != simdMaskModel)
    return std::nullopt;
  return randomDtypeIndirectStore(stage, *hardware.indirectMemoryParameters,
                                  model ==
                                      "random_store_no_fill_reuse_20261008");
}

} // namespace

double IndirectGatherMemoryCostModel::nativeChainCycles(
    const IndirectMemoryParameters &parameters, bool store,
    RankedTensorType type, int indexBits, int64_t warps, int64_t factor) {
  Type element = type.getElementType();
  return nativeMemoryChainCycles(
      parameters, store, type.getNumElements(), warps, factor, type.getRank(),
      type.getShape().back(), indexBits, element.getIntOrFloatBitWidth(),
      isa<FloatType>(element) && !element.isBF16(), element.isBF16(),
      isa<Float8E4M3FNType, Float8E5M2Type>(element));
}

StageMemoryCost IndirectGatherMemoryCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &implementation) const {
  if (!profile.indirectMemoryParameters)
    return {};
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
  StageMemoryCost result;
  if (implementation.mode == StageMode::SIMD) {
    result = simdMaskedEnvelope(stage, profile, implementation);
    if (result.nativeSimdEnvelope)
      return result;
    result.load = calibratedSimdIndirectLoad(stage, profile, implementation);
    result.store = calibratedIndirectStore(stage, profile, implementation);
    if (result.load)
      result.loadPricing = "random_dtype_matched_ab_20261008";
    if (result.store)
      result.storePricing = profile.simdIndirectStoreModel ==
                                    "random_store_no_fill_reuse_20261008"
                                ? "random_store_no_fill_reuse_20261008"
                                : "random_store_no_fill_first_20261008";
    return result;
  }
  if (implementation.mode != StageMode::SIMT)
    return result;
  for (bool store : {false, true}) {
    const auto &model =
        store ? profile.simtIndirectStoreModel : profile.simtIndirectLoadModel;
    const bool masked = model == prefixMaskModel || model == randomMaskModel;
    if (!masked && model != completeModel)
      continue;
    // Either masked selector enables both families; the actual predicate
    // chooses the coefficients and the diagnostic pricing name.
    const char *pricing = nullptr;
    auto value =
        simtIncrement(stage, profile, implementation, store, masked, pricing);
    if (value) {
      (store ? result.store : result.load) = value;
      (store ? result.storePricing : result.loadPricing) = pricing;
    }
  }
  return result;
}
