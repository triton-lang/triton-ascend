//===- StagePartitioner.cpp - Build semantic Stage IR -------------------===//

#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/Analysis/LoopWorkloadAnalysis.h"
#include "AscendModel/Analysis/StageMemoryPatternAnalysis.h"
#include "AscendModel/RouteModel/Reduction/ReductionAnalysis.h"
#include "ascend/include/Utils/SuperBlockFactor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static void recomputeIssueElements(StageWorkload &work) {
  double elements =
      work.scalarOperations + work.scalarLoadCount + work.scalarStoreCount;
  for (const auto &[size, count] : work.predicateCounts)
    elements += static_cast<double>(size) * count;
  for (const auto &entry : work.operationCounts)
    for (const auto &[size, count] : entry.second)
      elements += static_cast<double>(size) * count;
  elements +=
      32.0 *
      (work.loadWarpInstructions + work.storeWarpInstructions +
       work.sharedLocalLoadInstructions + work.sharedLocalStoreInstructions +
       work.privateStackLoadInstructions + work.privateStackStoreInstructions);
  for (const AtomicWorkload &atomic : work.atomicWorkloads)
    elements += atomic.logicalElements;
  work.issueElements = elements;
}

static double getTypeElementCount(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return 1.0;
    return static_cast<double>(std::max<int64_t>(1, shaped.getNumElements()));
  }
  return 1.0;
}

static Type getScalarElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

static std::string typeToString(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  stream.flush();
  return text;
}

static bool isPointerLikeType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return llvm::StringRef(typeToString(type)).contains("!tt.ptr");
}

static std::optional<unsigned> getMemoryMaskOperandIndex(Operation *operation);
static bool isMemoryMaskOnlyValue(Value value);
static bool shouldSplitLoopBodyIntoStages(Operation *operation);
static bool isGridStrideLoop(Operation *operation);

static bool isAddressOnlyLoopValue(Value root) {
  llvm::SmallVector<Value, 8> worklist{root};
  llvm::DenseSet<Value> visited;
  bool reachesAddressUse = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      const llvm::StringRef name = user->getName().getStringRef();
      if (name == "scf.yield" || name == "scf.condition")
        continue;
      if ((name == "tt.load" || name == "tt.store" ||
           name.starts_with("tt.atomic")) &&
          use.getOperandNumber() == 0) {
        reachesAddressUse = true;
        continue;
      }
      if (name == "arith.cmpi" || name == "arith.cmpf") {
        // A comparison feeding only memory masks is addressing boundary
        // logic, rather than a data recurrence.
        if (!llvm::all_of(user->getResults(), isMemoryMaskOnlyValue))
          return false;
        reachesAddressUse = true;
        continue;
      }
      const bool addressForwarding =
          name == "tt.addptr" || name == "tt.advance" || name == "tt.splat" ||
          name == "tt.broadcast" || name == "tt.expand_dims" ||
          name == "arith.addi" || name == "arith.subi" ||
          name == "arith.muli" || name == "arith.index_cast";
      if (!addressForwarding)
        return false;
      if (name == "tt.addptr" || name == "tt.advance")
        reachesAddressUse = true;
      llvm::append_range(worklist, user->getResults());
    }
  }
  return reachesAddressUse;
}

static int64_t getScalarBitWidth(Type type) {
  type = getScalarElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  if (isa<IndexType>(type))
    return 64;
  return 0;
}

static double getValueBytes(Value value) {
  const int64_t bits = getScalarBitWidth(value.getType());
  return bits > 0 ? getTypeElementCount(value.getType()) *
                        static_cast<double>(bits) / 8.0
                  : 0.0;
}

static double getOperationElements(Operation *operation) {
  // Histogram result width is the number of bins, while the helper consumes
  // every source element.
  if (operation->getName().getStringRef() == "tt.histogram" &&
      operation->getNumOperands() > 0)
    return getTypeElementCount(operation->getOperand(0).getType());
  double elements = 1.0;
  for (Type type : operation->getResultTypes())
    elements = std::max(elements, getTypeElementCount(type));
  if (operation->getNumResults() == 0)
    for (Value value : operation->getOperands())
      elements = std::max(elements, getTypeElementCount(value.getType()));
  return elements;
}

static std::optional<PartialContinuousTile>
getPartialContinuousTile(Operation *operation,
                         const StageMemoryPatternAnalysis *memoryPatterns) {
  if (!operation || operation->getNumOperands() == 0)
    return std::nullopt;
  if (memoryPatterns)
    return memoryPatterns->lookup(operation);
  ModuleOp module = operation->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;
  return StageMemoryPatternAnalysis(module).lookup(operation);
}

static bool
isIndirectMemoryOperation(Operation *operation,
                          const StageMemoryPatternAnalysis *memoryPatterns) {
  if (memoryPatterns)
    if (auto known = memoryPatterns->hasUnstructuredAxes(operation))
      return *known;
  return isLoadedIndexDependentMemoryOp(operation);
}

static bool hasTensorResult(Operation *operation) {
  return llvm::any_of(operation->getResultTypes(),
                      [](Type type) { return isa<ShapedType>(type); });
}

static bool isDotOperationName(llvm::StringRef name) {
  return name == "tt.dot" || name == "tt.dot_scaled";
}

static bool isSaturatedComputeFallback(Operation *operation) {
  if (!operation)
    return false;
  const llvm::StringRef name = operation->getName().getStringRef();
  if (name == "arith.constant" || name == "arith.cmpi" ||
      name == "arith.cmpf" || name == "arith.select" ||
      name == "arith.bitcast" || name == "tt.bitcast" ||
      name == "tt.int_to_ptr" || name == "tt.ptr_to_int" ||
      name == "tt.addptr" || name == "tt.advance" || name == "tt.splat" ||
      name == "tt.unsplat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" || name == "tt.cat" ||
      name == "tt.join" || name == "tt.split" || name == "tt.make_range" ||
      name == "tt.get_program_id" || name == "tt.get_num_programs" ||
      name == "tt.make_tensor_ptr" || name == "tt.make_tensor_descriptor" ||
      name == "tt.load" || name == "tt.store" || name == "tt.gather" ||
      name == "tt.atomic_rmw" || name == "tt.atomic_cas" || name == "tt.dot" ||
      name == "tt.dot_scaled" || name == "tt.reduce" || name == "tt.scan" ||
      name == "tt.histogram" || name == "tt.trans" ||
      name == "tt.descriptor_load" || name == "tt.descriptor_store" ||
      name == "tt.descriptor_reduce" || name == "tt.descriptor_gather" ||
      name == "tt.descriptor_scatter")
    return false;

  const bool hasNumericResult =
      llvm::any_of(operation->getResultTypes(), [](Type type) {
        Type scalar = getScalarElementType(type);
        return isa<FloatType, IntegerType, IndexType>(scalar);
      });
  const bool hasNumericOperand =
      llvm::any_of(operation->getOperandTypes(), [](Type type) {
        Type scalar = getScalarElementType(type);
        return isa<FloatType, IntegerType, IndexType>(scalar);
      });

  // Arithmetic and math dialect operations with a numeric result are executable
  // ALU work.
  if (name.starts_with("arith.") || name.starts_with("math."))
    return hasNumericResult;

  // Triton elementwise operations that survive into TTIR.
  if (name == "tt.mulhiui" || name == "tt.extern_elementwise")
    return hasNumericResult;
  if (name == "tt.elementwise_inline_asm")
    return hasNumericResult ||
           (operation->getNumResults() == 0 && hasNumericOperand);

  // Registered elementwise numeric ops and future TTIR numeric ops both keep a
  // non-zero execution floor; pointer-valued operations remain excluded.
  if (!operation->hasTrait<OpTrait::Elementwise>())
    // Keep future TTIR numeric execution visible even when the new operation
    // does not declare Elementwise.
    return name.starts_with("tt.") && hasNumericResult;
  return hasNumericResult;
}

static std::string getProfileOperationName(Operation *operation) {
  const llvm::StringRef name = operation->getName().getStringRef();
  // Tensor remainder keeps one rate per signedness across widths.
  if (name == "arith.remsi" || name == "arith.remui")
    return name == "arith.remsi" ? "srem" : "urem";
  if (name == "tt.histogram") {
    for (Type type : operation->getResultTypes())
      if (getScalarBitWidth(type) == 32)
        return "histogram.i32";
  }
  if (name == "tt.extern_elementwise") {
    // Extern elementwise is a transport op: the symbol carries the actual
    // numerical semantics.
    if (auto symbol = operation->getAttrOfType<StringAttr>("symbol")) {
      const llvm::StringRef value = symbol.getValue();
      if (value.contains("relu"))
        return "f32.relu";
      if (value.contains("isinf") || value.contains("isnan") ||
          value.contains("finite"))
        return "predicate.cmp";
      if (value.contains("fast_divide") || value.contains("recip") ||
          value.contains("rcp_") || value.contains("div_r"))
        return "f32.div";
      if (value.contains("sqrt_"))
        return "f32.sqrt";
      if (value.contains("fast_exp") || value.contains("exp10"))
        return "f32.exp";
      if (value.contains("log1p") || value.contains("fast_log"))
        return "f32.log";
      if (value.contains("fast_cos"))
        return "f32.cos";
      if (value.contains("saturate"))
        return "f32.clamp";
      if (value.contains("add_r"))
        return "f32.add";
      if (value.contains("sub_r"))
        return "f32.sub";
      if (value.contains("mul_r"))
        return "f32.mul";
      return (llvm::Twine("fp32.saturated.tt.extern_elementwise.") + value)
          .str();
    }
    return "fp32.saturated.tt.extern_elementwise";
  }
  return llvm::StringSwitch<std::string>(name)
      .Cases("arith.addf", "tt.add", "f32.add")
      .Case("arith.subf", "f32.sub")
      .Case("arith.mulf", "f32.mul")
      .Case("arith.divf", "f32.div")
      .Cases("arith.maximumf", "arith.maxnumf", "arith.minimumf",
             "arith.minnumf", "f32.max")
      .Cases("math.absf", "tt.abs", "f32.abs")
      .Cases("math.exp", "tt.exp", "f32.exp")
      .Cases("math.log", "tt.log", "f32.log")
      .Cases("math.sqrt", "tt.precise_sqrt", "f32.sqrt")
      .Case("tt.precise_divf", "f32.div")
      .Case("math.cos", "f32.cos")
      .Cases("math.isfinite", "math.isnan", "math.isinf", "predicate.cmp")
      .Case("arith.select", "predicate.select")
      .Case("tt.clampf", "f32.clamp")
      .Cases("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp",
             "convert.cast")
      .Cases("arith.fptosi", "arith.fptoui", "arith.extsi", "arith.extui",
             "arith.trunci", "arith.index_cast", "arith.index_castui",
             "tt.fp_to_fp", "convert.cast")
      .Case("tt.trans", "layout.transpose")
      .Default(isSaturatedComputeFallback(operation)
                   ? (llvm::Twine("fp32.saturated.") + name).str()
                   : std::string("generic.issue"));
}

static RankedTensorType getRepresentativeTensorType(Operation *operation) {
  RankedTensorType representative;
  int64_t maximumElements = -1;
  auto consider = [&](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape())
      return;
    const int64_t elements = tensor.getNumElements();
    if (elements > maximumElements) {
      representative = tensor;
      maximumElements = elements;
    }
  };
  for (Type type : operation->getResultTypes())
    consider(type);
  if (!representative)
    for (Value operand : operation->getOperands())
      consider(operand.getType());
  return representative;
}

/// Project the dense static subset of HIVM's pointwise reassociation.
static int64_t getPostFlattenPointwiseRun(Operation *operation,
                                          RankedTensorType outputType) {
  SmallVector<int64_t> activeAxes;
  for (int64_t axis = 0; axis < outputType.getRank(); ++axis)
    if (outputType.getDimSize(axis) > 1)
      activeAxes.push_back(axis);
  if (activeAxes.empty())
    return 1;
  SmallVector<bool> mergeWithPrevious(activeAxes.size(), true);
  for (Value operand : operation->getOperands()) {
    if (Operation *producer = operand.getDefiningOp()) {
      const llvm::StringRef producerName = producer->getName().getStringRef();
      if (producerName == "tt.broadcast" && producer->getNumOperands() == 1)
        operand = producer->getOperand(0);
    }
    auto input = dyn_cast<RankedTensorType>(operand.getType());
    if (!input || !input.hasStaticShape() ||
        input.getRank() != outputType.getRank())
      continue;
    if (input.getNumElements() == 1)
      continue;
    for (size_t i = 1; i < activeAxes.size(); ++i) {
      const int64_t left = activeAxes[i - 1];
      const int64_t right = activeAxes[i];
      if (input.getDimSize(left) != outputType.getDimSize(left) ||
          input.getDimSize(right) != outputType.getDimSize(right))
        mergeWithPrevious[i] = false;
    }
  }
  int64_t run = outputType.getDimSize(activeAxes.back());
  for (size_t i = activeAxes.size() - 1; i > 0 && mergeWithPrevious[i]; --i)
    run *= outputType.getDimSize(activeAxes[i - 1]);
  return run;
}

static void accumulateTensorOperationWorkload(
    Operation *operation, llvm::StringRef profileName, double elements,
    StageWorkload &work, RankedTensorType mappedTensor = {}) {
  RankedTensorType tensor =
      mappedTensor ? mappedTensor : getRepresentativeTensorType(operation);
  if (!tensor || tensor.getRank() == 0 || !tensor.hasStaticShape())
    return;
  int64_t elementBits =
      mappedTensor && operation->getNumResults() > 0
          ? getScalarBitWidth(operation->getResult(0).getType())
          : getScalarBitWidth(tensor.getElementType());
  if (profileName == "predicate.cmp")
    for (Value operand : operation->getOperands())
      elementBits = std::max(elementBits, getScalarBitWidth(operand.getType()));
  const bool knownPointwise =
      mappedTensor || operation->hasTrait<OpTrait::Elementwise>();
  const int64_t contiguousElements =
      knownPointwise ? getPostFlattenPointwiseRun(operation, tensor)
                     : tensor.getShape().back();
  if (elementBits <= 0 || contiguousElements <= 0)
    return;
  const double segments = elements / static_cast<double>(contiguousElements);
  if (segments <= 0.0 || segments != std::floor(segments))
    return;
  for (TensorOperationWorkload &group : work.tensorOperationWorkloads) {
    if (group.operation == profileName &&
        group.elementBitWidth == elementBits &&
        group.contiguousElementsPerSegment == contiguousElements) {
      group.logicalElements += elements;
      group.segmentCount += segments;
      return;
    }
  }
  work.tensorOperationWorkloads.push_back(
      {profileName.str(), elementBits, elements, segments, contiguousElements});
}

static std::optional<std::string> describeDotOperandPath(Operation *operation);

static void accumulateDotWorkload(Operation *operation, StageWorkload &work) {
  if (operation->getNumOperands() < 2)
    return;
  auto lhs = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto rhs = dyn_cast<ShapedType>(operation->getOperand(1).getType());
  if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      lhs.getRank() < 2 || rhs.getRank() < 2)
    return;
  int64_t m = lhs.getShape()[lhs.getRank() - 2];
  int64_t k = lhs.getShape()[lhs.getRank() - 1];
  int64_t n = rhs.getShape()[rhs.getRank() - 1];
  if (operation->getNumResults() > 0) {
    auto result = dyn_cast<ShapedType>(operation->getResult(0).getType());
    if (result && result.hasStaticShape() && result.getRank() >= 2) {
      // The dot_scaled result always carries logical M/N even when an E2M1
      // input physically packs two FP4 values into one i8 dimension.
      m = result.getShape()[result.getRank() - 2];
      n = result.getShape()[result.getRank() - 1];
    }
  }
  if (auto scaled = dyn_cast<triton::DotScaledOp>(operation)) {
    if (scaled.getAElemType() == triton::ScaleDotElemType::E2M1 &&
        scaled.getLhsKPack()) {
      if (k > std::numeric_limits<int64_t>::max() / 2)
        return;
      k *= 2;
    }
  }
  double batch = 1.0;
  for (int64_t dim : lhs.getShape().drop_back(2))
    batch *= static_cast<double>(dim);
  int64_t batchShape = 1;
  for (int64_t dim : lhs.getShape().drop_back(2)) {
    if (dim <= 0 || dim > std::numeric_limits<int64_t>::max() / batchShape) {
      batchShape = 0;
      break;
    }
    batchShape *= dim;
  }
  if (m > 0 && n > 0 && k > 0) {
    // Keep the shape key exact.  Do not let a malformed or merely enormous
    // static tensor wrap the FLOP key and turn a non-tiny dot into a tiny one.
    const __int128 product = static_cast<__int128>(2) * m * n * k;
    if (product > std::numeric_limits<int64_t>::max())
      return;
    const int64_t flopsPerMatrix = static_cast<int64_t>(product);
    work.dotOperations += batch;
    work.dotFlopCounts[flopsPerMatrix] += batch;
    work.dotFlops += static_cast<double>(flopsPerMatrix) * batch;
    if (batchShape > 0) {
      Type accumulatorType;
      if (operation->getNumOperands() > 2)
        accumulatorType =
            getScalarElementType(operation->getOperand(2).getType());
      else if (operation->getNumResults() > 0)
        accumulatorType =
            getScalarElementType(operation->getResult(0).getType());
      std::string precision = "unknown";
      if (auto dot = dyn_cast<triton::DotOp>(operation))
        precision =
            triton::stringifyInputPrecision(dot.getInputPrecision()).str();
      const auto shape = std::make_tuple(
          m, n, k, batchShape, typeToString(lhs.getElementType()),
          typeToString(rhs.getElementType()),
          accumulatorType ? typeToString(accumulatorType) : "unknown",
          std::move(precision));
      work.dotShapeCounts[shape] += 1.0;
    }
    if (auto path = describeDotOperandPath(operation)) {
      auto &entry = work.dotOperandPaths[*path];
      entry.first = flopsPerMatrix;
      entry.second += batch;
    }
  }
}

/// Classify a scan from the operation that directly defines each tt.scan.return
/// operand.
static std::string classifyScanCombine(Operation *scan) {
  if (!scan || scan->getName().getStringRef() != "tt.scan" ||
      scan->getNumRegions() == 0)
    return "other";
  std::optional<std::string> result;
  for (Region &region : scan->getRegions()) {
    if (region.empty())
      return "other";
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator ||
          terminator->getName().getStringRef() != "tt.scan.return" ||
          terminator->getNumOperands() == 0)
        return "other";
      for (Value returned : terminator->getOperands()) {
        Operation *combine = returned.getDefiningOp();
        if (!combine)
          return "other";
        const llvm::StringRef name = combine->getName().getStringRef();
        std::string kind;
        if (name == "arith.addf" || name == "arith.addi")
          kind = "add";
        else if (name == "arith.mulf" || name == "arith.muli")
          kind = "mul";
        else if (name == "arith.maximumf" || name == "arith.minimumf" ||
                 name == "arith.maxnumf" || name == "arith.minnumf" ||
                 name == "arith.maxsi" || name == "arith.maxui" ||
                 name == "arith.minsi" || name == "arith.minui")
          kind = "minmax";
        else
          return "other";
        if (result && *result != kind)
          return "other";
        result = std::move(kind);
      }
    }
  }
  return result.value_or("other");
}

static std::string classifyReductionCombine(Operation *reduce) {
  if (!reduce || reduce->getName().getStringRef() != "tt.reduce" ||
      reduce->getNumRegions() == 0)
    return "other";
  std::optional<std::string> result;
  for (Region &region : reduce->getRegions()) {
    if (region.empty())
      return "other";
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator ||
          terminator->getName().getStringRef() != "tt.reduce.return" ||
          terminator->getNumOperands() == 0)
        return "other";
      for (Value returned : terminator->getOperands()) {
        Operation *combine = returned.getDefiningOp();
        if (!combine)
          return "other";
        const llvm::StringRef name = combine->getName().getStringRef();
        std::string kind;
        if (name == "arith.addf" || name == "arith.addi")
          kind = "add";
        else if (name == "arith.mulf" || name == "arith.muli")
          kind = "mul";
        else if (name == "arith.maximumf" || name == "arith.minimumf" ||
                 name == "arith.maxnumf" || name == "arith.minnumf" ||
                 name == "arith.maxsi" || name == "arith.minsi" ||
                 name == "arith.maxui" || name == "arith.minui")
          kind = "minmax";
        else
          return "other";
        if (result && *result != kind)
          return "other";
        result = std::move(kind);
      }
    }
  }
  return result.value_or("other");
}

static std::optional<int64_t> getConstantInteger(Value value);

static bool isZeroConstant(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "arith.constant")
    return false;
  Attribute attribute = definition->getAttr("value");
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attribute))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast_or_null<FloatAttr>(attribute))
    return floating.getValue().isZero();
  auto dense = dyn_cast_or_null<DenseElementsAttr>(attribute);
  if (!dense || !dense.isSplat())
    return false;
  if (isa<IntegerType>(dense.getElementType()))
    return dense.getSplatValue<APInt>().isZero();
  if (isa<FloatType>(dense.getElementType()))
    return dense.getSplatValue<APFloat>().isZero();
  return false;
}

static bool valueDependsOn(Value value, Value dependency, unsigned depth = 0) {
  if (value == dependency)
    return true;
  Operation *definition = value.getDefiningOp();
  if (!definition || depth >= 24)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOn(operand, dependency, depth + 1);
  });
}

static bool valueTreeHasOperation(Value value, llvm::StringRef operationName,
                                  unsigned depth = 0) {
  Operation *definition = value.getDefiningOp();
  if (!definition || depth >= 24)
    return false;
  if (definition->getName().getStringRef() == operationName)
    return true;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueTreeHasOperation(operand, operationName, depth + 1);
  });
}

static bool maskProvesPrefixBound(Value mask, Value bound, unsigned depth = 0) {
  Operation *definition = mask.getDefiningOp();
  if (!definition || depth >= 16)
    return false;
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition))
    return maskProvesPrefixBound(conjunction.getLhs(), bound, depth + 1) ||
           maskProvesPrefixBound(conjunction.getRhs(), bound, depth + 1);
  if ((definition->getName().getStringRef() == "tt.broadcast" ||
       definition->getName().getStringRef() == "tt.expand_dims") &&
      definition->getNumOperands() == 1)
    return maskProvesPrefixBound(definition->getOperand(0), bound, depth + 1);
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison)
    return false;
  const bool lhsIsRange =
      valueTreeHasOperation(comparison.getLhs(), "tt.make_range");
  const bool rhsIsRange =
      valueTreeHasOperation(comparison.getRhs(), "tt.make_range");
  const bool lhsIsBound = valueDependsOn(comparison.getLhs(), bound);
  const bool rhsIsBound = valueDependsOn(comparison.getRhs(), bound);
  switch (comparison.getPredicate()) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return lhsIsRange && rhsIsBound;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return lhsIsBound && rhsIsRange;
  default:
    return false;
  }
}

struct MaskedAccumulatorDerivation {
  bool valid = false;
  bool hasMatchingLoad = false;
};

static MaskedAccumulatorDerivation
analyzeMaskedAccumulatorContribution(Value value, Value bound,
                                     unsigned depth = 0) {
  if (depth >= 24)
    return {};
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return {};
  const llvm::StringRef name = definition->getName().getStringRef();
  if (name == "arith.constant")
    return {true, false};
  if (name == "tt.load") {
    auto maskIndex = getMemoryMaskOperandIndex(definition);
    if (!maskIndex || *maskIndex >= definition->getNumOperands() ||
        definition->getNumOperands() <= *maskIndex + 1 ||
        !maskProvesPrefixBound(definition->getOperand(*maskIndex), bound) ||
        !isZeroConstant(
            definition->getOperand(definition->getNumOperands() - 1)))
      return {};
    return {true, true};
  }
  const bool allowed = name.starts_with("arith.") || name == "tt.splat" ||
                       name == "tt.broadcast" || name == "tt.expand_dims" ||
                       name == "tt.reshape" || name == "tt.trans";
  if (!allowed || definition->getNumOperands() == 0)
    return {};
  MaskedAccumulatorDerivation result{true, false};
  for (Value operand : definition->getOperands()) {
    auto child =
        analyzeMaskedAccumulatorContribution(operand, bound, depth + 1);
    if (!child.valid)
      return {};
    result.hasMatchingLoad |= child.hasMatchingLoad;
  }
  return result;
}

static std::optional<int64_t>
getOneTripMaskedAccumulatorExtent(Operation *reduction,
                                  int64_t physicalExtent) {
  if (!reduction || reduction->getNumOperands() == 0 || physicalExtent <= 1)
    return std::nullopt;
  auto loopResult = dyn_cast<OpResult>(reduction->getOperand(0));
  if (!loopResult)
    return std::nullopt;
  Operation *loop = loopResult.getOwner();
  if (!loop || loop->getName().getStringRef() != "scf.for" ||
      loop->getNumOperands() < 4 || loop->getNumRegions() != 1 ||
      loop->getRegion(0).empty())
    return std::nullopt;
  const unsigned resultNumber = loopResult.getResultNumber();
  if (3 + resultNumber >= loop->getNumOperands() ||
      !isZeroConstant(loop->getOperand(3 + resultNumber)))
    return std::nullopt;
  const auto lower = getConstantInteger(loop->getOperand(0));
  const auto upper = getConstantInteger(loop->getOperand(1));
  const auto step = getConstantInteger(loop->getOperand(2));
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  const uint64_t distance = uint64_t(*upper) - uint64_t(*lower);
  if (distance > uint64_t(*step) || distance > uint64_t(physicalExtent))
    return std::nullopt;
  Block &body = loop->getRegion(0).front();
  Operation *yield = body.getTerminator();
  if (!yield || yield->getName().getStringRef() != "scf.yield" ||
      resultNumber >= yield->getNumOperands() ||
      1 + resultNumber >= body.getNumArguments())
    return std::nullopt;
  Value yielded = yield->getOperand(resultNumber);
  Operation *addition = yielded.getDefiningOp();
  if (!addition ||
      (addition->getName().getStringRef() != "arith.addf" &&
       addition->getName().getStringRef() != "arith.addi") ||
      addition->getNumOperands() != 2)
    return std::nullopt;
  Value carried = body.getArgument(1 + resultNumber);
  Value contribution;
  if (addition->getOperand(0) == carried)
    contribution = addition->getOperand(1);
  else if (addition->getOperand(1) == carried)
    contribution = addition->getOperand(0);
  else
    return std::nullopt;
  auto derivation =
      analyzeMaskedAccumulatorContribution(contribution, loop->getOperand(1));
  if (!derivation.valid || !derivation.hasMatchingLoad)
    return std::nullopt;
  return static_cast<int64_t>(distance);
}

static void accumulateReductionWorkload(Operation *operation,
                                        StageWorkload &work) {
  if (operation->getNumOperands() == 0)
    return;
  auto input = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto axis = operation->getAttrOfType<IntegerAttr>("axis");
  if (!input || !input.hasStaticShape() || !axis || input.getRank() == 0)
    return;
  int64_t dimension = axis.getInt();
  if (dimension < 0)
    dimension += input.getRank();
  if (dimension < 0 || dimension >= input.getRank())
    return;
  const int64_t extent = input.getShape()[dimension];
  if (extent <= 1)
    return;
  if (operation->getName().getStringRef() == "tt.scan") {
    const int64_t totalElements = input.getNumElements();
    const int64_t independentSequences = totalElements / extent;
    int64_t axisStride = 1;
    for (int64_t axisIndex = dimension + 1; axisIndex < input.getRank();
         ++axisIndex)
      axisStride *= input.getShape()[axisIndex];
    const auto reverse = operation->getAttrOfType<mlir::BoolAttr>("reverse");
    if (independentSequences > 0)
      work.prefixScanCounts[{
          extent, independentSequences, axisStride,
          static_cast<int64_t>(operation->getNumResults()),
          reverse && reverse.getValue(), classifyScanCombine(operation),
          typeToString(input.getElementType()), input.getRank(), dimension,
          getScalarBitWidth(input.getElementType())}] += 1.0;
    return;
  }
  const int64_t independentRows = input.getNumElements() / extent;
  int64_t effectiveExtent = extent;
  if (auto active = getOneTripMaskedAccumulatorExtent(operation, extent)) {
    effectiveExtent = *active;
    work.oneTripMaskedAccumulatorReductionExtents[{extent, effectiveExtent}] +=
        1.0;
  }
  const double depth =
      effectiveExtent > 1
          ? std::ceil(std::log2(static_cast<double>(effectiveExtent)))
          : 0.0;
  if (auto reduction = analyzeReduction(operation))
    work.reductionWorkloads.push_back(std::move(*reduction));
  work.reductionOperations += 1.0;
  if (operation->getName().getStringRef() == "tt.reduce")
    work.reductionCombineClassCounts[classifyReductionCombine(operation)] +=
        1.0;
  work.shuffleLaneSteps += static_cast<double>(independentRows) *
                           static_cast<double>(effectiveExtent) * depth;
  const int64_t elementBits = getScalarBitWidth(input.getElementType());
  if (independentRows > 0 && elementBits > 0) {
    work.reductionGeometryCounts[{independentRows, extent, elementBits}] += 1.0;
    if (operation->getName().getStringRef() == "tt.reduce" &&
        operation->getNumResults() > 1)
      work.scalarizedMultiResultReductionCounts[{
          independentRows, extent,
          static_cast<int64_t>(operation->getNumResults()), elementBits}] +=
          1.0;
  }
}

static bool blockPointerCanUseFlatMemoryCurve(Value pointer, Type valueType);
static StageMemoryAccess describeMemoryAccess(Operation *operation,
                                              Value pointer, Type valueType,
                                              bool store);

static double getNonNegativeNumericAttr(Operation *operation,
                                        llvm::ArrayRef<llvm::StringRef> names) {
  for (llvm::StringRef name : names) {
    Attribute attr = operation->getAttr(name);
    if (auto integer = dyn_cast_or_null<IntegerAttr>(attr))
      return std::max(0.0, static_cast<double>(integer.getInt()));
    if (auto floating = dyn_cast_or_null<FloatAttr>(attr))
      return std::max(0.0, floating.getValueAsDouble());
  }
  return 0.0;
}

static bool hasNumericAttr(Operation *operation,
                           llvm::ArrayRef<llvm::StringRef> names) {
  for (llvm::StringRef name : names)
    if (operation->hasAttr(name))
      return true;
  return false;
}

static void accumulateGeneratedTraffic(Operation *operation,
                                       StageWorkload &work) {
  work.sharedLocalLoadInstructions += getNonNegativeNumericAttr(
      operation, {"ascend.simt.shared_local_load_instructions",
                  "ascend.simt.shared_local_load_warp_instructions"});
  work.sharedLocalStoreInstructions += getNonNegativeNumericAttr(
      operation, {"ascend.simt.shared_local_store_instructions",
                  "ascend.simt.shared_local_store_warp_instructions"});
  work.privateStackLoadInstructions += getNonNegativeNumericAttr(
      operation, {"ascend.simt.private_stack_load_instructions",
                  "ascend.simt.private_stack_load_warp_instructions"});
  work.privateStackStoreInstructions += getNonNegativeNumericAttr(
      operation, {"ascend.simt.private_stack_store_instructions",
                  "ascend.simt.private_stack_store_warp_instructions"});
}

static bool isScalarLoadOperation(Operation *operation);
static bool isScalarStoreOperation(Operation *operation);

static std::string getAtomicEnumName(Operation *operation,
                                     llvm::StringRef attribute,
                                     llvm::ArrayRef<llvm::StringRef> names) {
  auto value = operation->getAttrOfType<IntegerAttr>(attribute);
  if (!value || value.getInt() < 0 ||
      static_cast<size_t>(value.getInt()) >= names.size() ||
      names[value.getInt()].empty())
    return "unknown";
  return names[value.getInt()].str();
}

static AtomicWorkload getAtomicWorkload(Operation *operation) {
  static const llvm::StringRef rmwNames[] = {"",     "and",  "or",  "xor",
                                             "add",  "fadd", "max", "min",
                                             "umax", "umin", "exch"};
  static const llvm::StringRef semanticNames[] = {"", "relaxed", "acquire",
                                                  "release", "acq_rel"};
  static const llvm::StringRef scopeNames[] = {"", "gpu", "cta", "sys"};
  const llvm::StringRef name = operation->getName().getStringRef();
  AtomicWorkload atomic;
  atomic.kind = name == "tt.atomic_cas"
                    ? "cas"
                    : getAtomicEnumName(operation, "atomic_rmw_op", rmwNames);
  atomic.memorySemantic = getAtomicEnumName(operation, "sem", semanticNames);
  atomic.memoryScope = getAtomicEnumName(operation, "scope", scopeNames);
  const unsigned valueOperand = name == "tt.atomic_cas" ? 2 : 1;
  Value value = operation->getNumOperands() > valueOperand
                    ? operation->getOperand(valueOperand)
                    : Value{};
  Type elementType = value ? getScalarElementType(value.getType()) : Type{};
  atomic.dataType = elementType ? typeToString(elementType) : "unknown";
  atomic.logicalElements = value ? getTypeElementCount(value.getType()) : 0.0;
  atomic.logicalOperationInstances = 1.0;
  atomic.resultUsed =
      operation->getNumResults() > 0 && !operation->getResult(0).use_empty();
  atomic.addressDependsOnLoadedIndex =
      isLoadedIndexDependentMemoryOp(operation);
  if (operation->getNumOperands() > 0 && value) {
    const StageMemoryAccess access = describeMemoryAccess(
        operation, operation->getOperand(0), value.getType(), /*store=*/true);
    atomic.provenContiguousRunWidth =
        static_cast<double>(access.provenContiguousElements());
  }
  atomic.contentionUnknown = true;
  return atomic;
}

static bool isDiagnosticAssertionSink(Operation *operation) {
  if (!operation)
    return false;
  const llvm::StringRef name = operation->getName().getStringRef();
  return name == "tt.assert" || name == "cf.assert";
}

static bool isDiagnosticAssertionSupport(Operation *operation) {
  if (!operation || operation->getNumRegions() != 0)
    return false;
  return llvm::StringSwitch<bool>(operation->getName().getStringRef())
      .Cases("arith.constant", "arith.extsi", "arith.extui", true)
      .Cases("arith.trunci", "arith.index_cast", "arith.index_castui", true)
      .Cases("arith.addi", "arith.subi", "arith.muli", true)
      .Cases("arith.cmpi", "arith.andi", "arith.ori", true)
      .Cases("arith.xori", "arith.select", "tt.splat", true)
      .Case("tt.broadcast", true)
      .Default(false);
}

static bool isElidedDiagnosticOnlyOperationImpl(
    Operation *operation, llvm::SmallPtrSetImpl<Operation *> &visiting) {
  if (isDiagnosticAssertionSink(operation))
    return true;
  if (!isDiagnosticAssertionSupport(operation) ||
      operation->getNumResults() == 0 || !visiting.insert(operation).second)
    return false;

  bool hasUse = false;
  for (Value result : operation->getResults()) {
    for (OpOperand &use : result.getUses()) {
      hasUse = true;
      if (!isElidedDiagnosticOnlyOperationImpl(use.getOwner(), visiting)) {
        visiting.erase(operation);
        return false;
      }
    }
  }
  visiting.erase(operation);
  return hasUse;
}

static bool isElidedDiagnosticOnlyOperation(Operation *operation) {
  llvm::SmallPtrSet<Operation *, 16> visiting;
  return isElidedDiagnosticOnlyOperationImpl(operation, visiting);
}

struct FoldedAddressProof {
  bool valid = false;
  bool reachesFlatMemory = false;
};

static bool isStaticAddressForwardingOperation(Operation *operation) {
  if (!operation || operation->getNumRegions() != 0)
    return false;
  return llvm::StringSwitch<bool>(operation->getName().getStringRef())
      .Cases("arith.constant", "arith.extsi", "arith.extui", true)
      .Cases("arith.trunci", "arith.index_cast", "arith.index_castui", true)
      .Cases("arith.addi", "arith.subi", "arith.muli", true)
      .Cases("tt.make_range", "tt.splat", "tt.broadcast", true)
      .Cases("tt.expand_dims", "tt.addptr", "tt.advance", true)
      .Default(false);
}

/// Prove that a TTIR value is only static affine address scaffolding for a flat
/// contiguous transfer.
static FoldedAddressProof
proveFoldedFlatAddressUses(Operation *operation,
                           llvm::SmallPtrSetImpl<Operation *> &visiting) {
  if (!isStaticAddressForwardingOperation(operation) ||
      operation->getNumResults() == 0 || !visiting.insert(operation).second)
    return {};

  bool hasUse = false;
  bool reachesFlatMemory = false;
  for (Value result : operation->getResults()) {
    for (OpOperand &use : result.getUses()) {
      hasUse = true;
      Operation *user = use.getOwner();
      const llvm::StringRef name = user->getName().getStringRef();
      if (isElidedDiagnosticOnlyOperation(user))
        continue;
      if ((name == "tt.load" || name == "tt.store") &&
          use.getOperandNumber() == 0) {
        Type valueType;
        const bool store = name == "tt.store";
        if (store && user->getNumOperands() > 1)
          valueType = user->getOperand(1).getType();
        else if (!store && user->getNumResults() > 0)
          valueType = user->getResult(0).getType();
        if (!valueType) {
          visiting.erase(operation);
          return {};
        }
        const StageMemoryAccess access =
            describeMemoryAccess(user, user->getOperand(0), valueType, store);
        if (access.addressDependsOnLoadedIndex || !access.isFlatContiguous()) {
          visiting.erase(operation);
          return {};
        }
        reachesFlatMemory = true;
        continue;
      }
      const FoldedAddressProof child =
          proveFoldedFlatAddressUses(user, visiting);
      if (!child.valid) {
        visiting.erase(operation);
        return {};
      }
      reachesFlatMemory |= child.reachesFlatMemory;
    }
  }
  visiting.erase(operation);
  return {hasUse && reachesFlatMemory, reachesFlatMemory};
}

static bool isFoldedFlatAddressScaffolding(Operation *operation) {
  llvm::SmallPtrSet<Operation *, 16> visiting;
  return proveFoldedFlatAddressUses(operation, visiting).valid;
}

static void accumulateOneOperation(
    Operation *operation, StageWorkload &work,
    RankedTensorType mappedTensor = {},
    const StageMemoryPatternAnalysis *memoryPatterns = nullptr) {
  if (!operation || operation->hasTrait<OpTrait::IsTerminator>())
    return;
  const llvm::StringRef name = operation->getName().getStringRef();
  if (name.ends_with(".return") || name.ends_with(".yield"))
    return;
  if (isElidedDiagnosticOnlyOperation(operation))
    return;
  accumulateGeneratedTraffic(operation, work);
  if (name == "tt.reshape" || name == "tt.expand_dims" ||
      name == "tt.map_elementwise")
    return;
  const double elements = getOperationElements(operation);
  const double executionElements =
      mappedTensor ? static_cast<double>(mappedTensor.getNumElements())
                   : elements;
  if (isFoldedFlatAddressScaffolding(operation)) {
    work.operationCounts["simd.folded_flat_address"]
                        [static_cast<int64_t>(elements)] += 1.0;
    return;
  }

  // Some post-layout IRs expose shared/local traffic directly.
  const double localWarpInstructions = std::ceil(elements / 32.0);
  if ((name == "ttg.local_load" || name == "hivm.hir.local_load") &&
      !hasNumericAttr(operation,
                      {"ascend.simt.shared_local_load_instructions",
                       "ascend.simt.shared_local_load_warp_instructions"})) {
    work.sharedLocalLoadInstructions += localWarpInstructions;
    return;
  }
  if ((name == "ttg.local_store" || name == "hivm.hir.local_store") &&
      !hasNumericAttr(operation,
                      {"ascend.simt.shared_local_store_instructions",
                       "ascend.simt.shared_local_store_warp_instructions"})) {
    work.sharedLocalStoreInstructions += localWarpInstructions;
    return;
  }

  if ((name == "tt.load" || name == "tt.gather") &&
      operation->getNumResults() > 0) {
    Value result = operation->getResult(0);
    if (isScalarLoadOperation(operation)) {
      work.scalarLoadCount += 1.0;
      return;
    }
    const int64_t warpInstructions =
        static_cast<int64_t>(std::ceil(elements / 32.0));
    const double bytes = getValueBytes(result);
    work.loadBytes += bytes;
    work.loadOperations += 1.0;
    const auto partial = getPartialContinuousTile(operation, memoryPatterns);
    const double logicalMemoryGroups = warpInstructions;
    // Keep #2528's tensor-wide instruction accounting for resource pricing;
    // row geometry below is diagnostic, not a different execution count.
    work.loadWarpInstructions += warpInstructions;
    if (partial) {
      work.partialContinuousLoadRows += partial->rows;
      work.partialContinuousLoadBytes += bytes;
      work.partialContinuousLoadWarpInstructions += logicalMemoryGroups;
    } else if (name == "tt.gather" ||
               isIndirectMemoryOperation(operation, memoryPatterns)) {
      work.indirectLoadBytes += bytes;
      work.indirectLoadTransactions += logicalMemoryGroups;
    }
    work.memoryAccesses.push_back(
        describeMemoryAccess(operation, operation->getOperand(0),
                             result.getType(), /*store=*/false));
    // Credit a load only if all consumers use a supported dot operand path
    // in the same dynamic region. An escape or non-dot use retains GM cost.
    auto &paths = work.memoryAccesses.back().dotConsumerPaths;
    for (OpOperand &use : result.getUses()) {
      Operation *consumer = use.getOwner();
      auto path = describeDotOperandPath(consumer);
      if (consumer->getBlock() != operation->getBlock() ||
          use.getOperandNumber() >= 2 || !path) {
        paths.clear();
        break;
      }
      paths.push_back(*path);
    }
    const bool indirect =
        name == "tt.gather" || isLoadedIndexDependentMemoryOp(operation);
    work.memoryAccesses.back().addressDependsOnLoadedIndex = indirect;
    const int64_t contiguousElements =
        work.memoryAccesses.back().provenContiguousElements();
    if (indirect && contiguousElements > 1)
      work.segmentedIndirectLoadCounts[{
          static_cast<int64_t>(std::llround(bytes)),
          static_cast<int64_t>(elements), contiguousElements}] += 1.0;
    else if (indirect)
      work.indirectLoadCounts[{static_cast<int64_t>(std::llround(bytes)),
                               static_cast<int64_t>(elements)}] += 1.0;
    else if (blockPointerCanUseFlatMemoryCurve(operation->getOperand(0),
                                               result.getType()))
      work.continuousLoadCounts[{static_cast<int64_t>(std::llround(bytes)),
                                 static_cast<int64_t>(elements)}] += 1.0;
    return;
  }
  if (name == "tt.store" && operation->getNumOperands() > 1) {
    Value value = operation->getOperand(1);
    if (isScalarStoreOperation(operation)) {
      work.scalarStoreCount += 1.0;
      return;
    }
    work.storeBytes += getValueBytes(value);
    work.storeOperations += 1.0;
    const int64_t warpInstructions = static_cast<int64_t>(
        std::ceil(getTypeElementCount(value.getType()) / 32.0));
    const auto partial = getPartialContinuousTile(operation, memoryPatterns);
    const double logicalMemoryGroups = warpInstructions;
    work.storeWarpInstructions += warpInstructions;
    if (partial) {
      work.partialContinuousStoreRows += partial->rows;
      work.partialContinuousStoreBytes += getValueBytes(value);
      work.partialContinuousStoreWarpInstructions += logicalMemoryGroups;
    } else if (isIndirectMemoryOperation(operation, memoryPatterns)) {
      work.indirectStoreBytes += getValueBytes(value);
      work.indirectStoreTransactions += logicalMemoryGroups;
    }
    work.memoryAccesses.push_back(describeMemoryAccess(
        operation, operation->getOperand(0), value.getType(), /*store=*/true));
    const bool indirect = isLoadedIndexDependentMemoryOp(operation);
    work.memoryAccesses.back().addressDependsOnLoadedIndex = indirect;
    const int64_t contiguousElements =
        work.memoryAccesses.back().provenContiguousElements();
    const int64_t bytes =
        static_cast<int64_t>(std::llround(getValueBytes(value)));
    const int64_t valueElements =
        static_cast<int64_t>(getTypeElementCount(value.getType()));
    if (indirect && contiguousElements > 1)
      work.segmentedIndirectStoreCounts[{bytes, valueElements,
                                         contiguousElements}] += 1.0;
    else if (indirect)
      work.indirectStoreCounts[{bytes, valueElements}] += 1.0;
    else if (blockPointerCanUseFlatMemoryCurve(operation->getOperand(0),
                                               value.getType()))
      work.continuousStoreCounts[{bytes, valueElements}] += 1.0;
    return;
  }
  if (name.starts_with("tt.atomic")) {
    work.atomicWorkloads.push_back(getAtomicWorkload(operation));
    return;
  }
  if (isDotOperationName(name)) {
    accumulateDotWorkload(operation, work);
    return;
  }
  if (name == "tt.reduce" || name == "tt.scan") {
    accumulateReductionWorkload(operation, work);
    if (name == "tt.scan")
      return;
  }
  if (name == "ascend.sort" && operation->getNumOperands() == 1) {
    auto input = dyn_cast<ShapedType>(operation->getOperand(0).getType());
    auto axis = operation->getAttrOfType<IntegerAttr>("dim");
    if (input && input.hasStaticShape() && axis && input.getRank() > 0) {
      int64_t dimension = axis.getInt();
      if (dimension < 0)
        dimension += input.getRank();
      if (dimension >= 0 && dimension < input.getRank()) {
        const int64_t extent = input.getShape()[dimension];
        const int64_t rows = extent > 0 ? input.getNumElements() / extent : 0;
        const int64_t bits = getScalarBitWidth(input.getElementType());
        if (rows > 0 && extent > 1 && bits > 0)
          work.sortCounts[{rows, extent, bits}] += 1.0;
      }
    }
  }
  if (name == "arith.cmpi" || name == "arith.cmpf" || name == "math.isfinite" ||
      name == "math.isnan" || name == "math.isinf") {
    work.predicateCounts[static_cast<int64_t>(executionElements)] += 1.0;
    accumulateTensorOperationWorkload(operation, "predicate.cmp",
                                      executionElements, work, mappedTensor);
    return;
  }
  if (name == "scf.for" || name == "scf.if" || name == "scf.while")
    return;

  if (name == "tt.trans" && operation->getNumOperands() > 0) {
    auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    auto order = operation->getAttrOfType<DenseI32ArrayAttr>("order");
    if (input && input.hasStaticShape() && input.getRank() > 1 && order &&
        static_cast<int64_t>(order.size()) == input.getRank()) {
      bool identity = true;
      for (auto indexed : llvm::enumerate(order.asArrayRef()))
        identity &= indexed.value() == static_cast<int32_t>(indexed.index());
      if (!identity) {
        // The lowering exchanges every logical element across the moved
        // innermost axis.
        const int64_t inner = input.getDimSize(input.getRank() - 1);
        const int64_t logical = input.getNumElements();
        if (inner > 0 && logical > 0 &&
            logical <= std::numeric_limits<int64_t>::max() / inner) {
          work.operationCounts["layout.transpose"][logical * inner] += 1.0;
          return;
        }
      }
    }
  }

  const std::string profileName = getProfileOperationName(operation);
  const bool hasTensorExecutionGeometry =
      hasTensorResult(operation) ||
      (profileName != "generic.issue" &&
       static_cast<bool>(getRepresentativeTensorType(operation)));
  if (!mappedTensor && !hasTensorExecutionGeometry) {
    if (profileName != "generic.issue")
      work.operationCounts[profileName][1] += 1.0;
    else
      work.scalarOperations += 1.0;
    return;
  }
  work.operationCounts[profileName][static_cast<int64_t>(executionElements)] +=
      1.0;
  accumulateTensorOperationWorkload(operation, profileName, executionElements,
                                    work, mappedTensor);
}

static void mergeWorkload(StageWorkload &into, StageWorkload from);

static constexpr double StageWorkload::*workloadScalarFields[] = {
    &StageWorkload::scalarOperations,
    &StageWorkload::loadBytes,
    &StageWorkload::storeBytes,
    &StageWorkload::loadOperations,
    &StageWorkload::storeOperations,
    &StageWorkload::loadWarpInstructions,
    &StageWorkload::storeWarpInstructions,
    &StageWorkload::indirectLoadBytes,
    &StageWorkload::indirectStoreBytes,
    &StageWorkload::indirectLoadTransactions,
    &StageWorkload::indirectStoreTransactions,
    &StageWorkload::partialContinuousLoadRows,
    &StageWorkload::partialContinuousStoreRows,
    &StageWorkload::partialContinuousLoadBytes,
    &StageWorkload::partialContinuousStoreBytes,
    &StageWorkload::partialContinuousLoadWarpInstructions,
    &StageWorkload::partialContinuousStoreWarpInstructions,
    &StageWorkload::sharedLocalLoadInstructions,
    &StageWorkload::sharedLocalStoreInstructions,
    &StageWorkload::privateStackLoadInstructions,
    &StageWorkload::privateStackStoreInstructions,
    &StageWorkload::reductionOperations,
    &StageWorkload::shuffleLaneSteps,
    &StageWorkload::dotOperations,
    &StageWorkload::dotFlops,
    &StageWorkload::loopBackedges,
    &StageWorkload::conditionalBranches,
    &StageWorkload::divergentBranches,
    &StageWorkload::synchronizations,
    &StageWorkload::scalarLoadCount,
    &StageWorkload::scalarStoreCount,
};

template <typename Map> static void scaleCounts(Map &counts, double scale) {
  for (auto &entry : counts)
    entry.second *= scale;
}

template <typename Map> static void mergeCounts(Map &into, const Map &from) {
  for (const auto &[key, count] : from)
    into[key] += count;
}

static void scaleWorkload(StageWorkload &work, double scale) {
  for (auto &reduction : work.reductionWorkloads)
    reduction.instances *= scale;
  for (auto field : workloadScalarFields)
    work.*field *= scale;
  for (auto &access : work.memoryAccesses)
    access.count *= scale;
  for (AtomicWorkload &atomic : work.atomicWorkloads) {
    atomic.logicalElements *= scale;
    atomic.logicalOperationInstances *= scale;
  }
  scaleCounts(work.indirectLoadCounts, scale);
  scaleCounts(work.segmentedIndirectLoadCounts, scale);
  scaleCounts(work.indirectStoreCounts, scale);
  scaleCounts(work.segmentedIndirectStoreCounts, scale);
  scaleCounts(work.continuousLoadCounts, scale);
  scaleCounts(work.continuousStoreCounts, scale);
  scaleCounts(work.predicateCounts, scale);
  scaleCounts(work.prefixScanCounts, scale);
  scaleCounts(work.reductionGeometryCounts, scale);
  scaleCounts(work.reductionCombineClassCounts, scale);
  scaleCounts(work.scalarizedMultiResultReductionCounts, scale);
  scaleCounts(work.sortCounts, scale);
  scaleCounts(work.oneTripMaskedAccumulatorReductionExtents, scale);
  scaleCounts(work.dotFlopCounts, scale);
  scaleCounts(work.dotShapeCounts, scale);
  for (auto &[key, entry] : work.dotOperandPaths)
    entry.second *= scale;
  for (auto &entry : work.operationCounts)
    for (auto &[size, count] : entry.second)
      count *= scale;
  for (TensorOperationWorkload &tensor : work.tensorOperationWorkloads) {
    tensor.logicalElements *= scale;
    tensor.segmentCount *= scale;
  }
  recomputeIssueElements(work);
}

static std::optional<int64_t> getConstantInteger(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto function = dyn_cast_or_null<FunctionOpInterface>(
            argument.getOwner()->getParentOp()))
      if (auto attribute = function.getArgAttrOfType<IntegerAttr>(
              argument.getArgNumber(), "ascend.costmodel.runtime_value"))
        return attribute.getInt();
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  auto attribute = definition->getAttrOfType<IntegerAttr>("value");
  if (!attribute || attribute.getValue().getBitWidth() > 64)
    return std::nullopt;
  return attribute.getInt();
}

static std::optional<int64_t> getSplatConstantInteger(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return getConstantInteger(value);
  if (definition->getName().getStringRef() == "tt.splat" &&
      definition->getNumOperands() == 1)
    return getConstantInteger(definition->getOperand(0));
  Attribute attribute = definition->getAttr("value");
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attribute))
    return integer.getInt();
  if (auto dense = dyn_cast_or_null<DenseIntElementsAttr>(attribute)) {
    if (!dense.isSplat())
      return std::nullopt;
    return dense.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

using TensorAffineStrides = llvm::SmallVector<std::optional<int64_t>, 4>;

/// Infer the coefficient of each tensor index in a tensor-of-pointer offset.
static std::optional<TensorAffineStrides>
inferTensorAffineStrides(Value value, unsigned depth = 0) {
  auto outputType = dyn_cast<RankedTensorType>(value.getType());
  if (!outputType || !outputType.hasStaticShape() || depth >= 32)
    return std::nullopt;
  const int64_t rank = outputType.getRank();
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  const llvm::StringRef name = definition->getName().getStringRef();

  if (name == "tt.make_range") {
    if (rank != 1)
      return std::nullopt;
    return TensorAffineStrides{int64_t(1)};
  }
  if (name == "tt.splat" || name == "arith.constant")
    return TensorAffineStrides(rank, int64_t(0));
  if (name == "tt.load" || name == "tt.gather") {
    TensorAffineStrides result;
    result.reserve(rank);
    for (int64_t extent : outputType.getShape())
      result.push_back(extent == 1 ? std::optional<int64_t>(int64_t(0))
                                   : std::nullopt);
    return result;
  }

  const bool shapePreservingCast =
      name == "arith.extsi" || name == "arith.extui" ||
      name == "arith.trunci" || name == "arith.index_cast" ||
      name == "arith.index_castui" || name == "arith.sitofp" ||
      name == "arith.uitofp" || name == "arith.fptosi" ||
      name == "arith.fptoui";
  if (shapePreservingCast && definition->getNumOperands() == 1) {
    auto input =
        dyn_cast<RankedTensorType>(definition->getOperand(0).getType());
    if (!input || input.getShape() != outputType.getShape())
      return std::nullopt;
    return inferTensorAffineStrides(definition->getOperand(0), depth + 1);
  }

  if (name == "tt.expand_dims" && definition->getNumOperands() == 1) {
    auto axis = definition->getAttrOfType<IntegerAttr>("axis");
    auto input = inferTensorAffineStrides(definition->getOperand(0), depth + 1);
    if (!axis || !input || static_cast<int64_t>(input->size()) + 1 != rank ||
        axis.getInt() < 0 || axis.getInt() >= rank)
      return std::nullopt;
    input->insert(input->begin() + axis.getInt(), int64_t(0));
    return input;
  }

  if (name == "tt.broadcast" && definition->getNumOperands() == 1) {
    auto inputType =
        dyn_cast<RankedTensorType>(definition->getOperand(0).getType());
    auto input = inferTensorAffineStrides(definition->getOperand(0), depth + 1);
    if (!inputType || !inputType.hasStaticShape() || !input ||
        inputType.getRank() != rank)
      return std::nullopt;
    for (int64_t axis = 0; axis < rank; ++axis) {
      const int64_t from = inputType.getDimSize(axis);
      const int64_t to = outputType.getDimSize(axis);
      if (from == to)
        continue;
      if (from != 1 || to <= 0)
        return std::nullopt;
      (*input)[axis] = int64_t(0);
    }
    return input;
  }

  auto combine = [&](bool subtract) -> std::optional<TensorAffineStrides> {
    if (definition->getNumOperands() != 2)
      return std::nullopt;
    auto lhs = inferTensorAffineStrides(definition->getOperand(0), depth + 1);
    auto rhs = inferTensorAffineStrides(definition->getOperand(1), depth + 1);
    if (!lhs || !rhs || lhs->size() != rhs->size() ||
        static_cast<int64_t>(lhs->size()) != rank)
      return std::nullopt;
    TensorAffineStrides result;
    result.reserve(rank);
    for (int64_t axis = 0; axis < rank; ++axis) {
      if (!(*lhs)[axis] || !(*rhs)[axis]) {
        result.push_back(std::nullopt);
        continue;
      }
      const int64_t left = *(*lhs)[axis];
      const int64_t right = *(*rhs)[axis];
      int64_t coefficient = 0;
      if (subtract ? llvm::SubOverflow(left, right, coefficient)
                   : llvm::AddOverflow(left, right, coefficient))
        result.push_back(std::nullopt);
      else
        result.push_back(coefficient);
    }
    return result;
  };
  if (name == "arith.addi" || name == "tt.addptr")
    return combine(false);
  if (name == "arith.subi")
    return combine(true);

  if (name == "arith.muli" && definition->getNumOperands() == 2) {
    std::optional<int64_t> scale =
        getSplatConstantInteger(definition->getOperand(0));
    Value varying = definition->getOperand(1);
    if (!scale) {
      scale = getSplatConstantInteger(definition->getOperand(1));
      varying = definition->getOperand(0);
    }
    auto input = inferTensorAffineStrides(varying, depth + 1);
    if (scale && input && static_cast<int64_t>(input->size()) == rank) {
      for (auto &coefficient : *input) {
        if (!coefficient)
          continue;
        int64_t product = 0;
        if (llvm::MulOverflow(*coefficient, *scale, product))
          coefficient = std::nullopt;
        else
          coefficient = product;
      }
      return input;
    }

    auto lhs = inferTensorAffineStrides(definition->getOperand(0), depth + 1);
    auto rhs = inferTensorAffineStrides(definition->getOperand(1), depth + 1);
    auto isUniform = [rank](const auto &coefficients) {
      return coefficients &&
             static_cast<int64_t>(coefficients->size()) == rank &&
             llvm::all_of(*coefficients, [](const auto &coefficient) {
               return coefficient && *coefficient == 0;
             });
    };
    const TensorAffineStrides *varyingCoefficients = nullptr;
    if (isUniform(lhs) && rhs && static_cast<int64_t>(rhs->size()) == rank)
      varyingCoefficients = &*rhs;
    else if (isUniform(rhs) && lhs && static_cast<int64_t>(lhs->size()) == rank)
      varyingCoefficients = &*lhs;
    if (!varyingCoefficients)
      return std::nullopt;
    TensorAffineStrides result;
    result.reserve(rank);
    for (const auto &coefficient : *varyingCoefficients)
      result.push_back(coefficient && *coefficient == 0
                           ? std::optional<int64_t>(int64_t(0))
                           : std::nullopt);
    return result;
  }
  return std::nullopt;
}

/// Detect address expressions that the SIMD backend currently expands into a
/// scalar lane loop.
static bool addressHasNonAffineTensorComputation(
    Value value, llvm::SmallPtrSetImpl<Operation *> &visited,
    unsigned depth = 0) {
  if (depth >= 32)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  const llvm::StringRef name = definition->getName().getStringRef();
  if (name == "tt.load" || name == "tt.gather")
    return false;

  const bool tensorResult = llvm::any_of(definition->getResults(), [](Value r) {
    return isa<RankedTensorType>(r.getType());
  });
  if (tensorResult &&
      (name == "math.floor" || name == "math.ceil" || name == "math.round" ||
       name == "tt.clampf" || name == "arith.mulf" || name == "arith.divf" ||
       name == "arith.remf" || name == "arith.fptosi" ||
       name == "arith.fptoui"))
    return true;

  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return addressHasNonAffineTensorComputation(operand, visited, depth + 1);
  });
}

static bool addressHasNonAffineTensorComputation(Value value) {
  llvm::SmallPtrSet<Operation *, 32> visited;
  return addressHasNonAffineTensorComputation(value, visited);
}

static bool addressHasTensorControlFlowSelection(
    Value value, llvm::SmallPtrSetImpl<Operation *> &visited,
    unsigned depth = 0) {
  if (depth >= 32)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  const llvm::StringRef name = definition->getName().getStringRef();
  if (name == "tt.load" || name == "tt.gather")
    return false;
  const bool tensorResult = llvm::any_of(definition->getResults(), [](Value r) {
    return isa<RankedTensorType>(r.getType());
  });
  if (tensorResult && name == "scf.if")
    return true;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return addressHasTensorControlFlowSelection(operand, visited, depth + 1);
  });
}

static bool addressHasTensorControlFlowSelection(Value value) {
  llvm::SmallPtrSet<Operation *, 32> visited;
  return addressHasTensorControlFlowSelection(value, visited);
}

static int64_t getElementBits(Type type) {
  type = getScalarElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  if (type.isIndex())
    return 64;
  return 0;
}

static bool isPredicateType(Type type) {
  type = getScalarElementType(type);
  auto integer = dyn_cast<IntegerType>(type);
  return integer && integer.getWidth() == 1;
}

/// A shape-preserving elementwise operation inherits its nonconstant tensor
/// inputs' layout.
static bool fromDotAccumulator(Value value, unsigned depth = 0) {
  Operation *producer = value.getDefiningOp();
  if (!producer || depth >= 16)
    return false;
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || !tensor.hasStaticShape())
    return false;
  llvm::StringRef name = producer->getName().getStringRef();
  if (isDotOperationName(name))
    return true;
  if (name != "arith.addf" && name != "arith.subf" && name != "arith.mulf" &&
      name != "arith.negf")
    return false;
  bool foundDot = false;
  for (Value input : producer->getOperands()) {
    if (!isa<RankedTensorType>(input.getType()))
      continue;
    if (input.getType() != value.getType())
      return false;
    Operation *definition = input.getDefiningOp();
    if (definition && definition->getName().getStringRef() == "arith.constant")
      continue;
    if (!fromDotAccumulator(input, depth + 1))
      return false;
    foundDot = true;
  }
  return foundDot;
}

static Value stripPredicateShapeOps(Value value) {
  for (unsigned depth = 0; depth < 16; ++depth) {
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getNumOperands() != 1)
      break;
    const llvm::StringRef name = definition->getName().getStringRef();
    if (name != "tt.broadcast" && name != "tt.expand_dims" &&
        name != "tt.splat")
      break;
    value = definition->getOperand(0);
  }
  return value;
}

static std::optional<std::pair<int64_t, int64_t>> getStaticRange(Value value) {
  Operation *definition = stripPredicateShapeOps(value).getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "tt.make_range")
    return std::nullopt;
  auto start = definition->getAttrOfType<IntegerAttr>("start");
  auto end = definition->getAttrOfType<IntegerAttr>("end");
  if (!start || !end || end.getInt() <= start.getInt())
    return std::nullopt;
  return std::make_pair(start.getInt(), end.getInt());
}

static std::optional<int64_t> getStaticSplatInteger(Value value) {
  Operation *definition = stripPredicateShapeOps(value).getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "arith.constant")
    return std::nullopt;
  Attribute attribute = definition->getAttr("value");
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attribute))
    return integer.getInt();
  auto dense = dyn_cast_or_null<DenseIntElementsAttr>(attribute);
  if (!dense || !dense.isSplat())
    return std::nullopt;
  return dense.getSplatValue<llvm::APInt>().getSExtValue();
}

static arith::CmpIPredicate reversePredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return predicate;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unknown integer comparison predicate");
}

static std::optional<double> staticPredicateTrueRatio(Value value,
                                                      unsigned depth = 0) {
  if (depth >= 16)
    return std::nullopt;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  const llvm::StringRef name = definition->getName().getStringRef();
  if ((name == "tt.broadcast" || name == "tt.expand_dims") &&
      definition->getNumOperands() == 1)
    return staticPredicateTrueRatio(definition->getOperand(0), depth + 1);
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition)) {
    auto left = staticPredicateTrueRatio(conjunction.getLhs(), depth + 1);
    auto right = staticPredicateTrueRatio(conjunction.getRhs(), depth + 1);
    if (left && right)
      return std::min(*left, *right);
    return left ? left : right;
  }
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison)
    return std::nullopt;
  auto range = getStaticRange(comparison.getLhs());
  auto bound = getStaticSplatInteger(comparison.getRhs());
  arith::CmpIPredicate predicate = comparison.getPredicate();
  if (!range || !bound) {
    range = getStaticRange(comparison.getRhs());
    bound = getStaticSplatInteger(comparison.getLhs());
    predicate = reversePredicate(predicate);
  }
  if (!range || !bound)
    return std::nullopt;
  const auto [start, end] = *range;
  const int64_t count = end - start;
  int64_t active = 0;
  switch (predicate) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    active = std::clamp(*bound - start, int64_t(0), count);
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    active = std::clamp(*bound - start + 1, int64_t(0), count);
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    active = std::clamp(end - *bound - 1, int64_t(0), count);
    break;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    active = std::clamp(end - *bound, int64_t(0), count);
    break;
  case arith::CmpIPredicate::eq:
    active = *bound >= start && *bound < end ? 1 : 0;
    break;
  case arith::CmpIPredicate::ne:
    active = count - (*bound >= start && *bound < end ? 1 : 0);
    break;
  }
  return static_cast<double>(active) / static_cast<double>(count);
}

static bool
valueDependsOnLoadWithExactMask(Value value, Value mask,
                                llvm::SmallPtrSetImpl<Operation *> &visited,
                                unsigned depth = 0) {
  if (depth >= 32)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  const llvm::StringRef name = definition->getName().getStringRef();
  if (name == "tt.load") {
    auto maskIndex = getMemoryMaskOperandIndex(definition);
    return maskIndex && *maskIndex < definition->getNumOperands() &&
           definition->getOperand(*maskIndex) == mask;
  }
  if (name == "tt.store" || name.starts_with("tt.atomic"))
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOnLoadWithExactMask(operand, mask, visited, depth + 1);
  });
}

static bool valueDependsOnLoadWithExactMask(Value value, Value mask) {
  llvm::SmallPtrSet<Operation *, 32> visited;
  return valueDependsOnLoadWithExactMask(value, mask, visited);
}

static StageMemoryAccess describeMemoryAccess(Operation *operation,
                                              Value pointer, Type valueType,
                                              bool store) {
  StageMemoryAccess access;
  access.store = store;
  access.addressDependsOnLoadedIndex =
      operation->getName().getStringRef() == "tt.gather" ||
      isLoadedIndexDependentMemoryOp(operation);
  access.fromDotAccumulator =
      store && operation->getName().getStringRef() == "tt.store" &&
      fromDotAccumulator(operation->getOperand(1));
  access.elementBits = getElementBits(valueType);
  if (auto shaped = dyn_cast<ShapedType>(valueType))
    if (shaped.hasStaticShape())
      access.shape.assign(shaped.getShape().begin(), shaped.getShape().end());
  if (access.shape.empty())
    access.shape.push_back(1);

  Operation *definition = pointer.getDefiningOp();
  while (definition && definition->getName().getStringRef() == "tt.advance") {
    pointer = definition->getOperand(0);
    definition = pointer.getDefiningOp();
  }
  const int64_t rank = static_cast<int64_t>(access.shape.size());
  access.strides.resize(access.shape.size());
  bool hasCompleteAffineStrideProof = false;
  if (definition &&
      definition->getName().getStringRef() == "tt.make_tensor_ptr" &&
      definition->getNumOperands() == 1 + 3 * rank) {
    for (int64_t axis = 0; axis < rank; ++axis)
      access.strides[axis] =
          getConstantInteger(definition->getOperand(1 + rank + axis));
    hasCompleteAffineStrideProof = llvm::all_of(
        access.strides, [](const auto &stride) { return stride.has_value(); });
  } else if (auto inferred = inferTensorAffineStrides(pointer)) {
    if (static_cast<int64_t>(inferred->size()) == rank) {
      access.strides.assign(inferred->begin(), inferred->end());
      hasCompleteAffineStrideProof =
          llvm::all_of(access.strides,
                       [](const auto &stride) { return stride.has_value(); });
    }
  }
  access.addressHasNonAffineTensorComputation =
      !hasCompleteAffineStrideProof &&
      addressHasNonAffineTensorComputation(pointer);
  access.addressHasTensorControlFlowSelection =
      !hasCompleteAffineStrideProof &&
      addressHasTensorControlFlowSelection(pointer);
  access.masked = llvm::any_of(operation->getOperands(), [](Value operand) {
    return isPredicateType(operand.getType());
  });
  if (auto boundary =
          operation->getAttrOfType<DenseI32ArrayAttr>("boundaryCheck"))
    access.masked |= !boundary.empty();
  if (auto boundary =
          operation->getAttrOfType<DenseI32ArrayAttr>("boundary_check"))
    access.masked |= !boundary.empty();
  if (auto mask = getMemoryMaskOperandIndex(operation);
      mask && *mask < operation->getNumOperands() &&
      isPredicateType(operation->getOperand(*mask).getType())) {
    Value maskValue = operation->getOperand(*mask);
    if (auto ratio = staticPredicateTrueRatio(maskValue))
      access.activeElementRatio = std::clamp(*ratio, 0.0, 1.0);
    access.maskedStoreHasMatchingLoadDependency =
        store && operation->getNumOperands() > 1 &&
        valueDependsOnLoadWithExactMask(operation->getOperand(1), maskValue);
  }
  return access;
}

/// Provenance at the TTIR boundary, not a claim that arbitrary values reside in
/// shared memory.
static std::optional<std::string> describeDotOperandPath(Operation *operation) {
  if (operation->getName().getStringRef() != "tt.dot" ||
      operation->getNumOperands() != 3 || operation->getNumResults() != 1)
    return std::nullopt;
  auto lhs = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto rhs = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
  auto out = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!lhs || !rhs || !out || lhs.getRank() != 2 || rhs.getRank() != 2 ||
      out.getRank() != 2 || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      !lhs.getElementType().isF32() || !rhs.getElementType().isF32() ||
      !out.getElementType().isF32() || lhs.getDimSize(1) != rhs.getDimSize(0))
    return std::nullopt;
  auto operandKey = [&](Value operand) -> std::optional<std::string> {
    Operation *producer = operand.getDefiningOp();
    if (!producer || producer->getBlock() != operation->getBlock())
      return std::nullopt;
    llvm::StringRef name = producer->getName().getStringRef();
    if (name == "tt.load") {
      auto access = describeMemoryAccess(producer, producer->getOperand(0),
                                         operand.getType(), false);
      if (!access.hasStaticLayout() || !access.hasUnitInnerStride())
        return std::nullopt;
      return "gm{" + access.layoutKey() + "}";
    }
    if (isDotOperationName(name) || name == "arith.addf" ||
        name == "arith.subf" || name == "arith.mulf" ||
        name == "arith.select" || name == "scf.for")
      return std::string("computed");
    return std::nullopt;
  };
  auto left = operandKey(operation->getOperand(0));
  auto right = operandKey(operation->getOperand(1));
  if (!left || !right)
    return std::nullopt;
  return "f32:" + std::to_string(lhs.getDimSize(0)) + "x" +
         std::to_string(rhs.getDimSize(1)) + "x" +
         std::to_string(lhs.getDimSize(1)) + ":lhs=" + *left + ":rhs=" + *right;
}

/// A unit inner stride does not make a rectangular block a flat transfer.
static bool blockPointerCanUseFlatMemoryCurve(Value pointer, Type valueType) {
  Operation *definition = pointer.getDefiningOp();
  while (definition && definition->getName().getStringRef() == "tt.advance") {
    pointer = definition->getOperand(0);
    definition = pointer.getDefiningOp();
  }
  // Plain pointers and tensor-of-pointer accesses do not carry a complete
  // rectangular descriptor.
  if (!definition ||
      definition->getName().getStringRef() != "tt.make_tensor_ptr")
    return false;
  auto type = dyn_cast<RankedTensorType>(valueType);
  if (!type || !type.hasStaticShape())
    return false;
  const int64_t rank = type.getRank();
  // make_tensor_ptr operands: base, rank shapes, rank strides, rank offsets.
  if (definition->getNumOperands() != 1 + 3 * rank)
    return false;
  std::vector<std::pair<int64_t, int64_t>> dimensions;
  for (int64_t axis = 0; axis < rank; ++axis) {
    const int64_t extent = type.getDimSize(axis);
    if (extent == 1)
      continue;
    auto stride = getConstantInteger(definition->getOperand(1 + rank + axis));
    if (!stride || *stride <= 0 || extent <= 0)
      return false;
    dimensions.emplace_back(*stride, extent);
  }
  llvm::sort(dimensions);
  __int128 expectedStride = 1;
  for (auto [stride, extent] : dimensions) {
    if (stride != expectedStride)
      return false;
    expectedStride *= extent;
  }
  return true;
}

static int64_t getLoopTripCount(Operation *operation,
                                int64_t stageIterationCount,
                                StageModelFeatures *estimates = nullptr,
                                int64_t logicalProgramCountHint = 0) {
  const llvm::StringRef name = operation->getName().getStringRef();
  if (name != "scf.for" && name != "scf.while")
    return 1;
  const LoopTripCountInfo solution =
      analyzeLoopTripCount(operation, logicalProgramCountHint);
  if (estimates && solution.workloadUsesUpperBound)
    ++estimates->upperBoundLoopCount;
  if (estimates && solution.workloadUsesFallback)
    ++estimates->unknownTripCountLoopCount;
  return solution.workloadUsesFallback
             ? std::max<int64_t>(1, stageIterationCount)
             : solution.workloadTripCount;
}

static bool hasKnownAtMostOneIteration(Operation *operation) {
  if (!operation)
    return false;
  const LoopTripCountInfo solution = analyzeLoopTripCount(operation);
  return solution.constantTripCount && *solution.constantTripCount <= 1;
}

/// Accumulate dynamic work, not merely the number of syntactic TTIR ops.
static void accumulateDynamicOperationTree(
    Operation *operation, StageWorkload &work, double multiplicity,
    int64_t fallbackLoopTripCount, StageModelFeatures &estimates,
    int64_t logicalProgramCountHint, RankedTensorType mappedTensor = {},
    bool skipReductionBodies = false,
    std::vector<StageWorkload> *reductions = nullptr,
    const StageMemoryPatternAnalysis *memoryPatterns = nullptr) {
  if (!operation)
    return;
  const auto operationName = operation->getName().getStringRef();
  if (operationName == "tt.reduce") {
    if (skipReductionBodies)
      return;
    if (reductions) {
      StageWorkload contribution;
      auto ignored = estimates;
      accumulateDynamicOperationTree(operation, contribution, multiplicity,
                                     fallbackLoopTripCount, ignored,
                                     logicalProgramCountHint, mappedTensor,
                                     false, nullptr, memoryPatterns);
      reductions->push_back(std::move(contribution));
    }
  }
  StageWorkload local;
  accumulateOneOperation(operation, local, mappedTensor, memoryPatterns);
  const llvm::StringRef name = operation->getName().getStringRef();
  const bool schedulingLoop = operation->hasAttr("ta.auto_blockify_v1.loop");
  const double trips = schedulingLoop
                           ? 1.0
                           : static_cast<double>(getLoopTripCount(
                                 operation, fallbackLoopTripCount, &estimates,
                                 logicalProgramCountHint));
  if (name == "scf.for" || name == "scf.while")
    local.loopBackedges = trips;
  if (name == "scf.if" || name == "cf.cond_br")
    local.conditionalBranches = local.divergentBranches = 1.0;
  if (name.contains("barrier") || name.contains("sync"))
    local.synchronizations = 1.0;
  scaleWorkload(local, multiplicity);
  mergeWorkload(work, std::move(local));

  if (schedulingLoop)
    return;
  // A split loop shell owns only control and carried-state overhead; its body
  // operations are separate roots and must not be double-counted.
  if (shouldSplitLoopBodyIntoStages(operation))
    return;
  if (trips == 0)
    return;
  const double childMultiplicity = multiplicity * trips;
  RankedTensorType childMappedTensor = mappedTensor;
  if (name == "tt.map_elementwise")
    childMappedTensor = getRepresentativeTensorType(operation);
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block.getOperations())
        accumulateDynamicOperationTree(
            &nested, work, childMultiplicity, fallbackLoopTripCount, estimates,
            logicalProgramCountHint, childMappedTensor, skipReductionBodies,
            reductions, memoryPatterns);
}

static int64_t countAlgorithmLoops(const LogicalStage &stage) {
  int64_t count = 0;
  for (Operation *root : stage.operations) {
    if (!root || root->hasAttr("ta.auto_blockify_v1.loop"))
      continue;
    root->walk([&](Operation *operation) {
      const llvm::StringRef name = operation->getName().getStringRef();
      if ((name == "scf.for" || name == "scf.while") &&
          !operation->hasAttr("ta.auto_blockify_v1.loop"))
        ++count;
    });
  }
  return count;
}

static int64_t
inferRuntimeGridStrideIterations(const LogicalStage &stage,
                                 int64_t logicalProgramCountHint) {
  int64_t iterations = 0;
  if (logicalProgramCountHint <= 0)
    return iterations;
  for (Operation *root : stage.operations) {
    if (!root)
      continue;
    root->walk([&](Operation *operation) {
      if (operation->hasAttr("ta.auto_blockify_v1.loop"))
        return;
      if (operation->getName().getStringRef() != "scf.for")
        return;
      const LoopTripCountInfo solution =
          analyzeLoopTripCount(operation, logicalProgramCountHint);
      if (solution.gridStrideTripCount)
        iterations = std::max(iterations, *solution.gridStrideTripCount);
    });
  }
  return iterations;
}

static void mergeWorkload(StageWorkload &into, StageWorkload from) {
  llvm::append_range(into.reductionWorkloads,
                     std::move(from.reductionWorkloads));
  for (auto field : workloadScalarFields)
    into.*field += from.*field;
  into.memoryAccesses.insert(into.memoryAccesses.end(),
                             from.memoryAccesses.begin(),
                             from.memoryAccesses.end());
  llvm::append_range(into.atomicWorkloads, std::move(from.atomicWorkloads));
  mergeCounts(into.indirectLoadCounts, from.indirectLoadCounts);
  mergeCounts(into.segmentedIndirectLoadCounts,
              from.segmentedIndirectLoadCounts);
  mergeCounts(into.indirectStoreCounts, from.indirectStoreCounts);
  mergeCounts(into.segmentedIndirectStoreCounts,
              from.segmentedIndirectStoreCounts);
  mergeCounts(into.continuousLoadCounts, from.continuousLoadCounts);
  mergeCounts(into.continuousStoreCounts, from.continuousStoreCounts);
  mergeCounts(into.predicateCounts, from.predicateCounts);
  mergeCounts(into.prefixScanCounts, from.prefixScanCounts);
  mergeCounts(into.reductionGeometryCounts, from.reductionGeometryCounts);
  mergeCounts(into.reductionCombineClassCounts,
              from.reductionCombineClassCounts);
  mergeCounts(into.scalarizedMultiResultReductionCounts,
              from.scalarizedMultiResultReductionCounts);
  mergeCounts(into.sortCounts, from.sortCounts);
  mergeCounts(into.oneTripMaskedAccumulatorReductionExtents,
              from.oneTripMaskedAccumulatorReductionExtents);
  mergeCounts(into.dotFlopCounts, from.dotFlopCounts);
  mergeCounts(into.dotShapeCounts, from.dotShapeCounts);
  for (const auto &[key, entry] : from.dotOperandPaths) {
    into.dotOperandPaths[key].first = entry.first;
    into.dotOperandPaths[key].second += entry.second;
  }
  for (const auto &[name, counts] : from.operationCounts)
    for (const auto &[size, count] : counts)
      into.operationCounts[name][size] += count;
  for (TensorOperationWorkload &source : from.tensorOperationWorkloads) {
    auto destination =
        llvm::find_if(into.tensorOperationWorkloads,
                      [&](const TensorOperationWorkload &item) {
                        return item.operation == source.operation &&
                               item.elementBitWidth == source.elementBitWidth &&
                               item.contiguousElementsPerSegment ==
                                   source.contiguousElementsPerSegment;
                      });
    if (destination == into.tensorOperationWorkloads.end()) {
      into.tensorOperationWorkloads.push_back(std::move(source));
      continue;
    }
    destination->logicalElements += source.logicalElements;
    destination->segmentCount += source.segmentCount;
  }
  recomputeIssueElements(into);
}

static void makePerIteration(LogicalStage &stage) {
  const double count =
      static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
  scaleWorkload(stage.workload, 1.0 / count);
}

static Operation *getTopLevelSemanticRoot(Operation *operation);
static Operation *getPartitionSemanticRoot(Operation *operation);

static bool stageOwnsAnchor(const LogicalStage &stage,
                            const SimtAnchorDescriptor &anchor) {
  if (!anchor.materializable || !stage.localSimtMaterializable)
    return false;
  auto owns = [&](Operation *operation) {
    Operation *root = getPartitionSemanticRoot(operation);
    return root && llvm::is_contained(stage.operations, root);
  };
  if (anchor.scopeOperations.empty())
    return owns(anchor.operation);
  return llvm::all_of(anchor.scopeOperations, owns);
}

static void attachExactAnchorOwnership(StagePartition &partition,
                                       const SimtAnchorPlan &anchorPlan) {
  for (LogicalStage &stage : partition.stages) {
    if (!stage.localSimtMaterializable)
      continue;
    stage.simtAnchorIndices.clear();
    stage.localSuperblockMaterializable = false;
    bool allAnchorsDirectlyOwnedByV1Loop = true;
    for (auto indexedAnchor : llvm::enumerate(anchorPlan.anchors)) {
      const SimtAnchorDescriptor &anchor = indexedAnchor.value();
      if (anchor.materializable && stageOwnsAnchor(stage, anchor)) {
        stage.simtAnchorIndices.push_back(
            static_cast<unsigned>(indexedAnchor.index()));
        Operation *insertionPoint = anchor.scopeOperations.size() > 1
                                        ? anchor.scopeInsertionPoint
                                        : anchor.operation;
        Operation *blockOwner = insertionPoint && insertionPoint->getBlock()
                                    ? insertionPoint->getBlock()->getParentOp()
                                    : nullptr;
        allAnchorsDirectlyOwnedByV1Loop &=
            blockOwner && blockOwner->hasAttr("ta.auto_blockify_v1.loop");
      }
    }
    stage.localSimtMaterializable = !stage.simtAnchorIndices.empty();
    stage.localSuperblockMaterializable =
        stage.simtAnchorIndices.size() == 1 && allAnchorsDirectlyOwnedByV1Loop;
    if (!stage.localSimtMaterializable)
      stage.localSimtFactors.clear();
  }
}

static bool isFunctionLikeTTIROp(Operation *operation) {
  if (!operation)
    return false;
  const llvm::StringRef name = operation->getName().getStringRef();
  return name == "tt.func" || name == "func.func";
}

static Operation *getTopLevelSemanticRoot(Operation *operation) {
  if (!operation)
    return nullptr;
  Operation *root = operation;
  while (Operation *parent = root->getParentOp()) {
    const llvm::StringRef parentName = parent->getName().getStringRef();
    // AutoBlockify V1 may wrap a multi-block original function body in a newly
    // created scf.execute_region.
    if (isFunctionLikeTTIROp(parent) ||
        parent->hasAttr("ta.auto_blockify_v1.loop") ||
        (parentName == "scf.execute_region" &&
         parent->hasAttr("ta.auto_blockify_v1.schedule")))
      return root;
    root = parent;
  }
  return nullptr;
}

static bool isInsideAutoBlockifyV1Loop(Operation *operation) {
  if (!operation || operation->hasAttr("ta.auto_blockify_v1.loop"))
    return false;
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->hasAttr("ta.auto_blockify_v1.loop"))
      return true;
  return false;
}

static std::vector<Operation *> collectTopLevelSemanticRoots(ModuleOp module) {
  std::vector<Operation *> result;
  auto appendOps = [&](auto &&self, Block &block, bool insideV1Loop) -> void {
    for (Operation &nested : block.getOperations()) {
      if (nested.hasTrait<OpTrait::IsTerminator>())
        continue;

      const llvm::StringRef name = nested.getName().getStringRef();
      const bool isV1Loop = nested.hasAttr("ta.auto_blockify_v1.loop");
      // AutoBlockify V1 wraps a multi-block original function body in a new
      // scf.execute_region.
      const bool isV1ExecuteRegion =
          insideV1Loop && name == "scf.execute_region" &&
          nested.hasAttr("ta.auto_blockify_v1.schedule");
      if (isV1ExecuteRegion) {
        for (Region &region : nested.getRegions())
          for (Block &body : region)
            self(self, body, insideV1Loop);
        continue;
      }

      result.push_back(&nested);
      // AutoBlockify V1's scf.for is a scheduling shell.
      if (!isV1Loop || nested.getNumRegions() == 0)
        continue;
      for (Region &region : nested.getRegions())
        for (Block &body : region)
          self(self, body, /*insideV1Loop=*/true);
    }
  };
  for (Operation &operation : module.getBody()->getOperations()) {
    if (!isFunctionLikeTTIROp(&operation) || operation.getNumRegions() == 0)
      continue;
    for (Block &block : operation.getRegion(0))
      appendOps(appendOps, block, /*insideV1Loop=*/false);
  }
  return result;
}

static bool operationTreeContainsName(Operation *root, llvm::StringRef name) {
  bool found = root && root->getName().getStringRef() == name;
  if (!root || found)
    return found;
  root->walk([&](Operation *nested) {
    found |= nested->getName().getStringRef() == name;
  });
  return found;
}

static bool hasAllDAddressPattern(const StageWorkload &work) {
  return llvm::any_of(work.addressPatterns, [](const AddressPatternSummary &p) {
    return (p.memoryOp == "tt.load" || p.memoryOp == "tt.store") &&
           !p.axes.empty() &&
           llvm::all_of(p.axes, [](const AddressAxisSummary &a) {
             return a.offsetAxisInfo == "unstructured";
           });
  });
}

static bool
operationTreeContainsAllDMemory(Operation *root,
                                const StageMemoryPatternAnalysis *patterns) {
  if (!root || !patterns)
    return false;
  bool found = false;
  root->walk([&](Operation *op) {
    const auto name = op->getName().getStringRef();
    if (name != "tt.load" && name != "tt.store")
      return;
    const auto *p = patterns->lookupSummary(op);
    found |= p && !p->axes.empty() &&
             llvm::all_of(p->axes, [](const AddressAxisSummary &a) {
               return a.offsetAxisInfo == "unstructured";
             });
  });
  return found;
}

static bool isMixedAxisMemory(Operation *op,
                              const StageMemoryPatternAnalysis *patterns) {
  const auto *p = patterns ? patterns->lookupSummary(op) : nullptr;
  if (!p || p->axes.empty())
    return false;
  bool structured = false, unstructured = false;
  for (const auto &axis : p->axes) {
    if (axis.offsetAxisInfo == "structured" ||
        axis.offsetAxisInfo == "scalar" || axis.offsetAxisInfo == "scalarlike")
      structured = true;
    else if (axis.offsetAxisInfo == "unstructured")
      unstructured = true;
    else
      return false;
  }
  return structured && unstructured;
}

static bool operationTreeHasOnlyPartialContinuousMemory(
    Operation *root, const StageMemoryPatternAnalysis *memoryPatterns) {
  if (!root)
    return false;
  bool hasPartial = false;
  bool hasOtherIndirect = false;
  root->walk([&](Operation *operation) {
    const llvm::StringRef name = operation->getName().getStringRef();
    if (name == "tt.gather") {
      hasOtherIndirect = true;
      return;
    }
    if (name != "tt.load" && name != "tt.store")
      return;
    if (isMixedAxisMemory(operation, memoryPatterns))
      hasPartial = true;
    else if (isIndirectMemoryOperation(operation, memoryPatterns))
      hasOtherIndirect = true;
  });
  return hasPartial && !hasOtherIndirect;
}

static bool loopHasTrueCarriedDataDependency(Operation *operation) {
  if (!operation || operation->hasAttr("ta.auto_blockify_v1.loop"))
    return false;
  const llvm::StringRef name = operation->getName().getStringRef();
  if ((name != "scf.for" && name != "scf.while") ||
      operation->getNumRegions() == 0 || operation->getRegion(0).empty() ||
      hasKnownAtMostOneIteration(operation))
    return false;
  Block &body = operation->getRegion(0).front();
  const unsigned firstCarriedArgument = name == "scf.for" ? 1 : 0;
  for (unsigned index = firstCarriedArgument; index < body.getNumArguments();
       ++index) {
    BlockArgument argument = body.getArgument(index);
    if (!argument.use_empty() && !isPointerLikeType(argument.getType()) &&
        !isAddressOnlyLoopValue(argument))
      return true;
  }
  return false;
}

static bool operationTreeHasTrueLoopCarriedDependency(Operation *root) {
  bool found = false;
  if (root)
    root->walk([&](Operation *operation) {
      found |= loopHasTrueCarriedDataDependency(operation);
    });
  return found;
}

static bool operationTreeHasAnyName(Operation *root,
                                    llvm::ArrayRef<llvm::StringRef> names) {
  return llvm::any_of(names, [&](llvm::StringRef name) {
    return operationTreeContainsName(root, name);
  });
}

/// A scalar GM access has a non-shaped value/result; tile accesses are shaped.
static bool isScalarLoadOperation(Operation *operation) {
  return operation && operation->getName().getStringRef() == "tt.load" &&
         operation->getNumResults() > 0 &&
         !isa<ShapedType>(operation->getResult(0).getType());
}

static bool isScalarStoreOperation(Operation *operation) {
  return operation && operation->getName().getStringRef() == "tt.store" &&
         operation->getNumOperands() > 1 &&
         !isa<ShapedType>(operation->getOperand(1).getType());
}

static bool operationTreeHasScalarMemory(Operation *root, bool wantStore) {
  auto matches = [&](Operation *operation) {
    return wantStore ? isScalarStoreOperation(operation)
                     : isScalarLoadOperation(operation);
  };
  bool found = matches(root);
  if (found || !root)
    return found;
  root->walk([&](Operation *nested) {
    if (!found)
      found = matches(nested);
  });
  return found;
}

static bool operationTreeHasVectorCompute(Operation *root) {
  bool found = false;
  root->walk([&](Operation *operation) {
    // A one-element tensor is still vector work; a scalar arith.subf is not.
    found |= hasTensorResult(operation) &&
             operation->hasTrait<OpTrait::Elementwise>() &&
             !operation->hasTrait<OpTrait::ConstantLike>();
  });
  return found;
}

/// Operand index of the optional mask predicate of a memory operation.
static std::optional<unsigned> getMemoryMaskOperandIndex(Operation *operation) {
  const llvm::StringRef name = operation->getName().getStringRef();
  if (name == "tt.load")
    return 1u;
  if (name == "tt.store" || name.starts_with("tt.atomic"))
    return 2u;
  return std::nullopt;
}

/// True when every use of `value` is a memory mask, allowing forwarding
/// through shape helpers and predicate boolean algebra.
static bool isMemoryMaskOnlyValue(Value value) {
  llvm::SmallVector<Value, 8> worklist{value};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      const std::optional<unsigned> maskOperand =
          getMemoryMaskOperandIndex(user);
      if (maskOperand && use.getOperandNumber() == *maskOperand)
        continue;
      const llvm::StringRef name = user->getName().getStringRef();
      const bool maskForwarding =
          name == "tt.splat" || name == "tt.broadcast" ||
          name == "tt.expand_dims" || name == "arith.andi" ||
          name == "arith.ori" || name == "arith.xori";
      if (!maskForwarding)
        return false;
      llvm::append_range(worklist, user->getResults());
    }
  }
  return true;
}

/// A structured algorithmic loop with no true loop-carried dependency.
static bool isIndependentStructuredLoop(Operation *operation) {
  if (!operation || operation->hasAttr("ta.auto_blockify_v1.loop"))
    return false;
  const llvm::StringRef name = operation->getName().getStringRef();
  if ((name != "scf.for" && name != "scf.while") ||
      operation->getNumRegions() == 0 || operation->getRegion(0).empty())
    return false;
  return !operationTreeHasTrueLoopCarriedDependency(operation);
}

/// Follow scalar cast/shape helpers to a Triton launch builtin.
static bool valueComesFrom(Value value, llvm::StringRef operationName) {
  llvm::SmallVector<Value, 8> worklist{value};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    Operation *definition = current.getDefiningOp();
    if (!definition)
      continue;
    if (definition->getName().getStringRef() == operationName)
      return true;
    const llvm::StringRef name = definition->getName().getStringRef();
    if (name == "arith.index_cast" || name == "arith.bitcast" ||
        name == "arith.extsi" || name == "arith.extui" ||
        name == "arith.trunci" || name == "arith.addi" ||
        name == "arith.subi" || name == "arith.muli" || name == "arith.divsi" ||
        name == "arith.divui" || name == "arith.remsi" || name == "arith.remui")
      llvm::append_range(worklist, definition->getOperands());
  }
  return false;
}

static bool isGridStrideLoop(Operation *operation) {
  if (!operation || operation->getName().getStringRef() != "scf.for" ||
      operation->getNumOperands() < 3 ||
      operation->hasAttr("ta.auto_blockify_v1.loop"))
    return false;
  Value lower = operation->getOperand(0);
  const bool directProgramId = valueComesFrom(lower, "tt.get_program_id");
  // AutoBlockify rewrites the resident program id to a remainder modulo the
  // original program count.
  Operation *lowerDefinition = lower.getDefiningOp();
  const bool remappedProgramId =
      lowerDefinition &&
      (lowerDefinition->getName().getStringRef() == "arith.remui" ||
       lowerDefinition->getName().getStringRef() == "arith.remsi") &&
      valueComesFrom(lower, "tt.get_num_programs");
  bool autoBlockifyProgramId = false;
  if (auto blockArgument = llvm::dyn_cast<BlockArgument>(lower)) {
    Block *owner = blockArgument.getOwner();
    Operation *blockOwner = owner ? owner->getParentOp() : nullptr;
    autoBlockifyProgramId =
        blockOwner && blockOwner->hasAttr("ta.auto_blockify_v1.loop");
  }
  return (directProgramId || remappedProgramId || autoBlockifyProgramId) &&
         valueComesFrom(operation->getOperand(2), "tt.get_num_programs");
}

static bool isInsideGridStrideLoop(Operation *operation) {
  for (Operation *parent = operation ? operation->getParentOp() : nullptr;
       parent; parent = parent->getParentOp())
    if (isGridStrideLoop(parent))
      return true;
  return false;
}

/// True when one structured loop owns dominant structures that cannot share a
/// leaf Stage cost formula.  Real recurrence is kept on the loop shell while
/// dot/reduction/memory work is exposed as body roots.  Restrict recurrence
/// splitting to scf.for for now: scf.while has separate before/after regions
/// whose execution order cannot be represented by the current flat root list.
static bool loopRequiresDominantStructureSplit(Operation *operation) {
  if (!operation || operation->hasAttr("ta.auto_blockify_v1.loop") ||
      operation->getName().getStringRef() != "scf.for" ||
      operation->getNumRegions() == 0 || operation->getRegion(0).empty())
    return false;

  const bool hasDot = operationTreeHasAnyName(
      operation, {"tt.dot", "linalg.matmul", "linalg.batch_matmul"});
  if (!hasDot)
    return false;
  const bool hasReduction = operationTreeHasAnyName(
      operation, {"tt.reduce", "tt.scan", "linalg.reduce"});
  const bool hasAtomicMemory =
      operationTreeHasAnyName(operation, {"tt.atomic_rmw", "tt.atomic_cas"});
  return hasReduction || hasAtomicMemory;
}

/// PR #2112 splits independent loops only when a body SIMT anchor needs exact
/// ownership.  Also split an scf.for whose transitive operation tree would
/// otherwise fail StageKindClassifier with `requires_split`.  This is a
/// structural, conservative split: cross-iteration overlap remains a later
/// schedule-model concern.
static bool shouldSplitLoopBodyIntoStages(Operation *operation) {
  return loopRequiresDominantStructureSplit(operation) ||
         (isIndependentStructuredLoop(operation) &&
          hasSimtAnchorInBody(operation));
}

static Operation *getEnclosingSplitLoop(Operation *operation) {
  if (!operation)
    return nullptr;
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp())
    if (shouldSplitLoopBodyIntoStages(parent))
      return parent;
  return nullptr;
}

static Operation *getPartitionSemanticRoot(Operation *operation) {
  Operation *topLevelRoot = getTopLevelSemanticRoot(operation);
  if (!topLevelRoot)
    return topLevelRoot;

  Operation *root = operation;
  while (root != topLevelRoot) {
    Operation *parent = root->getParentOp();
    if (shouldSplitLoopBodyIntoStages(parent))
      return root;
    root = parent;
  }
  return topLevelRoot;
}

static void appendSplitLoopBodyRoots(Operation *operation,
                                     std::vector<Operation *> &roots) {
  if (!shouldSplitLoopBodyIntoStages(operation) ||
      operation->hasAttr("ta.auto_blockify_v1.loop"))
    return;

  for (Region &region : operation->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block.getOperations()) {
        if (nested.hasTrait<OpTrait::IsTerminator>())
          continue;
        roots.push_back(&nested);
        appendSplitLoopBodyRoots(&nested, roots);
      }
    }
  }
}

static std::vector<Operation *> collectPartitionSemanticRoots(ModuleOp module) {
  std::vector<Operation *> roots = collectTopLevelSemanticRoots(module);
  std::vector<Operation *> expandedRoots;
  for (Operation *root : roots) {
    expandedRoots.push_back(root);
    appendSplitLoopBodyRoots(root, expandedRoots);
  }
  return expandedRoots;
}

/// Trip count of the split structured loops enclosing `root`. Nested loop
/// bodies execute by the product of their enclosing trip counts, not the
/// maximum.
static int64_t multiplyTripCounts(int64_t left, int64_t right) {
  if (left == 0 || right == 0)
    return 0;
  if (left > std::numeric_limits<int64_t>::max() / right)
    return std::numeric_limits<int64_t>::max();
  return left * right;
}

static int64_t enclosingSplitLoopTripCount(Operation *root) {
  int64_t trips = 1;
  if (!root)
    return trips;
  for (Operation *parent = root->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (!shouldSplitLoopBodyIntoStages(parent))
      continue;
    const int64_t factor = getLoopTripCount(parent, 1);
    trips = multiplyTripCounts(trips, factor);
  }
  return trips;
}

/// Entry multiplicity for one root when accumulating dynamic Stage work.
static double semanticRootEntryMultiplicity(Operation *root) {
  if (!root)
    return 1.0;
  // A split loop shell is entered once per enclosing invocation. Its own
  // backedges are already counted by accumulateDynamicOperationTree; adding
  // its trip count here would charge a t-trip shell t² times.
  return static_cast<double>(enclosingSplitLoopTripCount(root));
}

/// Classify one transitive semantic ownership unit.
static StageCostModelKind
classifySemanticRoot(Operation *root,
                     const StageMemoryPatternAnalysis *memoryPatterns) {
  if (root->hasAttr("ta.auto_blockify_v1.loop"))
    return StageCostModelKind::AutoBlockifyLoop;
  if (root->hasAttr("ta.auto_blockify_v1.schedule"))
    return StageCostModelKind::AutoBlockifyDispatch;
  if (operationTreeHasTrueLoopCarriedDependency(root))
    return StageCostModelKind::LoopCarriedRecurrence;
  if (operationTreeHasAnyName(root, {"tt.scan"}))
    return StageCostModelKind::PrefixScan;
  if (operationTreeHasAnyName(root, {"tt.reduce", "linalg.reduce"}))
    return StageCostModelKind::RowwiseReduction;
  if (operationTreeHasAnyName(root, {"tt.dot", "tt.dot_scaled"}))
    return StageCostModelKind::CubeRoofline;
  if (operationTreeHasAnyName(root, {"tt.atomic_rmw", "tt.atomic_cas"}))
    return StageCostModelKind::AtomicMemory;
  if (operationTreeHasAnyName(root, {"tt.histogram"}))
    return StageCostModelKind::VectorIssue;
  if (operationTreeHasOnlyPartialContinuousMemory(root, memoryPatterns))
    return StageCostModelKind::PartialContinuousTileMemory;
  if (operationTreeContainsAllDMemory(root, memoryPatterns))
    return StageCostModelKind::IndirectGatherMemory;
  if (operationTreeHasAnyName(root, {"scf.for", "scf.while"}))
    return StageCostModelKind::IndependentPipelinedLoop;
  if (operationTreeHasAnyName(
          root, {"tt.fp_to_fp", "arith.extf", "arith.truncf", "arith.fptosi",
                 "arith.fptoui", "arith.sitofp", "arith.uitofp"}))
    return StageCostModelKind::ConversionPack;
  const bool hasScalarLoad = operationTreeHasScalarMemory(root, false);
  const bool hasScalarStore = operationTreeHasScalarMemory(root, true);
  if (hasScalarStore && !hasScalarLoad)
    return StageCostModelKind::ScalarStore;
  if (hasScalarLoad || hasScalarStore)
    return StageCostModelKind::ScalarLoad;
  const bool hasLoad = operationTreeHasAnyName(root, {"tt.load"});
  const bool hasStore = operationTreeHasAnyName(root, {"tt.store"});
  if (hasStore && !hasLoad)
    return StageCostModelKind::ContinuousTileStore;
  if (hasLoad || hasStore)
    return StageCostModelKind::ContinuousTileMemory;
  if (operationTreeHasAnyName(root, {"arith.cmpi", "arith.cmpf"}))
    return StageCostModelKind::PredicateMask;
  if (operationTreeHasAnyName(root, {"tt.get_program_id", "tt.addptr",
                                     "tt.advance", "arith.index_cast"}))
    return StageCostModelKind::IndexGeneration;
  if (operationTreeHasAnyName(root, {"scf.if", "cf.cond_br"}))
    return StageCostModelKind::ScalarControl;
  if (operationTreeHasVectorCompute(root))
    return StageCostModelKind::VectorIssue;
  return StageCostModelKind::ScalarIssue;
}

static StageScheduleKind scheduleForSemanticRoot(Operation *root,
                                                 StageCostModelKind kind) {
  if (kind == StageCostModelKind::LoopCarriedRecurrence)
    return StageScheduleKind::LoopCarriedSerial;
  if (kind == StageCostModelKind::IndirectGatherMemory ||
      kind == StageCostModelKind::PartialContinuousTileMemory)
    return StageScheduleKind::PartiallyDependent;
  if (kind == StageCostModelKind::AtomicMemory)
    return StageScheduleKind::PartiallyDependent;
  if (kind == StageCostModelKind::AutoBlockifyLoop ||
      kind == StageCostModelKind::IndependentPipelinedLoop ||
      operationTreeHasAnyName(root, {"scf.for", "scf.while"}))
    return StageScheduleKind::IndependentPipelined;
  return StageScheduleKind::StraightLine;
}

static int64_t semanticRootIterationCount(Operation *root) {
  if (!root || root->hasAttr("ta.auto_blockify_v1.loop"))
    return 1;
  const int64_t enclosingIterations = enclosingSplitLoopTripCount(root);
  if (shouldSplitLoopBodyIntoStages(root))
    return multiplyTripCounts(enclosingIterations, getLoopTripCount(root, 1));
  if (enclosingIterations == 0)
    return 0;
  int64_t iterations = 1;
  root->walk([&](Operation *operation) {
    if (!operation->hasAttr("ta.auto_blockify_v1.loop"))
      iterations = std::max(iterations, getLoopTripCount(operation, 1));
  });
  // Exposed body roots inherit their enclosing split loops' trip counts.
  return std::max(iterations, enclosingIterations);
}

static bool hasOrderedStageBoundary(Operation *root) {
  bool ordered = false;
  if (!root)
    return ordered;
  root->walk([&](Operation *operation) {
    const llvm::StringRef name = operation->getName().getStringRef();
    ordered |= operation->getNumRegions() > 0 ||
               name.starts_with("tt.atomic") || name.contains("barrier") ||
               name.contains("sync") || name == "cf.cond_br";
  });
  return ordered;
}

static std::string makeStageId(size_t ordinal, StageCostModelKind kind) {
  return ("stage_" + llvm::Twine(ordinal) + "_" + stringifyStageCostModel(kind))
      .str();
}

static void collectOwnedOperationTree(Operation *root,
                                      llvm::DenseSet<Operation *> &owned) {
  if (!root)
    return;
  owned.insert(root);
  // The AutoBlockify loop is intentionally split into a scheduling shell and
  // direct semantic body roots.
  if (root->hasAttr("ta.auto_blockify_v1.loop") ||
      shouldSplitLoopBodyIntoStages(root))
    return;
  root->walk([&](Operation *nested) {
    if (nested != root)
      owned.insert(nested);
  });
}

static bool isValueDefinedInside(Value value,
                                 const llvm::DenseSet<Operation *> &owned) {
  if (Operation *definition = value.getDefiningOp())
    return owned.contains(definition);
  auto argument = dyn_cast<BlockArgument>(value);
  Operation *parent = argument ? argument.getOwner()->getParentOp() : nullptr;
  return parent && owned.contains(parent);
}

static int64_t staticTensorBytes(Value value) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  Type elementType = shaped.getElementType();
  if (!isa<IntegerType, FloatType>(elementType))
    return 0;
  const int64_t elements = shaped.getNumElements();
  const int64_t bits = elementType.getIntOrFloatBitWidth();
  if (elements <= 0 || bits <= 0)
    return 0;
  constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
  if (elements > (maximum - 7) / bits)
    return maximum;
  return (elements * bits + 7) / 8;
}

static int64_t staticTensorBytes(llvm::ArrayRef<Value> values) {
  int64_t total = 0;
  for (Value value : values) {
    const int64_t bytes = staticTensorBytes(value);
    if (bytes > std::numeric_limits<int64_t>::max() - total)
      return std::numeric_limits<int64_t>::max();
    total += bytes;
  }
  return total;
}

/// Derive the exact SSA contract of every Stage from operation ownership.
static void deriveStageLiveValues(StagePartition &partition) {
  for (LogicalStage &stage : partition.stages) {
    llvm::DenseSet<Operation *> owned;
    for (Operation *root : stage.operations)
      collectOwnedOperationTree(root, owned);

    llvm::SetVector<Value> liveIns;
    llvm::SetVector<Value> liveOuts;
    for (Operation *operation : owned) {
      for (Value operand : operation->getOperands())
        if (!isValueDefinedInside(operand, owned))
          liveIns.insert(operand);
      for (Value result : operation->getResults())
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !owned.contains(user);
            }))
          liveOuts.insert(result);
    }
    stage.liveIns.assign(liveIns.begin(), liveIns.end());
    stage.liveOuts.assign(liveOuts.begin(), liveOuts.end());
    stage.liveInBytes = staticTensorBytes(stage.liveIns);
    stage.liveOutBytes = staticTensorBytes(stage.liveOuts);
  }
}

/// Derive the physical tensor traffic of the exact local SIMT scopes that the
/// materializer will create.
static void deriveLocalSimtScopeTraffic(StagePartition &partition,
                                        const SimtAnchorPlan &anchorPlan) {
  for (LogicalStage &stage : partition.stages) {
    stage.scopeInputTensorBytes = 0;
    stage.scopeOutputTensorBytes = 0;
    if (stage.costModelKind == StageCostModelKind::AtomicMemory) {
      stage.localSimtMaterializable = false;
      stage.localSuperblockMaterializable = false;
      stage.localSimtFactors.clear();
      stage.simtAnchorIndices.clear();
      continue;
    }
    auto merged = mergeSimtStageAnchors(anchorPlan, stage.simtAnchorIndices);
    if (!merged) {
      // The selector materializes exactly the same merged descriptor.
      stage.localSimtMaterializable = false;
      stage.localSuperblockMaterializable = false;
      stage.localSimtFactors.clear();
      stage.simtAnchorIndices.clear();
      continue;
    }
    {
      const SimtAnchorDescriptor &anchor = *merged;
      llvm::SmallVector<Operation *> roots;
      const bool isRange = anchor.scopeOperations.size() > 1;
      if (isRange)
        llvm::append_range(roots, anchor.scopeOperations);
      else
        roots.push_back(anchor.operation);

      llvm::DenseSet<Operation *> inside;
      for (Operation *root : roots) {
        if (!root)
          continue;
        inside.insert(root);
        root->walk([&](Operation *nested) { inside.insert(nested); });
      }

      llvm::SetVector<Value> captured;
      for (Operation *operation : inside)
        for (Value operand : operation->getOperands())
          if (!isValueDefinedInside(operand, inside))
            captured.insert(operand);

      llvm::SetVector<Value> returned;
      for (Operation *root : roots) {
        if (!root)
          continue;
        for (Value result : root->getResults()) {
          // A single-op scope returns every result.  A range scope mirrors
          // wrapAnchorRange and returns only values with an outside user.
          if (!isRange || llvm::any_of(result.getUsers(), [&](Operation *user) {
                return !inside.contains(user);
              }))
            returned.insert(result);
        }
      }

      if (llvm::any_of(returned, [](Value value) {
            return isPointerLikeType(value.getType());
          })) {
        stage.localSimtMaterializable = false;
        stage.localSimtFactors.clear();
        stage.simtAnchorIndices.clear();
        continue;
      }

      stage.scopeInputTensorBytes = staticTensorBytes(captured.getArrayRef());
      stage.scopeOutputTensorBytes = staticTensorBytes(returned.getArrayRef());
    }
  }
}

} // namespace

llvm::Expected<ProgramStructure>
ProgramStructureAnalysis::analyze(ModuleOp module,
                                  const SimtAnchorPlan &anchorPlan) const {
  if (!module)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ProgramStructureAnalysis requires ModuleOp");
  ProgramStructure structure;
  structure.rootOperations = collectPartitionSemanticRoots(module);
  if (structure.rootOperations.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ProgramStructureAnalysis found no top-level TTIR operation");

  // Compound scopes may legally move pure tensor setup across input loads.
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable || anchor.scopeOperations.size() < 2 ||
        !anchor.scopeInsertionPoint)
      continue;

    llvm::SmallVector<Operation *, 8> scopeRoots;
    for (Operation *operation : anchor.scopeOperations) {
      Operation *root = getPartitionSemanticRoot(operation);
      if (root && llvm::is_contained(structure.rootOperations, root) &&
          !llvm::is_contained(scopeRoots, root))
        scopeRoots.push_back(root);
    }
    Operation *insertionRoot =
        getPartitionSemanticRoot(anchor.scopeInsertionPoint);
    auto insertionIt = llvm::find(structure.rootOperations, insertionRoot);
    if (scopeRoots.empty() || insertionIt == structure.rootOperations.end())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ProgramStructureAnalysis cannot normalize a compound SIMT scope");

    const size_t insertionPosition =
        static_cast<size_t>(insertionIt - structure.rootOperations.begin());
    std::vector<Operation *> reordered;
    reordered.reserve(structure.rootOperations.size());
    size_t normalizedInsertionPosition = 0;
    for (auto indexedRoot : llvm::enumerate(structure.rootOperations)) {
      if (indexedRoot.index() < insertionPosition &&
          !llvm::is_contained(scopeRoots, indexedRoot.value()))
        ++normalizedInsertionPosition;
      if (!llvm::is_contained(scopeRoots, indexedRoot.value()))
        reordered.push_back(indexedRoot.value());
    }
    reordered.insert(reordered.begin() + normalizedInsertionPosition,
                     scopeRoots.begin(), scopeRoots.end());
    structure.rootOperations = std::move(reordered);
  }

  return structure;
}

static int semanticKindPriority(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::AutoBlockifyDispatch:
  case StageCostModelKind::AutoBlockifyLoop:
    return 100;
  case StageCostModelKind::LoopCarriedRecurrence:
    return 90;
  case StageCostModelKind::RowwiseReduction:
  case StageCostModelKind::PrefixScan:
    return 80;
  case StageCostModelKind::CubeRoofline:
  case StageCostModelKind::TinyCubeRoofline:
    return 70;
  case StageCostModelKind::AtomicMemory:
    return 65;
  case StageCostModelKind::IndirectScalarMemory:
  case StageCostModelKind::IndirectGatherMemory:
  case StageCostModelKind::PartialContinuousTileMemory:
    return 60;
  case StageCostModelKind::IndependentPipelinedLoop:
    return 50;
  case StageCostModelKind::ConversionPack:
    return 40;
  case StageCostModelKind::ScalarLoad:
  case StageCostModelKind::ScalarStore:
    return 35;
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
    return 30;
  case StageCostModelKind::VectorIssue:
    return 25;
  case StageCostModelKind::PredicateMask:
  case StageCostModelKind::LoopPredicate:
    return 20;
  case StageCostModelKind::IndexGeneration:
    return 10;
  default:
    return 0;
  }
}

static bool isSupportingSemanticKind(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::ScalarIssue:
  case StageCostModelKind::ScalarControl:
  case StageCostModelKind::ScalarMath:
  case StageCostModelKind::VectorIssue:
  case StageCostModelKind::IndexGeneration:
  case StageCostModelKind::PredicateMask:
  case StageCostModelKind::LoopPredicate:
    return true;
  default:
    return false;
  }
}

static bool hasDataDependencyOnStage(ArrayRef<Operation *> stageOperations,
                                     Operation *candidate) {
  if (!candidate)
    return false;
  return llvm::any_of(candidate->getOperands(), [&](Value operand) {
    Operation *producer = operand.getDefiningOp();
    if (!producer)
      return false;
    return llvm::is_contained(stageOperations,
                              getPartitionSemanticRoot(producer));
  });
}

static std::string getCarriedValueName(Location location) {
  std::string name;
  if (auto named = dyn_cast<NameLoc>(location))
    name = named.getName().str();
  else if (auto callsite = dyn_cast<CallSiteLoc>(location)) {
    name = getCarriedValueName(callsite.getCallee());
    if (name.empty())
      name = getCarriedValueName(callsite.getCaller());
  } else if (auto fused = dyn_cast<FusedLoc>(location)) {
    for (Location child : fused.getLocations()) {
      name = getCarriedValueName(child);
      if (!name.empty())
        break;
    }
  }
  if (llvm::StringRef(name).ends_with("_new"))
    name.resize(name.size() - 4);
  return name;
}

static std::string printDependencySourceLocation(Location location) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  location.print(stream);
  stream.flush();
  return result;
}

static llvm::Error deriveStageLoopMembership(StagePartition &partition) {
  llvm::DenseMap<Operation *, std::string> loopStages;
  for (const LogicalStage &stage : partition.stages)
    for (Operation *root : stage.operations)
      if (shouldSplitLoopBodyIntoStages(root))
        loopStages.try_emplace(root, stage.id);

  for (LogicalStage &stage : partition.stages) {
    std::optional<std::vector<std::string>> expected;
    for (Operation *root : stage.operations) {
      std::vector<std::string> enclosing;
      for (Operation *parent = root->getParentOp(); parent;
           parent = parent->getParentOp())
        if (auto loop = loopStages.find(parent); loop != loopStages.end())
          enclosing.push_back(loop->second);
      std::reverse(enclosing.begin(), enclosing.end());
      if (!expected)
        expected = enclosing;
      else if (*expected != enclosing)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' crosses split-loop membership", stage.id.c_str());
    }
    stage.enclosingLoopStages = expected.value_or(std::vector<std::string>{});
  }
  return llvm::Error::success();
}

static std::string innermostCommonLoop(const LogicalStage &source,
                                       const LogicalStage &target) {
  std::string common;
  const size_t count = std::min(source.enclosingLoopStages.size(),
                                target.enclosingLoopStages.size());
  for (size_t index = 0; index < count; ++index) {
    if (source.enclosingLoopStages[index] != target.enclosingLoopStages[index])
      break;
    common = source.enclosingLoopStages[index];
  }
  return common;
}

/// Convert inter-Stage SSA uses and scf.for iter-arg/yield semantics into
/// explicit same-iteration and distance-one edges. Pointer-only induction
/// remains a scheduling/address concern and is deliberately excluded.
static llvm::Error deriveStageDependencies(StagePartition &partition) {
  partition.dependencies.clear();
  llvm::DenseMap<Operation *, size_t> operationOwners;
  for (auto indexedStage : llvm::enumerate(partition.stages)) {
    llvm::DenseSet<Operation *> owned;
    for (Operation *root : indexedStage.value().operations)
      collectOwnedOperationTree(root, owned);
    for (Operation *operation : owned) {
      auto inserted =
          operationOwners.try_emplace(operation, indexedStage.index());
      if (!inserted.second)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage dependency analysis found overlapping operation ownership");
    }
  }

  llvm::StringSet<> intraStagePairs;
  for (auto indexedStage : llvm::enumerate(partition.stages)) {
    llvm::DenseSet<Operation *> owned;
    for (Operation *root : indexedStage.value().operations)
      collectOwnedOperationTree(root, owned);
    for (Operation *consumer : owned) {
      for (OpOperand &operand : consumer->getOpOperands()) {
        Value value = operand.get();
        Operation *producer = value.getDefiningOp();
        if (!producer)
          continue;
        auto sourceOwner = operationOwners.find(producer);
        if (sourceOwner == operationOwners.end() ||
            sourceOwner->second == indexedStage.index())
          continue;
        const LogicalStage &source = partition.stages[sourceOwner->second];
        const LogicalStage &target = indexedStage.value();
        const std::string loopStage = innermostCommonLoop(source, target);
        const std::string key =
            (llvm::Twine(loopStage) + ":" + source.id + ":" + target.id).str();
        if (!intraStagePairs.insert(key).second)
          continue;
        StageDependency dependency;
        dependency.loopStage = loopStage;
        dependency.sourceStage = source.id;
        dependency.targetStage = target.id;
        dependency.kind = StageDependencyKind::IntraIterationData;
        dependency.iterationDistance = 0;
        dependency.carriedValueName = "inter_stage_ssa";
        dependency.carriedValueBytes = staticTensorBytes(value);
        dependency.sourceLocation =
            printDependencySourceLocation(producer->getLoc());
        partition.dependencies.push_back(std::move(dependency));
      }
    }
  }

  for (auto indexedStage : llvm::enumerate(partition.stages)) {
    const LogicalStage &loopShell = indexedStage.value();
    for (Operation *root : loopShell.operations) {
      if (!shouldSplitLoopBodyIntoStages(root) ||
          root->getName().getStringRef() != "scf.for" ||
          !loopHasTrueCarriedDataDependency(root) ||
          root->getNumRegions() == 0 || root->getRegion(0).empty())
        continue;

      Block &body = root->getRegion(0).front();
      Operation *terminator = body.getTerminator();
      if (!terminator || terminator->getName().getStringRef() != "scf.yield")
        return llvm::createStringError(
            std::errc::invalid_argument,
            "split scf.for does not terminate with scf.yield");
      const unsigned carriedCount =
          body.getNumArguments() > 0 ? body.getNumArguments() - 1 : 0;
      if (terminator->getNumOperands() != carriedCount)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "split scf.for iter-arg/yield arity mismatch");

      for (unsigned carriedIndex = 0; carriedIndex < carriedCount;
           ++carriedIndex) {
        BlockArgument argument = body.getArgument(carriedIndex + 1);
        if (argument.use_empty() || isPointerLikeType(argument.getType()) ||
            isAddressOnlyLoopValue(argument))
          continue;

        Value yielded = terminator->getOperand(carriedIndex);
        if (yielded == argument)
          continue;
        Operation *producer = yielded.getDefiningOp();
        if (!producer)
          continue;
        auto producerOwner = operationOwners.find(producer);
        if (producerOwner == operationOwners.end())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "loop-carried value producer has no owning Stage");
        const size_t producerStageIndex = producerOwner->second;
        const Location producerLocation = producer->getLoc();

        llvm::DenseSet<size_t> consumerSet;
        for (OpOperand &use : argument.getUses()) {
          Operation *consumer = use.getOwner();
          if (consumer == terminator ||
              consumer->getName().getStringRef() == "scf.condition")
            continue;
          auto consumerOwner = operationOwners.find(consumer);
          if (consumerOwner == operationOwners.end())
            return llvm::createStringError(
                std::errc::invalid_argument,
                "loop-carried value consumer has no owning Stage");
          consumerSet.insert(consumerOwner->second);
        }

        llvm::SmallVector<size_t, 4> consumers(consumerSet.begin(),
                                               consumerSet.end());
        llvm::sort(consumers);
        std::string valueName = getCarriedValueName(producerLocation);
        if (valueName.empty())
          valueName = ("iter_arg_" + llvm::Twine(carriedIndex)).str();
        for (size_t consumerIndex : consumers) {
          StageDependency dependency;
          dependency.loopStage = loopShell.id;
          dependency.sourceStage = partition.stages[producerStageIndex].id;
          dependency.targetStage = partition.stages[consumerIndex].id;
          dependency.iterationDistance = 1;
          dependency.carriedValueIndex = carriedIndex;
          dependency.carriedValueName = valueName;
          dependency.carriedValueBytes = staticTensorBytes(argument);
          dependency.sourceLocation =
              printDependencySourceLocation(producerLocation);
          partition.dependencies.push_back(std::move(dependency));
        }
      }
    }
  }
  return llvm::Error::success();
}

static llvm::Expected<std::vector<int64_t>>
buildAnchorGroups(const ProgramStructure &structure,
                  const SimtAnchorPlan &anchorPlan) {
  std::vector<int64_t> groups(structure.rootOperations.size(), -1);
  int64_t nextGroup = 0;
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable)
      continue;
    llvm::SmallVector<size_t, 8> positions;
    auto addPosition = [&](Operation *operation) {
      Operation *root = getPartitionSemanticRoot(operation);
      auto iterator = llvm::find(structure.rootOperations, root);
      if (iterator == structure.rootOperations.end())
        return;
      const size_t position =
          static_cast<size_t>(iterator - structure.rootOperations.begin());
      if (!llvm::is_contained(positions, position))
        positions.push_back(position);
    };
    if (anchor.scopeOperations.empty())
      addPosition(anchor.operation);
    else
      for (Operation *operation : anchor.scopeOperations)
        addPosition(operation);
    if (positions.empty())
      continue;
    llvm::sort(positions);
    for (size_t index = 1; index < positions.size(); ++index)
      if (positions[index] != positions[index - 1] + 1)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "compound SIMT anchor roots are not contiguous after "
            "ProgramStructureAnalysis");

    int64_t group = -1;
    for (size_t position : positions) {
      if (groups[position] < 0)
        continue;
      if (group < 0)
        group = groups[position];
      else if (group != groups[position])
        return llvm::createStringError(
            std::errc::invalid_argument,
            "overlapping SIMT anchors define incompatible Stage boundaries");
    }
    if (group < 0)
      group = nextGroup++;
    for (size_t position : positions)
      groups[position] = group;
  }
  return groups;
}

llvm::Expected<StagePartition> StageBoundaryAnalysis::analyze(
    const ProgramStructure &structure, const SimtAnchorPlan &anchorPlan,
    const StageMemoryPatternAnalysis *memoryPatterns) const {
  if (structure.rootOperations.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis requires ordered semantic roots");
  auto anchorGroups = buildAnchorGroups(structure, anchorPlan);
  if (!anchorGroups)
    return anchorGroups.takeError();

  StagePartition partition;
  llvm::DenseSet<Operation *> owned;
  for (size_t index = 0; index < structure.rootOperations.size();) {
    Operation *root = structure.rootOperations[index];
    if (!root || !owned.insert(root).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StageBoundaryAnalysis received duplicate or null semantic root");

    const int64_t anchorGroup = (*anchorGroups)[index];
    StageCostModelKind kind = classifySemanticRoot(root, memoryPatterns);
    // A split loop's shell owns only control and carried-state semantics;
    // classification must not reach into the body it no longer owns.  Keep a
    // real recurrence on the shell, while independent PR #2112 loops retain
    // their pipelined-loop kind.
    if (shouldSplitLoopBodyIntoStages(root))
      kind = loopHasTrueCarriedDataDependency(root)
                 ? StageCostModelKind::LoopCarriedRecurrence
                 : StageCostModelKind::IndependentPipelinedLoop;
    StageScheduleKind schedule = scheduleForSemanticRoot(root, kind);
    LogicalStage stage;
    stage.operations.push_back(root);
    stage.iterationCount = semanticRootIterationCount(root);
    stage.localSimtMaterializable = anchorGroup >= 0;
    if (stage.localSimtMaterializable)
      stage.localSimtFactors = {1};

    size_t next = index + 1;
    while (next < structure.rootOperations.size()) {
      Operation *candidate = structure.rootOperations[next];
      const int64_t candidateAnchorGroup = (*anchorGroups)[next];
      const StageCostModelKind candidateKind =
          classifySemanticRoot(candidate, memoryPatterns);
      const StageScheduleKind candidateSchedule =
          scheduleForSemanticRoot(candidate, candidateKind);
      const bool sameCompoundAnchor =
          anchorGroup >= 0 && candidateAnchorGroup == anchorGroup;
      // Debug/source locations are deliberately excluded from Stage ownership.
      // Fold heterogeneous supporting work into a dominant resource Stage only
      // when the IR itself proves a producer-consumer relationship.
      const bool mergeStructuralSupportingWork =
          (isSupportingSemanticKind(kind) ||
           isSupportingSemanticKind(candidateKind)) &&
          schedule == StageScheduleKind::StraightLine &&
          candidateSchedule == StageScheduleKind::StraightLine &&
          hasDataDependencyOnStage(stage.operations, candidate);
      const bool sameSplitLoopRegion =
          getEnclosingSplitLoop(stage.operations.back()) ==
          getEnclosingSplitLoop(candidate);
      const bool mergePlainStage =
          anchorGroup < 0 && candidateAnchorGroup < 0 && sameSplitLoopRegion &&
          ((candidateKind == kind && candidateSchedule == schedule) ||
           mergeStructuralSupportingWork) &&
          !hasOrderedStageBoundary(stage.operations.back()) &&
          !hasOrderedStageBoundary(candidate);
      if (!sameCompoundAnchor && !mergePlainStage)
        break;
      if (!candidate || !owned.insert(candidate).second)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "StageBoundaryAnalysis overlaps semantic root ownership");
      stage.operations.push_back(candidate);
      stage.iterationCount =
          std::max(stage.iterationCount, semanticRootIterationCount(candidate));
      if (semanticKindPriority(candidateKind) > semanticKindPriority(kind)) {
        kind = candidateKind;
        schedule = candidateSchedule;
      }
      ++next;
    }
    stage.costModelKind = kind;
    stage.scheduleKind = schedule;
    stage.id = makeStageId(partition.stages.size(), kind);
    partition.stages.push_back(std::move(stage));
    index = next;
  }
  partition.operationOwnershipComplete = true;
  partition.modeledOperationCount =
      static_cast<int64_t>(structure.rootOperations.size());
  if (!partition.stages.empty())
    partition.stages.front().workload.paysKernelSetup = true;

  attachExactAnchorOwnership(partition, anchorPlan);
  deriveStageLiveValues(partition);
  deriveLocalSimtScopeTraffic(partition, anchorPlan);
  return partition;
}

llvm::Error StageFeatureAnalysis::analyze(
    StagePartition &partition,
    const StageMemoryPatternAnalysis *memoryPatterns) const {
  for (LogicalStage &stage : partition.stages) {
    StageModelFeatures &facts = stage.features;
    const double activeLaneRatio = facts.activeLaneRatio;
    const int64_t upperBoundLoopCount = facts.upperBoundLoopCount;
    const int64_t unknownTripCountLoopCount = facts.unknownTripCountLoopCount;
    facts = StageModelFeatures{};
    facts.activeLaneRatio = activeLaneRatio;
    facts.upperBoundLoopCount = upperBoundLoopCount;
    facts.unknownTripCountLoopCount = unknownTripCountLoopCount;
    llvm::DenseSet<Operation *> owned;
    for (Operation *root : stage.operations)
      collectOwnedOperationTree(root, owned);
    bool hasMemory = false;
    bool hasContiguousMemory = false;
    bool hasPartialContinuousMemory = false;
    bool hasOtherIndirectMemory = false;
    for (Operation *root : stage.operations) {
      facts.insideAutoBlockifyV1Loop |= isInsideAutoBlockifyV1Loop(root);
      facts.isGridStrideLoopShell |= isGridStrideLoop(root);
      facts.insideGridStrideLoopBody |= isInsideGridStrideLoop(root);
    }
    for (Operation *operation : owned) {
      const llvm::StringRef name = operation->getName().getStringRef();
      if (name == "scf.for" || name == "scf.while") {
        facts.hasLoop = true;
        ++facts.loopBackedgeCount;
        if (!operation->hasAttr("ta.auto_blockify_v1.loop") &&
            operation->getNumRegions() > 0 &&
            !operation->getRegion(0).empty() &&
            !hasKnownAtMostOneIteration(operation)) {
          Block &body = operation->getRegion(0).front();
          const unsigned firstCarriedArgument = name == "scf.for" ? 1 : 0;
          for (unsigned argumentIndex = firstCarriedArgument;
               argumentIndex < body.getNumArguments(); ++argumentIndex) {
            BlockArgument argument = body.getArgument(argumentIndex);
            if (argument.use_empty())
              continue;
            if (isPointerLikeType(argument.getType()) ||
                isAddressOnlyLoopValue(argument))
              facts.hasPointerInduction = true;
            else
              facts.hasLoopCarriedDataDependency = true;
          }
          if (name == "scf.for" && body.getNumArguments() > 0 &&
              isAddressOnlyLoopValue(body.getArgument(0)))
            facts.hasPointerInduction = true;
        }
      }
      if (name == "scf.if" || name == "cf.cond_br") {
        ++facts.conditionalBranchCount;
        ++facts.divergentBranchCount;
      }
      if (name.contains("barrier") || name.contains("sync"))
        ++facts.synchronizationCount;
      if (name == "tt.load" || name == "tt.store" || name == "tt.gather") {
        const bool scalarLoad = isScalarLoadOperation(operation);
        const bool scalarStore = isScalarStoreOperation(operation);
        if (!scalarLoad && !scalarStore) {
          hasMemory = true;
          const bool indirect =
              isIndirectMemoryOperation(operation, memoryPatterns) ||
              name == "tt.gather";
          facts.hasIndirectMemory |= indirect;
          hasContiguousMemory |= !indirect;
          if (indirect) {
            if (isMixedAxisMemory(operation, memoryPatterns))
              hasPartialContinuousMemory = true;
            else
              hasOtherIndirectMemory = true;
          }
        }
      }
      if (name.starts_with("tt.atomic")) {
        hasMemory = true;
        facts.hasAtomicMemory = true;
        facts.hasIndirectMemory |= isLoadedIndexDependentMemoryOp(operation);
        hasOtherIndirectMemory |= isLoadedIndexDependentMemoryOp(operation);
      }
      facts.hasReduction |=
          name == "tt.reduce" || name == "tt.scan" || name == "linalg.reduce";
      facts.hasPrefixScan |= name == "tt.scan";
      facts.hasDot |= isDotOperationName(name) || name.contains("matmul") ||
                      name.contains("mmad");
      facts.hasConversionPack |=
          name == "arith.extf" || name == "arith.truncf" ||
          name == "arith.fptosi" || name == "arith.fptoui" ||
          name == "arith.sitofp" || name == "arith.uitofp" ||
          name == "tt.fp_to_fp" || name.contains("convert") ||
          name.contains("pack") || name.contains("unpack");
    }
    facts.hasContiguousMemory = hasMemory && hasContiguousMemory;
    facts.hasPartialContinuousMemory =
        hasPartialContinuousMemory && !hasOtherIndirectMemory;
    if (!facts.isValid())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Stage '%s' has invalid features",
                                     stage.id.c_str());
  }
  return llvm::Error::success();
}

llvm::Error StageKindClassifier::analyze(StagePartition &partition,
                                         int64_t tinyDotFlopsMax) const {
  if (!partition.operationOwnershipComplete)
    return llvm::Error::success();
  auto compatible = [](StageCostModelKind kind, const StageModelFeatures &facts,
                       const StageWorkload &workload) {
    switch (kind) {
    case StageCostModelKind::LoopCarriedRecurrence:
      return facts.hasLoopCarriedDataDependency;
    case StageCostModelKind::IndependentPipelinedLoop:
      return facts.hasLoop && !facts.hasLoopCarriedDataDependency;
    case StageCostModelKind::RowwiseReduction:
      return facts.hasReduction;
    case StageCostModelKind::PrefixScan:
      return facts.hasPrefixScan;
    case StageCostModelKind::CubeRoofline:
    case StageCostModelKind::TinyCubeRoofline:
      return facts.hasDot;
    case StageCostModelKind::IndirectScalarMemory:
      return facts.hasIndirectMemory;
    case StageCostModelKind::IndirectGatherMemory:
      return hasAllDAddressPattern(workload);
    case StageCostModelKind::PartialContinuousTileMemory:
      return facts.hasPartialContinuousMemory;
    case StageCostModelKind::AtomicMemory:
      return facts.hasAtomicMemory;
    case StageCostModelKind::ContinuousTileMemory:
    case StageCostModelKind::ContinuousTileStore:
    case StageCostModelKind::ContinuousShortLoad:
    case StageCostModelKind::CachePolicyStore:
      return facts.hasContiguousMemory;
    case StageCostModelKind::ConversionPack:
      return facts.hasConversionPack;
    case StageCostModelKind::ScalarLoad:
      return workload.scalarLoadCount > 0.0 && !facts.hasContiguousMemory;
    case StageCostModelKind::ScalarStore:
      return workload.scalarStoreCount > 0.0 && !facts.hasContiguousMemory;
    default:
      return true;
    }
  };
  for (LogicalStage &stage : partition.stages) {
    const StageModelFeatures &facts = stage.features;
    const bool autoBlockifyKind =
        stage.costModelKind == StageCostModelKind::AutoBlockifyDispatch ||
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop;
    const bool hasAlgorithmicPayload =
        stage.workload.loadBytes > 0.0 || stage.workload.storeBytes > 0.0 ||
        stage.workload.loadOperations > 0.0 ||
        stage.workload.storeOperations > 0.0 ||
        !stage.workload.atomicWorkloads.empty() ||
        stage.workload.reductionOperations > 0.0 ||
        stage.workload.dotOperations > 0.0 || stage.workload.dotFlops > 0.0 ||
        !stage.workload.operationCounts.empty() ||
        !stage.workload.prefixScanCounts.empty() ||
        stage.workload.scalarLoadCount > 0.0 ||
        stage.workload.scalarStoreCount > 0.0;
    // collectTopLevelSemanticRoots recursively exposes execute_region bodies.
    const bool autoBlockifyBody =
        (autoBlockifyKind || facts.insideAutoBlockifyV1Loop) &&
        hasAlgorithmicPayload;
    if (autoBlockifyKind && !autoBlockifyBody)
      continue;
    if (!autoBlockifyBody && facts.hasDot &&
        (facts.hasReduction || facts.hasIndirectMemory ||
         facts.hasAtomicMemory || facts.hasLoopCarriedDataDependency))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "requires_split: Stage '%s' owns incompatible dominant structures",
          stage.id.c_str());

    auto derive = [&]() {
      if (facts.hasLoopCarriedDataDependency)
        return StageCostModelKind::LoopCarriedRecurrence;
      if (facts.hasPrefixScan)
        return StageCostModelKind::PrefixScan;
      if (facts.hasReduction)
        return StageCostModelKind::RowwiseReduction;
      if (facts.hasDot) {
        const int64_t tinyLimit = std::max<int64_t>(1, tinyDotFlopsMax);
        // `dotFlops` is the sum of all matrix instances in this Stage.
        const bool allMatricesTiny =
            !stage.workload.dotFlopCounts.empty()
                ? llvm::all_of(stage.workload.dotFlopCounts,
                               [&](const auto &entry) {
                                 return entry.first <= tinyLimit;
                               })
                : stage.workload.dotFlops * stage.iterationCount <=
                      static_cast<double>(tinyLimit);
        return allMatricesTiny ? StageCostModelKind::TinyCubeRoofline
                               : StageCostModelKind::CubeRoofline;
      }
      if (facts.hasLoop)
        return StageCostModelKind::IndependentPipelinedLoop;
      if (facts.hasAtomicMemory)
        return StageCostModelKind::AtomicMemory;
      if (facts.hasPartialContinuousMemory)
        return StageCostModelKind::PartialContinuousTileMemory;
      if (hasAllDAddressPattern(stage.workload))
        return StageCostModelKind::IndirectGatherMemory;
      if (facts.hasConversionPack)
        return StageCostModelKind::ConversionPack;
      if (stage.workload.scalarLoadCount > 0.0 && !facts.hasContiguousMemory)
        return StageCostModelKind::ScalarLoad;
      if (stage.workload.scalarStoreCount > 0.0 && !facts.hasContiguousMemory)
        return StageCostModelKind::ScalarStore;
      if (facts.hasContiguousMemory)
        return stage.workload.storeBytes > 0.0 &&
                       stage.workload.loadBytes == 0.0
                   ? StageCostModelKind::ContinuousTileStore
                   : StageCostModelKind::ContinuousTileMemory;
      if (llvm::any_of(stage.operations, operationTreeHasVectorCompute)) {
        // Predicate/cast/index models remain specialized. Ordinary tensor
        // arithmetic must not fall back to scalar_issue after partitioning.
        if (stage.costModelKind == StageCostModelKind::PredicateMask ||
            stage.costModelKind == StageCostModelKind::IndexGeneration)
          return stage.costModelKind;
        return StageCostModelKind::VectorIssue;
      }
      return StageCostModelKind::ScalarIssue;
    };
    const StageCostModelKind derived = derive();
    // Strong operation-graph semantics are authoritative.  Scalar
    // sub-kinds remain useful only when no dominant structure is present.
    if (semanticKindPriority(derived) > 0 ||
        !compatible(stage.costModelKind, facts, stage.workload))
      stage.costModelKind = derived;
    const bool tinyDotShapeValid = [&]() {
      if (stage.costModelKind != StageCostModelKind::TinyCubeRoofline)
        return true;
      const int64_t tinyLimit = std::max<int64_t>(1, tinyDotFlopsMax);
      if (!stage.workload.dotFlopCounts.empty())
        return llvm::all_of(
            stage.workload.dotFlopCounts,
            [&](const auto &entry) { return entry.first <= tinyLimit; });
      return stage.workload.dotFlops * stage.iterationCount <=
             static_cast<double>(tinyLimit);
    }();
    if (!compatible(stage.costModelKind, facts, stage.workload) ||
        !tinyDotShapeValid)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage '%s' operation graph does not match StageCostModelKind '%s'",
          stage.id.c_str(),
          stringifyStageCostModel(stage.costModelKind).str().c_str());
    if (stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop)
      stage.scheduleKind = StageScheduleKind::IndependentPipelined;
    else if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence)
      stage.scheduleKind = StageScheduleKind::LoopCarriedSerial;
  }
  for (auto indexedStage : llvm::enumerate(partition.stages))
    indexedStage.value().id =
        makeStageId(indexedStage.index(), indexedStage.value().costModelKind);
  return llvm::Error::success();
}

llvm::Error StageWorkloadAnalysis::analyze(
    StagePartition &partition, int64_t logicalProgramCountHint,
    const StageMemoryPatternAnalysis *memoryPatterns) const {
  if (!partition.operationOwnershipComplete)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageWorkloadAnalysis requires complete operation ownership");
  for (LogicalStage &stage : partition.stages) {
    stage.reductionContributions.clear();
    StageWorkload work;
    work.paysKernelSetup = stage.workload.paysKernelSetup;
    if (memoryPatterns)
      for (Operation *root : stage.operations)
        root->walk([&](Operation *operation) {
          const auto name = operation->getName().getStringRef();
          if (name != "tt.load" && name != "tt.store")
            return;
          if (const auto *summary = memoryPatterns->lookupSummary(operation)) {
            work.addressPatterns.push_back(*summary);
            work.addressPatterns.back().stageId = stage.id;
          }
        });
    if (stage.iterationCount > 0)
      stage.iterationCount = std::max(
          stage.iterationCount,
          inferRuntimeGridStrideIterations(stage, logicalProgramCountHint));
    const int64_t loopCount = countAlgorithmLoops(stage);
    const int64_t fallbackLoopTripCount =
        loopCount > 0 ? std::max<int64_t>(1, stage.iterationCount / loopCount)
                      : 1;
    stage.features.upperBoundLoopCount = 0;
    stage.features.unknownTripCountLoopCount = 0;
    for (Operation *root : stage.operations)
      accumulateDynamicOperationTree(
          root, work, semanticRootEntryMultiplicity(root),
          fallbackLoopTripCount, stage.features, logicalProgramCountHint, {},
          false, &stage.reductionContributions, memoryPatterns);
    StageWorkload external;
    external.paysKernelSetup = work.paysKernelSetup;
    // Ordinary fitted reductions remove only pure combine work; their external
    // memory operations retain the same static address descriptions.
    external.addressPatterns = work.addressPatterns;
    auto ignoredFeatures = stage.features;
    for (Operation *root : stage.operations)
      accumulateDynamicOperationTree(
          root, external, semanticRootEntryMultiplicity(root),
          fallbackLoopTripCount, ignoredFeatures, logicalProgramCountHint, {},
          true, nullptr, memoryPatterns);
    scaleWorkload(external, 1.0 / std::max<int64_t>(1, stage.iterationCount));
    stage.reductionExternalWorkload = std::move(external);
    for (auto &contribution : stage.reductionContributions)
      scaleWorkload(contribution,
                    1.0 / std::max<int64_t>(1, stage.iterationCount));
    recomputeIssueElements(work);
    stage.workload = std::move(work);
    makePerIteration(stage);
    stage.features.activeLaneRatio = 1.0;
    for (const StageMemoryAccess &access : stage.workload.memoryAccesses)
      stage.features.activeLaneRatio =
          std::min(stage.features.activeLaneRatio, access.activeElementRatio);
    if (!stage.workload.isFiniteAndNonNegative())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage '%s' has invalid operation-derived workload",
          stage.id.c_str());
  }
  return llvm::Error::success();
}

llvm::Error
StagePartitionVerifier::verify(const StagePartition &partition) const {
  if (partition.stages.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StagePartition has no Stage");
  llvm::StringSet<> stageIds;
  llvm::DenseSet<Operation *> ownedOperations;
  llvm::DenseSet<unsigned> ownedAnchors;
  for (const LogicalStage &stage : partition.stages) {
    if (stage.id.empty() || !stageIds.insert(stage.id).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "StagePartition has duplicate Stage id");
    if (stage.iterationCount < 0)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Stage '%s' has invalid iteration count",
                                     stage.id.c_str());
    if (stage.localSimtMaterializable && partition.operationOwnershipComplete &&
        stage.operations.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "materializable Stage '%s' has no operation ownership",
          stage.id.c_str());
    if (stage.localSimtMaterializable && partition.operationOwnershipComplete &&
        stage.simtAnchorIndices.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "materializable Stage '%s' has no exact SIMT anchor ownership",
          stage.id.c_str());
    if (partition.operationOwnershipComplete)
      for (unsigned index : stage.simtAnchorIndices)
        if (!ownedAnchors.insert(index).second)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "StagePartition SIMT anchor ownership overlaps");
    if (partition.operationOwnershipComplete)
      for (Operation *operation : stage.operations)
        if (!operation || !ownedOperations.insert(operation).second)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "StagePartition operation ownership overlaps");
  }
  for (const LogicalStage &stage : partition.stages)
    for (const std::string &loopStage : stage.enclosingLoopStages)
      if (!stageIds.contains(loopStage))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' references an unknown enclosing loop Stage",
            stage.id.c_str());
  llvm::StringSet<> dependencyKeys;
  for (const StageDependency &dependency : partition.dependencies) {
    if (!dependency.isValid() ||
        (dependency.kind == StageDependencyKind::LoopCarriedData &&
         dependency.iterationDistance != 1) ||
        (!dependency.loopStage.empty() &&
         !stageIds.contains(dependency.loopStage)) ||
        !stageIds.contains(dependency.sourceStage) ||
        !stageIds.contains(dependency.targetStage))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StagePartition has an invalid loop-carried dependency");
    if (dependency.kind == StageDependencyKind::LoopCarriedData) {
      auto loop =
          llvm::find_if(partition.stages, [&](const LogicalStage &stage) {
            return stage.id == dependency.loopStage;
          });
      if (loop == partition.stages.end() ||
          !loop->features.hasLoopCarriedDataDependency)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage dependency loop shell has no recurrence feature");
      const auto belongsToLoop = [&](llvm::StringRef stageId) {
        auto owner =
            llvm::find_if(partition.stages, [&](const LogicalStage &stage) {
              return stage.id == stageId;
            });
        return owner != partition.stages.end() &&
               llvm::is_contained(owner->enclosingLoopStages,
                                  dependency.loopStage);
      };
      if (!belongsToLoop(dependency.sourceStage) ||
          !belongsToLoop(dependency.targetStage))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "loop-carried dependency endpoints are outside its loop Stage");
    }
    std::string key = (llvm::Twine(static_cast<unsigned>(dependency.kind)) +
                       ":" + dependency.loopStage + ":" +
                       dependency.sourceStage + ":" + dependency.targetStage +
                       ":" + llvm::Twine(dependency.carriedValueIndex) + ":" +
                       llvm::Twine(dependency.iterationDistance) + ":" +
                       dependency.carriedValueName)
                          .str();
    if (!dependencyKeys.insert(key).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StagePartition has duplicate loop-carried dependencies");
  }
  if (partition.operationOwnershipComplete &&
      static_cast<int64_t>(ownedOperations.size()) !=
          partition.modeledOperationCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition operation ownership is incomplete");
  if (!partition.operationOwnershipComplete)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition requires complete TTIR operation ownership");
  return llvm::Error::success();
}

llvm::Error
StageModeLegalityAnalysis::analyze(StagePartition &partition,
                                   int64_t maximumSuperblockFactor,
                                   bool scopeSuperblockMaterializable,
                                   int64_t maximumScopeSuperblockFactor) const {
  const int64_t maximum = std::clamp<int64_t>(maximumSuperblockFactor, 1, 64);
  // Local and whole-kernel SuperBlock candidates consume the same SIMT warp
  // resources.
  const int64_t localMaximum =
      scopeSuperblockMaterializable
          ? std::min(maximum,
                     std::clamp<int64_t>(maximumScopeSuperblockFactor, 1, 64))
          : 1;
  for (LogicalStage &stage : partition.stages) {
    stage.simdLegal = true;
    stage.simtLegal = true;
    bool requiresFactorOneLocalSimt = false;
    for (Operation *root : stage.operations) {
      root->walk([&](Operation *operation) {
        // ascend.sort lowers through the SIMD-only native vsort template.
        if (operation->getName().getStringRef() == "ascend.sort")
          stage.simtLegal = false;
        if (operation->getName().getStringRef() == "tt.histogram")
          requiresFactorOneLocalSimt = true;
        if (operation->getName().getStringRef() != "tt.scan")
          return;
        // Judge the combine by the values returned from the region.
        if (classifyScanCombine(operation) == "other")
          stage.simdLegal = false;
      });
    }
    stage.legalSimtFactors = {1};
    // A pure-SIMT SuperBlock factor is a whole-kernel schedule, not a
    // recurrence-only annotation.
    for (int64_t factor = 2; factor <= maximum; factor *= 2)
      stage.legalSimtFactors.push_back(factor);
    if (stage.localSimtMaterializable) {
      // The ABI-v2 scope materializer batches complete logical programs around
      // this Stage.
      stage.localSimtFactors = {1};
      if (scopeSuperblockMaterializable && stage.localSuperblockMaterializable)
        for (int64_t factor = 2; factor <= localMaximum; factor *= 2)
          stage.localSimtFactors.push_back(factor);
    }
    if (requiresFactorOneLocalSimt && stage.localSimtMaterializable) {
      stage.localSuperblockMaterializable = false;
      stage.localSimtFactors = {1};
    }
    if (stage.localSimtMaterializable &&
        (stage.localSimtFactors.empty() ||
         llvm::any_of(stage.localSimtFactors, [&](int64_t factor) {
           return factor < 1 || factor > localMaximum ||
                  (factor & (factor - 1)) != 0;
         })))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "local SIMT factors are invalid for Stage '%s'", stage.id.c_str());
  }
  return llvm::Error::success();
}

llvm::Error StageResourceLegalityAnalysis::applySimtUbCapacity(
    StagePartition &partition, int64_t usableUbBytes) const {
  if (usableUbBytes < 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMT usable UB capacity must be nonnegative");
  if (usableUbBytes == 0)
    return llvm::Error::success();

  for (LogicalStage &stage : partition.stages) {
    stage.simtUsableUbBytes = usableUbBytes;
    stage.simtUbWorkingSetBytesPerProgram = 0;
    stage.simtFactorsRejectedByUb.clear();
    if (!stage.wholeSimtRegisterPressure)
      continue;
    const SimtRegisterPressure &facts = *stage.wholeSimtRegisterPressure;
    if (!facts.ttirMaximumWeakDotComponentGeometryKnown)
      continue;
    const int64_t m = facts.ttirMaximumWeakDotComponentM;
    const int64_t n = facts.ttirMaximumWeakDotComponentN;
    const int64_t k = facts.ttirMaximumWeakDotComponentK;
    const int64_t elementBits = facts.ttirMaximumWeakDotComponentElementBits;
    if (m <= 0 || n <= 0 || k <= 0 || elementBits <= 0)
      continue;

    // lhs[M,K], rhs[K,N] and accumulator[M,N] are simultaneously visible at a
    // dot in TTIR.
    const __int128 elements = static_cast<__int128>(m) * k +
                              static_cast<__int128>(k) * n +
                              static_cast<__int128>(m) * n;
    const __int128 elementBytes = (static_cast<__int128>(elementBits) + 7) / 8;
    const __int128 workingSet = elements * elementBytes;
    stage.simtUbWorkingSetBytesPerProgram =
        workingSet > std::numeric_limits<int64_t>::max()
            ? std::numeric_limits<int64_t>::max()
            : static_cast<int64_t>(workingSet);

    auto exceedsCapacity = [&](int64_t factor) {
      return factor <= 0 || workingSet * factor > usableUbBytes;
    };
    auto filter = [&](std::vector<int64_t> &factors) {
      llvm::erase_if(factors, [&](int64_t factor) {
        if (!exceedsCapacity(factor))
          return false;
        if (!llvm::is_contained(stage.simtFactorsRejectedByUb, factor))
          stage.simtFactorsRejectedByUb.push_back(factor);
        return true;
      });
    };
    filter(stage.legalSimtFactors);
    filter(stage.localSimtFactors);
    llvm::sort(stage.simtFactorsRejectedByUb);
    stage.simtLegal =
        !stage.legalSimtFactors.empty() ||
        (stage.localSimtMaterializable && !stage.localSimtFactors.empty());
  }
  return llvm::Error::success();
}

llvm::Expected<StagePartition>
StagePartitioner::partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
                            const StagePartitionerOptions &options) const {
  auto structure = ProgramStructureAnalysis().analyze(module, anchorPlan);
  if (!structure)
    return structure.takeError();
  StageMemoryPatternAnalysis memoryPatterns(module);
  auto result =
      StageBoundaryAnalysis().analyze(*structure, anchorPlan, &memoryPatterns);
  if (!result)
    return result.takeError();
  StageWorkloadAnalysis workloadAnalysis;
  if (llvm::Error error = workloadAnalysis.analyze(
          *result, options.logicalProgramCountHint, &memoryPatterns))
    return std::move(error);
  StageFeatureAnalysis featureAnalysis;
  if (llvm::Error error = featureAnalysis.analyze(*result, &memoryPatterns))
    return std::move(error);
  if (llvm::Error error =
          StageKindClassifier().analyze(*result, options.tinyDotFlopsMax))
    return std::move(error);
  if (llvm::Error error = deriveStageLoopMembership(*result))
    return std::move(error);
  if (llvm::Error error = deriveStageDependencies(*result))
    return std::move(error);
  StageModeLegalityAnalysis legalityAnalysis;
  if (llvm::Error error =
          legalityAnalysis.analyze(*result, options.maximumSuperblockFactor,
                                   options.scopeSuperblockMaterializable,
                                   options.maximumScopeSuperblockFactor))
    return std::move(error);
  if (llvm::Error error = StagePartitionVerifier().verify(*result))
    return std::move(error);
  return std::move(*result);
}

void mlir::ascend::mergeReductionResidualWorkload(StageWorkload &into,
                                                  StageWorkload from) {
  mergeWorkload(into, std::move(from));
}
