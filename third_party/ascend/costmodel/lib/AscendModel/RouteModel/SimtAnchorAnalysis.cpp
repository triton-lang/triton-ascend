//===- SimtAnchorAnalysis.cpp - Materializable SIMT anchors --------------===//

#include "AscendModel/RouteModel/SimtAnchorAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static int64_t getStaticNumElements(RankedTensorType type) {
  if (!type || !type.hasStaticShape())
    return 0;
  return std::accumulate(type.getShape().begin(), type.getShape().end(),
                         int64_t{1}, std::multiplies<int64_t>());
}

static std::string getScalarTypeName(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto integer = dyn_cast<IntegerType>(type))
    return ("i" + std::to_string(integer.getWidth()));
  if (type.isIndex())
    return "index";
  return "unknown";
}

static std::optional<PlainCumsumFacts>
analyzePlainOneDimensionalCumsum(Operation *op) {
  if (!op || op->getName().getStringRef() != "tt.scan" ||
      op->getNumOperands() != 1 || op->getNumResults() != 1)
    return std::nullopt;

  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  auto sourceType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!axis || !sourceType || !sourceType.hasStaticShape())
    return std::nullopt;

  int64_t axisValue = axis.getInt();
  if (axisValue < 0 || axisValue >= sourceType.getRank())
    return std::nullopt;
  for (auto [index, extent] : llvm::enumerate(sourceType.getShape()))
    if (static_cast<int64_t>(index) != axisValue && extent != 1)
      return std::nullopt;

  int64_t realCombineOps = 0;
  bool isAdd = false;
  if (op->getNumRegions() != 1 || op->getRegion(0).empty())
    return std::nullopt;
  Block &body = op->getRegion(0).front();
  Operation *terminator = body.empty() ? nullptr : &body.back();
  for (Operation &nested : body) {
    if (&nested == terminator)
      continue;
    llvm::StringRef name = nested.getName().getStringRef();
    if (name == "arith.extf" || name == "arith.truncf" ||
        name == "arith.bitcast")
      continue;
    ++realCombineOps;
    isAdd = name == "arith.addf" || name == "arith.addi";
  }
  if (realCombineOps != 1 || !isAdd || !terminator ||
      terminator->getName().getStringRef() != "tt.scan.return")
    return std::nullopt;

  PlainCumsumFacts facts;
  facts.axisExtent = sourceType.getShape()[axisValue];
  facts.elementType = getScalarTypeName(sourceType.getElementType());
  if (auto reverse = op->getAttrOfType<BoolAttr>("reverse"))
    facts.reverse = reverse.getValue();
  return facts;
}

static bool hasTensorPointerOperand(Operation *op) {
  if (!op || op->getNumOperands() == 0)
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  return type && type.hasStaticShape() && type.getRank() <= 5;
}

static bool pointerDependsOnLoadedIndex(Operation *memoryOp) {
  if (!memoryOp || memoryOp->getNumOperands() == 0)
    return false;
  llvm::SmallVector<Value> worklist{memoryOp->getOperand(0)};
  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      unsigned number = argument.getArgNumber();
      if (parent && parent->getName().getStringRef() == "scf.for" &&
          number > 0 && number + 2 < parent->getNumOperands()) {
        worklist.push_back(parent->getOperand(number + 2));
        Operation *yield = argument.getOwner()->getTerminator();
        if (yield && number - 1 < yield->getNumOperands())
          worklist.push_back(yield->getOperand(number - 1));
      }
      continue;
    }

    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    llvm::StringRef name = producer->getName().getStringRef();
    if (name == "tt.load" || name == "tt.gather")
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

static Value findPointerOffset(Value pointer) {
  llvm::SmallVector<Value> worklist{pointer};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    if (producer->getName().getStringRef() == "tt.addptr" &&
        producer->getNumOperands() > 1)
      return producer->getOperand(1);
    llvm::append_range(worklist, producer->getOperands());
  }
  return {};
}

static std::optional<double> getStaticMaskActiveFraction(Value mask) {
  if (!mask)
    return std::nullopt;
  Operation *producer = mask.getDefiningOp();
  if (!producer)
    return std::nullopt;
  if (producer->getName().getStringRef() == "arith.constant") {
    Attribute value = producer->getAttr("value");
    if (auto dense = dyn_cast_or_null<DenseElementsAttr>(value)) {
      if (!dense.getElementType().isInteger(1) || dense.getNumElements() == 0)
        return std::nullopt;
      int64_t active = 0;
      for (bool item : dense.getValues<bool>())
        active += item;
      return static_cast<double>(active) / dense.getNumElements();
    }
    if (auto integer = dyn_cast_or_null<IntegerAttr>(value))
      return integer.getValue().isZero() ? 0.0 : 1.0;
  }
  if (producer->getName().getStringRef() == "tt.splat" &&
      producer->getNumOperands() == 1)
    return getStaticMaskActiveFraction(producer->getOperand(0));
  return std::nullopt;
}

static std::string getAtomicOperation(Operation *op) {
  if (op->getName().getStringRef() == "tt.atomic_cas")
    return "cas";
  Attribute attribute = op->getAttr("atomic_rmw_op");
  if (!attribute)
    return "unknown";
  if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
    switch (integer.getInt()) {
    case 1:
      return "and";
    case 2:
      return "or";
    case 3:
      return "xor";
    case 4:
      return "add";
    case 5:
      return "fadd";
    case 6:
      return "max";
    case 7:
      return "min";
    case 8:
      return "umax";
    case 9:
      return "umin";
    case 10:
      return "xchg";
    default:
      return "unknown";
    }
  }
  std::string text;
  llvm::raw_string_ostream os(text);
  attribute.print(os);
  os.flush();
  for (char &character : text)
    character = llvm::toLower(character);
  for (llvm::StringRef name : {"fadd", "umax", "umin", "xchg", "exch", "and",
                               "xor", "add", "max", "min", "or"})
    if (llvm::StringRef(text).contains(name))
      return name == "exch" ? "xchg" : name.str();
  return "unknown";
}

static bool isSupportedCumsumType(llvm::StringRef type) {
  return type == "i8" || type == "i16" || type == "i32" || type == "f16" ||
         type == "bf16" || type == "f32";
}

static bool isSupportedAtomicType(llvm::StringRef type,
                                  llvm::StringRef operation) {
  if (operation == "and" || operation == "or" || operation == "xor")
    return type == "i32" || type == "i64";
  if (operation == "xchg" || operation == "cas")
    return type == "i32" || type == "i64" || type == "f32";
  if (operation == "add" || operation == "fadd" || operation == "max" ||
      operation == "min" || operation == "umax" || operation == "umin")
    return type == "i32" || type == "i64" || type == "f16" || type == "bf16" ||
           type == "f32";
  return false;
}

static bool isTriangularSolveLoop(Operation *op) {
  if (!op || op->getName().getStringRef() != "scf.for" ||
      op->getNumRegions() != 1 || op->getRegion(0).empty())
    return false;

  bool hasVectorLoad = false;
  bool hasAxisZeroReduce = false;
  bool hasMaskedUpdate = false;
  Block &body = op->getRegion(0).front();
  body.walk([&](Operation *nested) {
    llvm::StringRef name = nested->getName().getStringRef();
    if (name == "tt.load" && nested->getNumResults() == 1) {
      if (auto type =
              dyn_cast<RankedTensorType>(nested->getResult(0).getType()))
        hasVectorLoad |= type.hasStaticShape() && type.getRank() == 1 &&
                         type.getShape()[0] == 16;
    } else if (name == "tt.reduce") {
      auto axis = nested->getAttrOfType<IntegerAttr>("axis");
      hasAxisZeroReduce |= axis && axis.getInt() == 0;
    } else if (name == "arith.select") {
      hasMaskedUpdate = true;
    }
  });
  if (!hasVectorLoad || !hasAxisZeroReduce || !hasMaskedUpdate)
    return false;

  // The production solve_tril kernel has one block-update loop.  Its
  // iter_args carry the 16x16 triangular state; requiring that fixed state
  // shape keeps this single-loop form distinct from a generic vector reduce.
  bool hasTriangularState = false;
  for (BlockArgument argument : body.getArguments()) {
    if (auto type = dyn_cast<RankedTensorType>(argument.getType()))
      hasTriangularState |= type.hasStaticShape() && type.getRank() == 2 &&
                            type.getShape()[0] == 16 &&
                            type.getShape()[1] == 16;
  }

  // Multi-loop lowering variants use two or more sibling block-update loops;
  // retain that form for kernels whose state shape is not explicit in the
  // loop arguments.
  Block *parentBlock = op->getBlock();
  if (!parentBlock)
    return false;
  unsigned siblingMatches = 0;
  for (Operation &sibling : *parentBlock) {
    if (&sibling == op || sibling.getName().getStringRef() != "scf.for")
      continue;
    bool siblingLoad = false;
    bool siblingReduce = false;
    bool siblingSelect = false;
    sibling.walk([&](Operation *nested) {
      llvm::StringRef name = nested->getName().getStringRef();
      if (name == "tt.load" && nested->getNumResults() == 1)
        if (auto type =
                dyn_cast<RankedTensorType>(nested->getResult(0).getType()))
          siblingLoad |= type.hasStaticShape() && type.getRank() == 1 &&
                         type.getShape()[0] == 16;
      siblingReduce |= name == "tt.reduce";
      siblingSelect |= name == "arith.select";
    });
    siblingMatches += siblingLoad && siblingReduce && siblingSelect;
  }
  return hasTriangularState || siblingMatches >= 1;
}

static bool isMovableTriangularTensorSetup(Operation *op) {
  if (!op || op->getNumResults() == 0 || op->getNumRegions() != 0)
    return false;
  if (!llvm::all_of(op->getResultTypes(),
                    [](Type type) { return isa<RankedTensorType>(type); }))
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  // Triton hoists both integer offset splats and floating-point zero splats
  // out of a hand-written scope. Keep all arith.constant ops as captures so
  // the automatically materialized scope has the same boundary contract.
  return name == "tt.make_range" || name == "tt.expand_dims" ||
         name == "tt.broadcast" || name == "tt.splat" || name == "arith.cmpi" ||
         name == "arith.cmpf" || name == "arith.andi" || name == "arith.ori" ||
         name == "arith.xori" || name == "arith.addi" || name == "arith.subi" ||
         name == "arith.muli" || name == "arith.index_cast";
}

static Operation *getTopLevelOwnerInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

/// Return the exact TTIR operations represented by the hand-written
/// triangular-solve scope. The four 16x16 input loads stay outside. Pure
/// ranked-tensor mask setup is moved after those loads, then the sibling
/// recurrences and final diagonal updates execute inside the scope. The dense
/// tt.dot tail remains outside. Keeping this in analysis makes the plan
/// authoritative; the materializer must not rediscover a different set.
static llvm::SmallVector<Operation *, 0>
collectTriangularSolveScopeOperations(Operation *anchor,
                                      Operation *&insertionPoint) {
  llvm::SmallVector<Operation *, 0> result;
  insertionPoint = nullptr;
  if (!isTriangularSolveLoop(anchor) || !anchor->getBlock())
    return result;

  Block *block = anchor->getBlock();
  Operation *firstLoop = nullptr;
  Operation *lastLoop = nullptr;
  for (Operation &nested : *block) {
    if (!isTriangularSolveLoop(&nested))
      continue;
    if (!firstLoop)
      firstLoop = &nested;
    lastLoop = &nested;
  }
  if (!firstLoop || !lastLoop)
    return result;

  Operation *lastInputLoad = nullptr;
  Operation *cursor = firstLoop->getPrevNode();
  while (cursor && cursor->getName().getStringRef() != "tt.load")
    cursor = cursor->getPrevNode();
  if (cursor)
    lastInputLoad = cursor;
  Operation *start = lastInputLoad ? lastInputLoad->getNextNode() : firstLoop;
  insertionPoint = start;

  Operation *end = lastLoop;
  cursor = lastLoop->getNextNode();
  while (cursor) {
    llvm::StringRef name = cursor->getName().getStringRef();
    if (name != "arith.uitofp" && name != "arith.addf" &&
        name != "arith.select")
      break;
    end = cursor;
    cursor = cursor->getNextNode();
  }

  llvm::SmallVector<Operation *, 0> recurrenceRange;
  llvm::DenseSet<Operation *> selected;
  for (Operation *op = start; op; op = op->getNextNode()) {
    recurrenceRange.push_back(op);
    selected.insert(op);
    if (op == end)
      break;
  }
  if (recurrenceRange.empty() || recurrenceRange.back() != end)
    return {};

  // In the unscope source, o_i/m_A/m_I are constructed before the initial
  // block loads. The hand-written scope constructs them after the loads.
  // Pull in only a side-effect-free ranked-tensor backward slice whose uses
  // are entirely inside the selected scope; scalar/pointer setup and all
  // memory operations remain captures outside.
  llvm::DenseSet<Operation *> setupCandidates;
  llvm::SmallVector<Value, 16> worklist;
  for (Operation *op : recurrenceRange) {
    llvm::append_range(worklist, op->getOperands());
    op->walk([&](Operation *nested) {
      if (nested != op)
        llvm::append_range(worklist, nested->getOperands());
    });
  }
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || definingOp->getBlock() != block ||
        definingOp == insertionPoint || !lastInputLoad ||
        !definingOp->isBeforeInBlock(lastInputLoad) ||
        !isMovableTriangularTensorSetup(definingOp) ||
        !setupCandidates.insert(definingOp).second)
      continue;
    llvm::append_range(worklist, definingOp->getOperands());
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *candidate : llvm::make_early_inc_range(setupCandidates)) {
      bool hasOutsideUse =
          llvm::any_of(candidate->getResults(), [&](Value resultValue) {
            return llvm::any_of(resultValue.getUses(), [&](OpOperand &use) {
              Operation *owner = getTopLevelOwnerInBlock(use.getOwner(), block);
              return !selected.contains(owner) &&
                     !setupCandidates.contains(owner);
            });
          });
      if (!hasOutsideUse)
        continue;
      setupCandidates.erase(candidate);
      changed = true;
    }
  }

  for (Operation &op : *block)
    if (setupCandidates.contains(&op))
      result.push_back(&op);
  llvm::append_range(result, recurrenceRange);
  return result;
}

static CandidateLowerability
simpleMixedLowerability(llvm::StringRef allSimtReason) {
  CandidateLowerability result;
  result.allSimtOnly = CandidateLoweringStatus::BackendConditional;
  result.allSimtOnlyReasons.push_back(allSimtReason.str());
  return result;
}

static std::optional<SimtAnchorDescriptor> analyzeAnchor(Operation *op,
                                                         bool compileOn91095) {
  if (!op)
    return std::nullopt;
  SimtAnchorDescriptor descriptor;
  descriptor.operation = op;
  llvm::StringRef name = op->getName().getStringRef();

  if (name == "tt.gather") {
    descriptor.kind = SimtAnchorKind::DirectGather;
    descriptor.lowerability = simpleMixedLowerability(
        "whole_module_pure_simt_requires_backend_check");
  } else if (name == "tt.histogram") {
    descriptor.kind = SimtAnchorKind::Histogram;
    HistogramFacts facts;
    auto input = op->getNumOperands() > 0
                     ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                     : RankedTensorType();
    auto result = op->getNumResults() > 0
                      ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                      : RankedTensorType();
    facts.inputElements = getStaticNumElements(input);
    facts.numBins = result && result.hasStaticShape() && result.getRank() == 1
                        ? result.getShape()[0]
                        : 0;
    facts.inputType = input ? getScalarTypeName(input) : "unknown";
    facts.resultType = result ? getScalarTypeName(result) : "unknown";
    descriptor.facts = facts;
    descriptor.lowerability.allSimd = CandidateLoweringStatus::Unsupported;
    descriptor.lowerability.allSimdReasons.push_back(
        "ascend950_plain_histogram_aliases_backend_mixed_template");
    descriptor.lowerability.allSimtOnly = CandidateLoweringStatus::Unsupported;
    descriptor.lowerability.allSimtOnlyReasons.push_back(
        "tt.histogram_not_legalized_by_pure_simt_pipeline");
    bool supported = input && result && input.hasStaticShape() &&
                     result.hasStaticShape() && input.getRank() == 1 &&
                     result.getRank() == 1 &&
                     (facts.inputType == "i8" || facts.inputType == "i16" ||
                      facts.inputType == "i32" || facts.inputType == "i64") &&
                     facts.resultType == "i32" && facts.inputElements > 0 &&
                     facts.numBins > 0;
    if (!supported) {
      descriptor.lowerability.mixed = CandidateLoweringStatus::Unsupported;
      descriptor.lowerability.mixedReasons.push_back(
          "histogram_requires_static_rank1_integer_input_and_rank1_i32_bins");
    }
  } else if (name == "tt.scan") {
    auto facts = analyzePlainOneDimensionalCumsum(op);
    if (!facts)
      return std::nullopt;
    descriptor.kind = SimtAnchorKind::PlainOneDimensionalCumsum;
    descriptor.facts = *facts;
    descriptor.lowerability.allSimd = CandidateLoweringStatus::AliasesMixed;
    descriptor.lowerability.allSimdReasons.push_back(
        "ascend950_plain_1d_cumsum_symbol_is_simt_template");
    descriptor.lowerability.allSimtOnly =
        CandidateLoweringStatus::BackendConditional;
    descriptor.lowerability.allSimtOnlyReasons.push_back(
        "pure_simt_cumsum_requires_whole_module_backend_check");
    if (facts->axisExtent <= 0 || !isSupportedCumsumType(facts->elementType)) {
      descriptor.lowerability.mixed = CandidateLoweringStatus::Unsupported;
      descriptor.lowerability.mixedReasons.push_back(
          "plain_1d_cumsum_dtype_or_axis_extent_not_supported");
    } else if (facts->axisExtent <= 64) {
      descriptor.lowerability.mixedReasons.push_back(
          "template_uses_small_register_path_without_simt_async_invoke");
    }
  } else if (name == "tt.atomic_rmw" || name == "tt.atomic_cas") {
    descriptor.kind = SimtAnchorKind::TensorAtomic;
    TensorAtomicFacts facts;
    auto result = op->getNumResults() > 0
                      ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                      : RankedTensorType();
    facts.updateElements = getStaticNumElements(result);
    facts.valueType = result ? getScalarTypeName(result) : "unknown";
    facts.operation = getAtomicOperation(op);
    if (op->getNumOperands() > 0) {
      if (auto pointer =
              dyn_cast<RankedTensorType>(op->getOperand(0).getType())) {
        facts.addressRank = pointer.getRank();
        facts.addressIsLaneVarying = getStaticNumElements(pointer) > 1;
      }
      Value offset = findPointerOffset(op->getOperand(0));
      if (offset) {
        facts.offsetType = getScalarTypeName(offset.getType());
        if (auto offsetTensor = dyn_cast<RankedTensorType>(offset.getType()))
          facts.addressIsLaneVarying |= getStaticNumElements(offsetTensor) > 1;
      }
      facts.addressDependsOnLoadedIndex = pointerDependsOnLoadedIndex(op);
    }
    facts.hasMask = name == "tt.atomic_rmw" && op->getNumOperands() >= 3;
    if (facts.hasMask)
      facts.staticMaskActiveFraction =
          getStaticMaskActiveFraction(op->getOperand(2));
    facts.resultUsed = op->getNumResults() > 0 && !op->getResult(0).use_empty();
    facts.contention = "unknown";
    descriptor.facts = facts;
    descriptor.lowerability =
        simpleMixedLowerability("pure_simt_atomic_requires_backend_check");

    bool supported = result && result.hasStaticShape() &&
                     facts.updateElements > 0 &&
                     isSupportedAtomicType(facts.valueType, facts.operation) &&
                     (facts.offsetType == "i32" || facts.offsetType == "i64");
    if (facts.resultUsed &&
        (facts.valueType == "f16" || facts.valueType == "bf16")) {
      supported = false;
      descriptor.lowerability.mixedReasons.push_back(
          "f16_bf16_atomic_old_value_semantics_require_validation");
    }
    if (!supported) {
      descriptor.lowerability.mixed = CandidateLoweringStatus::Unsupported;
      descriptor.lowerability.mixedReasons.push_back(
          "atomic_shape_dtype_operation_or_offset_not_supported");
    }
  } else if (isTriangularSolveLoop(op)) {
    descriptor.kind = SimtAnchorKind::TriangularSolveLoop;
    descriptor.lowerability = simpleMixedLowerability(
        "pure_simt_triangular_solve_requires_cube_tail_partition");
    descriptor.lowerability.mixedReasons.push_back(
        "manual_scope_shape_vector16_masked_reduce_loop");
  } else if (isLoadedIndexDependentMemoryOp(op)) {
    descriptor.kind = SimtAnchorKind::LoadedIndexDependentMemory;
    descriptor.lowerability = simpleMixedLowerability(
        "whole_module_pure_simt_requires_backend_check");
  } else {
    return std::nullopt;
  }

  descriptor.materializable =
      compileOn91095 &&
      descriptor.lowerability.mixed == CandidateLoweringStatus::Native;
  return descriptor;
}

static CandidateLoweringStatus
combineWholeKernelStatus(CandidateLoweringStatus lhs,
                         CandidateLoweringStatus rhs) {
  auto rank = [](CandidateLoweringStatus status) {
    switch (status) {
    case CandidateLoweringStatus::Native:
      return 0;
    case CandidateLoweringStatus::BackendConditional:
      return 1;
    case CandidateLoweringStatus::AliasesMixed:
      return 2;
    case CandidateLoweringStatus::Unsupported:
      return 3;
    }
    llvm_unreachable("unknown lowering status");
  };
  return rank(lhs) >= rank(rhs) ? lhs : rhs;
}

} // namespace

llvm::StringRef mlir::ascend::stringifySimtAnchorKind(SimtAnchorKind kind) {
  switch (kind) {
  case SimtAnchorKind::DirectGather:
    return "direct_gather";
  case SimtAnchorKind::LoadedIndexDependentMemory:
    return "loaded_index_dependent_memory";
  case SimtAnchorKind::Histogram:
    return "histogram";
  case SimtAnchorKind::PlainOneDimensionalCumsum:
    return "plain_1d_cumsum";
  case SimtAnchorKind::TensorAtomic:
    return "tensor_atomic";
  case SimtAnchorKind::TriangularSolveLoop:
    return "triangular_solve_loop";
  }
  llvm_unreachable("unknown SIMT anchor kind");
}

llvm::StringRef
mlir::ascend::stringifyCandidateLoweringStatus(CandidateLoweringStatus status) {
  switch (status) {
  case CandidateLoweringStatus::Unsupported:
    return "unsupported";
  case CandidateLoweringStatus::Native:
    return "native";
  case CandidateLoweringStatus::BackendConditional:
    return "backend_conditional";
  case CandidateLoweringStatus::AliasesMixed:
    return "aliases_mixed";
  }
  llvm_unreachable("unknown candidate lowering status");
}

llvm::SmallVector<Operation *> SimtAnchorPlan::materializableRoots() const {
  llvm::SmallVector<Operation *> result;
  result.reserve(anchors.size());
  for (const SimtAnchorDescriptor &anchor : anchors)
    if (anchor.materializable)
      result.push_back(anchor.operation);
  return result;
}

int64_t SimtAnchorPlan::materializableCount() const {
  return llvm::count_if(anchors, [](const SimtAnchorDescriptor &anchor) {
    return anchor.materializable;
  });
}

bool mlir::ascend::isLoadedIndexDependentMemoryOp(Operation *op) {
  if (!op)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return (name == "tt.load" || name == "tt.store") &&
         hasTensorPointerOperand(op) && pointerDependsOnLoadedIndex(op);
}

std::optional<SimtAnchorKind>
mlir::ascend::classifyMixedSimtAnchor(Operation *op) {
  auto descriptor = analyzeAnchor(op, /*compileOn91095=*/true);
  return descriptor ? std::optional<SimtAnchorKind>(descriptor->kind)
                    : std::nullopt;
}

SimtAnchorPlan mlir::ascend::buildMixedSimtAnchorPlan(ModuleOp module,
                                                      bool compileOn91095) {
  SimtAnchorPlan plan;
  llvm::DenseSet<Operation *> operationsInPlannedScope;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (operationsInPlannedScope.contains(op))
      return WalkResult::skip();
    std::optional<SimtAnchorDescriptor> descriptor =
        analyzeAnchor(op, compileOn91095);
    if (!descriptor)
      return WalkResult::advance();

    if (descriptor->kind == SimtAnchorKind::TriangularSolveLoop) {
      descriptor->scopeOperations = collectTriangularSolveScopeOperations(
          op, descriptor->scopeInsertionPoint);
      if (descriptor->scopeOperations.empty()) {
        descriptor->materializable = false;
        descriptor->lowerability.mixed = CandidateLoweringStatus::Unsupported;
        descriptor->lowerability.mixedReasons.push_back(
            "triangular_solve_has_no_materializable_scope_operations");
      }
    } else {
      descriptor->scopeOperations.push_back(op);
      descriptor->scopeInsertionPoint = op;
    }
    for (Operation *scopeOperation : descriptor->scopeOperations)
      operationsInPlannedScope.insert(scopeOperation);
    plan.anchors.push_back(std::move(*descriptor));
    return WalkResult::skip();
  });

  bool anyMixedNative = false;
  bool mixedBlocked = false;
  for (const SimtAnchorDescriptor &anchor : plan.anchors) {
    plan.kernelLowerability.allSimd = combineWholeKernelStatus(
        plan.kernelLowerability.allSimd, anchor.lowerability.allSimd);
    plan.kernelLowerability.allSimtOnly = combineWholeKernelStatus(
        plan.kernelLowerability.allSimtOnly, anchor.lowerability.allSimtOnly);
    llvm::append_range(plan.kernelLowerability.allSimdReasons,
                       anchor.lowerability.allSimdReasons);
    llvm::append_range(plan.kernelLowerability.allSimtOnlyReasons,
                       anchor.lowerability.allSimtOnlyReasons);
    if (anchor.lowerability.mixed == CandidateLoweringStatus::Native)
      anyMixedNative = true;
    else if (anchor.lowerability.allSimd != CandidateLoweringStatus::Native)
      mixedBlocked = true;
    llvm::append_range(plan.kernelLowerability.mixedReasons,
                       anchor.lowerability.mixedReasons);
  }
  if (plan.anchors.empty()) {
    plan.kernelLowerability.mixed = CandidateLoweringStatus::Unsupported;
    plan.kernelLowerability.mixedReasons.push_back("no_recognized_simt_anchor");
  } else if (anyMixedNative && !mixedBlocked) {
    plan.kernelLowerability.mixed = CandidateLoweringStatus::Native;
  } else {
    plan.kernelLowerability.mixed =
        mixedBlocked ? CandidateLoweringStatus::Unsupported
                     : CandidateLoweringStatus::BackendConditional;
  }
  return plan;
}

bool mlir::ascend::isMixedSimtAnchor(Operation *op, bool compileOn91095) {
  auto descriptor = analyzeAnchor(op, compileOn91095);
  return descriptor && descriptor->materializable;
}

llvm::SmallVector<Operation *>
mlir::ascend::collectMixedSimtAnchors(ModuleOp module, bool compileOn91095) {
  return buildMixedSimtAnchorPlan(module, compileOn91095).materializableRoots();
}
