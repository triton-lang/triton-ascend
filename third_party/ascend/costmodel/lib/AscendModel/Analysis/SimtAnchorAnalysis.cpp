//===- SimtAnchorAnalysis.cpp - Materializable SIMT anchors --------------===//

#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/Support/CostModelLogger.h"

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

/// True for scalar or tensor-of-pointer state.  A scope.scope region may
/// capture such values but must not return them: TritonToUnstructure cannot
/// reconstruct the offset information for a returned pointer.
static bool isPointerLikeType(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  stream.flush();
  return llvm::StringRef(text).contains("!tt.ptr");
}

struct PlainCumsumLegality {
  int64_t axisExtent;
  std::string elementType;
};

static std::optional<PlainCumsumLegality>
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

  return PlainCumsumLegality{sourceType.getShape()[axisValue],
                             getScalarTypeName(sourceType.getElementType())};
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

static TriangularSolveFacts analyzeTriangularSolveFacts(Operation *anchor) {
  TriangularSolveFacts facts;
  if (!anchor || !anchor->getBlock())
    return facts;

  auto recordStateType = [&](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getRank() != 2)
      return;
    if (facts.blockRows == 0 && facts.blockColumns == 0) {
      facts.blockRows = tensor.getShape()[0];
      facts.blockColumns = tensor.getShape()[1];
      facts.accumulatorType = getScalarTypeName(tensor.getElementType());
    }
  };
  if (anchor->getNumRegions() == 1 && !anchor->getRegion(0).empty())
    for (BlockArgument argument : anchor->getRegion(0).front().getArguments())
      recordStateType(argument.getType());
  for (Type type : anchor->getResultTypes())
    recordStateType(type);

  Operation *lastRecurrence = nullptr;
  int64_t recurrenceLoopOps = 0;
  for (Operation &candidate : *anchor->getBlock()) {
    if (!isTriangularSolveLoop(&candidate))
      continue;
    ++recurrenceLoopOps;
    lastRecurrence = &candidate;
  }

  // The recognized 16x16 recurrence updates rows [2, 16), hence 14 modeled
  // trips per full block.  Keep the start row explicit so structural matching
  // and reports do not hide that structural assumption in a magic trip count.
  facts.recurrenceStartRow = 2;
  // recurrenceLoopCount is consumed by StageCostModel as N_iter.  It must be
  // the dynamic body execution count, not the number of sibling scf.for ops.
  // A full 16x16 block updates rows [2, 16), hence 14 iterations.
  facts.recurrenceLoopCount =
      facts.blockRows > facts.recurrenceStartRow
          ? recurrenceLoopOps * (facts.blockRows - facts.recurrenceStartRow)
          : recurrenceLoopOps;

  for (Operation *candidate = lastRecurrence ? lastRecurrence->getNextNode()
                                             : nullptr;
       candidate; candidate = candidate->getNextNode()) {
    candidate->walk([&](Operation *nested) {
      facts.denseDotTailOps +=
          nested->getName().getStringRef() == "tt.dot" ? 1 : 0;
    });
  }
  facts.requiresCubeTailPartition = facts.denseDotTailOps > 0;
  return facts;
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

static std::optional<SimtAnchorDescriptor> analyzeAnchor(Operation *op,
                                                         bool compileOn91095) {
  if (!op)
    return std::nullopt;
  SimtAnchorDescriptor descriptor;
  descriptor.operation = op;
  llvm::StringRef name = op->getName().getStringRef();

  if (name == "tt.gather") {
    descriptor.kind = SimtAnchorKind::DirectGather;
  } else if (name == "tt.histogram") {
    descriptor.kind = SimtAnchorKind::Histogram;
    auto input = op->getNumOperands() > 0
                     ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                     : RankedTensorType();
    auto result = op->getNumResults() > 0
                      ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                      : RankedTensorType();
    const int64_t inputElements = getStaticNumElements(input);
    const int64_t numBins =
        result && result.hasStaticShape() && result.getRank() == 1
            ? result.getShape()[0]
            : 0;
    const std::string inputType = input ? getScalarTypeName(input) : "unknown";
    const std::string resultType =
        result ? getScalarTypeName(result) : "unknown";
    descriptor.lowerability.allSimd = false;
    descriptor.lowerability.allSimtOnly = false;
    bool supported = input && result && input.hasStaticShape() &&
                     result.hasStaticShape() && input.getRank() == 1 &&
                     result.getRank() == 1 &&
                     (inputType == "i8" || inputType == "i16" ||
                      inputType == "i32" || inputType == "i64") &&
                     resultType == "i32" && inputElements > 0 && numBins > 0;
    if (!supported)
      descriptor.lowerability.mixed = false;
  } else if (name == "tt.scan") {
    auto facts = analyzePlainOneDimensionalCumsum(op);
    if (!facts)
      return std::nullopt;
    descriptor.kind = SimtAnchorKind::PlainOneDimensionalCumsum;
    if (facts->axisExtent <= 0 || !isSupportedCumsumType(facts->elementType))
      descriptor.lowerability.mixed = false;
  } else if (name == "tt.atomic_rmw" || name == "tt.atomic_cas") {
    descriptor.kind = SimtAnchorKind::TensorAtomic;
    auto result = op->getNumResults() > 0
                      ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                      : RankedTensorType();
    const int64_t updateElements = getStaticNumElements(result);
    const std::string valueType =
        result ? getScalarTypeName(result) : "unknown";
    const std::string operation = getAtomicOperation(op);
    std::string offsetType = "unknown";
    if (op->getNumOperands() > 0) {
      Value offset = findPointerOffset(op->getOperand(0));
      if (offset)
        offsetType = getScalarTypeName(offset.getType());
    }
    bool supported = result && result.hasStaticShape() && updateElements > 0 &&
                     isSupportedAtomicType(valueType, operation) &&
                     (offsetType == "i32" || offsetType == "i64");
    const bool resultUsed =
        op->getNumResults() > 0 && !op->getResult(0).use_empty();
    if (resultUsed && (valueType == "f16" || valueType == "bf16"))
      supported = false;
    if (!supported)
      descriptor.lowerability.mixed = false;
  } else if (isTriangularSolveLoop(op)) {
    descriptor.kind = SimtAnchorKind::TriangularSolveLoop;
    descriptor.triangularSolve = analyzeTriangularSolveFacts(op);
  } else if (isLoadedIndexDependentMemoryOp(op)) {
    descriptor.kind = SimtAnchorKind::LoadedIndexDependentMemory;
  } else {
    return std::nullopt;
  }

  descriptor.materializable = compileOn91095 && descriptor.lowerability.mixed;
  return descriptor;
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
  case SimtAnchorKind::StageOwnedScope:
    return "stage_owned_scope";
  }
  llvm_unreachable("unknown SIMT anchor kind");
}

llvm::SmallVector<Operation *> SimtAnchorPlan::materializableRoots() const {
  llvm::SmallVector<Operation *> result;
  result.reserve(anchors.size());
  for (const SimtAnchorDescriptor &anchor : anchors)
    if (anchor.materializable)
      result.push_back(anchor.operation);
  return result;
}

std::optional<SimtAnchorDescriptor>
mlir::ascend::mergeSimtStageAnchors(const SimtAnchorPlan &plan,
                                    llvm::ArrayRef<unsigned> anchorIndices) {
  llvm::SmallVector<const SimtAnchorDescriptor *> anchors;
  llvm::DenseSet<unsigned> seen;
  for (unsigned index : anchorIndices) {
    if (index >= plan.anchors.size() || !seen.insert(index).second)
      continue;
    const SimtAnchorDescriptor &anchor = plan.anchors[index];
    if (!anchor.materializable || !anchor.operation)
      return std::nullopt;
    anchors.push_back(&anchor);
  }
  if (anchors.empty())
    return std::nullopt;
  if (anchors.size() == 1)
    return *anchors.front();

  Block *block = nullptr;
  llvm::DenseSet<Operation *> selected;
  for (const SimtAnchorDescriptor *anchor : anchors) {
    llvm::ArrayRef<Operation *> operations = anchor->scopeOperations;
    if (operations.empty())
      operations = llvm::ArrayRef(anchor->operation);
    for (Operation *operation : operations) {
      if (!operation || !operation->getBlock() ||
          (block && block != operation->getBlock()))
        return std::nullopt;
      block = operation->getBlock();
      selected.insert(operation);
    }
  }
  if (!block)
    return std::nullopt;

  Operation *first = nullptr;
  Operation *last = nullptr;
  for (Operation &operation : *block) {
    if (!selected.contains(&operation))
      continue;
    if (!first)
      first = &operation;
    last = &operation;
  }
  if (!first || !last)
    return std::nullopt;

  SimtAnchorDescriptor merged = *anchors.front();
  merged.scopeOperations.clear();
  bool inRange = false;
  for (Operation &operation : *block) {
    if (&operation == first)
      inRange = true;
    if (inRange) {
      if (operation.hasTrait<OpTrait::IsTerminator>() ||
          operation.hasTrait<OpTrait::IsIsolatedFromAbove>() ||
          operation.getName().getStringRef() == "scope.scope" ||
          operation.getName().getStringRef() == "scope.return")
        return std::nullopt;
      merged.scopeOperations.push_back(&operation);
    }
    if (&operation == last)
      break;
  }
  merged.scopeInsertionPoint = first;
  return merged;
}

bool mlir::ascend::isLoadedIndexDependentMemoryOp(Operation *op) {
  if (!op)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return (name == "tt.load" || name == "tt.store") &&
         hasTensorPointerOperand(op) && pointerDependsOnLoadedIndex(op);
}

bool mlir::ascend::isStageScopeWrappable(llvm::ArrayRef<Operation *> roots) {
  if (roots.empty())
    return false;
  Block *block = nullptr;
  llvm::DenseSet<Operation *> planned;
  for (Operation *root : roots) {
    if (!root || !root->getBlock())
      return false;
    if (block && root->getBlock() != block)
      return false;
    block = root->getBlock();
    if (!planned.insert(root).second)
      return false;
    const llvm::StringRef name = root->getName().getStringRef();
    if (root->hasTrait<OpTrait::IsTerminator>() ||
        root->hasTrait<OpTrait::IsIsolatedFromAbove>() ||
        name == "scope.scope" || name == "scope.return")
      return false;
  }
  if (!block)
    return false;

  // The materializer moves exactly `roots` into one scope.  A foreign
  // operation lexically inside the [first, last] range would be reordered
  // behind the new scope and could break SSA dominance for moved consumers,
  // so require the range to contain only planned operations.
  Operation *first = nullptr;
  Operation *last = nullptr;
  for (Operation &nested : *block) {
    if (!planned.contains(&nested))
      continue;
    if (!first)
      first = &nested;
    last = &nested;
  }
  for (Operation *cursor = first; cursor && cursor != last;
       cursor = cursor->getNextNode())
    if (!planned.contains(cursor))
      return false;
  return true;
}

std::optional<SimtAnchorDescriptor>
mlir::ascend::buildStageOwnedScopeDescriptor(llvm::ArrayRef<Operation *> roots,
                                             bool compileOn91095) {
  if (!compileOn91095 || !isStageScopeWrappable(roots))
    return std::nullopt;

  // Mirror wrapAnchorRange: the scope returns exactly the values with an
  // outside user.  Returned pointer-like state cannot be reconstructed by
  // TritonToUnstructure, so reject that implementation before it is scored.
  llvm::DenseSet<Operation *> inside;
  for (Operation *root : roots) {
    inside.insert(root);
    root->walk([&](Operation *nested) { inside.insert(nested); });
  }
  auto isInside = [&](Operation *user) {
    for (Operation *owner = user; owner; owner = owner->getParentOp())
      if (inside.contains(owner))
        return true;
    return false;
  };
  for (Operation *root : roots)
    for (Value result : root->getResults())
      if (isPointerLikeType(result.getType()) &&
          llvm::any_of(result.getUses(), [&](OpOperand &use) {
            return !isInside(use.getOwner());
          }))
        return std::nullopt;

  SimtAnchorDescriptor descriptor;
  descriptor.kind = SimtAnchorKind::StageOwnedScope;
  descriptor.operation = roots.front();
  llvm::append_range(descriptor.scopeOperations, roots);
  descriptor.scopeInsertionPoint = roots.front();
  descriptor.lowerability = CandidateLowerability{};
  descriptor.materializable = true;
  return descriptor;
}

SimtAnchorPlan mlir::ascend::buildMixedSimtAnchorPlan(ModuleOp module,
                                                      bool compileOn91095) {
  COSTMODEL_TRACE("buildMixedSimtAnchorPlan");
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
        descriptor->lowerability.mixed = false;
      }
    } else {
      descriptor->scopeOperations.push_back(op);
      descriptor->scopeInsertionPoint = op;
    }
    for (Operation *scopeOperation : descriptor->scopeOperations)
      operationsInPlannedScope.insert(scopeOperation);
    costModelDebug()
        << "anchor: op=" << descriptor->operation->getName().getStringRef()
        << " kind="
        << stringifySimtAnchorKind(descriptor->kind).str()
        << " materializable=" << descriptor->materializable
        << " scopeOperations=" << descriptor->scopeOperations.size() << "\n";
    plan.anchors.push_back(std::move(*descriptor));
    return WalkResult::skip();
  });

  bool anyMixed = false;
  bool mixedBlocked = false;
  for (const SimtAnchorDescriptor &anchor : plan.anchors) {
    plan.kernelLowerability.allSimd &= anchor.lowerability.allSimd;
    plan.kernelLowerability.allSimtOnly &= anchor.lowerability.allSimtOnly;
    anyMixed |= anchor.lowerability.mixed;
    mixedBlocked |= !anchor.lowerability.mixed && !anchor.lowerability.allSimd;
  }
  // A kernel with no primitive anchor can still run mixed: every anchor-free
  // Stage whose roots form a contiguous same-block range is materializable
  // as a StageOwnedScope, so absence of anchors alone must not disable the
  // mixed candidate.
  plan.kernelLowerability.mixed =
      (anyMixed || plan.anchors.empty()) && !mixedBlocked;
  costModelLog() << "output: anchors=" << plan.anchors.size()
                 << " (kernel lowerability: all_simd="
                 << plan.kernelLowerability.allSimd
                 << " all_simt_only="
                 << plan.kernelLowerability.allSimtOnly
                 << " mixed=" << plan.kernelLowerability.mixed
                 << (plan.anchors.empty() ? " [stage_owned_scope fallback]"
                                          : "")
                 << ")\n";
  return plan;
}
