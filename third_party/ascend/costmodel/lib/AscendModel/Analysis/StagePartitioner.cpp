//===- StagePartitioner.cpp - Build semantic Stage IR -------------------===//

#include "AscendModel/Analysis/StagePartitioner.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
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
  double elements = work.scalarOperations + work.predicateElements;
  for (const auto &entry : work.operationElements)
    elements += entry.second;
  elements += 32.0 * (work.loadWarpInstructions + work.storeWarpInstructions);
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

/// True when a loop argument only participates in address induction.  Such a
/// value is an implementation recurrence that later pointer canonicalization
/// can eliminate; it is not an algorithmic loop-carried dependency and must
/// not disable the SIMD independent-loop roofline model.
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
  double elements = 1.0;
  for (Type type : operation->getResultTypes())
    elements = std::max(elements, getTypeElementCount(type));
  if (operation->getNumResults() == 0)
    for (Value value : operation->getOperands())
      elements = std::max(elements, getTypeElementCount(value.getType()));
  return elements;
}

static bool hasTensorResult(Operation *operation) {
  return llvm::any_of(operation->getResultTypes(),
                      [](Type type) { return isa<ShapedType>(type); });
}

static llvm::StringRef getProfileOperationName(Operation *operation) {
  const llvm::StringRef name = operation->getName().getStringRef();
  return llvm::StringSwitch<llvm::StringRef>(name)
      .Cases("arith.addf", "tt.add", "f32.add")
      .Case("arith.subf", "f32.sub")
      .Case("arith.mulf", "f32.mul")
      .Case("arith.divf", "f32.div")
      .Cases("arith.maximumf", "arith.maxnumf", "f32.max")
      .Cases("math.absf", "tt.abs", "f32.abs")
      .Cases("math.exp", "tt.exp", "f32.exp")
      .Cases("math.log", "tt.log", "f32.log")
      .Cases("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp",
             "convert.cast")
      .Cases("arith.fptosi", "arith.fptoui", "convert.cast")
      .Default("generic.issue");
}

static void accumulateDotWorkload(Operation *operation, StageWorkload &work) {
  if (operation->getNumOperands() < 2)
    return;
  auto lhs = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto rhs = dyn_cast<ShapedType>(operation->getOperand(1).getType());
  if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      lhs.getRank() < 2 || rhs.getRank() < 2)
    return;
  const int64_t m = lhs.getShape()[lhs.getRank() - 2];
  const int64_t k = lhs.getShape()[lhs.getRank() - 1];
  const int64_t n = rhs.getShape()[rhs.getRank() - 1];
  if (m > 0 && n > 0 && k > 0)
    work.dotFlops += 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                     static_cast<double>(k);
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
  const double depth = std::ceil(std::log2(static_cast<double>(extent)));
  work.shuffleLaneSteps += getTypeElementCount(input) * depth;
}

static void accumulateOneOperation(Operation *operation, StageWorkload &work) {
  if (!operation || operation->hasTrait<OpTrait::IsTerminator>())
    return;
  const llvm::StringRef name = operation->getName().getStringRef();
  const double elements = getOperationElements(operation);

  if ((name == "tt.load" || name == "tt.gather") &&
      operation->getNumResults() > 0) {
    Value result = operation->getResult(0);
    work.loadBytes += getValueBytes(result);
    work.loadWarpInstructions += std::ceil(elements / 32.0);
    return;
  }
  if ((name == "tt.store" || name.starts_with("tt.atomic")) &&
      operation->getNumOperands() > 1) {
    Value value = operation->getOperand(1);
    work.storeBytes += getValueBytes(value);
    work.storeWarpInstructions +=
        std::ceil(getTypeElementCount(value.getType()) / 32.0);
    return;
  }
  if (name == "tt.dot") {
    accumulateDotWorkload(operation, work);
    return;
  }
  if (name == "tt.reduce" || name == "tt.scan")
    accumulateReductionWorkload(operation, work);
  if (name == "arith.cmpi" || name == "arith.cmpf") {
    work.predicateElements += elements;
    return;
  }
  if (name == "scf.for" || name == "scf.if" || name == "scf.while")
    return;

  if (!hasTensorResult(operation)) {
    work.scalarOperations += 1.0;
    return;
  }
  work.operationElements[getProfileOperationName(operation)] += elements;
}

static void mergeWorkload(StageWorkload &into, StageWorkload from);

static void scaleWorkload(StageWorkload &work, double scale) {
  work.scalarOperations *= scale;
  work.loadBytes *= scale;
  work.storeBytes *= scale;
  work.loadWarpInstructions *= scale;
  work.storeWarpInstructions *= scale;
  work.predicateElements *= scale;
  work.shuffleLaneSteps *= scale;
  work.dotFlops *= scale;
  work.estimatedSpillTransactions *= scale;
  for (auto &entry : work.operationElements)
    entry.second *= scale;
  recomputeIssueElements(work);
}

static std::optional<int64_t> getConstantInteger(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  auto attribute = definition->getAttrOfType<IntegerAttr>("value");
  if (!attribute)
    return std::nullopt;
  return attribute.getInt();
}

static int64_t getLoopTripCount(Operation *operation,
                                int64_t stageIterationCount) {
  const llvm::StringRef name = operation->getName().getStringRef();
  if (name == "scf.for" && operation->getNumOperands() >= 3) {
    const std::optional<int64_t> lower =
        getConstantInteger(operation->getOperand(0));
    const std::optional<int64_t> upper =
        getConstantInteger(operation->getOperand(1));
    const std::optional<int64_t> step =
        getConstantInteger(operation->getOperand(2));
    if (lower && upper && step && *step > 0 && *upper > *lower)
      return (*upper - *lower + *step - 1) / *step;
  }
  if (name == "scf.for" || name == "scf.while")
    return std::max<int64_t>(1, stageIterationCount);
  return 1;
}

/// Accumulate dynamic work, not merely the number of syntactic TTIR ops.
/// A loop body appears once in TTIR but executes `tripCount` times.  The
/// resulting total dynamic work is normalized to one Stage iteration by
/// makePerIteration().  Consequently N_iter * C_body accounts for every
/// loop iteration instead of accidentally counting the body once.
/// AutoBlockify V1 is the exception: its loop is a scheduling shell and its
/// direct body operations are already separate semantic roots.
static void accumulateDynamicOperationTree(Operation *operation,
                                           StageWorkload &work,
                                           double multiplicity,
                                           int64_t fallbackLoopTripCount) {
  if (!operation)
    return;
  StageWorkload local;
  accumulateOneOperation(operation, local);
  scaleWorkload(local, multiplicity);
  mergeWorkload(work, std::move(local));

  if (operation->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  const double childMultiplicity =
      multiplicity *
      static_cast<double>(getLoopTripCount(operation, fallbackLoopTripCount));
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block.getOperations())
        accumulateDynamicOperationTree(&nested, work, childMultiplicity,
                                       fallbackLoopTripCount);
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

static void mergeWorkload(StageWorkload &into, StageWorkload from) {
  into.scalarOperations += from.scalarOperations;
  into.loadBytes += from.loadBytes;
  into.storeBytes += from.storeBytes;
  into.loadWarpInstructions += from.loadWarpInstructions;
  into.storeWarpInstructions += from.storeWarpInstructions;
  into.predicateElements += from.predicateElements;
  into.shuffleLaneSteps += from.shuffleLaneSteps;
  into.dotFlops += from.dotFlops;
  into.estimatedSpillTransactions += from.estimatedSpillTransactions;
  for (const auto &[name, elements] : from.operationElements)
    into.operationElements[name] += elements;
  recomputeIssueElements(into);
}

static void makePerIteration(LogicalStage &stage) {
  const double count =
      static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
  scaleWorkload(stage.workload, 1.0 / count);
}

static Operation *getTopLevelSemanticRoot(Operation *operation);

static bool stageOwnsAnchor(const LogicalStage &stage,
                            const SimtAnchorDescriptor &anchor) {
  if (!anchor.materializable || !stage.localSimtMaterializable)
    return false;
  auto owns = [&](Operation *operation) {
    Operation *root = getTopLevelSemanticRoot(operation);
    return root && llvm::is_contained(stage.operations, root);
  };
  if (anchor.scopeOperations.empty())
    return owns(anchor.operation);
  return llvm::all_of(anchor.scopeOperations, owns);
}

/// Associate exact anchor descriptors with their owning Stage after complete
/// operation ownership has been established.  This is deliberately separate
/// from operation assignment: anchors are materialization evidence, not a
/// second source of Stage boundaries.
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
    // AutoBlockify V1 may wrap a multi-block original function body in a
    // newly created scf.execute_region.  That region is only a scheduling
    // shell, so semantic roots inside it should be exposed directly instead
    // of being owned by the execute_region.
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
      // scf.execute_region.  That region is only a scheduling shell; expose
      // its inner operations as semantic roots so the real algorithm is not
      // hidden behind an auto_blockify_dispatch stage.
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
      // AutoBlockify V1's scf.for is a scheduling shell.  Own the shell as
      // loop control, then expose its direct body operations as semantic
      // roots.  Other structured operations remain atomic roots so their
      // nested recurrence/reduction work is not double-owned.
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

static bool operationTreeContainsLoadedIndexMemory(Operation *root) {
  bool found = root && isLoadedIndexDependentMemoryOp(root);
  if (!root || found)
    return found;
  root->walk([&](Operation *nested) {
    if (!found)
      found = isLoadedIndexDependentMemoryOp(nested);
  });
  return found;
}

static bool operationTreeHasTrueLoopCarriedDependency(Operation *root) {
  bool found = false;
  if (!root)
    return found;
  root->walk([&](Operation *operation) {
    if (found || operation->hasAttr("ta.auto_blockify_v1.loop"))
      return;
    const llvm::StringRef name = operation->getName().getStringRef();
    if ((name != "scf.for" && name != "scf.while") ||
        operation->getNumRegions() == 0 || operation->getRegion(0).empty())
      return;
    Block &body = operation->getRegion(0).front();
    const unsigned firstCarriedArgument = name == "scf.for" ? 1 : 0;
    for (unsigned index = firstCarriedArgument; index < body.getNumArguments();
         ++index) {
      BlockArgument argument = body.getArgument(index);
      if (!argument.use_empty() && !isPointerLikeType(argument.getType()) &&
          !isAddressOnlyLoopValue(argument)) {
        found = true;
        return;
      }
    }
  });
  return found;
}

static bool operationTreeHasAnyName(Operation *root,
                                    llvm::ArrayRef<llvm::StringRef> names) {
  return llvm::any_of(names, [&](llvm::StringRef name) {
    return operationTreeContainsName(root, name);
  });
}

/// Classify one transitive semantic ownership unit.  This function consumes
/// only TTIR structure; it does not inspect a kernel name, workload name,
/// measured performance, or route score.
static StageCostModelKind classifySemanticRoot(Operation *root) {
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
  if (operationTreeHasAnyName(root, {"tt.dot"}))
    return StageCostModelKind::CubeRoofline;
  if (operationTreeContainsLoadedIndexMemory(root) ||
      operationTreeHasAnyName(root, {"tt.gather"}))
    return StageCostModelKind::IndirectGatherMemory;
  if (operationTreeHasAnyName(root, {"scf.for", "scf.while"}))
    return StageCostModelKind::IndependentPipelinedLoop;
  if (operationTreeHasAnyName(
          root, {"tt.fp_to_fp", "arith.extf", "arith.truncf", "arith.fptosi",
                 "arith.fptoui", "arith.sitofp", "arith.uitofp"}))
    return StageCostModelKind::ConversionPack;
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
  return StageCostModelKind::ScalarIssue;
}

static StageScheduleKind scheduleForSemanticRoot(Operation *root,
                                                 StageCostModelKind kind) {
  if (kind == StageCostModelKind::LoopCarriedRecurrence)
    return StageScheduleKind::LoopCarriedSerial;
  if (kind == StageCostModelKind::IndirectGatherMemory)
    return StageScheduleKind::PartiallyDependent;
  if (kind == StageCostModelKind::AutoBlockifyLoop ||
      kind == StageCostModelKind::IndependentPipelinedLoop ||
      operationTreeHasAnyName(root, {"scf.for", "scf.while"}))
    return StageScheduleKind::IndependentPipelined;
  return StageScheduleKind::StraightLine;
}

static int64_t semanticRootIterationCount(Operation *root) {
  int64_t iterations = 1;
  if (!root || root->hasAttr("ta.auto_blockify_v1.loop"))
    return iterations;
  root->walk([&](Operation *operation) {
    if (!operation->hasAttr("ta.auto_blockify_v1.loop"))
      iterations = std::max(iterations, getLoopTripCount(operation, 1));
  });
  return iterations;
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
  // direct semantic body roots.  Treating the shell as the owner of its body
  // would double-own every algorithm operation.
  if (root->hasAttr("ta.auto_blockify_v1.loop"))
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
/// Values defined outside and consumed inside are live-ins; values defined
/// inside and consumed by any operation outside are live-outs.
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
/// materializer will create.  Stage live values are intentionally not used:
/// a Stage can own SIMD operations around a much smaller local scope, and
/// charging its complete live-out footprint would invent UB traffic.
static void deriveLocalSimtScopeTraffic(StagePartition &partition,
                                        const SimtAnchorPlan &anchorPlan) {
  for (LogicalStage &stage : partition.stages) {
    stage.localSimtScopeCount = 0;
    stage.scopeInputTensorBytes = 0;
    stage.scopeOutputTensorBytes = 0;
    auto merged = mergeSimtStageAnchors(anchorPlan, stage.simtAnchorIndices);
    if (!merged)
      continue;
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

      // TritonToUnstructure cannot reconstruct offset information for a
      // tensor-of-pointer returned by scope.scope.  Capturing pointers is
      // legal (the scope is not isolated from above), but returning pointer
      // state would make this local Mixed implementation fail after route
      // selection.  Reject that implementation before it is scored; the
      // same Stage remains legal in a whole-kernel pure-SIMT route.
      if (llvm::any_of(returned, [](Value value) {
            return isPointerLikeType(value.getType());
          })) {
        stage.localSimtMaterializable = false;
        stage.localSimtFactors.clear();
        stage.simtAnchorIndices.clear();
        continue;
      }

      ++stage.localSimtScopeCount;
      stage.scopeInputTensorBytes += staticTensorBytes(captured.getArrayRef());
      stage.scopeOutputTensorBytes += staticTensorBytes(returned.getArrayRef());
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
  structure.rootOperations = collectTopLevelSemanticRoots(module);
  if (structure.rootOperations.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ProgramStructureAnalysis found no top-level TTIR operation");

  // Compound scopes may legally move pure tensor setup across input loads.
  // Stage boundaries must describe the program that the selected route will
  // materialize, rather than treating the pre-materialization textual order
  // as immutable.  Normalize each compound anchor to its planned insertion
  // point before StageBoundaryAnalysis performs its semantic cut.  The
  // operation objects are not mutated here; only the analysis view is
  // reordered.
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable || anchor.scopeOperations.size() < 2 ||
        !anchor.scopeInsertionPoint)
      continue;

    llvm::SmallVector<Operation *, 8> scopeRoots;
    for (Operation *operation : anchor.scopeOperations) {
      Operation *root = getTopLevelSemanticRoot(operation);
      if (root && llvm::is_contained(structure.rootOperations, root) &&
          !llvm::is_contained(scopeRoots, root))
        scopeRoots.push_back(root);
    }
    Operation *insertionRoot =
        getTopLevelSemanticRoot(anchor.scopeInsertionPoint);
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
  case StageCostModelKind::IndirectScalarMemory:
  case StageCostModelKind::IndirectGatherMemory:
    return 60;
  case StageCostModelKind::IndependentPipelinedLoop:
    return 50;
  case StageCostModelKind::ConversionPack:
    return 40;
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
    return 30;
  case StageCostModelKind::PredicateMask:
  case StageCostModelKind::LoopPredicate:
    return 20;
  case StageCostModelKind::IndexGeneration:
    return 10;
  default:
    return 0;
  }
}

/// Scalar/index/predicate work is a supporting resource of a Stage, rather
/// than necessarily a Stage boundary of its own.  StageCostModel accounts for
/// those resources inside every dominant model.  Keeping this distinction
/// here prevents one Triton statement such as a masked load from being split
/// into artificial index, predicate and memory Stages.
static bool isSupportingSemanticKind(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::ScalarIssue:
  case StageCostModelKind::ScalarControl:
  case StageCostModelKind::ScalarMath:
  case StageCostModelKind::IndexGeneration:
  case StageCostModelKind::PredicateMask:
  case StageCostModelKind::LoopPredicate:
    return true;
  default:
    return false;
  }
}

struct SourceStatement {
  StringAttr file;
  unsigned line = 0;

  explicit operator bool() const { return file && line != 0; }
};

static SourceStatement getSourceStatement(Location location) {
  if (auto file = dyn_cast<FileLineColLoc>(location))
    return {file.getFilename(), file.getLine()};
  if (auto name = dyn_cast<NameLoc>(location))
    return getSourceStatement(name.getChildLoc());
  if (auto callsite = dyn_cast<CallSiteLoc>(location)) {
    SourceStatement callee = getSourceStatement(callsite.getCallee());
    return callee ? callee : getSourceStatement(callsite.getCaller());
  }
  if (auto fused = dyn_cast<FusedLoc>(location))
    for (Location child : fused.getLocations()) {
      SourceStatement statement = getSourceStatement(child);
      if (statement)
        return statement;
    }
  return {};
}

static bool haveSameSourceStatement(Operation *left, Operation *right) {
  if (!left || !right)
    return false;
  SourceStatement lhs = getSourceStatement(left->getLoc());
  SourceStatement rhs = getSourceStatement(right->getLoc());
  return lhs && rhs && lhs.file == rhs.file && lhs.line == rhs.line;
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
      Operation *root = getTopLevelSemanticRoot(operation);
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

llvm::Expected<StagePartition>
StageBoundaryAnalysis::analyze(const ProgramStructure &structure,
                               const SimtAnchorPlan &anchorPlan) const {
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
    StageCostModelKind kind = classifySemanticRoot(root);
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
      const StageCostModelKind candidateKind = classifySemanticRoot(candidate);
      const StageScheduleKind candidateSchedule =
          scheduleForSemanticRoot(candidate, candidateKind);
      const bool sameCompoundAnchor =
          anchorGroup >= 0 && candidateAnchorGroup == anchorGroup;
      const bool mergePlainStage =
          anchorGroup < 0 && candidateAnchorGroup < 0 &&
          ((candidateKind == kind && candidateSchedule == schedule) ||
           (haveSameSourceStatement(stage.operations.back(), candidate) &&
            (isSupportingSemanticKind(kind) ||
             isSupportingSemanticKind(candidateKind)))) &&
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

llvm::Error StageFeatureAnalysis::analyze(StagePartition &partition) const {
  for (LogicalStage &stage : partition.stages) {
    StageModelFeatures &facts = stage.features;
    const double activeLaneRatio = facts.activeLaneRatio;
    facts = StageModelFeatures{};
    facts.activeLaneRatio = activeLaneRatio;
    llvm::DenseSet<Operation *> owned;
    for (Operation *root : stage.operations)
      collectOwnedOperationTree(root, owned);
    bool hasMemory = false;
    int64_t algorithmLoopCount = 0;
    if (stage.costModelKind != StageCostModelKind::AutoBlockifyDispatch &&
        stage.costModelKind != StageCostModelKind::AutoBlockifyLoop)
      for (Operation *root : stage.operations)
        facts.replicatedByLocalSuperBlock |= isInsideAutoBlockifyV1Loop(root);
    for (Operation *operation : owned) {
      const llvm::StringRef name = operation->getName().getStringRef();
      if (name == "scf.for" || name == "scf.while") {
        facts.hasLoop = true;
        ++facts.loopBackedgeCount;
        if (!operation->hasAttr("ta.auto_blockify_v1.loop"))
          ++algorithmLoopCount;
        if (!operation->hasAttr("ta.auto_blockify_v1.loop") &&
            operation->getNumRegions() > 0 &&
            !operation->getRegion(0).empty()) {
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
      if (name == "tt.load" || name == "tt.store" || name == "tt.gather" ||
          name.starts_with("tt.atomic")) {
        hasMemory = true;
        facts.hasIndirectMemory |= isLoadedIndexDependentMemoryOp(operation) ||
                                   name == "tt.gather" ||
                                   name.starts_with("tt.atomic");
      }
      facts.hasReduction |=
          name == "tt.reduce" || name == "tt.scan" || name == "linalg.reduce";
      facts.hasPrefixScan |= name == "tt.scan";
      facts.hasDot |=
          name == "tt.dot" || name.contains("matmul") || name.contains("mmad");
      facts.hasConversionPack |=
          name == "arith.extf" || name == "arith.truncf" ||
          name == "arith.fptosi" || name == "arith.fptoui" ||
          name == "arith.sitofp" || name == "arith.uitofp" ||
          name == "tt.fp_to_fp" || name.contains("convert") ||
          name.contains("pack") || name.contains("unpack");
    }
    facts.hasContiguousMemory = hasMemory && !facts.hasIndirectMemory;
    if (algorithmLoopCount > 0 && stage.iterationCount > 1) {
      if (facts.hasLoopCarriedDataDependency)
        facts.parallelRecurrenceGroupCount = algorithmLoopCount;
      facts.loopBackedgeCount = 1;
      facts.conditionalBranchCount =
          std::max<int64_t>(facts.conditionalBranchCount > 0 ? 1 : 0,
                            facts.conditionalBranchCount / algorithmLoopCount);
      facts.divergentBranchCount =
          std::max<int64_t>(facts.divergentBranchCount > 0 ? 1 : 0,
                            facts.divergentBranchCount / algorithmLoopCount);
    }
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
  auto compatible = [](StageCostModelKind kind,
                       const StageModelFeatures &facts) {
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
    case StageCostModelKind::IndirectGatherMemory:
      return facts.hasIndirectMemory;
    case StageCostModelKind::ContinuousTileMemory:
    case StageCostModelKind::ContinuousTileStore:
    case StageCostModelKind::ContinuousShortLoad:
    case StageCostModelKind::CachePolicyStore:
      return facts.hasContiguousMemory;
    case StageCostModelKind::ConversionPack:
      return facts.hasConversionPack;
    default:
      return true;
    }
  };
  for (LogicalStage &stage : partition.stages) {
    const StageModelFeatures &facts = stage.features;
    if (stage.costModelKind == StageCostModelKind::AutoBlockifyDispatch ||
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop)
      continue;
    if (facts.hasDot && (facts.hasReduction || facts.hasIndirectMemory ||
                         facts.hasLoopCarriedDataDependency))
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
      if (facts.hasDot)
        return stage.workload.dotFlops * stage.iterationCount <=
                       static_cast<double>(
                           std::max<int64_t>(1, tinyDotFlopsMax))
                   ? StageCostModelKind::TinyCubeRoofline
                   : StageCostModelKind::CubeRoofline;
      if (facts.hasLoop)
        return StageCostModelKind::IndependentPipelinedLoop;
      if (facts.hasIndirectMemory)
        return StageCostModelKind::IndirectGatherMemory;
      if (facts.hasConversionPack)
        return StageCostModelKind::ConversionPack;
      if (facts.hasContiguousMemory)
        return stage.workload.storeBytes > 0.0 &&
                       stage.workload.loadBytes == 0.0
                   ? StageCostModelKind::ContinuousTileStore
                   : StageCostModelKind::ContinuousTileMemory;
      return StageCostModelKind::ScalarIssue;
    };
    const StageCostModelKind derived = derive();
    // Strong operation-graph semantics are authoritative.  Scalar
    // sub-kinds remain useful only when no dominant structure is present.
    if (semanticKindPriority(derived) > 0 ||
        !compatible(stage.costModelKind, facts))
      stage.costModelKind = derived;
    if (!compatible(stage.costModelKind, facts) ||
        (stage.costModelKind == StageCostModelKind::TinyCubeRoofline &&
         stage.workload.dotFlops * stage.iterationCount >
             static_cast<double>(std::max<int64_t>(1, tinyDotFlopsMax))))
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

llvm::Error StageWorkloadAnalysis::analyze(StagePartition &partition) const {
  if (!partition.operationOwnershipComplete)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageWorkloadAnalysis requires complete operation ownership");
  for (LogicalStage &stage : partition.stages) {
    StageWorkload work;
    work.paysKernelSetup = stage.workload.paysKernelSetup;
    const int64_t loopCount = countAlgorithmLoops(stage);
    const int64_t fallbackLoopTripCount =
        loopCount > 0 ? std::max<int64_t>(1, stage.iterationCount / loopCount)
                      : 1;
    for (Operation *root : stage.operations)
      accumulateDynamicOperationTree(root, work, 1.0, fallbackLoopTripCount);
    recomputeIssueElements(work);
    stage.workload = std::move(work);
    makePerIteration(stage);
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
    if (stage.iterationCount < 1)
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
                                   bool scopeSuperblockMaterializable) const {
  const int64_t maximum = std::clamp<int64_t>(maximumSuperblockFactor, 1, 4);
  // Local and whole-kernel SuperBlock candidates consume the same SIMT warp
  // resources.  Do not regenerate F4 here after evaluateStageModel has
  // already reduced the target maximum to F2 for num_warps=32 (or to F1 for
  // a smaller runtime grid).
  const int64_t localMaximum = scopeSuperblockMaterializable ? maximum : 1;
  for (LogicalStage &stage : partition.stages) {
    stage.simdLegal = true;
    stage.simtLegal = true;
    stage.legalSimtFactors = {1};
    // A pure-SIMT SuperBlock factor is a whole-kernel schedule, not a
    // recurrence-only annotation.  Every SIMT Stage must therefore expose
    // the same factor candidates; KernelRouteSolver keeps the chosen factor
    // uniform.  Local mixed scopes stay restricted by localSimtFactors
    // (F1 unless Scope SuperBlock materialization is explicitly available).
    if (maximum >= 2)
      stage.legalSimtFactors.push_back(2);
    if (maximum >= 4)
      stage.legalSimtFactors.push_back(4);
    if (stage.localSimtMaterializable) {
      // The ABI-v2 scope materializer batches complete logical programs
      // around this Stage.  F2/F4 therefore does not require multiple
      // recurrence groups inside one logical program; that older W2/W4
      // interpretation was only warp widening, not a SuperBlock.
      stage.localSimtFactors = {1};
      if (scopeSuperblockMaterializable && stage.localSuperblockMaterializable)
        for (int64_t factor : {2, 4})
          if (factor <= localMaximum)
            stage.localSimtFactors.push_back(factor);
    }
    if (stage.localSimtMaterializable &&
        (stage.localSimtFactors.empty() ||
         llvm::any_of(stage.localSimtFactors, [&](int64_t factor) {
           return factor < 1 || factor > localMaximum ||
                  (factor != 1 && factor != 2 && factor != 4);
         })))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "local SIMT factors are invalid for Stage '%s'", stage.id.c_str());
  }
  return llvm::Error::success();
}

llvm::Expected<StagePartition>
StagePartitioner::partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
                            const StagePartitionerOptions &options) const {
  auto structure = ProgramStructureAnalysis().analyze(module, anchorPlan);
  if (!structure)
    return structure.takeError();
  auto result = StageBoundaryAnalysis().analyze(*structure, anchorPlan);
  if (!result)
    return result.takeError();
  StageWorkloadAnalysis workloadAnalysis;
  if (llvm::Error error = workloadAnalysis.analyze(*result))
    return std::move(error);
  StageFeatureAnalysis featureAnalysis;
  if (llvm::Error error = featureAnalysis.analyze(*result))
    return std::move(error);
  if (llvm::Error error =
          StageKindClassifier().analyze(*result, options.tinyDotFlopsMax))
    return std::move(error);
  StageModeLegalityAnalysis legalityAnalysis;
  if (llvm::Error error =
          legalityAnalysis.analyze(*result, options.maximumSuperblockFactor,
                                   options.scopeSuperblockMaterializable))
    return std::move(error);
  if (llvm::Error error = StagePartitionVerifier().verify(*result))
    return std::move(error);
  return std::move(*result);
}
