/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"
#include "TritonToGraph/ProgramGridSpecialization.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr int32_t kGroupAxis = 2;
constexpr llvm::StringLiteral kStaticFusionStableId =
    "static-program-axis-fusion";

// This rule deliberately recognizes one narrow, auditable shape: an unrolled
// logits-like group suffix with a single group-major store. It is preferable
// to reject a superficially similar program than to move an unproven effect
// into a sequential loop.
struct StaticFusionStructure {
  triton::FuncOp function;
  triton::GetProgramIdOp groupPid;
  triton::LoadOp keyLoad;
  triton::StoreOp store;
  Operation *firstSuffix = nullptr;
  int64_t groups = 1;
  uint64_t keyLoadBytes = 0;
  uint64_t suffixLoadBytes = 0;
  uint64_t storeBytes = 0;
};

struct StaticFusionCandidate {
  StaticFusionStructure structure;
  unsigned factor = 1;
  CandidateEvaluation evaluation;
};

bool isPublicEntry(triton::FuncOp function) {
  auto visibility = function->getAttrOfType<StringAttr>("sym_visibility");
  return !visibility || visibility.getValue() == "public";
}

bool isOnlyPublicEntry(ModuleOp module, triton::FuncOp function) {
  unsigned publicEntries = 0;
  for (triton::FuncOp candidate : module.getOps<triton::FuncOp>()) {
    if (!isPublicEntry(candidate))
      continue;
    ++publicEntries;
    if (candidate != function)
      return false;
  }
  return publicEntries == 1;
}

bool hasDirectCall(triton::FuncOp function) {
  bool hasCall = false;
  function.walk([&](Operation *operation) {
    if (isa<CallOpInterface>(operation))
      hasCall = true;
  });
  return hasCall;
}

bool hasDisallowedEffect(triton::FuncOp function) {
  bool disallowed = false;
  function.walk([&](Operation *operation) {
    // Operation::walk includes the function root. Its region is the normal
    // entry body, not nested control flow introduced by the candidate.
    if (operation == function.getOperation())
      return;
    const StringRef name = operation->getName().getStringRef();
    // Atomics, barriers, random state, printing, and nested control flow all
    // make execution order observable. SPAF has no synchronization or RNG
    // contract, so each is an unconditional legality rejection.
    if (operation->getNumRegions() != 0 || isa<CallOpInterface>(operation) ||
        name.contains_insensitive("atomic") ||
        name.contains_insensitive("barrier") ||
        name.contains_insensitive("random") ||
        name.contains_insensitive("rng") || name.contains_insensitive("print"))
      disallowed = true;
  });
  return disallowed;
}

bool valueDependsOnImpl(Value value, Value root, DenseSet<Value> &seen) {
  if (!value)
    return false;
  if (value == root)
    return true;
  if (!seen.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOnImpl(operand, root, seen);
  });
}

bool valueDependsOn(Value value, Value root) {
  DenseSet<Value> seen;
  return valueDependsOnImpl(value, root, seen);
}

bool containsZeroBasedRangeImpl(Value value, DenseSet<Value> &seen) {
  if (!value || !seen.insert(value).second)
    return false;
  if (auto range =
          dyn_cast_or_null<triton::MakeRangeOp>(value.getDefiningOp())) {
    auto start = range->getAttrOfType<IntegerAttr>("start");
    return start && start.getInt() == 0;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return containsZeroBasedRangeImpl(operand, seen);
  });
}

bool containsZeroBasedRange(Value value) {
  DenseSet<Value> seen;
  return containsZeroBasedRangeImpl(value, seen);
}

bool derivesFrom(Value value, Value source) {
  return valueDependsOn(value, source);
}

bool hasBoundedRangeMaskImpl(Value value, Value span, DenseSet<Value> &seen) {
  if (!value || !seen.insert(value).second)
    return false;
  if (auto compare = dyn_cast_or_null<arith::CmpIOp>(value.getDefiningOp())) {
    const arith::CmpIPredicate predicate = compare.getPredicate();
    if ((predicate == arith::CmpIPredicate::slt ||
         predicate == arith::CmpIPredicate::ult) &&
        derivesFrom(compare.getRhs(), span) &&
        containsZeroBasedRange(compare.getLhs()))
      return true;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return hasBoundedRangeMaskImpl(operand, span, seen);
  });
}

bool hasBoundedRangeMask(Value value, Value span) {
  DenseSet<Value> seen;
  return hasBoundedRangeMaskImpl(value, span, seen);
}

void collectGroupSpansImpl(Value value, Value group, DenseSet<Value> &seen,
                           SmallVectorImpl<Value> &spans) {
  if (!value || !seen.insert(value).second)
    return;
  if (auto multiply = dyn_cast_or_null<arith::MulIOp>(value.getDefiningOp())) {
    if (multiply.getLhs() == group && multiply.getRhs() != group) {
      spans.push_back(multiply.getRhs());
      return;
    }
    if (multiply.getRhs() == group && multiply.getLhs() != group) {
      spans.push_back(multiply.getLhs());
      return;
    }
  }
  if (Operation *definition = value.getDefiningOp())
    for (Value operand : definition->getOperands())
      collectGroupSpansImpl(operand, group, seen, spans);
}

bool hasRawGroupLeafImpl(Value value, Value group, DenseSet<Value> &seen) {
  if (!value)
    return false;
  if (value == group)
    return true;
  if (!seen.insert(value).second)
    return false;
  if (auto multiply = dyn_cast_or_null<arith::MulIOp>(value.getDefiningOp())) {
    if (multiply.getLhs() == group || multiply.getRhs() == group)
      return false;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return hasRawGroupLeafImpl(operand, group, seen);
  });
}

// A store is accepted only when its pointer contains one group*span term and
// its mask bounds a zero-based logical row range by that exact span. The
// standard logits row-major pointer may have an additional dynamic row stride
// and key offset, but those are downstream of the disjoint logical row IDs.
// This intentionally rejects a pointer with another raw group contribution.
bool hasProvenGroupMajorStore(triton::StoreOp store, Value group) {
  Value pointer = store.getPtr();
  Value mask = store.getMask();
  if (!pointer || !mask || !valueDependsOn(pointer, group))
    return false;

  SmallVector<Value> spans;
  DenseSet<Value> seen;
  collectGroupSpansImpl(pointer, group, seen, spans);
  if (spans.size() != 1 || !spans.front() || spans.front() == group)
    return false;

  DenseSet<Value> rawSeen;
  if (hasRawGroupLeafImpl(pointer, group, rawSeen))
    return false;
  return hasBoundedRangeMask(mask, spans.front());
}

bool isInsideSuffix(Operation *operation, const DenseSet<Operation *> &suffix) {
  return operation && suffix.contains(operation);
}

std::optional<uint64_t> getStaticResultBytes(Operation *operation) {
  if (!operation || operation->getNumResults() != 1)
    return std::nullopt;
  return getStaticTensorBytes(operation->getResult(0).getType());
}

bool isF32Tensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return false;
  auto element = dyn_cast<FloatType>(type.getElementType());
  return element && element.getWidth() == 32;
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool collectSuffixLoadsAndDots(StaticFusionStructure &structure,
                               ArrayRef<Operation *> suffix) {
  Value group = structure.groupPid.getResult();
  uint64_t suffixLoads = 0;
  bool foundGroupLoad = false;
  bool foundDot = false;
  Value keyValue;
  uint64_t keyBytes = 0;

  for (Operation *operation : suffix) {
    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      std::optional<uint64_t> bytes = getStaticResultBytes(operation);
      if (!bytes)
        return false;
      if (valueDependsOn(load.getPtr(), group)) {
        if (!checkedAdd(suffixLoads, *bytes, suffixLoads))
          return false;
        foundGroupLoad = true;
      } else {
        // Any load moved through the new loop would be duplicated. Only
        // group-dependent Q/weight loads are allowed in the suffix; the K
        // tile must be the one retained in the outer program body.
        return false;
      }
    }

    if (auto dot = dyn_cast<triton::DotOp>(operation)) {
      if (!isF32Tensor(dot.getResult()))
        return false;
      Operation *keyDefinition = dot.getB().getDefiningOp();
      auto keyLoad = dyn_cast_or_null<triton::LoadOp>(keyDefinition);
      if (!keyLoad || valueDependsOn(keyLoad.getPtr(), group))
        return false;
      std::optional<uint64_t> bytes = getStaticResultBytes(keyLoad);
      if (!bytes)
        return false;
      if (!keyValue) {
        keyValue = keyLoad.getResult();
        keyBytes = *bytes;
        structure.keyLoad = keyLoad;
      } else if (keyValue != keyLoad.getResult()) {
        return false;
      }
      foundDot = true;
    }
  }

  if (!foundGroupLoad || !foundDot || !structure.keyLoad || keyBytes == 0)
    return false;
  structure.keyLoadBytes = keyBytes;
  structure.suffixLoadBytes = suffixLoads;
  return true;
}

std::optional<StaticFusionStructure>
matchStaticFusion(triton::FuncOp function,
                  const ProgramAxisDependenceAnalysis *axisAnalysis = nullptr) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || !isPublicEntry(function) ||
      !isOnlyPublicEntry(module, function) || function->getNumRegions() != 1 ||
      function->getRegion(0).empty() || hasDirectCall(function) ||
      hasDisallowedEffect(function) ||
      module->hasAttr(kProgramGridTransformsAttr))
    return std::nullopt;

  auto moduleSpecialization = parseProgramGridSpecialization(
      module->getAttr(kProgramGridSpecializationAttr));
  auto functionSpecialization = parseProgramGridSpecialization(
      function->getAttr(kProgramGridSpecializationAttr));
  if (failed(moduleSpecialization) || failed(functionSpecialization) ||
      moduleSpecialization->grid != functionSpecialization->grid ||
      moduleSpecialization->ruleMask != functionSpecialization->ruleMask ||
      (moduleSpecialization->ruleMask &
       getGraphOptimizationRuleMask(
           GraphOptimizationRuleId::StaticProgramAxisFusion)) == 0)
    return std::nullopt;

  const int64_t groups = moduleSpecialization->grid[kGroupAxis];
  // G=1 is intentionally a no-op: do not change the grid or introduce a
  // loop merely to replay one original program.
  if (groups <= 1)
    return std::nullopt;

  if (function->getRegion(0).getBlocks().size() != 1)
    return std::nullopt;
  Block &block = function->getRegion(0).front();
  if (!block.mightHaveTerminator() ||
      !isa<triton::ReturnOp>(block.getTerminator()))
    return std::nullopt;

  SmallVector<triton::GetProgramIdOp> groupPids;
  function.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() == kGroupAxis)
      groupPids.push_back(pid);
  });
  if (groupPids.size() != 1)
    return std::nullopt;
  triton::GetProgramIdOp groupPid = groupPids.front();

  ProgramAxisDependenceAnalysis localAxisAnalysis(function);
  const ProgramAxisDependence &groupInfo =
      axisAnalysis ? axisAnalysis->get(kGroupAxis)
                   : localAxisAnalysis.get(kGroupAxis);
  if (groupInfo.programIds.size() != 1 || groupInfo.escapes ||
      groupInfo.hasUnsupportedSideEffects || groupInfo.readsNumPrograms)
    return std::nullopt;

  SmallVector<Operation *> operations;
  operations.reserve(block.getOperations().size());
  for (Operation &operation : block.without_terminator()) {
    if (operation.getNumRegions() != 0)
      return std::nullopt;
    operations.push_back(&operation);
  }

  Operation *firstSuffix = nullptr;
  for (Operation *user : groupPid.getResult().getUsers()) {
    if (user->getBlock() != &block)
      return std::nullopt;
    if (!firstSuffix || user->isBeforeInBlock(firstSuffix))
      firstSuffix = user;
  }
  if (!firstSuffix)
    return std::nullopt;

  auto firstIt = llvm::find(operations, firstSuffix);
  if (firstIt == operations.end())
    return std::nullopt;
  SmallVector<Operation *> suffix(firstIt, operations.end());
  if (suffix.empty() || !isa<triton::StoreOp>(suffix.back()))
    return std::nullopt;

  DenseSet<Operation *> suffixSet;
  for (Operation *operation : suffix)
    suffixSet.insert(operation);
  for (Operation *operation : suffix) {
    if (operation != suffix.back() && isa<triton::StoreOp>(operation))
      return std::nullopt;
    for (Value result : operation->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!isInsideSuffix(user, suffixSet))
          return std::nullopt;
      }
    }
  }

  auto store = dyn_cast<triton::StoreOp>(suffix.back());
  if (!store || !hasProvenGroupMajorStore(store, groupPid.getResult()))
    return std::nullopt;

  unsigned storeCount = 0;
  function.walk([&](triton::StoreOp candidate) { ++storeCount; });
  if (storeCount != 1)
    return std::nullopt;

  StaticFusionStructure structure;
  structure.function = function;
  structure.groupPid = groupPid;
  structure.store = store;
  structure.firstSuffix = firstSuffix;
  structure.groups = groups;
  if (!collectSuffixLoadsAndDots(structure, suffix))
    return std::nullopt;

  std::optional<uint64_t> storeBytes =
      getStaticTensorBytes(store.getValue().getType());
  if (!storeBytes)
    return std::nullopt;
  structure.storeBytes = *storeBytes;
  return structure;
}

bool buildCandidateCost(const StaticFusionStructure &structure, unsigned factor,
                        const LiveByteEstimate &liveBytes,
                        CandidateCost &cost) {
  if (!liveBytes.known || factor < 2 || structure.groups % factor != 0)
    return false;
  ModuleOp module = structure.function->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto specialization = parseProgramGridSpecialization(
      module->getAttr(kProgramGridSpecializationAttr));
  if (failed(specialization))
    return false;

  uint64_t xyPrograms = 0;
  if (!checkedMul(static_cast<uint64_t>(specialization->grid[0]),
                  static_cast<uint64_t>(specialization->grid[1]), xyPrograms))
    return false;
  uint64_t programsBefore = 0;
  uint64_t programsAfter = 0;
  if (!checkedMul(xyPrograms, static_cast<uint64_t>(structure.groups),
                  programsBefore) ||
      !checkedMul(xyPrograms, static_cast<uint64_t>(structure.groups / factor),
                  programsAfter))
    return false;

  uint64_t perGroupRead = 0;
  if (!checkedAdd(structure.keyLoadBytes, structure.suffixLoadBytes,
                  perGroupRead))
    return false;
  uint64_t readsBefore = 0;
  uint64_t suffixReadsAfter = 0;
  uint64_t keyReadsAfter = 0;
  uint64_t readsAfter = 0;
  uint64_t writes = 0;
  if (!checkedMul(perGroupRead, programsBefore, readsBefore) ||
      !checkedMul(structure.suffixLoadBytes, programsBefore,
                  suffixReadsAfter) ||
      !checkedMul(structure.keyLoadBytes, programsAfter, keyReadsAfter) ||
      !checkedAdd(suffixReadsAfter, keyReadsAfter, readsAfter) ||
      !checkedMul(structure.storeBytes, programsBefore, writes))
    return false;

  cost.plan.tensorizeFactor = 1;
  cost.plan.blockT = 1;
  cost.plan.staticAxisFusionFactor = factor;
  cost.plan.stableId =
      (llvm::Twine(kStaticFusionStableId) + ".f" + llvm::Twine(factor)).str();
  cost.logicalTasksBefore = programsBefore;
  cost.logicalTasksAfter = programsAfter;
  cost.actualProgramsBefore = programsBefore;
  cost.actualProgramsAfter = programsAfter;
  cost.launchesBefore = 1;
  cost.launchesAfter = 1;
  cost.gmReadBytesBefore = readsBefore;
  cost.gmReadBytesAfter = readsAfter;
  cost.gmWriteBytesBefore = writes;
  cost.gmWriteBytesAfter = writes;
  cost.storeCountBefore = programsBefore;
  cost.storeCountAfter = programsBefore;
  // The K address and load are executed once per physical program after
  // fusion. All group-local address work remains once per logical group.
  cost.addressCalculationsBefore = programsBefore;
  cost.addressCalculationsAfter = programsAfter;
  cost.baselinePeakLiveBytes = liveBytes.peakLiveBytes;
  cost.estimatedPeakLiveBytes = liveBytes.peakLiveBytes;
  cost.workPerProgramBefore = 1;
  cost.workPerProgramAfter = factor;
  cost.hasPeakLiveBytes = true;
  cost.persistent = false;
  return true;
}

std::optional<StaticFusionCandidate>
selectStaticFusionCandidate(GraphOptimizationContext &context,
                            bool emitRemarks = true) {
  std::optional<StaticFusionStructure> structure = matchStaticFusion(
      context.getFunction(), &context.getProgramAxisDependenceAnalysis());
  if (!structure)
    return std::nullopt;

  const LiveByteEstimate &liveBytes =
      context.getResourceCostAnalysis().getLiveByteEstimate();
  SmallVector<unsigned> factors;
  if (structure->groups % 2 == 0)
    factors.push_back(2);
  if (structure->groups % 4 == 0)
    factors.push_back(4);
  if (factors.empty())
    return std::nullopt;

  SmallVector<CandidateEvaluation, 2> evaluations;
  evaluations.reserve(factors.size());
  for (unsigned factor : factors) {
    CandidateCost cost;
    if (!buildCandidateCost(*structure, factor, liveBytes, cost))
      continue;
    CandidateEvaluation evaluation =
        context.getResourceCostAnalysis().evaluate(cost);
    // This is intentionally a remark rather than a hidden heuristic: a
    // rejected UB/parallelism candidate must be inspectable in a compiler log.
    if (emitRemarks)
      emitCandidateRemark(structure->keyLoad, evaluation);
    evaluations.push_back(std::move(evaluation));
  }
  if (evaluations.empty())
    return std::nullopt;

  sortCandidateEvaluations(evaluations);
  const CandidateEvaluation &selected = evaluations.front();
  if (!selected.accepted || selected.benefitScore <= 0)
    return std::nullopt;
  return StaticFusionCandidate{
      *structure,
      static_cast<unsigned>(selected.candidate.plan.staticAxisFusionFactor),
      selected};
}

bool sameStructure(const StaticFusionStructure &lhs,
                   const StaticFusionStructure &rhs) {
  return lhs.function == rhs.function && lhs.groupPid == rhs.groupPid &&
         lhs.keyLoad == rhs.keyLoad && lhs.store == rhs.store &&
         lhs.firstSuffix == rhs.firstSuffix && lhs.groups == rhs.groups;
}

bool sameCandidate(const StaticFusionCandidate &lhs,
                   const StaticFusionCandidate &rhs) {
  return lhs.factor == rhs.factor &&
         sameStructure(lhs.structure, rhs.structure);
}

LogicalResult materializeStaticFusion(triton::FuncOp function,
                                      StaticFusionStructure &structure,
                                      unsigned factor) {
  if (factor < 2 || structure.groups % factor != 0)
    return failure();
  Block &block = function->getRegion(0).front();
  SmallVector<Operation *> suffix;
  bool inSuffix = false;
  for (Operation &operation : block.without_terminator()) {
    if (&operation == structure.firstSuffix)
      inSuffix = true;
    if (inSuffix)
      suffix.push_back(&operation);
  }
  if (suffix.empty() || suffix.back() != structure.store.getOperation())
    return failure();

  IRRewriter rewriter(function.getContext());
  const Location loc = structure.firstSuffix->getLoc();
  rewriter.setInsertionPoint(structure.firstSuffix);
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  Value upper = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(factor)));
  Value one =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
  Value groupBase;
  if (factor != structure.groups) {
    Value factorValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(factor)));
    groupBase = rewriter.create<arith::MulIOp>(
        loc, structure.groupPid.getResult(), factorValue);
  }

  auto loop = rewriter.create<scf::ForOp>(loc, zero, upper, one);
  Block *body = loop.getBody();
  if (!body || !body->mightHaveTerminator())
    return failure();
  rewriter.setInsertionPointToStart(body);
  // In both full and partial fusion the induction variable identifies the
  // logical group handled by this loop iteration.  Full fusion starts at
  // group zero, while partial fusion adds the physical-program base.  Do not
  // substitute the zero lower bound for full fusion: that would replay group
  // zero for every iteration and leave the remaining group-major rows stale.
  Value fusedGroup = loop.getInductionVar();
  if (factor != structure.groups)
    fusedGroup = rewriter.create<arith::AddIOp>(loc, groupBase, fusedGroup);

  for (Operation *operation : suffix)
    operation->moveBefore(body->getTerminator());

  structure.groupPid.getResult().replaceUsesWithIf(
      fusedGroup,
      [&](OpOperand &use) { return loop->isAncestor(use.getOwner()); });
  if (factor == structure.groups && structure.groupPid.getResult().use_empty())
    rewriter.eraseOp(structure.groupPid.getOperation());

  // No loop-carried values are created. In particular, each moved f32 dot
  // accumulator is defined in the body and is dead before the next IV, while
  // the K tile remains defined before the scf.for and is captured read-only.
  return mlir::verify(function.getOperation());
}

class StaticProgramAxisFusionPlan final : public RewritePlan {
public:
  StaticProgramAxisFusionPlan(StaticFusionCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::StaticProgramAxisFusion;
  }

  unsigned getBenefit() const override {
    const int64_t score = candidate.evaluation.benefitScore;
    if (score <= 0)
      return 1;
    return static_cast<unsigned>(std::min<int64_t>(
        score, static_cast<int64_t>(std::numeric_limits<unsigned>::max())));
  }

  Operation *getAnchor() const override {
    return candidate.structure.firstSuffix;
  }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.structure.function)
      return failure();
    std::optional<StaticFusionCandidate> current =
        selectStaticFusionCandidate(context, /*emitRemarks=*/false);
    return current && sameCandidate(candidate, *current) ? success()
                                                         : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.structure.function->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    // Clone into a detached one-function module. A late verifier failure must
    // leave both the original IR and launcher metadata unchanged.
    ModuleOp sandbox = ModuleOp::create(candidate.structure.function.getLoc());
    Attribute specialization = module->getAttr(kProgramGridSpecializationAttr);
    if (!specialization)
      return failure();
    sandbox->setAttr(kProgramGridSpecializationAttr, specialization);
    sandbox.getBody()->push_back(candidate.structure.function->clone());
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    if (!clonedFunction)
      return failure();

    ProgramAxisDependenceAnalysis clonedAxisAnalysis(clonedFunction);
    std::optional<StaticFusionStructure> clonedStructure =
        matchStaticFusion(clonedFunction, &clonedAxisAnalysis);
    if (!clonedStructure ||
        clonedStructure->groups != candidate.structure.groups ||
        failed(materializeStaticFusion(clonedFunction, *clonedStructure,
                                       candidate.factor)))
      return failure();

    ProgramGridTransformContract contract;
    contract.transforms.push_back(ProgramGridTransform{
        /*order=*/0,
        /*axis=*/kGroupAxis,
        /*factor=*/static_cast<int64_t>(candidate.factor),
        /*logicalExtent=*/candidate.structure.groups,
        /*persistentCoverage=*/false,
        /*gridStrideAbiVerified=*/false,
    });
    if (failed(setProgramGridTransformContract(sandbox, contract)) ||
        failed(mlir::verify(sandbox.getOperation())))
      return failure();

    candidate.structure.function->getRegion(0).takeBody(
        clonedFunction->getRegion(0));
    return setProgramGridTransformContract(module, contract);
  }

private:
  StaticFusionCandidate candidate;
  unsigned epoch;
};

class StaticProgramAxisFusionRule final : public GraphOptimizationRule {
public:
  explicit StaticProgramAxisFusionRule(bool enabledForCompileMode)
      : enabledForCompileMode(enabledForCompileMode) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::StaticProgramAxisFusion;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::ProgramAxisDependence |
           AnalysisRequirement::ResourceCost;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (!enabledForCompileMode)
      return success();
    std::optional<StaticFusionCandidate> candidate =
        selectStaticFusionCandidate(context);
    if (!candidate)
      return success();
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] matched graph optimization rule "
               << static_cast<unsigned>(getId()) << " ("
               << getGraphOptimizationRuleName(getId()) << ") in @"
               << candidate->structure.function.getName() << ": groups="
               << candidate->structure.groups << " factor=" << candidate->factor
               << " " << candidate->evaluation.remark << "\n");
    plans.push_back(std::make_unique<StaticProgramAxisFusionPlan>(
        std::move(*candidate), context.getEpoch()));
    return success();
  }

private:
  bool enabledForCompileMode;
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createStaticProgramAxisFusionRule(
    const StaticProgramAxisFusionRuleOptions &options) {
  return std::make_unique<StaticProgramAxisFusionRule>(
      options.enabledForCompileMode);
}
