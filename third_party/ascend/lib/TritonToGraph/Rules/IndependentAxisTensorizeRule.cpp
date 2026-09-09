/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to do so, subject to the
 * following conditions:
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

#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"
#include "TritonToGraph/ProgramGridSpecialization.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr llvm::StringLiteral kIndependentAxisTensorizeMarkerAttr =
    "hacc.independent_axis_tensorize";
constexpr llvm::StringLiteral kPersistentTaskStripMiningMarkerAttr =
    "hacc.persistent_task_strip_mining";
constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

// Keep the accepted MergeSplit policy isolated from the Norm+RoPE joint
// planner. Factor 16 is available only to the latter; merely expanding a
// global array must not alter existing Merge primary selection.
constexpr std::array<unsigned, 3> kMergeTensorizeFactors = {2, 4, 8};
// These factors are intentionally a second, MergeSplit-only candidate set.
// Norm+RoPE's standalone and joint paths keep their existing factor domains.
constexpr std::array<unsigned, 2> kMergeLargeTensorizeFactors = {16, 32};
constexpr std::array<unsigned, 4> kNormTensorizeFactors = {2, 4, 8, 16};
constexpr std::array<unsigned, 7> kNormPersistentBlockTCandidates = {
    2, 4, 8, 16, 32, 64, 128};

// MergeSplit keeps the split-reduction dimension and the newly tensorized head
// dimension live in the same state tile.  On Ascend950PR, a 2x head fusion at
// the primary shapes left too many small physical programs, while the resource
// model still selected it because it priced only the extra live bytes.  Bound
// the product of those two live dimensions and, within the existing UB and
// parallelism gates, prefer the largest legal factor.  This gives 2x8 lanes
// for BLOCK_S=2 and 4x4 lanes for BLOCK_S=4, but falls back to 2x for
// BLOCK_S=8 and rejects larger unvalidated planes instead of over-tiling.
constexpr uint64_t kMergeSplitMaxTensorizedPlanes = 16;

// Phase B is deliberately narrower than the large-factor mapping policy: it
// removes the known padding tail only after the original DSL has materialized
// the exact ``range(0, 256) < 192`` data-tile form.  Do not turn this into a
// generic non-power-of-two tile rewrite; the matcher below validates the full
// local dataflow before changing any tensor type.
constexpr int64_t kMergeSplitD192SourceTile = 256;
constexpr int64_t kMergeSplitD192TargetTile = 192;

enum class TensorizeForm : uint8_t {
  MergeSplit,
  NormRope,
};

enum class IATCandidateScope : uint8_t {
  // Existing F2/F4/F8 MergeSplit and Norm+RoPE candidate selection.
  Default,
  // The only path allowed to bypass default min-programs-per-core: a
  // statically small, nonpersistent MergeSplit F16/F32 launch.
  MergeSplitLarge,
};

struct IATCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  TensorizeForm form = TensorizeForm::MergeSplit;
  int32_t axis = 1;
  // The split and D extents make the merge form's lane placement explicit:
  // [S, D] becomes [S, F, D], while values already reduced over S become
  // [F, D].  A single global insertion index cannot represent both shapes.
  int64_t splitExtent = 0;
  int64_t dimExtent = 0;
  int64_t logicalExtent = 0;
  unsigned factor = 1;
  // This is set only after the static small-grid checks for MergeSplit F16/F32
  // succeed. It is consumed by CandidateCost, not by any global graph option.
  bool usesMergeSplitSubCorePolicy = false;
  // Set only when a detached post-IAT sandbox proves that the primary D256
  // data-tile can be rewritten to its statically masked D192 extent.
  bool usesMergeSplitD192Tile = false;
  // With both mapping bits enabled a Norm+RoPE IAT candidate is a joint IAT
  // plus PTSM transaction. No launcher-visible IAT-only intermediate is ever
  // committed when that required second half cannot validate.
  bool requiresPersistentChaining = false;
  ResourceSnapshot resources;
  CandidateEvaluation evaluation;
};

// This is scheduler state for two existing rules, not a fourth rule or a new
// rule-mask bit. The IAT id remains the owning phase so the later standalone
// PTSM phase sees the committed marker and becomes a no-op.
struct JointProgramMappingCandidate {
  IATCandidate iat;
  unsigned blockT = 0;
  std::array<uint64_t, 3> finalLogicalGrid = {1, 1, 1};
  std::array<uint64_t, 3> finalPhysicalGrid = {1, 1, 1};
  CandidateEvaluation evaluation;
};

struct TensorizeReductionShape {
  int64_t splitExtent = 0;
  int64_t dimExtent = 0;
};

struct TensorizeFormMatch {
  TensorizeForm form;
  TensorizeReductionShape reduction;
};

struct MappedValue {
  Value value;
  bool tensorized = false;
  // Position of the newly introduced F dimension in `value` when tensorized.
  // Scalars tensorize to tensor<F>, so their lane axis is always zero.
  int64_t laneAxis = 0;
};

bool multiplyNoOverflow(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  return !__builtin_mul_overflow(lhs, rhs, &result);
}

bool getGridTaskProduct(const std::array<int64_t, 3> &grid, uint64_t &product) {
  product = 1;
  for (int64_t extent : grid) {
    if (extent < 1 ||
        !multiplyNoOverflow(product, static_cast<uint64_t>(extent), product))
      return false;
  }
  return true;
}

bool isMergeSplitLargeFactor(unsigned factor) {
  return factor == 16 || factor == 32;
}

// This is a performance-selection gate for the statically specialized
// MergeSplit primary form, not a generic nonpersistent legality rule. A
// nonpersistent transform must already prove that it launches every logical
// program; only then may its complete small grid occupy fewer than all cores.
bool hasMergeSplitSmallGridEligibility(
    TensorizeForm form, const TensorizeReductionShape &reduction,
    unsigned factor, const ProgramMappingLaunchProjection &after,
    const ResourceSnapshot &resources) {
  return form == TensorizeForm::MergeSplit && isMergeSplitLargeFactor(factor) &&
         reduction.splitExtent > 0 && reduction.dimExtent > 0 &&
         resources.deviceCoreCount != 0 && !after.persistentCoverage &&
         !after.legacyAutoMap && after.logicalPrograms != 0 &&
         after.physicalPrograms == after.logicalPrograms &&
         after.physicalWaves == 1 &&
         after.logicalPrograms <= resources.deviceCoreCount;
}

std::optional<triton::GetProgramIdOp> findOnlyProgramId(triton::FuncOp function,
                                                        int32_t axis) {
  std::optional<triton::GetProgramIdOp> result;
  unsigned count = 0;
  function.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() != axis)
      return;
    ++count;
    if (count == 1)
      result = pid;
  });
  return count == 1 ? result : std::nullopt;
}

std::optional<TensorizeFormMatch>
classifyTensorizeForm(triton::FuncOp function) {
  std::optional<TensorizeReductionShape> mergeReduction;
  std::optional<TensorizeReductionShape> normReduction;
  function.walk([&](triton::ReduceOp reduce) {
    if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1 ||
        reduce.getAxis() != 0)
      return;
    auto source =
        dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
    if (!source || !source.hasStaticShape())
      return;
    Type result = reduce.getResults().front().getType();
    auto rankedResult = dyn_cast<RankedTensorType>(result);
    if (source.getRank() == 2 && rankedResult && rankedResult.getRank() == 1 &&
        rankedResult.hasStaticShape() &&
        source.getShape()[1] == rankedResult.getShape()[0]) {
      if (!mergeReduction)
        mergeReduction =
            TensorizeReductionShape{source.getShape()[0], source.getShape()[1]};
    } else if (source.getRank() == 1 && !isa<RankedTensorType>(result)) {
      if (!normReduction)
        normReduction =
            TensorizeReductionShape{/*splitExtent=*/0,
                                    /*dimExtent=*/source.getShape()[0]};
    }
  });
  // MergeSplit has auxiliary rank-1 scalar reductions for peak and total
  // before its rank-2 state reduction. The latter carries the actual split
  // and data dimensions and is therefore the more specific structural form.
  // Only fall back to the rank-1 scalar form when that merge signature is
  // absent; a function name never resolves this choice.
  if (mergeReduction)
    return TensorizeFormMatch{TensorizeForm::MergeSplit, *mergeReduction};
  if (normReduction)
    return TensorizeFormMatch{TensorizeForm::NormRope, *normReduction};
  return std::nullopt;
}

bool hasSupportedControlFlow(triton::FuncOp function) {
  Region &body = function.getBody();
  if (!body.hasOneBlock())
    return false;
  for (Operation &operation : body.front()) {
    if (isa<CallOpInterface>(operation))
      return false;
    StringRef name = operation.getName().getStringRef();
    if (name.contains("atomic") || name.contains("barrier") ||
        name == "cf.br" || name == "cf.cond_br" || name.starts_with("scf."))
      return false;
    if (operation.getNumRegions() != 0 &&
        !isa<triton::ReduceOp, triton::ScanOp>(operation))
      return false;
  }
  return true;
}

std::optional<ProgramGridSpecialization>
getGridSpecialization(triton::FuncOp function) {
  Attribute attribute =
      function->getAttr(mlir::triton::cfg::kProgramGridSpecializationAttr);
  if (!attribute) {
    if (ModuleOp module = function->getParentOfType<ModuleOp>())
      attribute =
          module->getAttr(mlir::triton::cfg::kProgramGridSpecializationAttr);
  }
  if (!attribute)
    return std::nullopt;
  FailureOr<ProgramGridSpecialization> parsed =
      parseProgramGridSpecialization(attribute);
  if (failed(parsed))
    return std::nullopt;
  return *parsed;
}

bool isIATRuleEnabled(const ProgramGridSpecialization &specialization) {
  return (specialization.ruleMask &
          getGraphOptimizationRuleMask(
              GraphOptimizationRuleId::IndependentAxisTensorize)) != 0;
}

bool isPTSMRuleEnabled(const ProgramGridSpecialization &specialization) {
  return (specialization.ruleMask &
          getGraphOptimizationRuleMask(
              GraphOptimizationRuleId::PersistentTaskStripMining)) != 0;
}

// The direct IAT launch limit is 32768 logical programs.  For a Norm+RoPE
// chain we may exceed it transiently only when a later PTSM block candidate
// can bring the final, launcher-visible logical grid back into that validated
// range.  Materialization is still revalidated transactionally below.
bool canPotentiallyFitPersistentNormLaunch(
    const std::array<int64_t, 3> &iatGrid) {
  constexpr uint64_t kNormRopeMaxTensorizedLaunchPrograms = 32768;
  if (iatGrid[0] < 1 || iatGrid[1] < 1 || iatGrid[2] < 1)
    return false;
  for (unsigned blockT : kNormPersistentBlockTCandidates) {
    std::array<int64_t, 3> finalGrid = iatGrid;
    finalGrid[0] = finalGrid[0] / static_cast<int64_t>(blockT) +
                   (finalGrid[0] % static_cast<int64_t>(blockT) != 0);
    uint64_t tasks = 0;
    if (getGridTaskProduct(finalGrid, tasks) &&
        tasks <= kNormRopeMaxTensorizedLaunchPrograms)
      return true;
  }
  return false;
}

bool hasConflictingLaunchContract(ModuleOp module) {
  return module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
         module->hasAttr(kProgramGridTransformsAttr) ||
         module->hasAttr(kCoalesceFactorAttr) ||
         module->hasAttr(kCoalesceAxisAttr) ||
         module->hasAttr(kCoalesceGridCeilDivAttr);
}

// Every transformed store has its own selected-axis disjointness proof from
// ProgramAxisDependenceAnalysis.  A write/read relation still needs a pointer
// root proof: an in-place or unknown-alias read could observe another lane's
// write even when each store interval is individually disjoint.
bool hasDisjointWriteReadRoots(
    const ProgramAxisDependence &dependence,
    const EntryArgPointerAliasAnalysis &entryPointerAliases) {
  for (const StoreAddressDependence &store : dependence.stores) {
    auto storeOp = dyn_cast_or_null<triton::StoreOp>(store.store);
    if (!storeOp)
      return false;
    for (Operation *operation : dependence.dependenceClosure) {
      auto load = dyn_cast<triton::LoadOp>(operation);
      if (!load)
        continue;
      if (entryPointerAliases.classify(storeOp.getPtr(), load.getPtr()) !=
          EntryArgPointerRelation::DistinctEntryRoots)
        return false;
    }
  }
  return true;
}

CandidateCost
buildResourceCandidate(const IATCandidate &candidate,
                       const ProgramAxisDependence &dependence,
                       const LiveByteEstimate &liveBytes,
                       const ProgramMappingLaunchProjection &before,
                       const ProgramMappingLaunchProjection &after,
                       const LiveByteEstimate *finalLiveBytes = nullptr) {
  CandidateCost cost;
  cost.plan.tensorizeFactor = candidate.factor;
  cost.plan.blockT = 1;
  cost.plan.staticAxisFusionFactor = 1;
  cost.parallelismPolicy =
      candidate.usesMergeSplitSubCorePolicy
          ? ParallelismPolicy::MergeSplitSmallGridAllowSubCore
          : ParallelismPolicy::DefaultMinProgramsPerCore;
  const llvm::StringRef stablePrefix =
      candidate.form == TensorizeForm::MergeSplit
          ? (candidate.usesMergeSplitSubCorePolicy ? "iat.merge.large"
                                                   : "iat.merge")
          : "iat.norm";
  cost.plan.stableId =
      (llvm::Twine(stablePrefix) + ".f" + llvm::Twine(candidate.factor) +
       (candidate.usesMergeSplitD192Tile ? ".d192" : ""))
          .str();
  cost.logicalTasksBefore = before.logicalPrograms;
  cost.logicalTasksAfter = after.logicalPrograms;
  cost.actualProgramsBefore = before.physicalPrograms;
  cost.actualProgramsAfter = after.physicalPrograms;
  cost.physicalWavesBefore = before.physicalWaves;
  cost.physicalWavesAfter = after.physicalWaves;
  cost.launchesBefore = 1;
  cost.launchesAfter = 1;
  cost.storeCountBefore = dependence.stores.size();
  cost.storeCountAfter = dependence.stores.size();
  cost.addressCalculationsBefore = dependence.dependenceClosure.size();
  cost.addressCalculationsAfter = dependence.dependenceClosure.size();
  cost.workPerProgramBefore = before.physicalWaves;
  cost.workPerProgramAfter = after.physicalWaves;
  cost.legacyAutoMapBefore = before.legacyAutoMap;
  cost.legacyAutoMapAfter = after.legacyAutoMap;
  cost.persistent = false;

  const LiveByteEstimate &candidateLiveBytes =
      finalLiveBytes ? *finalLiveBytes : liveBytes;
  if (!liveBytes.known || !candidateLiveBytes.known) {
    const ResourceCostRejectReason reason =
        !liveBytes.known ? liveBytes.reason : candidateLiveBytes.reason;
    cost.hasDynamicShape = reason == ResourceCostRejectReason::DynamicShape;
    cost.hasUnknownResource = !cost.hasDynamicShape;
    return cost;
  }

  cost.hasPeakLiveBytes = true;
  cost.baselinePeakLiveBytes = liveBytes.peakLiveBytes;
  if (finalLiveBytes) {
    cost.estimatedPeakLiveBytes = finalLiveBytes->peakLiveBytes;
    return cost;
  }

  uint64_t estimated = 0;
  if (!multiplyNoOverflow(liveBytes.peakLiveBytes, candidate.factor,
                          estimated)) {
    cost.hasUnknownResource = true;
    return cost;
  }
  cost.estimatedPeakLiveBytes = estimated;
  return cost;
}

std::optional<IATCandidate>
analyzeCandidate(GraphOptimizationContext &context, bool emitRejectRemark,
                 std::optional<unsigned> requestedFactor = std::nullopt,
                 IATCandidateScope scope = IATCandidateScope::Default) {
  triton::FuncOp function = context.getFunction();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || hasConflictingLaunchContract(module) ||
      !hasSupportedControlFlow(function))
    return std::nullopt;

  std::optional<TensorizeFormMatch> formMatch = classifyTensorizeForm(function);
  if (!formMatch)
    return std::nullopt;
  const TensorizeForm form = formMatch->form;
  const TensorizeReductionShape &reductionShape = formMatch->reduction;
  if (scope == IATCandidateScope::MergeSplitLarge &&
      form != TensorizeForm::MergeSplit)
    return std::nullopt;

  std::optional<ProgramGridSpecialization> specialization =
      getGridSpecialization(function);
  if (!specialization || !isIATRuleEnabled(*specialization))
    return std::nullopt;

  constexpr int32_t kTargetAxis = 1;
  // The Ascend Norm+RoPE IAT kernel has a bounded validated logical-launch
  // range.  A standalone transformed grid above this boundary reaches a
  // vector-core runtime failure (the first observed failing shape is
  // 32768 x 16 -> 32768 x 2 = 65536 logical programs).  The only exception is
  // a transactionally materialized IAT->PTSM chain whose final launcher grid
  // is back inside the same bound.
  constexpr uint64_t kNormRopeMaxTensorizedLaunchPrograms = 32768;
  const int64_t logicalExtent = specialization->grid[kTargetAxis];
  if (logicalExtent < 1)
    return std::nullopt;
  // The norm K form has a single head and must remain a stable no-op. Merge
  // intentionally still permits H=1 so tail-mask tests cover F-1.
  if (form == TensorizeForm::NormRope && logicalExtent == 1)
    return std::nullopt;

  const ProgramAxisDependence &dependence =
      context.getProgramAxisDependenceAnalysis().get(kTargetAxis);
  if (!dependence.isIndependentAxisTransformCandidate() ||
      !hasDisjointWriteReadRoots(dependence,
                                 context.getEntryArgPointerAliasAnalysis()))
    return std::nullopt;
  std::optional<triton::GetProgramIdOp> targetPid =
      findOnlyProgramId(function, kTargetAxis);
  if (!targetPid)
    return std::nullopt;
  // Large MergeSplit candidates deliberately preserve the token PID as the
  // outer address dimension. Do not infer a one-PID form from an arbitrary
  // structural merge reduction; the established regular IAT path remains
  // unchanged for its broader regression coverage.
  if (scope == IATCandidateScope::MergeSplitLarge &&
      !findOnlyProgramId(function, /*axis=*/0))
    return std::nullopt;

  const LiveByteEstimate &liveBytes =
      context.getResourceCostAnalysis().getLiveByteEstimate();
  const ResourceSnapshot &resources =
      context.getResourceCostAnalysis().getResourceSnapshot();
  std::optional<ProgramMappingLaunchProjection> beforeProjection =
      projectProgramMappingLaunch(*specialization, {}, resources);
  if (!beforeProjection)
    return std::nullopt;
  llvm::ArrayRef<unsigned> factors;
  if (scope == IATCandidateScope::MergeSplitLarge) {
    factors = llvm::ArrayRef<unsigned>(kMergeLargeTensorizeFactors.data(),
                                       kMergeLargeTensorizeFactors.size());
  } else if (form == TensorizeForm::MergeSplit) {
    factors = llvm::ArrayRef<unsigned>(kMergeTensorizeFactors.data(),
                                       kMergeTensorizeFactors.size());
  } else if (isPTSMRuleEnabled(*specialization)) {
    factors = llvm::ArrayRef<unsigned>(kNormTensorizeFactors.data(),
                                       kNormTensorizeFactors.size());
  } else {
    factors = llvm::ArrayRef<unsigned>(kMergeTensorizeFactors.data(),
                                       kMergeTensorizeFactors.size());
  }
  SmallVector<CandidateEvaluation, 4> evaluations;
  for (unsigned factor : factors) {
    if (requestedFactor && factor != *requestedFactor)
      continue;
    std::array<int64_t, 3> transformedGrid = specialization->grid;
    transformedGrid[kTargetAxis] =
        logicalExtent / factor + (logicalExtent % factor != 0);
    ProgramGridTransform transform{0,
                                   kTargetAxis,
                                   static_cast<int64_t>(factor),
                                   logicalExtent,
                                   /*persistentCoverage=*/false,
                                   /*gridStrideAbiVerified=*/false};
    std::optional<ProgramMappingLaunchProjection> afterProjection =
        projectProgramMappingLaunch(*specialization, {transform}, resources);
    if (!afterProjection)
      continue;
    const bool usesMergeSplitSubCorePolicy =
        scope == IATCandidateScope::MergeSplitLarge &&
        hasMergeSplitSmallGridEligibility(form, reductionShape, factor,
                                          *afterProjection, resources);
    if (scope == IATCandidateScope::MergeSplitLarge &&
        !usesMergeSplitSubCorePolicy)
      continue;
    const uint64_t tasksAfter = afterProjection->logicalPrograms;
    const bool requiresPersistentChaining =
        form == TensorizeForm::NormRope && isPTSMRuleEnabled(*specialization);
    const bool needsPersistentForLaunch =
        form == TensorizeForm::NormRope &&
        tasksAfter > kNormRopeMaxTensorizedLaunchPrograms;
    if (needsPersistentForLaunch &&
        (!requiresPersistentChaining ||
         !canPotentiallyFitPersistentNormLaunch(transformedGrid)))
      continue;

    IATCandidate prototype;
    prototype.function = function;
    prototype.anchor = targetPid->getOperation();
    prototype.form = form;
    prototype.axis = kTargetAxis;
    prototype.splitExtent = reductionShape.splitExtent;
    prototype.dimExtent = reductionShape.dimExtent;
    prototype.logicalExtent = logicalExtent;
    prototype.factor = factor;
    prototype.usesMergeSplitSubCorePolicy = usesMergeSplitSubCorePolicy;
    prototype.requiresPersistentChaining = requiresPersistentChaining;
    prototype.resources = resources;
    evaluations.push_back(context.getResourceCostAnalysis().evaluate(
        buildResourceCandidate(prototype, dependence, liveBytes,
                               *beforeProjection, *afterProjection)));
  }
  if (evaluations.empty())
    return std::nullopt;

  sortCandidateEvaluations(evaluations);
  const CandidateEvaluation *selected = nullptr;
  if (requestedFactor) {
    for (const CandidateEvaluation &evaluation : evaluations) {
      if (evaluation.candidate.plan.tensorizeFactor == *requestedFactor) {
        selected = &evaluation;
        break;
      }
    }
    if (!selected)
      return std::nullopt;
  } else if (!evaluations.front().accepted) {
    if (emitRejectRemark)
      emitCandidateRemark(targetPid->getOperation(), evaluations.front());
    return std::nullopt;
  } else {
    selected = &evaluations.front();
  }
  if (!requestedFactor && scope == IATCandidateScope::Default &&
      form == TensorizeForm::MergeSplit) {
    if (reductionShape.splitExtent <= 0 ||
        static_cast<uint64_t>(reductionShape.splitExtent) >
            kMergeSplitMaxTensorizedPlanes)
      return std::nullopt;
    const uint64_t maxFactor =
        kMergeSplitMaxTensorizedPlanes /
        static_cast<uint64_t>(reductionShape.splitExtent);
    selected = nullptr;
    for (const CandidateEvaluation &evaluation : evaluations) {
      const uint64_t factor = evaluation.candidate.plan.tensorizeFactor;
      if (!evaluation.accepted || factor == 0 || factor > maxFactor)
        continue;
      if (!selected || factor > selected->candidate.plan.tensorizeFactor)
        selected = &evaluation;
    }
    if (!selected)
      return std::nullopt;
  }

  IATCandidate candidate;
  candidate.function = function;
  candidate.anchor = targetPid->getOperation();
  candidate.form = form;
  candidate.axis = kTargetAxis;
  candidate.splitExtent = reductionShape.splitExtent;
  candidate.dimExtent = reductionShape.dimExtent;
  candidate.logicalExtent = logicalExtent;
  candidate.factor =
      static_cast<unsigned>(selected->candidate.plan.tensorizeFactor);
  candidate.usesMergeSplitSubCorePolicy =
      scope == IATCandidateScope::MergeSplitLarge;
  candidate.requiresPersistentChaining =
      form == TensorizeForm::NormRope && isPTSMRuleEnabled(*specialization);
  candidate.resources = resources;
  candidate.evaluation = *selected;
  return candidate;
}

bool sameCandidate(const IATCandidate &lhs, const IATCandidate &rhs) {
  return lhs.function == rhs.function && lhs.form == rhs.form &&
         lhs.axis == rhs.axis && lhs.splitExtent == rhs.splitExtent &&
         lhs.dimExtent == rhs.dimExtent &&
         lhs.logicalExtent == rhs.logicalExtent && lhs.factor == rhs.factor &&
         lhs.usesMergeSplitSubCorePolicy == rhs.usesMergeSplitSubCorePolicy &&
         lhs.usesMergeSplitD192Tile == rhs.usesMergeSplitD192Tile &&
         lhs.requiresPersistentChaining == rhs.requiresPersistentChaining;
}

// Keep the lane next to the logical dimension which carries the data, rather
// than applying a fixed insertion point to every value.  In merge, S-shaped
// values carry the split reduction and become [S, F, ...]; D-shaped values
// live after that reduction and become [F, D].  Norm has only the D dimension
// and always puts F first.
int64_t getPreferredLaneAxis(Type originalType, const IATCandidate &candidate) {
  auto tensor = dyn_cast<RankedTensorType>(originalType);
  if (!tensor || candidate.form == TensorizeForm::NormRope)
    return 0;
  const int64_t rank = tensor.getRank();
  if (rank == 0)
    return 0;
  if (rank == 1)
    return tensor.getShape().front() == candidate.splitExtent ? 1 : 0;
  // Scalars broadcast over a leading split dimension retain that dimension on
  // the left of F as well (for example tensor<1xD> -> tensor<1xF xD>).
  if (tensor.getShape().front() == candidate.splitExtent ||
      tensor.getShape().front() == 1)
    return 1;
  return 0;
}

std::optional<RankedTensorType>
getLiftedTensorType(Type originalType, const IATCandidate &candidate,
                    int64_t laneAxis) {
  Type elementType = originalType;
  SmallVector<int64_t> shape;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType)) {
    if (laneAxis < 0 || laneAxis > tensor.getRank())
      return std::nullopt;
    shape.assign(tensor.getShape().begin(), tensor.getShape().end());
    elementType = tensor.getElementType();
  } else if (laneAxis != 0) {
    return std::nullopt;
  }
  shape.insert(shape.begin() + laneAxis, candidate.factor);
  return RankedTensorType::get(shape, elementType);
}

std::optional<RankedTensorType>
getSameShapeTensorType(Type originalType, ArrayRef<int64_t> shape) {
  Type elementType = originalType;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType))
    elementType = tensor.getElementType();
  return RankedTensorType::get(shape, elementType);
}

bool hasSameShape(RankedTensorType lhs, RankedTensorType rhs) {
  return lhs.getRank() == rhs.getRank() &&
         llvm::equal(lhs.getShape(), rhs.getShape());
}

Value makeZero(IRRewriter &rewriter, Location loc, Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  Type elementType = tensor ? tensor.getElementType() : type;
  if (auto integer = dyn_cast<IntegerType>(elementType)) {
    TypedAttr zero = IntegerAttr::get(integer, 0);
    return tensor ? rewriter.create<arith::ConstantOp>(
                        loc, type, DenseElementsAttr::get(tensor, zero))
                  : rewriter.create<arith::ConstantOp>(loc, type, zero);
  }
  if (auto floating = dyn_cast<FloatType>(elementType)) {
    TypedAttr zero = FloatAttr::get(floating, 0.0);
    return tensor ? rewriter.create<arith::ConstantOp>(
                        loc, type, DenseElementsAttr::get(tensor, zero))
                  : rewriter.create<arith::ConstantOp>(loc, type, zero);
  }
  return Value();
}

bool rebuildTensorizedFunction(triton::FuncOp function,
                               const IATCandidate &candidate,
                               IRRewriter &rewriter) {
  if (!hasSupportedControlFlow(function))
    return false;
  std::optional<triton::GetProgramIdOp> targetPid =
      findOnlyProgramId(function, candidate.axis);
  if (!targetPid)
    return false;

  ProgramAxisDependenceAnalysis analysis(function);
  const ProgramAxisDependence &dependence = analysis.get(candidate.axis);
  if (!dependence.isIndependentAxisTransformCandidate())
    return false;
  DenseSet<Operation *> closure;
  for (Operation *operation : dependence.dependenceClosure)
    closure.insert(operation);

  Block &block = function.getBody().front();
  SmallVector<Operation *> originals;
  originals.reserve(block.getOperations().size());
  for (Operation &operation : block)
    originals.push_back(&operation);

  DenseMap<Value, MappedValue> values;
  auto lookup = [&](Value value) -> MappedValue {
    auto it = values.find(value);
    return it == values.end() ? MappedValue{value, false, 0} : it->second;
  };

  auto broadcastTo = [&](Location loc, Value value,
                         RankedTensorType target) -> Value {
    auto source = dyn_cast<RankedTensorType>(value.getType());
    if (!source || source.getElementType() != target.getElementType() ||
        source.getRank() != target.getRank())
      return Value();
    for (auto [sourceDim, targetDim] :
         llvm::zip(source.getShape(), target.getShape()))
      if (sourceDim != targetDim && sourceDim != 1)
        return Value();
    return hasSameShape(source, target)
               ? value
               : rewriter.create<triton::BroadcastOp>(loc, target, value);
  };

  // Embed an already tensorized value in a target shape without moving data:
  // only singleton dimensions are inserted and all non-singleton dimensions
  // must preserve their relative order.  This rejects instead of guessing a
  // transpose when an unfamiliar shape relation reaches the MVP.
  auto alignTensorized = [&](Location loc, MappedValue mapped,
                             RankedTensorType target,
                             int64_t targetLaneAxis) -> Value {
    auto source = dyn_cast<RankedTensorType>(mapped.value.getType());
    if (!source || mapped.laneAxis < 0 || mapped.laneAxis >= source.getRank() ||
        targetLaneAxis < 0 || targetLaneAxis >= target.getRank() ||
        source.getElementType() != target.getElementType())
      return Value();
    const int64_t sourceLaneAxis = mapped.laneAxis;
    const int64_t sourceTail = source.getRank() - sourceLaneAxis - 1;
    const int64_t targetTail = target.getRank() - targetLaneAxis - 1;
    if (sourceLaneAxis > targetLaneAxis || sourceTail > targetTail ||
        source.getShape()[sourceLaneAxis] != candidate.factor ||
        target.getShape()[targetLaneAxis] != candidate.factor)
      return Value();

    SmallVector<int64_t> sourceToTarget(source.getRank(), -1);
    for (int64_t index = 0; index < sourceLaneAxis; ++index)
      sourceToTarget[index] = index;
    sourceToTarget[sourceLaneAxis] = targetLaneAxis;
    for (int64_t index = 0; index < sourceTail; ++index)
      sourceToTarget[sourceLaneAxis + 1 + index] =
          target.getRank() - sourceTail + index;

    SmallVector<bool> targetMapped(target.getRank(), false);
    for (int64_t sourceIndex = 0; sourceIndex < source.getRank();
         ++sourceIndex) {
      const int64_t targetIndex = sourceToTarget[sourceIndex];
      if (targetIndex < 0 || targetIndex >= target.getRank() ||
          targetMapped[targetIndex])
        return Value();
      targetMapped[targetIndex] = true;
      const int64_t sourceDim = source.getShape()[sourceIndex];
      const int64_t targetDim = target.getShape()[targetIndex];
      if (sourceDim != targetDim && sourceDim != 1)
        return Value();
    }

    Value current = mapped.value;
    int64_t currentAxis = 0;
    for (int64_t targetIndex = 0; targetIndex < target.getRank();
         ++targetIndex) {
      if (targetMapped[targetIndex]) {
        ++currentAxis;
        continue;
      }
      current =
          rewriter.create<triton::ExpandDimsOp>(loc, current, currentAxis);
      ++currentAxis;
    }
    return broadcastTo(loc, current, target);
  };

  auto promoteUniform = [&](Location loc, Value value, Type oldType,
                            RankedTensorType target,
                            int64_t targetLaneAxis) -> Value {
    if (targetLaneAxis < 0 || targetLaneAxis >= target.getRank())
      return Value();
    if (auto oldTensor = dyn_cast<RankedTensorType>(oldType)) {
      if (oldTensor.getRank() != target.getRank() - 1 ||
          oldTensor.getElementType() != target.getElementType())
        return Value();
      SmallVector<int64_t> shape(target.getShape().begin(),
                                 target.getShape().end());
      shape.erase(shape.begin() + targetLaneAxis);
      auto unlaned = RankedTensorType::get(shape, target.getElementType());
      Value base = broadcastTo(loc, value, unlaned);
      if (!base)
        return Value();
      Value expanded =
          rewriter.create<triton::ExpandDimsOp>(loc, base, targetLaneAxis);
      return broadcastTo(loc, expanded, target);
    }
    return rewriter.create<triton::SplatOp>(loc, target, value);
  };

  auto liftOperandTo = [&](Value value, RankedTensorType target,
                           int64_t targetLaneAxis) -> Value {
    MappedValue mapped = lookup(value);
    return mapped.tensorized
               ? alignTensorized(value.getLoc(), mapped, target, targetLaneAxis)
               : promoteUniform(value.getLoc(), mapped.value, value.getType(),
                                target, targetLaneAxis);
  };

  auto copyMissingAttrs = [](Operation *from, Operation *to) {
    for (NamedAttribute attribute : from->getAttrs())
      if (!to->hasAttr(attribute.getName()))
        to->setAttr(attribute.getName(), attribute.getValue());
  };

  Value laneMask;
  auto maskForPointerType = [&](Location loc, RankedTensorType pointerType,
                                int64_t laneAxis) -> Value {
    if (!laneMask)
      return Value();
    auto maskType =
        RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
    return alignTensorized(loc, MappedValue{laneMask, true, 0}, maskType,
                           laneAxis);
  };

  auto createUnchanged = [&](Operation *operation) -> bool {
    // `tt.reduce` and `tt.scan` carry a combiner region.  Rebuilding a
    // non-tensorized operation through the generic OperationState overload
    // loses that region, which leaves malformed IR when a MergeSplit kernel
    // has an auxiliary scalar reduction next to the tensorized state
    // reduction.  Clone with an operand mapping instead so unchanged
    // operations retain every nested region while still consuming the
    // rewritten operands that dominate this insertion point.
    IRMapping mapping;
    for (Value operand : operation->getOperands()) {
      Value mapped = lookup(operand).value;
      if (!mapped)
        return false;
      mapping.map(operand, mapped);
    }
    Operation *replacement = rewriter.clone(*operation, mapping);
    if (replacement->getNumResults() != operation->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(operation->getResults(), replacement->getResults()))
      values[oldResult] = {newResult, false, 0};
    return true;
  };

  auto originalTypesHaveSameShape = [](Type lhs, Type rhs) {
    auto left = dyn_cast<RankedTensorType>(lhs);
    auto right = dyn_cast<RankedTensorType>(rhs);
    if (!left || !right)
      return !left && !right;
    return hasSameShape(left, right);
  };

  auto createElementwise = [&](Operation *operation) -> bool {
    if (operation->getNumResults() == 0)
      return false;
    const bool isReshape = isa<triton::ReshapeOp>(operation);
    if (!operation->hasTrait<OpTrait::Elementwise>() && !isReshape)
      return false;
    if (isReshape &&
        (!originalTypesHaveSameShape(operation->getOperand(0).getType(),
                                     operation->getResult(0).getType())))
      return false;

    int64_t laneAxis =
        getPreferredLaneAxis(operation->getResult(0).getType(), candidate);
    for (Value operand : operation->getOperands()) {
      MappedValue mapped = lookup(operand);
      if (!mapped.tensorized ||
          !originalTypesHaveSameShape(operand.getType(),
                                      operation->getResult(0).getType()))
        continue;
      laneAxis = mapped.laneAxis;
      break;
    }
    std::optional<RankedTensorType> firstResultType = getLiftedTensorType(
        operation->getResult(0).getType(), candidate, laneAxis);
    if (!firstResultType)
      return false;

    SmallVector<Type> resultTypes;
    resultTypes.reserve(operation->getNumResults());
    for (Type resultType : operation->getResultTypes()) {
      std::optional<RankedTensorType> lifted =
          getLiftedTensorType(resultType, candidate, laneAxis);
      if (!lifted || !hasSameShape(*lifted, *firstResultType))
        return false;
      resultTypes.push_back(*lifted);
    }

    SmallVector<Value> operands;
    operands.reserve(operation->getNumOperands());
    for (Value operand : operation->getOperands()) {
      std::optional<RankedTensorType> operandType = getSameShapeTensorType(
          operand.getType(), firstResultType->getShape());
      if (!operandType)
        return false;
      Value lifted = liftOperandTo(operand, *operandType, laneAxis);
      if (!lifted)
        return false;
      operands.push_back(lifted);
    }

    Operation *replacement = rewriter.create(
        operation->getLoc(), operation->getName().getIdentifier(), operands,
        resultTypes, operation->getAttrs());
    if (replacement->getNumResults() != operation->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(operation->getResults(), replacement->getResults()))
      values[oldResult] = {newResult, true, laneAxis};
    return true;
  };

  for (Operation *operation : originals) {
    rewriter.setInsertionPoint(operation);

    if (operation == targetPid->getOperation()) {
      Operation *scalarPid = rewriter.clone(*operation);
      if (scalarPid->getNumResults() != 1)
        return false;
      Value scalar = scalarPid->getResult(0);
      auto integer = dyn_cast<IntegerType>(scalar.getType());
      if (!integer || integer.getWidth() != 32)
        return false;
      Value factor = rewriter.create<arith::ConstantIntOp>(
          operation->getLoc(), candidate.factor, integer.getWidth());
      Value base =
          rewriter.create<arith::MulIOp>(operation->getLoc(), scalar, factor);
      auto laneType =
          RankedTensorType::get({candidate.factor}, scalar.getType());
      Value lane = rewriter.create<triton::MakeRangeOp>(
          operation->getLoc(), laneType, 0, candidate.factor);
      Value baseSplat =
          rewriter.create<triton::SplatOp>(operation->getLoc(), laneType, base);
      Value logicalIds =
          rewriter.create<arith::AddIOp>(operation->getLoc(), baseSplat, lane);
      Value extent = rewriter.create<arith::ConstantIntOp>(
          operation->getLoc(), candidate.logicalExtent, integer.getWidth());
      Value extentSplat = rewriter.create<triton::SplatOp>(operation->getLoc(),
                                                           laneType, extent);
      laneMask = rewriter.create<arith::CmpIOp>(operation->getLoc(),
                                                arith::CmpIPredicate::slt,
                                                logicalIds, extentSplat);
      values[operation->getResult(0)] = {logicalIds, true, 0};
      continue;
    }

    bool tensorized = closure.contains(operation);
    for (Value operand : operation->getOperands())
      tensorized |= lookup(operand).tensorized;

    if (auto splat = dyn_cast<triton::SplatOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      const int64_t laneAxis = getPreferredLaneAxis(splat.getType(), candidate);
      std::optional<RankedTensorType> resultType =
          getLiftedTensorType(splat.getType(), candidate, laneAxis);
      if (!resultType)
        return false;
      Value result = liftOperandTo(splat.getSrc(), *resultType, laneAxis);
      if (!result)
        return false;
      values[splat.getResult()] = {result, true, laneAxis};
      continue;
    }

    if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue sourceMapped = lookup(expand.getSrc());
      const int64_t oldLaneAxis =
          sourceMapped.tensorized
              ? sourceMapped.laneAxis
              : getPreferredLaneAxis(expand.getSrc().getType(), candidate);
      std::optional<RankedTensorType> sourceType = getLiftedTensorType(
          expand.getSrc().getType(), candidate, oldLaneAxis);
      if (!sourceType)
        return false;
      Value source = liftOperandTo(expand.getSrc(), *sourceType, oldLaneAxis);
      if (!source)
        return false;
      const int64_t newAxis =
          expand.getAxis() + (expand.getAxis() >= oldLaneAxis ? 1 : 0);
      const int64_t resultLaneAxis =
          oldLaneAxis + (expand.getAxis() < oldLaneAxis ? 1 : 0);
      Value result = rewriter.create<triton::ExpandDimsOp>(operation->getLoc(),
                                                           source, newAxis);
      std::optional<RankedTensorType> expected =
          getLiftedTensorType(expand.getType(), candidate, resultLaneAxis);
      if (!expected || result.getType() != *expected)
        return false;
      values[expand.getResult()] = {result, true, resultLaneAxis};
      continue;
    }

    if (auto broadcast = dyn_cast<triton::BroadcastOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue sourceMapped = lookup(broadcast.getSrc());
      const int64_t laneAxis =
          sourceMapped.tensorized
              ? sourceMapped.laneAxis
              : getPreferredLaneAxis(broadcast.getSrc().getType(), candidate);
      std::optional<RankedTensorType> resultType =
          getLiftedTensorType(broadcast.getType(), candidate, laneAxis);
      if (!resultType)
        return false;
      Value result = liftOperandTo(broadcast.getSrc(), *resultType, laneAxis);
      if (!result)
        return false;
      values[broadcast.getResult()] = {result, true, laneAxis};
      continue;
    }

    if (auto reduce = dyn_cast<triton::ReduceOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      if (reduce.getSrcs().empty())
        return false;
      auto oldInputType =
          dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
      if (!oldInputType)
        return false;
      MappedValue firstSource = lookup(reduce.getSrcs().front());
      const int64_t laneAxis =
          firstSource.tensorized
              ? firstSource.laneAxis
              : getPreferredLaneAxis(oldInputType, candidate);
      std::optional<RankedTensorType> firstSourceType =
          getLiftedTensorType(oldInputType, candidate, laneAxis);
      if (!firstSourceType)
        return false;
      SmallVector<Value> sources;
      for (Value source : reduce.getSrcs()) {
        std::optional<RankedTensorType> sourceType =
            getLiftedTensorType(source.getType(), candidate, laneAxis);
        if (!sourceType || !hasSameShape(*sourceType, *firstSourceType))
          return false;
        Value lifted = liftOperandTo(source, *sourceType, laneAxis);
        if (!lifted)
          return false;
        sources.push_back(lifted);
      }
      const int64_t axis =
          reduce.getAxis() + (reduce.getAxis() >= laneAxis ? 1 : 0);
      const int64_t resultLaneAxis = laneAxis - (axis < laneAxis ? 1 : 0);
      auto replacement =
          rewriter.create<triton::ReduceOp>(operation->getLoc(), sources, axis);
      rewriter.cloneRegionBefore(reduce.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      copyMissingAttrs(operation, replacement.getOperation());
      for (auto [oldResult, newResult] :
           llvm::zip(reduce.getResults(), replacement.getResults())) {
        std::optional<RankedTensorType> expected =
            getLiftedTensorType(oldResult.getType(), candidate, resultLaneAxis);
        if (!expected || newResult.getType() != *expected)
          return false;
        values[oldResult] = {newResult, true, resultLaneAxis};
      }
      continue;
    }

    if (auto scan = dyn_cast<triton::ScanOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      if (scan.getSrcs().empty())
        return false;
      auto oldInputType =
          dyn_cast<RankedTensorType>(scan.getSrcs().front().getType());
      if (!oldInputType)
        return false;
      MappedValue firstSource = lookup(scan.getSrcs().front());
      const int64_t laneAxis =
          firstSource.tensorized
              ? firstSource.laneAxis
              : getPreferredLaneAxis(oldInputType, candidate);
      std::optional<RankedTensorType> firstSourceType =
          getLiftedTensorType(oldInputType, candidate, laneAxis);
      if (!firstSourceType)
        return false;
      SmallVector<Value> sources;
      for (Value source : scan.getSrcs()) {
        std::optional<RankedTensorType> sourceType =
            getLiftedTensorType(source.getType(), candidate, laneAxis);
        if (!sourceType || !hasSameShape(*sourceType, *firstSourceType))
          return false;
        Value lifted = liftOperandTo(source, *sourceType, laneAxis);
        if (!lifted)
          return false;
        sources.push_back(lifted);
      }
      const int64_t axis =
          scan.getAxis() + (scan.getAxis() >= laneAxis ? 1 : 0);
      auto replacement = rewriter.create<triton::ScanOp>(
          operation->getLoc(), sources, static_cast<uint32_t>(axis),
          scan.getReverse());
      rewriter.cloneRegionBefore(scan.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      copyMissingAttrs(operation, replacement.getOperation());
      for (auto [oldResult, newResult] :
           llvm::zip(scan.getResults(), replacement.getResults())) {
        std::optional<RankedTensorType> expected =
            getLiftedTensorType(oldResult.getType(), candidate, laneAxis);
        if (!expected || newResult.getType() != *expected)
          return false;
        values[oldResult] = {newResult, true, laneAxis};
      }
      continue;
    }

    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue pointerMapped = lookup(load.getPtr());
      const int64_t laneAxis =
          pointerMapped.tensorized
              ? pointerMapped.laneAxis
              : getPreferredLaneAxis(load.getPtr().getType(), candidate);
      std::optional<RankedTensorType> pointerType =
          getLiftedTensorType(load.getPtr().getType(), candidate, laneAxis);
      if (!pointerType)
        return false;
      Value pointer = liftOperandTo(load.getPtr(), *pointerType, laneAxis);
      std::optional<RankedTensorType> maskType =
          getSameShapeTensorType(rewriter.getI1Type(), pointerType->getShape());
      Value mask = load.getMask()
                       ? liftOperandTo(load.getMask(), *maskType, laneAxis)
                       : Value();
      Value laneMaskForLoad =
          maskForPointerType(operation->getLoc(), *pointerType, laneAxis);
      if (laneMaskForLoad)
        mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                     laneMaskForLoad)
                    : laneMaskForLoad;
      std::optional<RankedTensorType> otherType = getSameShapeTensorType(
          load.getResult().getType(), pointerType->getShape());
      Value other = load.getOther()
                        ? liftOperandTo(load.getOther(), *otherType, laneAxis)
                        : makeZero(rewriter, operation->getLoc(), *otherType);
      if (!pointer || !maskType || !otherType || !other)
        return false;
      auto replacement = rewriter.create<triton::LoadOp>(
          operation->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      copyMissingAttrs(operation, replacement.getOperation());
      if (replacement.getResult().getType() != *otherType)
        return false;
      values[load.getResult()] = {replacement.getResult(), true, laneAxis};
      continue;
    }

    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue pointerMapped = lookup(store.getPtr());
      const int64_t laneAxis =
          pointerMapped.tensorized
              ? pointerMapped.laneAxis
              : getPreferredLaneAxis(store.getPtr().getType(), candidate);
      std::optional<RankedTensorType> pointerType =
          getLiftedTensorType(store.getPtr().getType(), candidate, laneAxis);
      if (!pointerType)
        return false;
      Value pointer = liftOperandTo(store.getPtr(), *pointerType, laneAxis);
      std::optional<RankedTensorType> valueType = getSameShapeTensorType(
          store.getValue().getType(), pointerType->getShape());
      if (!valueType)
        return false;
      Value value = liftOperandTo(store.getValue(), *valueType, laneAxis);
      auto maskType =
          RankedTensorType::get(pointerType->getShape(), rewriter.getI1Type());
      Value mask = store.getMask()
                       ? liftOperandTo(store.getMask(), maskType, laneAxis)
                       : Value();
      Value laneMaskForStore =
          maskForPointerType(operation->getLoc(), *pointerType, laneAxis);
      if (laneMaskForStore)
        mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                     laneMaskForStore)
                    : laneMaskForStore;
      if (!pointer || !value || !laneMaskForStore)
        return false;
      auto replacement = rewriter.create<triton::StoreOp>(
          operation->getLoc(), pointer, value, mask, store.getBoundaryCheck(),
          store.getCache(), store.getEvict());
      copyMissingAttrs(operation, replacement.getOperation());
      continue;
    }

    if (isa<triton::ReturnOp>(operation)) {
      // The target kernels have no return values. Returning a tensorized value
      // would require changing the callable ABI, so reject it.
      for (Value operand : operation->getOperands())
        if (lookup(operand).tensorized)
          return false;
      if (!createUnchanged(operation))
        return false;
      continue;
    }

    if (!tensorized) {
      if (!createUnchanged(operation))
        return false;
      continue;
    }
    if (!createElementwise(operation))
      return false;
  }

  if (!laneMask)
    return false;
  for (Operation *operation : llvm::reverse(originals)) {
    operation->dropAllUses();
    rewriter.eraseOp(operation);
  }
  return true;
}

LogicalResult runProgramMappingStructuralCleanup(ModuleOp module) {
  PassManager cleanup(module.getContext(), module.getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return cleanup.run(module);
}

bool hasBoundedPersistentNormLaunch(ModuleOp module, triton::FuncOp function) {
  constexpr uint64_t kNormRopeMaxTensorizedLaunchPrograms = 32768;
  std::optional<ProgramGridSpecialization> specialization =
      getGridSpecialization(function);
  Attribute rawContract = module->getAttr(kProgramGridTransformsAttr);
  if (!specialization || !rawContract)
    return false;
  FailureOr<ProgramGridTransformContract> parsed =
      parseProgramGridTransformContract(rawContract);
  if (failed(parsed) || parsed->transforms.size() != 2)
    return false;

  const ProgramGridTransform &iat = parsed->transforms[0];
  const ProgramGridTransform &ptsm = parsed->transforms[1];
  if (iat.order != 0 || iat.axis != 1 || iat.factor < 2 ||
      iat.logicalExtent != specialization->grid[1] || iat.persistentCoverage ||
      iat.gridStrideAbiVerified || ptsm.order != 1 || ptsm.axis != 0 ||
      ptsm.factor < 2 || ptsm.logicalExtent != specialization->grid[0] ||
      !ptsm.persistentCoverage || !ptsm.gridStrideAbiVerified)
    return false;

  std::array<int64_t, 3> finalGrid = specialization->grid;
  for (const ProgramGridTransform &transform : parsed->transforms) {
    const int32_t axis = transform.axis;
    if (axis < 0 || axis >= static_cast<int32_t>(finalGrid.size()) ||
        finalGrid[axis] < 1 || transform.factor < 2)
      return false;
    finalGrid[axis] = finalGrid[axis] / transform.factor +
                      (finalGrid[axis] % transform.factor != 0);
  }
  uint64_t finalTasks = 0;
  return getGridTaskProduct(finalGrid, finalTasks) &&
         finalTasks <= kNormRopeMaxTensorizedLaunchPrograms;
}

LogicalResult materializeIATCandidateToSandbox(ModuleOp module,
                                               triton::FuncOp function,
                                               const IATCandidate &candidate) {
  if (!module || module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
      module->hasAttr(kProgramGridTransformsAttr))
    return failure();
  IRRewriter rewriter(module.getContext());
  if (!rebuildTensorizedFunction(function, candidate, rewriter))
    return failure();
  ProgramGridTransformContract contract;
  contract.transforms.push_back(ProgramGridTransform{
      0, candidate.axis, static_cast<int64_t>(candidate.factor),
      candidate.logicalExtent,
      /*persistentCoverage=*/false,
      /*gridStrideAbiVerified=*/false});
  if (failed(setProgramGridTransformContract(module, contract)))
    return failure();
  module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                  UnitAttr::get(module.getContext()));
  return mlir::verify(module.getOperation());
}

ModuleOp createProgramMappingSandbox(ModuleOp module, triton::FuncOp function) {
  ModuleOp sandbox = ModuleOp::create(function.getLoc());
  if (Attribute specialization =
          module->getAttr(kProgramGridSpecializationAttr))
    sandbox->setAttr(kProgramGridSpecializationAttr, specialization);
  sandbox.getBody()->push_back(function->clone());
  return sandbox;
}

bool hasD192SourceTile(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return false;
  return llvm::any_of(tensor.getShape(), [](int64_t extent) {
    return extent == kMergeSplitD192SourceTile;
  });
}

std::optional<Type> getD192TileType(Type originalType) {
  auto tensor = dyn_cast<RankedTensorType>(originalType);
  if (!tensor)
    return originalType;

  SmallVector<int64_t> shape(tensor.getShape().begin(),
                             tensor.getShape().end());
  unsigned sourceTileCount = 0;
  for (int64_t &extent : shape) {
    if (extent != kMergeSplitD192SourceTile)
      continue;
    extent = kMergeSplitD192TargetTile;
    ++sourceTileCount;
  }
  // A rank-two [256, 256] value has no unambiguous D-axis interpretation in
  // this narrow rule. Keep the rule fail-closed instead of shrinking both.
  if (sourceTileCount > 1)
    return std::nullopt;
  if (sourceTileCount == 0)
    return originalType;
  return RankedTensorType::get(shape, tensor.getElementType());
}

bool isD192Bound(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat() || !isa<IntegerType>(dense.getElementType()))
    return false;
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  return tensor && tensor.getRank() == 1 &&
         tensor.getShape().front() == kMergeSplitD192SourceTile &&
         dense.getSplatValue<llvm::APInt>().getSExtValue() ==
             kMergeSplitD192TargetTile;
}

std::optional<triton::MakeRangeOp>
findMergeSplitD192Range(triton::FuncOp function) {
  std::optional<triton::MakeRangeOp> result;
  bool ambiguous = false;
  function.walk([&](triton::MakeRangeOp range) {
    auto tensor = dyn_cast<RankedTensorType>(range.getType());
    if (!tensor || tensor.getRank() != 1 ||
        tensor.getShape().front() != kMergeSplitD192SourceTile ||
        range.getStart() != 0 || range.getEnd() != kMergeSplitD192SourceTile)
      return;

    unsigned boundMasks = 0;
    for (Operation *user : range.getResult().getUsers()) {
      auto compare = dyn_cast<arith::CmpIOp>(user);
      if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt ||
          compare.getLhs() != range.getResult() ||
          !isD192Bound(compare.getRhs()))
        continue;
      ++boundMasks;
    }
    if (boundMasks != 1)
      return;
    if (result)
      ambiguous = true;
    else
      result = range;
  });
  return ambiguous ? std::nullopt : result;
}

bool operationTouchesD192SourceTile(Operation *operation) {
  for (Value operand : operation->getOperands())
    if (hasD192SourceTile(operand.getType()))
      return true;
  for (Value result : operation->getResults())
    if (hasD192SourceTile(result.getType()))
      return true;
  return false;
}

// Rebuild only the closed D256 dataflow rooted at the exact range/mask pair.
// This is intentionally not a generic shape substitution: any unrelated D256
// tensor, non-splat constant, control-flow edge, or multi-D256 value rejects
// the Phase-B tile while leaving the already-valid Phase-A IAT candidate
// intact.
bool rebuildMergeSplitD192Function(triton::FuncOp function) {
  if (!hasSupportedControlFlow(function))
    return false;
  std::optional<triton::MakeRangeOp> range = findMergeSplitD192Range(function);
  if (!range)
    return false;

  Block &block = function.getBody().front();
  SmallVector<Operation *> originals;
  originals.reserve(block.getOperations().size());
  for (Operation &operation : block)
    originals.push_back(&operation);

  DenseSet<Operation *> tileClosure;
  SmallVector<Operation *> worklist;
  auto addToClosure = [&](Operation *operation) {
    if (!operation || operation->getBlock() != &block ||
        !tileClosure.insert(operation).second)
      return;
    worklist.push_back(operation);
  };
  addToClosure(range->getOperation());
  for (size_t index = 0; index < worklist.size(); ++index) {
    Operation *operation = worklist[index];
    for (Value result : operation->getResults())
      for (Operation *user : result.getUsers())
        addToClosure(user);
    for (Value operand : operation->getOperands()) {
      if (!hasD192SourceTile(operand.getType()))
        continue;
      Operation *definingOperation = operand.getDefiningOp();
      if (!definingOperation || definingOperation->getBlock() != &block)
        return false;
      addToClosure(definingOperation);
    }
  }

  bool hasTileLoad = false;
  bool hasTileStore = false;
  bool hasTileReduce = false;
  for (Operation *operation : originals) {
    if (operationTouchesD192SourceTile(operation) &&
        !tileClosure.contains(operation))
      return false;
    if (!tileClosure.contains(operation))
      continue;
    hasTileLoad |= isa<triton::LoadOp>(operation);
    hasTileStore |= isa<triton::StoreOp>(operation);
    hasTileReduce |= isa<triton::ReduceOp>(operation);
    if (operation->getNumSuccessors() != 0 ||
        (operation->getNumRegions() != 0 && !isa<triton::ReduceOp>(operation)))
      return false;
  }
  if (!hasTileLoad || !hasTileStore || !hasTileReduce)
    return false;

  IRRewriter rewriter(function.getContext());
  DenseMap<Value, Value> values;
  auto mapValue = [&](Value value) -> std::optional<Value> {
    if (auto it = values.find(value); it != values.end())
      return it->second;
    if (isa<BlockArgument>(value))
      return value;
    return std::nullopt;
  };
  auto mapResults = [&](Operation *original, Operation *replacement) -> bool {
    if (!replacement ||
        original->getNumResults() != replacement->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(original->getResults(), replacement->getResults()))
      values[oldResult] = newResult;
    return true;
  };
  auto collectMappedOperands = [&](Operation *operation,
                                   SmallVectorImpl<Value> &operands) -> bool {
    operands.clear();
    operands.reserve(operation->getNumOperands());
    for (Value operand : operation->getOperands()) {
      std::optional<Value> mapped = mapValue(operand);
      if (!mapped)
        return false;
      operands.push_back(*mapped);
    }
    return true;
  };
  auto collectD192ResultTypes =
      [&](Operation *operation, SmallVectorImpl<Type> &resultTypes) -> bool {
    resultTypes.clear();
    resultTypes.reserve(operation->getNumResults());
    for (Value result : operation->getResults()) {
      std::optional<Type> type = getD192TileType(result.getType());
      if (!type)
        return false;
      resultTypes.push_back(*type);
    }
    return true;
  };

  for (Operation *operation : originals) {
    rewriter.setInsertionPoint(operation);
    if (!tileClosure.contains(operation)) {
      IRMapping mapping;
      for (Value operand : operation->getOperands()) {
        std::optional<Value> mapped = mapValue(operand);
        if (!mapped)
          return false;
        mapping.map(operand, *mapped);
      }
      if (!mapResults(operation, rewriter.clone(*operation, mapping)))
        return false;
      continue;
    }

    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      std::optional<Type> type =
          getD192TileType(constant.getResult().getType());
      RankedTensorType tensor =
          type ? dyn_cast<RankedTensorType>(*type) : RankedTensorType();
      if (!dense || !dense.isSplat() || !tensor)
        return false;
      SmallVector<Attribute, 1> elements = {dense.getSplatValue<Attribute>()};
      auto replacement = rewriter.create<arith::ConstantOp>(
          operation->getLoc(), tensor,
          DenseElementsAttr::get(tensor, elements));
      if (!mapResults(operation, replacement.getOperation()))
        return false;
      continue;
    }

    if (auto makeRange = dyn_cast<triton::MakeRangeOp>(operation)) {
      if (makeRange != *range)
        return false;
      std::optional<Type> type = getD192TileType(makeRange.getType());
      RankedTensorType tensor =
          type ? dyn_cast<RankedTensorType>(*type) : RankedTensorType();
      if (!tensor)
        return false;
      auto replacement = rewriter.create<triton::MakeRangeOp>(
          operation->getLoc(), tensor, makeRange.getStart(),
          kMergeSplitD192TargetTile);
      if (!mapResults(operation, replacement.getOperation()))
        return false;
      continue;
    }

    SmallVector<Value> operands;
    SmallVector<Type> resultTypes;
    if (!collectMappedOperands(operation, operands) ||
        !collectD192ResultTypes(operation, resultTypes))
      return false;

    if (auto reduce = dyn_cast<triton::ReduceOp>(operation)) {
      auto replacement = rewriter.create<triton::ReduceOp>(
          operation->getLoc(), operands, reduce.getAxis());
      rewriter.cloneRegionBefore(reduce.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      for (NamedAttribute attribute : operation->getAttrs())
        if (!replacement->hasAttr(attribute.getName()))
          replacement->setAttr(attribute.getName(), attribute.getValue());
      if (replacement.getResults().size() != resultTypes.size())
        return false;
      for (auto [result, expected] :
           llvm::zip(replacement.getResults(), resultTypes))
        if (result.getType() != expected)
          return false;
      if (!mapResults(operation, replacement.getOperation()))
        return false;
      continue;
    }

    if (operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0)
      return false;
    OperationState state(operation->getLoc(), operation->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(operation->getAttrs());
    if (!mapResults(operation, rewriter.create(state)))
      return false;
  }

  for (Operation *operation : llvm::reverse(originals)) {
    operation->dropAllUses();
    rewriter.eraseOp(operation);
  }
  return succeeded(mlir::verify(function.getOperation()));
}

bool tryMaterializeMergeSplitD192Tile(ModuleOp module, triton::FuncOp function,
                                      const IATCandidate &candidate) {
  if (!module || candidate.form != TensorizeForm::MergeSplit ||
      !candidate.usesMergeSplitSubCorePolicy ||
      !isMergeSplitLargeFactor(candidate.factor) ||
      candidate.dimExtent != kMergeSplitD192SourceTile)
    return false;

  // Phase B is an independent transaction inside the already detached IAT
  // sandbox. Its rejection must retain the valid D256 Phase-A body rather than
  // rejecting the complete mapping candidate.
  ModuleOp trial = createProgramMappingSandbox(module, function);
  auto trialFunction = dyn_cast<triton::FuncOp>(&trial.getBody()->front());
  if (!trialFunction || !rebuildMergeSplitD192Function(trialFunction) ||
      failed(runProgramMappingStructuralCleanup(trial)) ||
      failed(mlir::verify(trial.getOperation())))
    return false;
  function.getRegion().takeBody(trialFunction.getRegion());
  return true;
}

bool hasExpectedMergeSplitLargeLaunch(
    ModuleOp module, triton::FuncOp function, const IATCandidate &candidate,
    const ResourceSnapshot &resources,
    ProgramMappingLaunchProjection *projection = nullptr) {
  if (!candidate.usesMergeSplitSubCorePolicy ||
      candidate.form != TensorizeForm::MergeSplit ||
      !isMergeSplitLargeFactor(candidate.factor))
    return false;
  std::optional<ProgramGridSpecialization> specialization =
      getGridSpecialization(function);
  FailureOr<ProgramGridTransformContract> contract =
      parseProgramGridTransformContract(
          module->getAttr(kProgramGridTransformsAttr));
  if (!specialization || failed(contract) || contract->transforms.size() != 1)
    return false;
  const ProgramGridTransform &iat = contract->transforms.front();
  if (iat.order != 0 || iat.axis != candidate.axis ||
      iat.factor != static_cast<int64_t>(candidate.factor) ||
      iat.logicalExtent != candidate.logicalExtent || iat.persistentCoverage ||
      iat.gridStrideAbiVerified)
    return false;

  std::optional<ProgramMappingLaunchProjection> current =
      projectProgramMappingLaunch(*specialization, contract->transforms,
                                  resources);
  if (!current || current->persistentCoverage || current->legacyAutoMap ||
      current->logicalPrograms == 0 ||
      current->physicalPrograms != current->logicalPrograms ||
      current->physicalWaves != 1 ||
      current->logicalPrograms > resources.deviceCoreCount)
    return false;
  if (projection)
    *projection = *current;
  return true;
}

// The early candidate cost uses baseline-live-bytes * factor as a conservative
// pruning bound. The selected F16/F32 candidate must additionally survive the
// actual tensorized-and-cleaned sandbox liveness calculation before it can
// become a rewrite plan.
std::optional<IATCandidate>
evaluateMergeSplitLargeCandidate(GraphOptimizationContext &context,
                                 unsigned factor) {
  std::optional<IATCandidate> iat =
      analyzeCandidate(context, /*emitRejectRemark=*/false, factor,
                       IATCandidateScope::MergeSplitLarge);
  if (!iat || iat->form != TensorizeForm::MergeSplit ||
      !iat->usesMergeSplitSubCorePolicy)
    return std::nullopt;
  if (!iat->evaluation.accepted)
    return iat;

  ModuleOp module = iat->function->getParentOfType<ModuleOp>();
  if (!module || hasConflictingLaunchContract(module))
    return std::nullopt;
  std::optional<ProgramGridSpecialization> specialization =
      getGridSpecialization(iat->function);
  if (!specialization)
    return std::nullopt;
  const ResourceSnapshot &resources =
      context.getResourceCostAnalysis().getResourceSnapshot();
  std::optional<ProgramMappingLaunchProjection> before =
      projectProgramMappingLaunch(*specialization, {}, resources);
  if (!before)
    return std::nullopt;

  ModuleOp sandbox = createProgramMappingSandbox(module, iat->function);
  auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
  if (!clonedFunction ||
      failed(materializeIATCandidateToSandbox(sandbox, clonedFunction, *iat)) ||
      failed(runProgramMappingStructuralCleanup(sandbox)) ||
      failed(mlir::verify(sandbox.getOperation())))
    return std::nullopt;

  // D192 is a separate, fail-closed optimization transaction. A mismatch
  // leaves this candidate as the already-validated D256 Phase-A plan.
  iat->usesMergeSplitD192Tile =
      tryMaterializeMergeSplitD192Tile(sandbox, clonedFunction, *iat);

  ProgramMappingLaunchProjection after;
  if (!hasExpectedMergeSplitLargeLaunch(sandbox, clonedFunction, *iat,
                                        resources, &after))
    return std::nullopt;
  const ProgramAxisDependence &headDependence =
      context.getProgramAxisDependenceAnalysis().get(iat->axis);
  LiveByteEstimate finalLiveBytes =
      estimatePeakLiveBytes(clonedFunction.getOperation());
  iat->evaluation = evaluateCandidateCost(
      resources, buildResourceCandidate(
                     *iat, headDependence,
                     context.getResourceCostAnalysis().getLiveByteEstimate(),
                     *before, after, &finalLiveBytes));
  return iat;
}

bool isBetterMergeSplitLargeCandidate(const IATCandidate &lhs,
                                      const IATCandidate &rhs) {
  // This is deliberately a fixed compile-time policy. Profiling calibrates
  // its weights offline; it never enters candidate discovery at compile time.
  if (lhs.evaluation.benefitScore != rhs.evaluation.benefitScore)
    return lhs.evaluation.benefitScore > rhs.evaluation.benefitScore;
  if (lhs.evaluation.candidate.estimatedPeakLiveBytes !=
      rhs.evaluation.candidate.estimatedPeakLiveBytes)
    return lhs.evaluation.candidate.estimatedPeakLiveBytes <
           rhs.evaluation.candidate.estimatedPeakLiveBytes;
  if (lhs.evaluation.candidate.actualProgramsAfter !=
      rhs.evaluation.candidate.actualProgramsAfter)
    return lhs.evaluation.candidate.actualProgramsAfter >
           rhs.evaluation.candidate.actualProgramsAfter;
  return lhs.factor < rhs.factor;
}

std::optional<IATCandidate>
selectMergeSplitLargeCandidate(GraphOptimizationContext &context,
                               bool emitRemarks) {
  std::optional<IATCandidate> selected;
  for (unsigned factor : kMergeLargeTensorizeFactors) {
    std::optional<IATCandidate> candidate =
        evaluateMergeSplitLargeCandidate(context, factor);
    if (!candidate)
      continue;
    if (emitRemarks)
      emitCandidateRemark(candidate->anchor, candidate->evaluation);
    // The generic score prices the extra F lane as a live-byte penalty.  It
    // can therefore be negative even after this MergeSplit-only candidate has
    // passed the explicit small-grid, full-launch, alias, tail, cleanup, and
    // final-UB checks.  Its dedicated ordering below is the selection policy;
    // a positive generic score is not an additional legality requirement.
    if (!candidate->evaluation.accepted)
      continue;
    if (!selected || isBetterMergeSplitLargeCandidate(*candidate, *selected))
      selected = std::move(candidate);
  }
  return selected;
}

bool calculateTokenOnlyRepeatedBytes(uint64_t headGroups, uint64_t tokens,
                                     int64_t dimExtent, uint64_t &bytes) {
  bytes = 0;
  if (headGroups <= 1)
    return true;
  if (dimExtent <= 0)
    return false;
  // This is a conservative lower bound for one token-only FP32 vector. The
  // final IR ownership model prevents broadcasts/views from adding extra UB
  // allocations; the repeated-work term records only cross-head-group replay.
  uint64_t perTokenBytes = 0;
  if (!multiplyNoOverflow(static_cast<uint64_t>(dimExtent), sizeof(float),
                          perTokenBytes))
    return false;
  uint64_t repeatedGroups = headGroups - 1;
  uint64_t repeatedTokens = 0;
  return multiplyNoOverflow(repeatedGroups, tokens, repeatedTokens) &&
         multiplyNoOverflow(repeatedTokens, perTokenBytes, bytes);
}

CandidateCost buildJointResourceCandidate(
    const IATCandidate &iat, const ProgramAxisDependence &tokenDependence,
    const LiveByteEstimate &baselineLiveBytes,
    const LiveByteEstimate &finalLiveBytes,
    const ProgramMappingLaunchProjection &before,
    const ProgramMappingLaunchProjection &after, unsigned blockT) {
  CandidateCost cost;
  cost.plan.tensorizeFactor = iat.factor;
  cost.plan.blockT = blockT;
  cost.plan.staticAxisFusionFactor = 1;
  cost.plan.stableId = (llvm::Twine("program-mapping.iat") +
                        llvm::Twine(iat.factor) + ".ptsm" + llvm::Twine(blockT))
                           .str();
  cost.logicalTasksBefore = before.logicalPrograms;
  cost.logicalTasksAfter = after.logicalPrograms;
  cost.actualProgramsBefore = before.physicalPrograms;
  cost.actualProgramsAfter = after.physicalPrograms;
  cost.physicalWavesBefore = before.physicalWaves;
  cost.physicalWavesAfter = after.physicalWaves;
  cost.persistentLoopTripsBefore = before.physicalWaves;
  cost.persistentLoopTripsAfter = after.physicalWaves;
  cost.launchesBefore = 1;
  cost.launchesAfter = 1;
  cost.storeCountBefore = tokenDependence.stores.size();
  cost.storeCountAfter = tokenDependence.stores.size();
  cost.addressCalculationsBefore = tokenDependence.dependenceClosure.size();
  cost.addressCalculationsAfter = tokenDependence.dependenceClosure.size();
  cost.workPerProgramBefore = before.physicalWaves;
  cost.workPerProgramAfter = after.physicalWaves;
  cost.legacyAutoMapBefore = before.legacyAutoMap;
  cost.legacyAutoMapAfter = after.legacyAutoMap;
  cost.persistent = true;

  if (!baselineLiveBytes.known || !finalLiveBytes.known) {
    const ResourceCostRejectReason reason = !baselineLiveBytes.known
                                                ? baselineLiveBytes.reason
                                                : finalLiveBytes.reason;
    cost.hasDynamicShape = reason == ResourceCostRejectReason::DynamicShape;
    cost.hasUnknownResource = !cost.hasDynamicShape;
    return cost;
  }

  uint64_t repeatedBefore = 0;
  uint64_t repeatedAfter = 0;
  if (!calculateTokenOnlyRepeatedBytes(before.logicalGrid[1],
                                       before.logicalGrid[0], iat.dimExtent,
                                       repeatedBefore) ||
      !calculateTokenOnlyRepeatedBytes(after.logicalGrid[1],
                                       before.logicalGrid[0], iat.dimExtent,
                                       repeatedAfter)) {
    cost.hasUnknownResource = true;
    return cost;
  }
  cost.tokenOnlyRepeatedBytesBefore = repeatedBefore;
  cost.tokenOnlyRepeatedBytesAfter = repeatedAfter;
  cost.hasPeakLiveBytes = true;
  cost.baselinePeakLiveBytes = baselineLiveBytes.peakLiveBytes;
  cost.estimatedPeakLiveBytes = finalLiveBytes.peakLiveBytes;
  return cost;
}

bool hasExpectedJointLaunch(ModuleOp module, triton::FuncOp function,
                            unsigned iatFactor, unsigned blockT,
                            const ResourceSnapshot &resources,
                            ProgramMappingLaunchProjection *projection) {
  if (!hasBoundedPersistentNormLaunch(module, function))
    return false;
  std::optional<ProgramGridSpecialization> specialization =
      getGridSpecialization(function);
  FailureOr<ProgramGridTransformContract> contract =
      parseProgramGridTransformContract(
          module->getAttr(kProgramGridTransformsAttr));
  if (!specialization || failed(contract) || contract->transforms.size() != 2)
    return false;
  const ProgramGridTransform &iat = contract->transforms[0];
  const ProgramGridTransform &ptsm = contract->transforms[1];
  if (iat.factor != static_cast<int64_t>(iatFactor) ||
      ptsm.factor != static_cast<int64_t>(blockT))
    return false;
  std::optional<ProgramMappingLaunchProjection> current =
      projectProgramMappingLaunch(*specialization, contract->transforms,
                                  resources);
  if (!current || !current->persistentCoverage || current->legacyAutoMap)
    return false;
  if (projection)
    *projection = *current;
  return true;
}

std::optional<JointProgramMappingCandidate>
evaluateJointCandidate(GraphOptimizationContext &context, unsigned iatFactor,
                       unsigned blockT) {
  std::optional<IATCandidate> iat =
      analyzeCandidate(context, /*emitRejectRemark=*/false, iatFactor);
  if (!iat || iat->form != TensorizeForm::NormRope ||
      !iat->requiresPersistentChaining)
    return std::nullopt;
  ModuleOp module = iat->function->getParentOfType<ModuleOp>();
  if (!module || hasConflictingLaunchContract(module))
    return std::nullopt;
  std::optional<ProgramGridSpecialization> specialization =
      getGridSpecialization(iat->function);
  if (!specialization)
    return std::nullopt;
  const ResourceSnapshot &resources =
      context.getResourceCostAnalysis().getResourceSnapshot();
  std::optional<ProgramMappingLaunchProjection> before =
      projectProgramMappingLaunch(*specialization, {}, resources);
  if (!before)
    return std::nullopt;

  ModuleOp sandbox = createProgramMappingSandbox(module, iat->function);
  auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
  if (!clonedFunction ||
      failed(materializeIATCandidateToSandbox(sandbox, clonedFunction, *iat)))
    return std::nullopt;
  CandidateEvaluation ptsmEvaluation;
  if (failed(materializePersistentTaskStripMiningCandidate(
          sandbox, clonedFunction, resources, blockT, &ptsmEvaluation,
          /*deferIntermediateResourceRejection=*/true)) ||
      failed(runProgramMappingStructuralCleanup(sandbox)) ||
      failed(mlir::verify(sandbox.getOperation())))
    return std::nullopt;

  ProgramMappingLaunchProjection after;
  if (!hasExpectedJointLaunch(sandbox, clonedFunction, iatFactor, blockT,
                              resources, &after))
    return std::nullopt;
  LiveByteEstimate finalLiveBytes =
      estimatePeakLiveBytes(clonedFunction.getOperation());
  const ProgramAxisDependence &tokenDependence =
      context.getProgramAxisDependenceAnalysis().get(0);
  CandidateEvaluation evaluation = evaluateCandidateCost(
      resources, buildJointResourceCandidate(
                     *iat, tokenDependence,
                     context.getResourceCostAnalysis().getLiveByteEstimate(),
                     finalLiveBytes, *before, after, blockT));

  JointProgramMappingCandidate candidate;
  candidate.iat = std::move(*iat);
  candidate.blockT = blockT;
  candidate.finalLogicalGrid = after.logicalGrid;
  candidate.finalPhysicalGrid = after.physicalGrid;
  candidate.evaluation = std::move(evaluation);
  return candidate;
}

bool sameJointCandidate(const JointProgramMappingCandidate &lhs,
                        const JointProgramMappingCandidate &rhs) {
  return sameCandidate(lhs.iat, rhs.iat) && lhs.blockT == rhs.blockT &&
         lhs.finalLogicalGrid == rhs.finalLogicalGrid &&
         lhs.finalPhysicalGrid == rhs.finalPhysicalGrid;
}

bool isBetterJointCandidate(const JointProgramMappingCandidate &lhs,
                            const JointProgramMappingCandidate &rhs) {
  const CandidateEvaluation &left = lhs.evaluation;
  const CandidateEvaluation &right = rhs.evaluation;
  if (left.tokenOnlyRepeatedBytesAfter != right.tokenOnlyRepeatedBytesAfter)
    return left.tokenOnlyRepeatedBytesAfter < right.tokenOnlyRepeatedBytesAfter;
  if (left.persistentLoopTripsAfter != right.persistentLoopTripsAfter)
    return left.persistentLoopTripsAfter < right.persistentLoopTripsAfter;
  if (left.candidate.actualProgramsAfter != right.candidate.actualProgramsAfter)
    return left.candidate.actualProgramsAfter >
           right.candidate.actualProgramsAfter;
  if (left.candidate.estimatedPeakLiveBytes !=
      right.candidate.estimatedPeakLiveBytes)
    return left.candidate.estimatedPeakLiveBytes <
           right.candidate.estimatedPeakLiveBytes;
  if (left.benefitScore != right.benefitScore)
    return left.benefitScore > right.benefitScore;
  return std::tie(lhs.iat.factor, lhs.blockT) <
         std::tie(rhs.iat.factor, rhs.blockT);
}

std::optional<JointProgramMappingCandidate>
selectJointCandidate(GraphOptimizationContext &context, bool emitRemarks) {
  // An empty selection is the explicit factor-1/no-op candidate.  It leaves
  // both the IR and launcher contract untouched when no transformed pair is
  // profitable and resource-safe.
  std::optional<JointProgramMappingCandidate> selected;
  for (unsigned iatFactor : kNormTensorizeFactors) {
    for (unsigned blockT : kNormPersistentBlockTCandidates) {
      std::optional<JointProgramMappingCandidate> candidate =
          evaluateJointCandidate(context, iatFactor, blockT);
      if (!candidate)
        continue;
      if (emitRemarks)
        emitCandidateRemark(candidate->iat.anchor, candidate->evaluation);
      if (!candidate->evaluation.accepted ||
          candidate->evaluation.benefitScore <= 0)
        continue;
      if (!selected || isBetterJointCandidate(*candidate, *selected))
        selected = std::move(candidate);
    }
  }
  return selected;
}

class JointProgramMappingPlan final : public RewritePlan {
public:
  JointProgramMappingPlan(JointProgramMappingCandidate candidate,
                          unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    // This is still the IAT rule's scheduling phase; PTSM's public rule id
    // remains unchanged and observes the committed marker later.
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  unsigned getBenefit() const override {
    const int64_t score = candidate.evaluation.benefitScore;
    if (score <= 0)
      return 1;
    return static_cast<unsigned>(std::min<int64_t>(
        score, static_cast<int64_t>(std::numeric_limits<unsigned>::max())));
  }
  Operation *getAnchor() const override { return candidate.iat.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.iat.function)
      return failure();
    std::optional<JointProgramMappingCandidate> current =
        evaluateJointCandidate(context, candidate.iat.factor, candidate.blockT);
    return current && current->evaluation.accepted &&
                   current->evaluation.benefitScore > 0 &&
                   sameJointCandidate(candidate, *current)
               ? success()
               : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.iat.function->getParentOfType<ModuleOp>();
    if (!module || !candidate.iat.requiresPersistentChaining ||
        hasConflictingLaunchContract(module))
      return failure();

    ModuleOp sandbox =
        createProgramMappingSandbox(module, candidate.iat.function);
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    if (!clonedFunction ||
        failed(materializeIATCandidateToSandbox(sandbox, clonedFunction,
                                                candidate.iat)) ||
        failed(materializePersistentTaskStripMiningCandidate(
            sandbox, clonedFunction, candidate.iat.resources, candidate.blockT,
            nullptr, /*deferIntermediateResourceRejection=*/true)) ||
        failed(runProgramMappingStructuralCleanup(sandbox)) ||
        failed(mlir::verify(sandbox.getOperation())))
      return failure();

    ProgramMappingLaunchProjection finalProjection;
    if (!hasExpectedJointLaunch(sandbox, clonedFunction, candidate.iat.factor,
                                candidate.blockT, candidate.iat.resources,
                                &finalProjection) ||
        finalProjection.logicalGrid != candidate.finalLogicalGrid ||
        finalProjection.physicalGrid != candidate.finalPhysicalGrid)
      return failure();

    Attribute transforms = sandbox->getAttr(kProgramGridTransformsAttr);
    Attribute iatMarker = sandbox->getAttr(kIndependentAxisTensorizeMarkerAttr);
    Attribute ptsmMarker =
        sandbox->getAttr(kPersistentTaskStripMiningMarkerAttr);
    if (!transforms || !iatMarker || !ptsmMarker)
      return failure();

    // All fallible work was done in the detached sandbox. Publishing the body
    // and the three coupled metadata fields is consequently all-or-nothing.
    candidate.iat.function->getRegion(0).takeBody(clonedFunction->getRegion(0));
    module->setAttr(kProgramGridTransformsAttr, transforms);
    module->setAttr(kIndependentAxisTensorizeMarkerAttr, iatMarker);
    module->setAttr(kPersistentTaskStripMiningMarkerAttr, ptsmMarker);
    return success();
  }

private:
  JointProgramMappingCandidate candidate;
  unsigned epoch;
};

class IndependentAxisTensorizePlan final : public RewritePlan {
public:
  IndependentAxisTensorizePlan(IATCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  unsigned getBenefit() const override {
    return candidate.evaluation.benefitScore > 0
               ? static_cast<unsigned>(candidate.evaluation.benefitScore)
               : 1;
  }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    std::optional<IATCandidate> current =
        candidate.usesMergeSplitSubCorePolicy
            ? selectMergeSplitLargeCandidate(context, /*emitRemarks=*/false)
            : analyzeCandidate(context, false);
    if (!current || !sameCandidate(candidate, *current))
      return failure();
    if (candidate.usesMergeSplitSubCorePolicy &&
        (candidate.evaluation.benefitScore !=
             current->evaluation.benefitScore ||
         candidate.evaluation.candidate.estimatedPeakLiveBytes !=
             current->evaluation.candidate.estimatedPeakLiveBytes))
      return failure();
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module || candidate.requiresPersistentChaining ||
        hasConflictingLaunchContract(module))
      return failure();

    // Materialize in a detached module. All late failures (unsupported tensor
    // relation, verifier failure, or contract serialization failure) therefore
    // leave the source function and its metadata untouched.
    ModuleOp sandbox = createProgramMappingSandbox(module, candidate.function);
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    if (!clonedFunction)
      return failure();

    if (failed(materializeIATCandidateToSandbox(sandbox, clonedFunction,
                                                candidate)))
      return failure();

    if (candidate.usesMergeSplitSubCorePolicy) {
      if (failed(runProgramMappingStructuralCleanup(sandbox)) ||
          failed(mlir::verify(sandbox.getOperation())))
        return failure();
      if (candidate.usesMergeSplitD192Tile &&
          !tryMaterializeMergeSplitD192Tile(sandbox, clonedFunction, candidate))
        return failure();
      std::optional<ProgramGridSpecialization> specialization =
          getGridSpecialization(candidate.function);
      std::optional<ProgramMappingLaunchProjection> before;
      if (specialization)
        before = projectProgramMappingLaunch(*specialization, {},
                                             candidate.resources);
      ProgramMappingLaunchProjection after;
      if (!before ||
          !hasExpectedMergeSplitLargeLaunch(sandbox, clonedFunction, candidate,
                                            candidate.resources, &after))
        return failure();
      ProgramAxisDependenceAnalysis analysis(candidate.function);
      const ProgramAxisDependence &headDependence =
          analysis.get(candidate.axis);
      LiveByteEstimate finalLiveBytes =
          estimatePeakLiveBytes(clonedFunction.getOperation());
      CandidateEvaluation finalEvaluation = evaluateCandidateCost(
          candidate.resources,
          buildResourceCandidate(
              candidate, headDependence,
              estimatePeakLiveBytes(candidate.function.getOperation()), *before,
              after, &finalLiveBytes));
      if (!finalEvaluation.accepted ||
          finalEvaluation.benefitScore != candidate.evaluation.benefitScore ||
          finalEvaluation.candidate.estimatedPeakLiveBytes !=
              candidate.evaluation.candidate.estimatedPeakLiveBytes)
        return failure();
    }

    // All potentially failing large-factor work has completed in the detached
    // sandbox. The body transfer has no allocation path and the generic
    // attribute was created in the same MLIRContext.
    candidate.function->getRegion(0).takeBody(clonedFunction->getRegion(0));
    module->setAttr(kProgramGridTransformsAttr,
                    sandbox->getAttr(kProgramGridTransformsAttr));
    module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                    UnitAttr::get(module.getContext()));
    return success();
  }

private:
  IATCandidate candidate;
  unsigned epoch;
};

class IndependentAxisTensorizeRule final : public GraphOptimizationRule {
public:
  explicit IndependentAxisTensorizeRule(bool enabledForCompileMode)
      : enabledForCompileMode(enabledForCompileMode) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::EntryArgPointerAlias |
           AnalysisRequirement::ProgramAxisDependence |
           AnalysisRequirement::ResourceCost;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (!enabledForCompileMode)
      return success();
    triton::FuncOp function = context.getFunction();
    std::optional<TensorizeFormMatch> formMatch =
        classifyTensorizeForm(function);
    std::optional<ProgramGridSpecialization> specialization =
        getGridSpecialization(function);
    if (formMatch && formMatch->form == TensorizeForm::NormRope &&
        specialization && isPTSMRuleEnabled(*specialization)) {
      std::optional<JointProgramMappingCandidate> candidate =
          selectJointCandidate(context, /*emitRemarks=*/true);
      if (!candidate)
        return success();
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] selected joint program mapping in @"
                 << candidate->iat.function.getName()
                 << ": iat_factor=" << candidate->iat.factor
                 << " block_t=" << candidate->blockT << " logical="
                 << candidate->evaluation.candidate.logicalTasksBefore << "->"
                 << candidate->evaluation.candidate.logicalTasksAfter
                 << " physical="
                 << candidate->evaluation.candidate.actualProgramsBefore << "->"
                 << candidate->evaluation.candidate.actualProgramsAfter
                 << "\n");
      plans.push_back(std::make_unique<JointProgramMappingPlan>(
          std::move(*candidate), context.getEpoch()));
      return success();
    }
    if (formMatch && formMatch->form == TensorizeForm::MergeSplit) {
      std::optional<IATCandidate> candidate =
          selectMergeSplitLargeCandidate(context, /*emitRemarks=*/true);
      if (candidate) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] selected MergeSplit large IAT in @"
                   << candidate->function.getName()
                   << ": factor=" << candidate->factor << " logical="
                   << candidate->evaluation.candidate.logicalTasksBefore << "->"
                   << candidate->evaluation.candidate.logicalTasksAfter
                   << " physical="
                   << candidate->evaluation.candidate.actualProgramsBefore
                   << "->"
                   << candidate->evaluation.candidate.actualProgramsAfter
                   << "\n");
        plans.push_back(std::make_unique<IndependentAxisTensorizePlan>(
            std::move(*candidate), context.getEpoch()));
        return success();
      }
    }
    std::optional<IATCandidate> candidate = analyzeCandidate(context, true);
    if (!candidate)
      return success();

    emitCandidateRemark(candidate->anchor, candidate->evaluation);
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] matched graph optimization rule "
               << static_cast<unsigned>(getId()) << " ("
               << getGraphOptimizationRuleName(getId()) << ") in @"
               << candidate->function.getName() << ": axis=" << candidate->axis
               << " factor=" << candidate->factor
               << " logical_extent=" << candidate->logicalExtent << "\n");
    plans.push_back(std::make_unique<IndependentAxisTensorizePlan>(
        std::move(*candidate), context.getEpoch()));
    return success();
  }

private:
  bool enabledForCompileMode;
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createIndependentAxisTensorizeRule(
    const IndependentAxisTensorizeRuleOptions &options) {
  return std::make_unique<IndependentAxisTensorizeRule>(
      options.enabledForCompileMode);
}
