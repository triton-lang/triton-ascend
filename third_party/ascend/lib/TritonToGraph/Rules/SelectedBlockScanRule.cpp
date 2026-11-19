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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// SelectedBlockScanRule (#2686, rule bit 65536): the driver shell.
//
// Only the framework-facing layer: the RewritePlan that owns a pre-verified
// sandbox, the GraphOptimizationRule, the joint scan + row-lift commit contract
// and the factory. No IR pattern knowledge; the matcher and the rewriter are
// their own translation units.

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/QueryRowBatching.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

// Joint decision point: this rule never consults a read-only row-lift matcher.
// It hands its private sandbox to the one QueryRowBatchingRule entry declared
// in QueryRowBatching.h, which rewrites the row lift in place and verifies the
// result, so landing the sandbox means both rewrites are already in it and a
// failure means neither lands. There is no separate enable flag: the row
// lift's own implementation is the gate.

#include "TritonToGraph/SelectedBlockScanFacts.h"

namespace {

// Identity of a rewrite across re-matches. Driver-internal, so it keeps
// internal linkage and stays out of the shared header.
// Shape identity: two candidates emit the same rewrite even when their IR
// handles come from different modules. This is what compares the live match
// against the sandbox clone's match.
static bool sameScanShape(const ScanCandidate &a, const ScanCandidate &b) {
  return a.recurrence.index() == b.recurrence.index() &&
         a.tiles.index() == b.tiles.index() &&
         a.table.length == b.table.length &&
         tileBlockSize(a.tiles) == tileBlockSize(b.tiles);
}

static bool sameScanCandidate(const ScanCandidate &a, const ScanCandidate &b) {
  // The integers alone are not an identity: two different loops in the same
  // function can have the same shape, and the plan must anchor on the loop it
  // actually matched. The anchor handles are compared first, then the shape.
  if (a.function != b.function || a.loop.forOp != b.loop.forOp)
    return false;
  if (a.recurrence.index() != b.recurrence.index() ||
      a.tiles.index() != b.tiles.index())
    return false;
  if (a.rowPosition != b.rowPosition || a.table.root != b.table.root ||
      a.scanBound != b.scanBound)
    return false;
  if (kTileLoad(a.tiles) != kTileLoad(b.tiles) ||
      vTileLoad(a.tiles) != vTileLoad(b.tiles))
    return false;
  if (scoreDotOf(a.recurrence) != scoreDotOf(b.recurrence) ||
      valueDotOf(a.recurrence) != valueDotOf(b.recurrence))
    return false;
  return a.table.length == b.table.length &&
         tileBlockSize(a.tiles) == tileBlockSize(b.tiles);
}

// ---- Plan / Rule / factory ----

class SelectedBlockScanPlan final : public RewritePlan {
  ScanCandidate candidate;
  unsigned epoch;
  // A sandbox already accepted and verified on its own: it holds the scan
  // rewrite and may already carry the row lift and the launch contract.
  // The accepted sandbox: it already carries the scan rebuild and, when the
  // joint gate ran, the row lift and its launch contract.
  OwningOpRef<ModuleOp> prepared;

  SelectedBlockScanPlan(ScanCandidate candidate, unsigned epoch,
                        OwningOpRef<ModuleOp> prepared)
      : candidate(candidate), epoch(epoch), prepared(std::move(prepared)) {}

public:
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::SelectedBlockScan;
  }

  unsigned getBenefit() const override { return 1; }

  // The anchor stays the real function's gather loop: getAnchor() feeds the
  // driver's program-order map, which is built on the real function, so a
  // sandbox anchor would not be found.
  Operation *getAnchor() const override { return candidate.loop.forOp; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    // Re-run the full structural match and compare the anchor and parameters.
    std::optional<ScanCandidate> current =
        matchSelectedBlockScan(candidate.function);
    if (!current || current->loop.forOp != candidate.loop.forOp ||
        !sameScanCandidate(*current, candidate))
      return failure();
    // The row lift is not re-consulted here: the pre-commit function still has
    // the gather shape it does not match, so a second consult would always
    // decline. The single consult is sound because the body apply() commits IS
    // the sandbox body that was accepted, and the match just re-proved that
    // this candidate rebuilds that body.
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // The sandbox has already been verified; this only commits it.  With a
    // runtime grid the row lift appends a hidden extent argument, so the
    // signature has to be swapped too instead of just taking the body.
    // Commit only: the sandbox is already verified.  A runtime grid makes the
    // row lift append hidden extent arguments, so the function signature has to
    // travel with the body instead of a plain takeBody.  Nothing is emitted
    // here, so the rewriter is unused.
    static_cast<void>(rewriter);
    auto replacement = cast<triton::FuncOp>(&prepared->getBody()->front());
    ModuleOp preparedModule = *prepared;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (preparedModule->hasAttr(kProgramGridTransformsAttr)) {
      if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                      replacement)))
        return failure();
      module->setAttr(kProgramGridTransformsAttr,
                      preparedModule->getAttr(kProgramGridTransformsAttr));
      return success();
    }

    // The static-grid commit keeps the live signature: if that ever stops
    // being an invariant of the row lift, fail here instead of producing a
    // whose type disagrees with its entry block.
    if (replacement.getFunctionType() != candidate.function.getFunctionType())
      return failure();
    candidate.function.getBody().takeBody(replacement.getBody());
    for (StringLiteral name : {StringLiteral("hacc.coalesce_factor"),
                               StringLiteral("hacc.coalesce_axis"),
                               StringLiteral("hacc.coalesce_grid_ceil_div")})
      if (preparedModule->hasAttr(name))
        module->setAttr(name, preparedModule->getAttr(name));
    return success();
  }

  // Prepare and verify privately before offering a plan to the scheduler. An
  // unsupported rewrite must leave the input and its diagnostics untouched;
  // declining returns nullptr and never fails apply().
  static std::unique_ptr<SelectedBlockScanPlan>
  create(const ScanCandidate &candidate, unsigned epoch,
         const ResourceSnapshot &resources) {
    // The tablegen accessors used below are not const-qualified in this tree,
    // so take a non-const handle to the same op (not a copy of the function).
    triton::FuncOp function = candidate.function;
    ModuleOp module = function->getParentOfType<ModuleOp>();
    if (!module)
      return nullptr;

    OwningOpRef<ModuleOp> sandbox(ModuleOp::create(function.getLoc()));
    // The sandbox must carry the module attributes: the row lift's matcher
    // module-level gates, and a sandbox without them would decline forever.
    sandbox->getOperation()->setAttrs(module->getAttrs());
    sandbox->getBody()->push_back(function->clone());
    auto cloned = cast<triton::FuncOp>(&sandbox->getBody()->front());

    // Re-match on the clone so recorded operations point into the sandbox.
    std::optional<ScanCandidate> sandboxCandidate =
        matchSelectedBlockScan(cloned);
    if (!sandboxCandidate || !sameScanShape(*sandboxCandidate, candidate))
      return nullptr;

    // Sandbox diagnostics are only swallowed in this scope; the budget gate
    // has to warn on the live function, so it lives outside.
    // Sandbox diagnostics are swallowed only in this scope: the budget gate
    // warning must reach the user, so it runs after the scope.  The row lift
    // installs its own handler, so its rewrite stays quiet as before.
    {
      IRRewriter rewriter(module.getContext());
      ScopedDiagnosticHandler diagnostics(
          module.getContext(), [](Diagnostic &diag) {
            LLVM_DEBUG(llvm::dbgs()
                       << "selected-block-scan: skipped candidate: " << diag
                       << "\n");
            return success();
          });
      if (!buildSelectedBlockScan(rewriter, *sandboxCandidate))
        return nullptr;
      // Tidy and verify the private sandbox here so neither a malformed nor an
      // untidied body reaches the scheduler. The cleanup runs before the joint
      // consult. `cloned` stays valid: canonicalize/CSE/LICM rewrite operations
      // inside the function but never erase the tt.func op itself.
      if (failed(runSandboxStructuralCleanup(*sandbox)))
        return nullptr;
      if (failed(verify(*sandbox)))
        return nullptr;
    }

    // The scan rewrite's own shape goes through the budget gate first, and the
    // joint shape is checked once more on the final sandbox (the row lift
    // grows the tile and the live set).
    // The scan form passes the gate first; the joint form (the row lift widens
    // the tiles and the live set) is measured again before the commit.
    if (budgetOverflow(cloned, resources, "the scan rewrite", function))
      return nullptr;

    // Joint decision: run the row lift's real implementation (the materialize
    // entry in QueryRowBatching.h) directly on the sandbox.  Whether it
    // accepts is the only switch; there is no extra rule bit to keep in
    // sync.  materialize rewrites in place and verifies, so failing voids
    // the whole plan; on success the sandbox already holds the row lift and
    // the launch contract, and apply() only has to commit it.
    if (failed(materializeQueryRowBatchingCandidate(*sandbox, cloned,
                                                    resources))) {
      LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: joint query-row "
                                 "batching materialization declined @"
                              << function.getName() << "\n");
      return nullptr;
    }
    LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: joint query-row "
                               "batching materialized @"
                            << function.getName() << "\n");
    // The row lift rewrote the sandbox after the scan rewrite's cleanup+verify,
    // so re-run the same hygiene on the joint result before it can be
    // committed.  A failure declines the plan (never fails apply()), matching
    // the rule's contract.  Same hygiene scope as the scan build: the joint
    // cleanup and verify must also keep a decline silent, since a raw verify()
    // failure would print an error to the user's stderr for a plan being
    // dropped.
    LogicalResult jointHygiene = failure();
    {
      ScopedDiagnosticHandler diagnostics(
          module.getContext(), [](Diagnostic &diag) {
            LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: joint cleanup: "
                                    << diag << "\n");
            return success();
          });
      jointHygiene = runSandboxStructuralCleanup(*sandbox);
      if (succeeded(jointHygiene))
        jointHygiene = verify(*sandbox);
    }
    if (failed(jointHygiene)) {
      LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: joint rewrite failed "
                                 "post-batching verification @"
                              << function.getName() << "\n");
      return nullptr;
    }
    if (budgetOverflow(cloned, resources, "the joint scan + row-lift rewrite",
                       function))
      return nullptr;

    // Launch-contract mutual exclusion.  A joint rewrite may either hand the
    // grid to the launcher (hacc.program_grid_transforms / hacc.coalesce_*) or
    // decode a flat program id inside the kernel (tt.get_num_programs) -- never
    // both: the launcher would divide the grid while the kernel still assumes
    // the original one, silently computing a fraction of the queries.
    {
      bool hasGridContract = (*sandbox)->hasAttr(kProgramGridTransformsAttr) ||
                             (*sandbox)->hasAttr("hacc.coalesce_factor");
      bool readsProgramCount = false;
      cloned.walk([&](triton::GetNumProgramsOp) { readsProgramCount = true; });
      if (hasGridContract && readsProgramCount) {
        LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: grid contract and "
                                   "flat program-id decoding must not coexist @"
                                << function.getName() << "\n");
        return nullptr;
      }
    }

    return std::unique_ptr<SelectedBlockScanPlan>(
        new SelectedBlockScanPlan(candidate, epoch, std::move(sandbox)));
  }
};

class SelectedBlockScanRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::SelectedBlockScan;
  }

  // The scan rewrite needs the L0 budget snapshot and forwards it to the row
  // lift's (QueryRowBatchingRule) materialize entry.
  // The scan rule forwards the L0 budget snapshot to QueryRowBatchingRule's
  // materialize entry.
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::ResourceCost;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    // Pure read-only matching first: a miss is a silent continue with no
    // diagnostics and no IR change.
    std::optional<ScanCandidate> candidate =
        matchSelectedBlockScan(context.getFunction());
    if (!candidate)
      return success();

    LLVM_DEBUG(llvm::dbgs()
               << "selected-block-scan: matched gather loop in @"
               << candidate->function.getName()
               << " (BS=" << tileBlockSize(candidate->tiles) << " BN="
               << scanTileWidth(tileBlockSize(candidate->tiles),
                                std::holds_alternative<AdditiveRecurrence>(
                                    candidate->recurrence))
               << " S=" << candidate->table.length << ")\n");

    if (auto plan = SelectedBlockScanPlan::create(
            *candidate, context.getEpoch(),
            context.getResourceCostAnalysis().getResourceSnapshot()))
      plans.push_back(std::move(plan));
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createSelectedBlockScanRule() {
  return std::make_unique<SelectedBlockScanRule>();
}
