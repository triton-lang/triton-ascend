/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ESTIMATE_CV_PIPELINE_COST_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ESTIMATE_CV_PIPELINE_COST_H

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace triton {

/// Estimated cost of the CV-pipelined module, in hardware cycles (i64).
/// Consumers rank alternative CV-pipeline IR variants by this number: lower is
/// better. Absent when the costmodel is not built into this binary, or when the
/// estimate could not be produced.
///
/// Built from a schedule over the compute blocks. Each hardware pipe runs one
/// thing at a time, so blocks overlap wherever they need different pipes; what
/// orders them is the IR's own barriers -- a block waiting on a synchronisation
/// flag cannot start before the block that sets it finishes. That is what makes
/// two variants containing the same operations in different blocks score
/// differently.
///
/// It is the larger of a resource bound (the busiest core's busy time) and a
/// recurrence bound (the buffer held longest, divided by how many copies of it
/// the pipeline allocated). The second matters because the inter-core depth
/// defaults to one, so the next iteration cannot start writing a buffer until
/// this one has finished reading it.
inline constexpr llvm::StringLiteral kCVPipelineEstimatedCycles =
    "ascend.cv_pipeline_estimated_cycles";

/// The same module costed as if Cube and Vector overlapped unconditionally
/// (i64) -- the busiest hardware unit's total busy time. This is a lower bound
/// on any schedule, so the gap to kCVPipelineEstimatedCycles is exactly the
/// time the model attributes to waiting on barriers.
inline constexpr llvm::StringLiteral kCVPipelineCostRoofline =
    "ascend.cv_pipeline_cost_roofline";

/// The two bounds kCVPipelineEstimatedCycles is the larger of, recorded
/// separately (i64 each) so that two variants tying on the total can still be
/// ordered.
///
/// Both are needed because the total throws away which constraint was slack.
/// Two variants held back by the same wall are not equally good: the one whose
/// other bound sits further below that wall has more room before something the
/// model does not charge -- scalar time, L1->L0 traffic, barrier latency --
/// pushes that bound over the wall and starts costing real time.
///
/// Measured on flash attention: intra-core buffer depths 2 and 3 produced an
/// identical total of 288 578 196 cycles, recurrence bounds of 184 465 709 and
/// 154 792 120, and 168 825 us against 147 302 us on hardware. The total could
/// not tell them apart; the recurrence bound ranked them correctly.
inline constexpr llvm::StringLiteral kCVPipelineCostResource =
    "ascend.cv_pipeline_cost_resource";
inline constexpr llvm::StringLiteral kCVPipelineCostRecurrence =
    "ascend.cv_pipeline_cost_recurrence";

/// Bytes of Unified Buffer the module allocates (i64), summed over every
/// allocation rather than tracked by liveness.
///
/// Recorded because it is the constraint that decides whether a block
/// partition can be compiled at all, and the only one of that kind the
/// estimate can see. A finer partition buys overlap with UB -- each new block
/// boundary materialises the value crossing it -- so the candidates a search
/// likes best are systematically the ones most likely to be refused by the
/// binary compiler, which reports the overrun and nothing else.
///
/// Not comparable to that compiler's own figure: this counts buffers that
/// never coexist, and the compiler multi-buffers on top of what it is given.
/// It is comparable *between candidates of one kernel*, which is what a search
/// needs.
inline constexpr llvm::StringLiteral kCVPipelineCostUBBytes =
    "ascend.cv_pipeline_cost_ub_bytes";

/// The most Unified Buffer live at any one point of the block schedule (i64),
/// as opposed to the sum above.
///
/// This is the figure worth comparing. Every allocation's size is exact and its
/// live range is read off the schedule, so the *difference* between two
/// variants of one kernel is a computation: it is precisely the buffers that a
/// finer partition forced into existence. The absolute value still is not what
/// the binary compiler will ask for -- it multi-buffers again on top, aligns to
/// banks, and adds temporaries after this pass -- but those all apply equally
/// to both variants and cancel in the comparison.
inline constexpr llvm::StringLiteral kCVPipelineCostUBPeak =
    "ascend.cv_pipeline_cost_ub_peak";

/// How far software pipelining stretched the deepest loop past the iterations
/// that actually do work (i64): the rewritten bound minus the trip count, taken
/// over every loop and maximised.
///
/// Diagnostic only, and it mixes two causes that behave oppositely -- see the
/// two attributes below, which separate them. Kept because it is the one number
/// directly comparable with the loop bound a reader sees in the IR.
inline constexpr llvm::StringLiteral kCVPipelineCostLoopExtension =
    "ascend.cv_pipeline_cost_loop_extension";

/// The buffer rescaling factor software pipelining applied (i64): the
/// requiredBuffers field of ssbuffer.iter_extension, maximised over loops.
///
/// NOT the stage count, despite the name this attribute has carried since it
/// was introduced -- the stage count is the attribute below. requiredBuffers is
/// how many buffers a producer/consumer pair separated by stages needs, and the
/// bound is multiplied by it so the same work is spread over proportionally
/// more iterations. Work is conserved across that rescaling, which is
/// measurable: two configurations differing only in this factor, with ramps of
/// 68 and 4 iterations, run within 2.4% of each other.
///
/// What it is good for is predicting a compile failure. The binary compiler
/// refuses a module for reading a buffer before its first write when the
/// prologue outruns the buffers backing it, and measured, the untouched
/// pipeline runs at 1 while every candidate refused for that reason had gone
/// to 2. That is why a search guards on it: guards are for modules that will
/// not build.
inline constexpr llvm::StringLiteral kCVPipelineCostPipelineDepth =
    "ascend.cv_pipeline_cost_pipeline_depth";

/// How many predicated stages the deepest loop's software pipeline has (i64):
/// the ifCount field of ssbuffer.iter_extension, maximised over loops.
///
/// This is the real depth, and unlike the factor above it is a cost rather than
/// a correctness limit: a pipeline of N stages spends N iterations filling and
/// draining, and those iterations run. The estimate charges them at the
/// profile's prologue fraction, so nothing here needs a guard -- a variant that
/// buys short dependency chains by adding stages now pays for them in its own
/// score. Published for diagnosis, because when a variant is expensive this is
/// usually why. Measured on flash attention: the untouched pipeline has 3, and
/// a search that could not see this cost picked one with 138.
inline constexpr llvm::StringLiteral kCVPipelineCostPipelineStages =
    "ascend.cv_pipeline_cost_pipeline_stages";

/// Unified Buffer the hardware profile says the part has, in bytes (i64).
///
/// Published so a consumer can size an allowance without carrying a copy of the
/// profile: the room a candidate may spend is what the baseline leaves unused,
/// and both halves of that subtraction are here.
inline constexpr llvm::StringLiteral kCVPipelineCostUBCapacity =
    "ascend.cv_pipeline_cost_ub_capacity";

/// Name of the hardware profile the estimate was produced against (StringAttr).
/// Estimates are only comparable across modules sharing the same profile.
inline constexpr llvm::StringLiteral kCVPipelineCostHardware =
    "ascend.cv_pipeline_cost_hardware";

/// Number of operations whose own cost could not be computed (i64), because a
/// shape is not statically known. They contribute nothing, so a high count
/// relative to the module size means the estimate should not be trusted.
inline constexpr llvm::StringLiteral kCVPipelineCostUnknownOps =
    "ascend.cv_pipeline_cost_unknown_ops";

/// Per-compute-block results, as an array of dictionaries -- one per
/// ssbuffer.block_id, plus one with id -1 for operations the pipeline left
/// unassigned. Each entry carries:
///   id, core ("CUBE"/"VECTOR"/"MIXED"), cycles (all iterations),
///   iter_cycles (one iteration), work_cycles (the plain sum),
///   iterations (how many times the block runs),
///   iter_start / iter_finish (position in the one-iteration schedule),
///   segments (runs of work separated by a barrier inside the block),
///   ops, bottleneck (busiest unit), depends_on (blocks whose values it reads),
///   sync_depends_on (blocks whose flag it waits on).
///
/// The block is the unit the CV pipeline schedules and synchronises, so this is
/// the view that maps onto its decisions. Block cycles still do not sum to the
/// module estimate -- blocks on different cores overlap wherever no barrier
/// orders them -- but iter_start/iter_finish show exactly how much they did.
inline constexpr llvm::StringLiteral kCVPipelineBlocks =
    "ascend.cv_pipeline_blocks";

/// Unit attribute asking the estimate to record its results on the module but
/// print nothing. Set by the variant search, which evaluates one candidate per
/// attempt and would otherwise emit a full report for each of them; the search
/// prints its own one-line summary instead. Not for ordinary compilation.
inline constexpr llvm::StringLiteral kCVPipelineCostQuiet =
    "ascend.cv_pipeline_cost_quiet";

/// Number of loops whose trip count is only known at run time (i64), such as a
/// loop bounded by a sequence length passed to the kernel. Their bodies are
/// counted once, so each of them understates the estimate by however many
/// iterations actually execute -- usually the largest single source of error.
inline constexpr llvm::StringLiteral kCVPipelineCostDynamicLoops =
    "ascend.cv_pipeline_cost_dynamic_loops";

/// Number of operations charged by element count alone, because their kind has
/// no dedicated cost model (i64). Unlike the unknown count these do contribute
/// cycles, but the number is an order of magnitude rather than a model, so a
/// high value means the estimate is coarse. Set TRITON_ASCEND_CV_COST_VERBOSE=2
/// to see which operation kinds they are.
inline constexpr llvm::StringLiteral kCVPipelineCostGenericOps =
    "ascend.cv_pipeline_cost_generic_ops";

/// Estimates the execution cost of the dynamic CV pipeline IR and records it as
/// module attributes. Purely advisory: it never mutates compute IR and never
/// fails the pipeline, so a costmodel problem can not break compilation.
///
/// Must run before RemoveSsbufAttrPass, which strips the ssbuffer.* attributes
/// (notably ssbuffer.core_type) this pass reads.
class EstimateCVPipelineCostPass
    : public PassWrapper<EstimateCVPipelineCostPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EstimateCVPipelineCostPass);

  EstimateCVPipelineCostPass() = default;
  explicit EstimateCVPipelineCostPass(std::string hardwareConfigPath)
      : hardwareConfigPath(std::move(hardwareConfigPath)) {}

  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "estimate-cv-pipeline-cost";
  }

private:
  /// Path to a costmodel hardware JSON. Empty selects the built-in default.
  std::string hardwareConfigPath;
};

std::unique_ptr<OperationPass<ModuleOp>>
createEstimateCVPipelineCostPass(std::string hardwareConfigPath = "");
void registerEstimateCVPipelineCostPasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ESTIMATE_CV_PIPELINE_COST_H
