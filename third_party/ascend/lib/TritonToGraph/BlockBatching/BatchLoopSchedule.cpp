// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"
#include "TritonToGraph/BlockBatching/BatchAffine.h"
namespace mlir::triton::cfg {
namespace {
using K = BatchExpr;

// Same order of magnitude as the layout analysis uses for one predicate. Loop
// bounds are far smaller than address expressions, so this is never the
// binding constraint; it only stops a pathological DAG.
constexpr unsigned kBudget = 256;

BatchResult<BatchLogicalAxis> reject(const char *reason) {
  return BatchResult<BatchLogicalAxis>::decline(
      {reason, "loop_coordinates", {}, {}, {}, {}});
}

// The coefficient of exactly one named leaf, when the form holds nothing else.
// "Nothing else" is the point: p*16 + arg is not a shape we can scale, and
// silently ignoring the extra term would scale the wrong thing.
std::optional<int64_t> soleCoefficient(const BatchAffineForm &form,
                                       K::Kind kind, int64_t id) {
  if (form.offset.constant != 0 || form.terms.size() != 1)
    return std::nullopt;
  const auto &[key, coefficient] = *form.terms.begin();
  if (std::get<0>(key) != kind || std::get<1>(key) != id)
    return std::nullopt;
  return coefficient.constant;
}

// Look through integer extensions. TTIR pid and rows_per_program values
// can each be wrapped in arith.extsi; strip those wrappers before matching.
const BatchExprRef &stripWidening(const BatchExprRef &e) {
  const BatchExprRef *cursor = &e;
  while (*cursor &&
         ((*cursor)->kind == K::ExtSI || (*cursor)->kind == K::ExtUI))
    cursor = &(*cursor)->lhs;
  return *cursor;
}

// Uniform means independent of the selected program axis. Segment width
// must be the same for all F lanes so each can compute its own segment.
bool laneUniform(const BatchExprRef &e, int gridAxis) {
  return e && BatchExpr::isLaneUniform(e, gridAxis);
}

// Match p*R in either operand order and return R. R need only be uniform:
// it can be a constant or a runtime value, such as rows_per_program in
// rms_norm_backward. Use structural matching because that product need
// not fit batchLinearForm.
std::optional<BatchExprRef> matchProgramTimesWidth(const BatchExprRef &e,
                                                   int gridAxis) {
  const BatchExprRef &root = stripWidening(e);
  if (!root || root->kind != K::Mul)
    return std::nullopt;
  for (auto [maybeProgram, maybeWidth] :
       {std::pair{root->lhs, root->rhs}, std::pair{root->rhs, root->lhs}}) {
    const BatchExprRef &program = stripWidening(maybeProgram);
    if (program && program->kind == K::Program && program->value == gridAxis &&
        laneUniform(maybeWidth, gridAxis))
      return maybeWidth;
  }
  return std::nullopt;
}

// A segment end is uniform or min((p+1)*R, uniform expression).
// The latter includes min((pid+1)*rows_per_program, n_rows), as used by
// the blocked-distribution positive cases.
bool matchSegmentUpper(const BatchExprRef &upper, const BatchExprRef &width,
                       int gridAxis) {
  if (laneUniform(upper, gridAxis))
    return true;
  const BatchExprRef &root = stripWidening(upper);
  if (!root || root->kind != K::MinSI)
    return false;
  for (auto [maybeEnd, maybeCap] :
       {std::pair{root->lhs, root->rhs}, std::pair{root->rhs, root->lhs}}) {
    if (!laneUniform(maybeCap, gridAxis))
      continue;
    const BatchExprRef &end = stripWidening(maybeEnd);
    if (!end || end->kind != K::Mul)
      continue;
    for (auto [maybeNext, maybeWidth] :
         {std::pair{end->lhs, end->rhs}, std::pair{end->rhs, end->lhs}}) {
      // Use the same width as the segment start; otherwise adjacent segments
      // could overlap or leave gaps.
      if (!batchExprEquivalent(maybeWidth, width))
        continue;
      const BatchExprRef &next = stripWidening(maybeNext);
      if (!next || next->kind != K::Add)
        continue;
      for (auto [maybeProgram, maybeOne] :
           {std::pair{next->lhs, next->rhs}, std::pair{next->rhs, next->lhs}}) {
        const BatchExprRef &program = stripWidening(maybeProgram);
        const BatchExprRef &one = stripWidening(maybeOne);
        if (program && program->kind == K::Program &&
            program->value == gridAxis && one && one->kind == K::Constant &&
            one->value == 1)
          return true;
      }
    }
  }
  return false;
}
} // namespace

BatchSchedule selectBatchLoopLowering(const BatchLoopFact *loop,
                                      const BatchLogicalAxis *axis,
                                      int64_t factor) {
  if (!loop || !axis || loop->id != axis->loopID || !loop->carriesValues)
    return BatchSchedule::ProgramLanes;
  // nullopt trip counts include both dynamic inputs and unrepresentable
  // static counts. Only the former may defer to a runtime guard.
  if (batchLoopIterationArithmetic(*loop, factor) ==
      BatchLoopArithmetic::Unsafe)
    return BatchSchedule::ProgramLanes;
  // No complete iteration group exists. Retain the original program-lane
  // route when the existing trip-count analysis can prove this statically.
  if (auto trips = batchLoopConstantTripCount(*loop); trips && *trips < factor)
    return BatchSchedule::ProgramLanes;
  if ((axis->distribution == BatchLogicalAxis::Cyclic) ||
      (axis->distribution == BatchLogicalAxis::Blocked &&
       BatchExpr::isLaneUniform(loop->lower, axis->gridAxis) &&
       BatchExpr::isLaneUniform(loop->upper, axis->gridAxis) &&
       BatchExpr::isLaneUniform(loop->step, axis->gridAxis)))
    return BatchSchedule::SequentialIterations;
  return BatchSchedule::ProgramLanes;
}

BatchResult<BatchLogicalAxis>
normalizeBatchLoopCoordinates(const BatchLoopFact &loop, int gridAxis) {
  if (!loop.lower || !loop.upper || !loop.step || gridAxis < 0)
    return reject("UNPROVEN_LOOP_COALESCING");

  // Gate 1: prove a positive step. Cyclic G*R need not be constant, but
  // num_programs >= 1 allows a positive lower bound for its range.
  if (!batchLoopHasProvenPositiveStep(loop))
    return reject("UNPROVEN_POSITIVE_LOOP_STEP");

  // Gate 2: a constant positive step gives blocked distribution. Folded
  // lane u represents original program g*F+u and must retain its segment.
  // Accept only these bound forms:
  //   U: both bounds are uniform; all F lanes run the same loop.
  //   S: lower is p*R; upper is uniform or min((p+1)*R, uniform), with
  //      the same R at both ends.
  // Reject all other forms. Keeping arbitrary bounds and multiplying the
  // step by F belongs to an unfolded scheme, not this grid /= F mapping.
  if (batchLoopConstantStep(loop)) {
    bool ok =
        laneUniform(loop.lower, gridAxis)
            ? laneUniform(loop.upper, gridAxis)
            : [&] {
                auto width = matchProgramTimesWidth(loop.lower, gridAxis);
                return width && matchSegmentUpper(loop.upper, *width, gridAxis);
              }();
    if (!ok)
      return reject("UNPROVEN_LANE_SEGMENT_BOUND");
    return BatchResult<BatchLogicalAxis>::success(
        {BatchLogicalAxis::Blocked, loop.id, gridAxis, 1});
  }

  // Gate 3: cyclic distribution requires lower = p*R and step = G*R with
  // the same R. R is the rows per iteration; batching expands the chunk to
  // R*F.
  unsigned budget = kBudget;
  auto origin =
      BatchAffineForm::fromExpr(loop.lower, budget, kBatchLoopLeaves,
                                BatchAffineForm::Profile::Integer, -1);
  auto stride =
      BatchAffineForm::fromExpr(loop.step, budget, kBatchLoopLeaves,
                                BatchAffineForm::Profile::Integer, -1);
  if (!origin || !stride)
    return reject("UNPROVEN_LOOP_COALESCING");
  auto perProgram = soleCoefficient(*origin, K::Program, gridAxis);
  auto perStep = soleCoefficient(*stride, K::Grid, gridAxis);
  // No separate R >= 1 check is needed: soleCoefficient proves step = G*R,
  // G >= 1, and gate 1 already proved a positive step. Together these imply
  // R >= 1.
  if (!perProgram || !perStep || *perProgram != *perStep)
    return reject("UNPROVEN_LOOP_COALESCING");

  // Cyclic batching folds F programs' rows into one shared loop, so its
  // upper bound must be uniform across lanes.
  if (!BatchExpr::isLaneUniform(loop.upper, gridAxis))
    return reject("VARYING_LOOP_BOUND");

  return BatchResult<BatchLogicalAxis>::success(
      {BatchLogicalAxis::Cyclic, loop.id, gridAxis, *perProgram});
}

BatchExprRef normalizeBatchAccessOnLogicalAxis(const BatchExprRef &e,
                                               const BatchLoopFact &loop) {
  // Memoize by node identity to avoid exponential traversal of shared
  // subtrees in the address DAG.
  std::map<const BatchExpr *, BatchExprRef> seen;
  // Null means either a matching induction variable has no lower bound or
  // the input itself is null. Callers treat both as an unavailable proof.
  auto rebuild = [&](auto &self, const BatchExprRef &node) -> BatchExprRef {
    if (!node)
      return node;
    if (node->kind == BatchExpr::Induction && node->value == int64_t(loop.id))
      return loop.lower
                 ? BatchExpr::node(BatchExpr::Add, node->bits, loop.lower, node)
                 : BatchExprRef();
    if (!node->lhs && !node->rhs)
      return node; // Other leaves are independent of this loop.
    auto it = seen.find(node.get());
    if (it != seen.end())
      return it->second;
    BatchExprRef lhs = self(self, node->lhs);
    BatchExprRef rhs = self(self, node->rhs);
    if ((node->lhs && !lhs) || (node->rhs && !rhs))
      return seen[node.get()] = BatchExprRef();
    // Preserve the pointer when the entire subtree is independent of this
    // loop: downstream checks compare plan.coordinate and predicates by
    // pointer identity.
    if (lhs == node->lhs && rhs == node->rhs)
      return seen[node.get()] = node;
    // Rebuild range nodes with range(), not node(), to preserve rangeOrigin
    // and the rangeGroup proof. For example, _block_rms_norm_backward uses
    // row_idx = start + arange(0,16), where start is the induction variable.
    return seen[node.get()] =
               node->rangeOrigin
                   ? BatchExpr::range(lhs, rhs)
                   : BatchExpr::node(node->kind, node->bits, lhs, rhs);
  };
  return rebuild(rebuild, e);
}
} // namespace mlir::triton::cfg
