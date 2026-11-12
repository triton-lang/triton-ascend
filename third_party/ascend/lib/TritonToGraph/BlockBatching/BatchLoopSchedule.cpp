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

BatchLoopScheduleResult reject(const char *reason) {
  return {std::nullopt, reason};
}

// The coefficient of exactly one named leaf, when the form holds nothing else.
// "Nothing else" is the point: p*16 + arg is not a shape we can scale, and
// silently ignoring the extra term would scale the wrong thing.
std::optional<int64_t> soleCoefficient(const BatchLinearForm &form,
                                       K::Kind kind, int64_t id) {
  if (form.constant != 0 || form.terms.size() != 1)
    return std::nullopt;
  const auto &[key, coefficient] = *form.terms.begin();
  if (key.first != kind || key.second != id)
    return std::nullopt;
  return coefficient;
}

// 穿透位宽转换。TTIR 里 pid 和 rows_per_program 都是 i32，用之前各裹一层
// arith.extsi，结构匹配必须先把这层剥掉。
const BatchExprRef &stripWidening(const BatchExprRef &e) {
  const BatchExprRef *cursor = &e;
  while (*cursor &&
         ((*cursor)->kind == K::ExtSI || (*cursor)->kind == K::ExtUI))
    cursor = &(*cursor)->lhs;
  return *cursor;
}

// 各份相同：不随合并轴上的程序号变化。lane u 要能算出自己那一段，段宽就必须
// 对 F 条 lane 是同一个值。
bool laneUniform(const BatchExprRef &e, const BatchRangeFacts &ranges) {
  return e && ranges.isLaneUniform(e);
}

// 匹配 p*R（两种操作数次序都认），返回 R。R 只要求各份相同，可以是常数，也可以
// 是运行期参数——rms_norm_backward 的 rows_per_program 就是个 i32 参数，拆不进
// 仿射形式，所以这里走结构而不是 batchLinearForm。
std::optional<BatchExprRef>
matchProgramTimesWidth(const BatchExprRef &e, int gridAxis,
                       const BatchRangeFacts &ranges) {
  const BatchExprRef &root = stripWidening(e);
  if (!root || root->kind != K::Mul)
    return std::nullopt;
  for (auto [maybeProgram, maybeWidth] :
       {std::pair{root->lhs, root->rhs}, std::pair{root->rhs, root->lhs}}) {
    const BatchExprRef &program = stripWidening(maybeProgram);
    if (program && program->kind == K::Program && program->value == gridAxis &&
        laneUniform(maybeWidth, ranges))
      return maybeWidth;
  }
  return std::nullopt;
}

// 段尾：lane 一致（整条循环的终点各份相同），或 min((p+1)*R, 各份相同式)。
// 第二种是 min((pid+1)*rows_per_program, n_rows)，两个定长分块正例都长这样。
bool matchSegmentUpper(const BatchExprRef &upper, const BatchExprRef &width,
                       int gridAxis, const BatchRangeFacts &ranges) {
  if (laneUniform(upper, ranges))
    return true;
  const BatchExprRef &root = stripWidening(upper);
  if (!root || root->kind != K::MinSI)
    return false;
  for (auto [maybeEnd, maybeCap] :
       {std::pair{root->lhs, root->rhs}, std::pair{root->rhs, root->lhs}}) {
    if (!laneUniform(maybeCap, ranges))
      continue;
    const BatchExprRef &end = stripWidening(maybeEnd);
    if (!end || end->kind != K::Mul)
      continue;
    for (auto [maybeNext, maybeWidth] :
         {std::pair{end->lhs, end->rhs}, std::pair{end->rhs, end->lhs}}) {
      // 段宽必须和段首用的是同一个，否则各段首尾接不上，会重叠或漏行。
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

BatchLoopLowering selectBatchLoopLowering(const BatchLoopFact *loop,
                                          const BatchLogicalAxis *axis,
                                          const BatchRangeFacts &ranges,
                                          int64_t factor) {
  if (!loop || !axis || loop->id != axis->loopID || !loop->carriesValues)
    return BatchLoopLowering::ProgramLanes;
  // nullopt trip counts include both dynamic inputs and unrepresentable
  // static counts. Only the former may defer to a runtime guard.
  if (batchLoopIterationArithmetic(*loop, ranges, factor) ==
      BatchLoopArithmetic::Unsafe)
    return BatchLoopLowering::ProgramLanes;
  // No complete iteration group exists. Retain the original program-lane
  // route when the existing trip-count analysis can prove this statically.
  if (auto trips = batchLoopConstantTripCount(*loop, ranges);
      trips && *trips < factor)
    return BatchLoopLowering::ProgramLanes;
  if ((axis->distribution == BatchLogicalAxis::Cyclic) ||
      (axis->distribution == BatchLogicalAxis::Blocked &&
       ranges.isLaneUniform(loop->lower) && ranges.isLaneUniform(loop->upper) &&
       ranges.isLaneUniform(loop->step)))
    return BatchLoopLowering::SequentialIterations;
  return BatchLoopLowering::ProgramLanes;
}

BatchLoopScheduleResult
normalizeBatchLoopCoordinates(const BatchLoopFact &loop,
                              const BatchRangeFacts &ranges, int gridAxis) {
  if (!loop.lower || !loop.upper || !loop.step || gridAxis < 0)
    return reject("UNPROVEN_LOOP_COALESCING");

  // 关卡一：步长必须是正的。成块发的 G*R 不是常数，但 num_programs >= 1 让它的
  // 值域下界为正，所以这一关对两种分布都过得去。
  if (!batchLoopHasProvenPositiveStep(loop, ranges))
    return reject("UNPROVEN_POSITIVE_LOOP_STEP");

  // 关卡二：步长是常数就是定长分块。折叠之后 lane u 就是原程序 g*F+u，要守住
  // 它自己那一段，所以上下界必须能按 lane 算出来。只认两种形态：
  //   U 形  上下界都各份相同——F 条 lane 跑一模一样的循环，不分段。
  //   S 形  下界恰是 p*R，上界各份相同或 min((p+1)*R, 各份相同式)，两个 R
  //   同一个。
  // 其余拒。2026-11-02 之前这里对边界零要求，那条放开的依据是"合并只把步长乘
  // F、边界一个字不动"的不折叠手写版方案，与本设计服从的 grid /= F 不符。
  if (batchLoopConstantStep(loop, ranges)) {
    bool ok = laneUniform(loop.lower, ranges)
                  ? laneUniform(loop.upper, ranges)
                  : [&] {
                      auto width =
                          matchProgramTimesWidth(loop.lower, gridAxis, ranges);
                      return width && matchSegmentUpper(loop.upper, *width,
                                                        gridAxis, ranges);
                    }();
    if (!ok)
      return reject("UNPROVEN_LANE_SEGMENT_BOUND");
    return {BatchLogicalAxis{BatchLogicalAxis::Blocked, loop.id, gridAxis, 1},
            ""};
  }

  // 关卡三：成块发。起点必须恰好是 p*R、步长恰好是 G*R，且两个 R 相同——R 就是
  // 一圈覆盖的行数，合并时把它放大到 R*F。
  unsigned budget = kBudget;
  auto origin = batchLinearForm(loop.lower, budget, kBatchLoopLeaves);
  auto stride = batchLinearForm(loop.step, budget, kBatchLoopLeaves);
  if (!origin || !stride)
    return reject("UNPROVEN_LOOP_COALESCING");
  auto perProgram = soleCoefficient(*origin, K::Program, gridAxis);
  auto perStep = soleCoefficient(*stride, K::Grid, gridAxis);
  // R >= 1 不用在这里查：soleCoefficient 成功意味着步长恒等于 G*R，而 G >= 1，
  // 所以关卡一证出的正步长已经蕴含 R >= 1。多写一次只会是测不到的死分支。
  if (!perProgram || !perStep || *perProgram != *perStep)
    return reject("UNPROVEN_LOOP_COALESCING");

  // 成块发把 F 个程序的行折进一条循环，各份共用这一条循环，所以上界必须一致。
  if (!ranges.isLaneUniform(loop.upper))
    return reject("VARYING_LOOP_BOUND");

  return {BatchLogicalAxis{BatchLogicalAxis::Cyclic, loop.id, gridAxis,
                           *perProgram},
          ""};
}

BatchExprRef normalizeBatchAccessOnLogicalAxis(const BatchExprRef &e,
                                               const BatchLoopFact &loop) {
  // 记忆化按节点指针做。地址 DAG 里共享子树很常见，没有它这就是指数级的。
  std::map<const BatchExpr *, BatchExprRef> seen;
  // 返回空有两种含义，调用方都当"证不出"处理：命中了循环变量但 lower 缺失，
  // 或者入参本来就是空。
  auto rebuild = [&](auto &self, const BatchExprRef &node) -> BatchExprRef {
    if (!node)
      return node;
    if (node->kind == BatchExpr::Induction && node->value == int64_t(loop.id))
      return loop.lower
                 ? BatchExpr::node(BatchExpr::Add, node->bits, loop.lower, node)
                 : BatchExprRef();
    if (!node->lhs && !node->rhs)
      return node; // 其余叶子与这条循环无关。
    auto it = seen.find(node.get());
    if (it != seen.end())
      return it->second;
    BatchExprRef lhs = self(self, node->lhs);
    BatchExprRef rhs = self(self, node->rhs);
    if ((node->lhs && !lhs) || (node->rhs && !rhs))
      return seen[node.get()] = BatchExprRef();
    // 整条子树都与这条循环无关就原样返回，指针不变：下游拿 plan.coordinate 和
    // 谓词做指针相等比较（BatchLayoutAnalysis.cpp:282）。
    if (lhs == node->lhs && rhs == node->rhs)
      return seen[node.get()] = node;
    // range 节点要用 range() 重建：它带 rangeOrigin 标志，而 node() 不带，丢了
    // 它多轴来源的证明（rangeGroup）会无故证不出。_block_rms_norm_backward 的
    // row_idx = start + arange(0,16) 正是这种节点，且 start 就是循环变量。
    return seen[node.get()] =
               node->rangeOrigin
                   ? BatchExpr::range(lhs, rhs)
                   : BatchExpr::node(node->kind, node->bits, lhs, rhs);
  };
  return rebuild(rebuild, e);
}
} // namespace mlir::triton::cfg
