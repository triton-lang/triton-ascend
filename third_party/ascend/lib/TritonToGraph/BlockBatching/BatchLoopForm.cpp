// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopForm.h"

#include "TritonToGraph/BlockBatching/BatchRequest.h"

namespace mlir::triton::cfg {

BatchResult<scf::ForOp> checkBatchLoopForm(Operation *function,
                                           BatchVaryingPredicate isVarying) {
  scf::ForOp programLoop;
  BatchDiagnostic first;
  auto walk = function->walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
    scf::ForOp outer = programLoop;
    // 程序轴循环内部的循环：上下界与步长必须定义在它之外，否则并合之后 F 份
    // 共用同一个内层循环时长度不一，并不到一起。只看定义位置，不追数据流，所以
    // 一个与循环变量无关的常量只要写在外层体内也会被拒——保守的方向，规范化之后
    // 常量会被提到外面。
    if (outer && outer->isProperAncestor(loop)) {
      for (Value bound :
           {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()})
        if (outer->isAncestor(bound.getParentBlock()->getParentOp())) {
          first = snapshotBatchDiagnostic(loop, "loop_form",
                                          "INNER_TRIP_COUNT_NOT_INVARIANT");
          return WalkResult::interrupt();
        }
      // 界在外层之外定义但随 pid 变化：程序轴循环套程序轴循环，落到下面拒。
    }
    // 三个界里**任何一个**随程序轴变化，这条循环就归这里管。只看下界不够：
    // 下界是常量而上界随 pid 变的循环各 lane 圈数不同，合并之后同样并不到一起，
    // 而它既不是成块发也不是定长分块，关卡 1 会把它拒掉。放过去的话就没人拒了
    // ——S8b2 开闸之后控制流那一层不再看界。
    if (!isVarying(loop.getLowerBound()) && !isVarying(loop.getUpperBound()) &&
        !isVarying(loop.getStep()))
      return WalkResult::advance();
    // 只认一条程序轴循环，而且必须直接在函数体顶层：不在 scf.if 里，不在块内
    // 循环里，不与另一条程序轴循环并列。重分块要把"程序编号"和"这条循环的循环
    // 变量"合成一个行号，有两条就有两个行号要合，处理不了。
    if (outer || loop->getParentOp() != function) {
      first = snapshotBatchDiagnostic(loop, "loop_form",
                                      "UNPROVEN_LOOP_COALESCING");
      return WalkResult::interrupt();
    }
    programLoop = loop;
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return BatchResult<scf::ForOp>::decline(std::move(first));
  // A null handle is a valid straight-line program.
  return BatchResult<scf::ForOp>::success(programLoop);
}

bool batchLoopCarriesValues(scf::ForOp loop) {
  // 数 iter_args，不去辨认"这是不是一次累加"。零就意味着可证没有跨行携带，
  // 那正是关卡放行所需要的全部；而把真累加和单纯透传分开要顺着 yield 链走，
  // 今天多认出来的那点形态一个算子也放不进来。
  return !loop.getInitArgs().empty();
}

} // namespace mlir::triton::cfg
