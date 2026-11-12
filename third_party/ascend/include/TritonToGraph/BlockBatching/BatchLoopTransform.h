// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_TRANSFORM_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_TRANSFORM_H
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::triton::cfg {

// Sequential composition threads every yielded value into the next copy;
// neither original program ownership nor reduction order changes.
LogicalResult groupBatchLoopIterations(scf::ForOp loop, int64_t factor);

// 第 3 层：变换。这里只重算程序轴循环的三个界；循环体、iter_args、零迭代分支
// 都由既有的重建逻辑照常处理（BatchRewrite.cpp:457-493），不新建一条路径。
//
// 不认 triton::GetProgramIdOp，改为由调用方把"原来的程序号"和"合并后的标量
// group*F"一并传进来。驱动层本来就已经找到了那个
// op（BatchRewrite.cpp:298-303）， 这样本文件只依赖 arith 与 scf，不把 Triton
// dialect 拖进变换层。
//
// 没有 factor 入参：成块发的界只换程序号、步长不动，F 全部体现在调用方算好的
// foldedProgramId 里；块宽放大 F 倍是布局层按 axis.chunk * F 做的，不在这里。
struct BatchRetiledBounds {
  Value lower, upper, step;
  bool ok() const { return lower && upper && step; }
};

// 把界里的一个原值换成新函数体里的对应物。
//
// 存在的理由：界表达式读的是**原** IR，而原函数体在改写收尾时会被整体 erase
// （BatchRewrite.cpp）。与程序号无关的那些子链（`pid*R` 里的 R）如果原样留在新
// 界上，erase 之后就是悬空引用。调用方手里有原值到新值的映射，这里问它要。
//
// 返回空表示这个值在新体里没有对应物，整条界按失败处理。
using BatchValueRemap = llvm::function_ref<Value(Value)>;

// 失败时返回全空，调用方应当放弃这个候选而不是继续改写。
BatchRetiledBounds retileBatchLoopBounds(OpBuilder &builder, scf::ForOp loop,
                                         const BatchLogicalAxis &axis,
                                         Value programId, Value foldedProgramId,
                                         BatchValueRemap remap);

} // namespace mlir::triton::cfg
#endif
