// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_FACTS_JSON_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_FACTS_JSON_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "llvm/Support/JSON.h"

// 事实层的序列化，单独成一层。
//
// 这串 JSON 是**编译期到运行期的契约**，不是调试输出：block 合并能不能用，
// 取决于调用时的实际指针与 shape，编译期证不出来，所以把条件原样交给运行期。
// backend/compiler.py 把它 json.loads 出来，backend/block_batching_guard.py
// 的 RuntimeGuard 每次 launch 都代入实参算一遍访存区间，任何一对涉及写的区间
// 重叠就报 WRITE_ALIAS，退回未合并的 baseline。格式因此是对外契约，不可随意改。
//
// 只有报告路径 include 这个头。判定层与事实层只认 BatchExpr / BatchLoopFact
// 这两个纯数据类型，不认它们的 JSON 形态。
namespace mlir::triton::cfg {

llvm::json::Value batchExprJSON(const BatchExprRef &expr);

// 这五个键与它们的形状是 S2b 换类型之前就定下的，换类型不换格式。
// BatchLoopFact::carriesValues 刻意不在其中：它只给编译期的判定层用，运行期
// 守卫对每条循环只读 lower/upper/step（block_batching_guard.py:248-250）。
llvm::json::Value batchLoopJSON(const BatchLoopFact &loop);

} // namespace mlir::triton::cfg
#endif
