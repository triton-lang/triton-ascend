// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_ANALYSIS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_ANALYSIS_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/JSON.h"
#include <string>
namespace mlir::triton::cfg {
struct BatchAccessFact {
  // Valid only until the original body is erased. Consumers must copy plans
  // and map this operation while both original and generated bodies are alive.
  Operation *operation = nullptr;
  int root = -1;
  unsigned bytes = 0;
  bool write = false;
  SmallVector<int64_t> shape;
  BatchExprRef byteOffset, mask;
  SmallVector<BatchExprRef> pointerChecks;
  // 包着这条访存的 scf.for，由内向外。S2b 之前这里直接存 JSON，判定层要用就
  // 得先把字符串解回来；现在存事实本身，序列化留给报告路径（BatchFactsJSON.h）。
  // 零内联：绝大多数访存根本不在循环里，有循环时再上堆。默认内联会让
  // BatchAccessFact 直接撑过 SmallVector 的 256 字节门槛。
  SmallVector<BatchLoopFact, 0> loops;
  bool isVolatile = false;
  // Missing summaries retain a real device SSA effect, never an empty domain.
  bool deviceAddress = false;
  bool normalizedPointer = false;
};
struct BatchAccessReport {
  std::string guardJSON;
  std::string reason;
  Operation *operation = nullptr;
  SmallVector<BatchAccessFact> accesses;
  // 函数里的循环，每条一次，按 BatchLoopFact::id 升序。
  //
  // 和 BatchAccessFact::loops 装的是同一批事实，区别只在视角：那边按访存挂，
  // 同一条循环在每条访存下各出现一次；判定层问的是「这条程序轴循环安不安全」，
  // 要的是循环本身。
  //
  // 只收**包着某条访存**的循环。一条循环里一条访存都没有时这里没有它的事实，
  // 驱动层按证不出处理——那种形状合并了也没有收益，保守拒掉即可。这样做是为了
  // 不新增失败模式：界的表达式要过 Analysis::integer，而它证不出时会把整条
  // kernel 判拒，对一条本来就够不着的循环付这个代价不值得。
  SmallVector<BatchLoopFact, 0> loops;
};
unsigned getBatchIndexBitWidth(ModuleOp module);
BatchAccessReport analyzeBatchAccesses(triton::FuncOp function,
                                       unsigned indexBits,
                                       bool pointerRecurrences = false);
} // namespace mlir::triton::cfg
#endif
