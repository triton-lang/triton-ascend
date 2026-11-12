// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchExpr.h"

// 从 BatchAccessAnalysis.cpp 原样搬来，一个字符没改。
// 只依赖 <memory>，不碰 MLIR 也不碰 JSON——分出来之后判定层的单测才能
// 不经由 BatchAccessAnalysis.cpp 把整条 Triton/GPU 依赖链拖进来。
namespace mlir::triton::cfg {
BatchExprRef BatchExpr::leaf(Kind kind, int64_t value, unsigned bits,
                             int64_t extent) {
  return std::make_shared<BatchExpr>(
      BatchExpr{kind, bits, value, extent, {}, {}});
}
BatchExprRef BatchExpr::node(Kind kind, unsigned bits, BatchExprRef lhs,
                             BatchExprRef rhs, unsigned sourceBits) {
  return std::make_shared<BatchExpr>(
      BatchExpr{kind, bits, 0, sourceBits, std::move(lhs), std::move(rhs)});
}
BatchExprRef BatchExpr::range(BatchExprRef start, BatchExprRef projection) {
  return std::make_shared<BatchExpr>(
      BatchExpr{Add, 32, 0, 0, std::move(start), std::move(projection), true});
}
} // namespace mlir::triton::cfg
