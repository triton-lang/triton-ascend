// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchExpr.h"

// Expression support extracted from BatchAccessAnalysis.cpp without semantic
// changes. Depending only on <memory>, not MLIR or JSON, lets legality tests
// avoid the full Triton/GPU dependency chain.
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
