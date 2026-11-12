// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_EXPR_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_EXPR_H
#include <cstdint>
#include <memory>
namespace mlir::triton::cfg {
// Owned, immutable coordinate expressions. Coordinate IDs denote projections
// in this access, not make_range SSA identity. No erased SSA value is retained.
//
// Deliberately free of any serialization: the JSON form is the compile-time to
// run-time contract (see BatchFactsJSON.h), not a property of the data type.
// Keeping it out means the judgment layer can depend on BatchExpr without
// pulling in llvm/Support/JSON.h.
struct BatchExpr;
using BatchExprRef = std::shared_ptr<const BatchExpr>;
struct BatchExpr {
  enum Kind {
    Constant,
    Argument,
    Program,
    ProgramQuotient,
    ProgramRemainder,
    Grid,
    Coordinate,
    Induction,
    Add,
    Sub,
    Mul,
    ExtSI,
    ExtUI,
    Trunc,
    True,
    False,
    Unknown,
    And,
    SLT,
    ULT,
    EQ,
    MinSI,
    MaxSI
  };
  Kind kind;
  unsigned bits = 0;
  int64_t value = 0, extent = 0;
  BatchExprRef lhs, rhs;
  bool rangeOrigin = false;
  static BatchExprRef range(BatchExprRef start, BatchExprRef projection);
  static BatchExprRef leaf(Kind kind, int64_t value = 0, unsigned bits = 0,
                           int64_t extent = 0);
  static BatchExprRef node(Kind kind, unsigned bits, BatchExprRef lhs,
                           BatchExprRef rhs = {}, unsigned sourceBits = 0);
};
} // namespace mlir::triton::cfg
#endif
