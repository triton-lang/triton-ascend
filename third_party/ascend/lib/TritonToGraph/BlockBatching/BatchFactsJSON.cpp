// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"

namespace mlir::triton::cfg {
namespace {
llvm::json::Value exprJSON(const BatchExpr &self);
} // namespace

llvm::json::Value batchExprJSON(const BatchExprRef &expr) {
  return expr ? exprJSON(*expr) : llvm::json::Value(nullptr);
}
llvm::json::Value batchLoopJSON(const BatchLoopFact &loop) {
  return llvm::json::Object{{"id", loop.id},
                            {"bits", loop.bits},
                            {"lower", batchExprJSON(loop.lower)},
                            {"upper", batchExprJSON(loop.upper)},
                            {"step", batchExprJSON(loop.step)}};
}

namespace {
llvm::json::Value exprJSON(const BatchExpr &self) {
  using A = llvm::json::Array;
  switch (self.kind) {
  case BatchExpr::Constant:
    return A{"const", self.value, self.bits};
  case BatchExpr::Argument:
    return A{"arg", self.value, self.bits};
  case BatchExpr::Program:
    return A{"pid", self.value, self.bits};
  case BatchExpr::ProgramQuotient:
  case BatchExpr::ProgramRemainder:
    return A{self.kind == BatchExpr::ProgramQuotient ? "pid_div" : "pid_rem",
             self.value, self.extent, self.bits};
  case BatchExpr::Grid:
    return A{"grid", self.value, self.bits};
  case BatchExpr::Coordinate:
    return A{"coord", self.value, self.extent, self.bits};
  case BatchExpr::Induction:
    return A{"iv", self.value, self.bits};
  case BatchExpr::True:
    return A{"true"};
  case BatchExpr::False:
    return A{"false"};
  case BatchExpr::Unknown:
    return A{"unknown"};
  case BatchExpr::And:
    return A{"and", batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  case BatchExpr::SLT:
  case BatchExpr::ULT:
  case BatchExpr::EQ:
    return A{self.kind == BatchExpr::SLT   ? "slt"
             : self.kind == BatchExpr::ULT ? "ult"
                                           : "eq",
             batchExprJSON(self.lhs), batchExprJSON(self.rhs), self.bits};
  case BatchExpr::ExtSI:
  case BatchExpr::ExtUI:
  case BatchExpr::Trunc:
    return A{self.kind == BatchExpr::ExtSI   ? "extsi"
             : self.kind == BatchExpr::ExtUI ? "extui"
                                             : "trunci",
             self.bits, batchExprJSON(self.lhs), self.extent};
  case BatchExpr::MinSI:
  case BatchExpr::MaxSI:
    return A{self.kind == BatchExpr::MinSI ? "minsi" : "maxsi", self.bits,
             batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  case BatchExpr::Add:
  case BatchExpr::Sub:
  case BatchExpr::Mul:
    return A{self.kind == BatchExpr::Add   ? "addi"
             : self.kind == BatchExpr::Sub ? "subi"
                                           : "muli",
             self.bits, batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  }
  llvm_unreachable("unhandled batch coordinate expression");
}
} // namespace

} // namespace mlir::triton::cfg
