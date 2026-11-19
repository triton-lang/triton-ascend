// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_BLOCK_AUTO_IR_ANALYSIS_H
#define TRITON_ASCEND_BLOCK_AUTO_IR_ANALYSIS_H
#include "Selection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir::triton::ascend::blockauto {
// Allocation and lowering legality belong to the compiler passes.
class CandidateIRAnalysis {
public:
  CandidateIRAnalysis(Operation *op, AnalysisManager &manager);
  int64_t spec(StringRef key) const;
  void checkResources(const O &item) const;
  A launchArguments(const O &contract, const O &metadata) const;
  std::string metaLibrary() const;

private:
  ModuleOp module;
  const DataLayout &layout;
};
void registerCandidateDialects(DialectRegistry &registry);
OwningOpRef<ModuleOp> parseCandidateIR(StringRef source, MLIRContext &context);
} // namespace mlir::triton::ascend::blockauto
#endif
