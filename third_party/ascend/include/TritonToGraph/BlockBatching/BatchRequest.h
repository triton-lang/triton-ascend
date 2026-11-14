// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_BATCH_REQUEST_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_BATCH_REQUEST_H

#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "TritonToGraph/BlockBatching/BatchResult.h"
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <map>
#include <string>

namespace mlir::triton::cfg {

enum class BlockMappingOwner { None, BlockBatching, Legacy };

struct BatchRequest {
  int64_t factor = 1;
  int32_t axis = -1;
  bool semanticOnly = false;
  // CV supports independent lanes or a proved shared-operand spatial tile.
  // The dispatcher proves a full group before launching either candidate.
  bool cv = false;
  // Opt-in independent rank-three lowering. Preserve rectangular masks;
  // the guard proves complete reduction axes and unchanged padded footprints.
  bool cvBatchMatmul = false;
  bool isValid() const { return factor > 0 && axis >= -1 && axis <= 2; }
};

// No MLIR Value survives this read-only analysis. It may be serialized into
// compiler metadata, but must never authorize a transformed launch.
struct BatchAnalysisReport {
  std::string status = "REJECTED";
  std::string reason;
  std::string location;
  std::string operation;
  std::string context;
  std::string layoutJSON;
  std::string computationJSON;
  std::map<std::string, unsigned> operations;
};

struct BatchRewriteReport : BatchAnalysisReport {
  std::string guardJSON;
  std::string cvMode = "disabled";
  std::string cvLowering = "disabled";
  std::string vvLowering = "disabled";
  unsigned structuredMasks = 0;
  int selectedAxis = -1;
  int64_t laneWidth = 1;
};
// Describe original aggregation domains before the rewrite erases them.
std::string describeBatchComputation(Operation *root);
// Private sandbox provenance, installed only after source validation and
// removed before publishing IR. The finalizer observes actual generated
// axes/shapes.
void markVvAggregations(Operation *root);
LogicalResult finalizeVvComputation(Operation *root, std::string &description,
                                    bool sequential, int64_t factor);
BatchDiagnostic snapshotBatchDiagnostic(Operation *op, llvm::StringRef stage,
                                        llvm::StringRef reason);
void applyBatchDiagnostic(BatchAnalysisReport &report,
                          const BatchDiagnostic &detail);
void setBatchDiagnostic(BatchAnalysisReport &report, Operation *op,
                        llvm::StringRef reason);

// 成功时把改写结果提交回 module。
FailureOr<BatchRewriteReport> rewriteBlockBatching(ModuleOp module,
                                                   BatchRequest request);

// 同样的改写，但**绝不提交**：module 一个字节都不变。干跑用这条。
// 在它之前，干跑只能靠在外面再 clone 一层来躲开上面那次提交，于是一次干跑要把
// 整个模块 clone 两遍。
FailureOr<BatchRewriteReport> probeBlockBatching(ModuleOp module,
                                                 BatchRequest request);

FailureOr<BatchAnalysisReport> analyzeBlockBatching(ModuleOp module,
                                                    BatchRequest request);
} // namespace mlir::triton::cfg
#endif
