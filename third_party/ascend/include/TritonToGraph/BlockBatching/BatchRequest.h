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

// Commit the rewritten result to module on success.
FailureOr<BatchRewriteReport> rewriteBlockBatching(ModuleOp module,
                                                   BatchRequest request);

// Run the same rewrite without committing; module remains unchanged.
// Dry runs use this entry point to avoid cloning the whole module a second
// time merely to discard the committed result.
FailureOr<BatchRewriteReport> probeBlockBatching(ModuleOp module,
                                                 BatchRequest request);

FailureOr<BatchAnalysisReport> analyzeBlockBatching(ModuleOp module,
                                                    BatchRequest request);
} // namespace mlir::triton::cfg
#endif
