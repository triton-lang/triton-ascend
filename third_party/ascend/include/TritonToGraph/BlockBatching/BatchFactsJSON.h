// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_FACTS_JSON_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_FACTS_JSON_H
#include "TritonToGraph/BlockBatching/BatchFacts.h"
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "llvm/Support/JSON.h"

// Serialization is separate from the fact data structures.
//
// This JSON is the compile-time/runtime contract. Eligibility can depend on
// actual pointers and shapes that are unavailable during compilation, so the
// compiler passes those conditions to the runtime. backend/compiler.py decodes
// the JSON; RuntimeGuard in backend/block_batching_guard.py uses launch
// arguments to evaluate access intervals. Overlap involving a write reports
// WRITE_ALIAS and selects the unbatched baseline. Preserve this wire format.
//
// Only reporting code includes this header. The fact and legality layers use
// the BatchExpr and BatchLoopFact data types without depending on JSON.
namespace mlir::triton::cfg {

llvm::json::Value batchEffectsJSON(const BatchFacts &facts);
// Diagnostic-only pending proof; successful contracts require a finalized
// rewrite.
std::optional<std::string> encodeBatchPendingGuard(
    const BatchFacts &facts, llvm::ArrayRef<BatchEffectObligation> effects,
    llvm::ArrayRef<unsigned> terminalReads, int axis, int64_t factor);

llvm::json::Value batchExprJSON(const BatchExprRef &expr);

// These five keys and their shapes predate the in-memory type refactoring;
// changing the data types must not change the wire format.
// BatchLoopFact::carriesValues is intentionally omitted: it is a compile-time
// legality fact. Runtime guards consume the loop's lower/upper/step bounds.
llvm::json::Value batchLoopJSON(const BatchLoopFact &loop);

} // namespace mlir::triton::cfg
#endif
