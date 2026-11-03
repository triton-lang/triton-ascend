/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
// Buffer count thresholds for flowOpt condition
constexpr int CROSS_CORE_BUFFER_COUNT_THRESHOLD = 1;
constexpr int INTRA_CORE_BUFFER_COUNT_THRESHOLD = 2;

// Relationship between a tensor iter_arg and ssbuffer.if in the main_loop
struct TensorIterArgIfOpRelation {
  Value iterArg;
  scf::IfOp producer;
  llvm::SmallVector<scf::IfOp> consumers;
};

// One consumer op may appear in multiple groups.
// Each inner SmallVector is the producers of one group this consumer belongs
// to.
using ConsumerProducerMap =
    llvm::DenseMap<Operation *,
                   llvm::SmallVector<llvm::SmallVector<Operation *>>>;

inline int countProducerGroups(const ConsumerProducerMap &map) {
  int numProducerGroups = 0;
  for (const auto &entry : map)
    numProducerGroups += static_cast<int>(entry.second.size());
  return numProducerGroups;
}

// Variables to control when an ifOp is both producer and consumer of a tensor
// iter_args
struct TensorIterArgIfOpVars {
  // The variables that need to be controlled as a producer
  llvm::SmallVector<Value> producerVars;
  // The variables that need to be controlled as a consumer
  llvm::SmallVector<Value> consumerVars;
};

// Kind of dependency edge between two ifOps in ifBlockDAG.
enum class IfBlockDepKind : uint8_t { CrossCore = 0, IntraCore = 1 };

// Slot of the SSBuffer counter owned by one ssbuffer.if computation block.
// A vector block keeps one copy per vector core (byteOffset for core0 and
// byteOffset + 1024 for core1); a cube block keeps a single copy.
struct BlockCounterSlot {
  int slot = -1;         // slot index, 0 based
  int byteOffset = 0;    // byte offset of the counter in the ssbuf area
  bool isVector = false; // true when the block runs on the vector side
  int blockId = -1;      // block id of the owning ssbuffer.if
};

// Cross-core dependency edge between two ssbuffer.if computation blocks.
struct CrossCoreBlockEdge {
  scf::IfOp producer;
  scf::IfOp consumer;
  int bufferNum = 1; // buffer depth of this edge
};

// Per scf.while block-arg map: whileOp -> block_id -> (new_arg_idx ->
// old_arg_idx).
using WhileBlockArgMap =
    llvm::DenseMap<scf::WhileOp, llvm::DenseMap<int, llvm::DenseMap<int, int>>>;

struct ControlFlowConditionInfo {
  // Keys: main-loop op (scf.for/scf.while carrying ssbuffer.main_loop)
  llvm::DenseMap<Operation *, SmallVector<int>> blockCounters;
  llvm::DenseMap<Operation *, int> blockCounterNums;
  llvm::DenseMap<Operation *, SmallVector<int>> innerDepConds;

  ConsumerProducerMap crossCoreDependentMap;
  llvm::DenseMap<Operation *,
                 llvm::DenseMap<Operation *, SmallVector<Operation *>>>
      intraCoreDependentMap;
  // Stores producer/consumer relationship between tensor iter_args in main_loop
  // and ssbuffer.if; vector index corresponds to iter arg index in the
  // main-loop op
  llvm::DenseMap<Operation *, llvm::SmallVector<TensorIterArgIfOpRelation>>
      tensorIterArgDepsMap;
  // Records control condition variable index for newly created iter_args of
  // tensor iter_args
  llvm::DenseMap<Operation *, llvm::DenseMap<Value, SmallVector<int>>>
      tensorIterArgIndicesMap;

  // unique counter value for each ifblock scf.for only.
  llvm::DenseMap<scf::IfOp, Value> cntArgs;

  // DAG for if block dependencies.
  llvm::DenseMap<scf::IfOp,
                 llvm::SmallVector<std::pair<scf::IfOp, IfBlockDepKind>>>
      ifBlockDAG;
  llvm::DenseMap<scf::IfOp, scf::IfOp> flowOptIfOpPairs;

  // SSBuffer counter slot owned by a computation block. Only blocks taking
  // part in a cross-core dependency get a slot.
  llvm::DenseMap<scf::IfOp, BlockCounterSlot> blockCounterSlots;
  // Cross-core edges at computation-block granularity. Keyed by the if op;
  // the value holds every edge the block takes part in, in both directions.
  llvm::DenseMap<scf::IfOp, llvm::SmallVector<CrossCoreBlockEdge>>
      crossCoreBlockEdges;

  // Buffer counts for flowOpt condition
  int intraCoreBufferCount = 0;
  int crossCoreBufferCount = 0;

  // Per scf.while (with main_loop attr): records per-block new iter_args
  // mirroring iter_args used in scf.condition. Keys: whileOp -> block_id -> new
  // iter_arg index. Value: original iter_arg index.
  WhileBlockArgMap whileBlockArgMap;
};

class AddControlFlowConditionPass
    : public PassWrapper<AddControlFlowConditionPass, OperationPass<ModuleOp>> {
public:
  AddControlFlowConditionPass() = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "add-control-flow-condition";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowConditionPass();

void registerAddControlFlowConditionPasses();
} // namespace triton
} // namespace mlir
#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_CONTROLFLOW_CONDITION_PASS_H
