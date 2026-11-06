/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#ifndef TRITON_TO_GRAPH_ROW_LIFTING_H
#define TRITON_TO_GRAPH_ROW_LIFTING_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace triton {
namespace cfg {

// Whether the dependency chain of v uses pid directly; no nested regions.
bool dependsOnValue(Value v, Value pid);

// Whether value is independent of pid (the row id) along the full chain.
bool isPidInvariant(Value value, Value pid);

// Compile-time constant integer value; nullopt when not a constant.
std::optional<int64_t> getConstantIntValue(Value v);

// Storage bytes of one element.
int64_t getElementStorageBytes(Type element);

// Whether the loop bound can use the last row's value; the bound has to be
// monotone in pid and its addition must not wrap.
bool canUseLastRowLoopBound(scf::ForOp loop, Value pid, int64_t grid);

// Feature switches for row lifting; a disabled feature keeps the IR as is.
struct RowLiftFeatures {
  bool reduce = false;               // tt.reduce takes part
  bool loopInvariantIf = false;      // row-invariant scf.if takes part
  bool tensorPointer = false;        // expand make_tensor_ptr to pointers
  bool maskedStore = false;          // masked write-back
  bool maskedLoad = false;           // masked read
  bool splitWideAccumulator = false; // halve an over-wide accumulator
  bool flatCompute = false;          // flatten compute to [R*M, ..]
  bool groupedLoopBounds = false;    // share one bound across rows
};

// Largest byte count one lifted value may use; doubled in split mode.
constexpr int64_t kMaxLiftedValueBytes = 128 * 1024;
// Vector register width, used as the whole-row broadcast granularity.
constexpr int64_t kVectorRegisterBytes = 256;
// Conservative UB replication factor covering the auto multi-buffer copy.
constexpr uint64_t kUBBufferReplication = 2;

// Policy for picking the rows per program (R).
struct RowCountPolicy {
  int64_t targetDotRows = 0;                          // target dot rows
  int64_t maxLiftedValueBytes = kMaxLiftedValueBytes; // per-value budget
  uint64_t ubBudgetBytes = 0;                         // 0 = unconstrained
  bool splitWideAccumulator = false;                  // allow splitting
  int64_t maxL0ABytes = 0;                            // L0A operand limit
  int64_t maxL0CBytes = 0;                            // L0C limit
  uint64_t l0ABudgetBytes = 0;                        // total L0A budget
  uint64_t l0CBudgetBytes = 0;                        // total L0C budget
  uint64_t coreNum = 0;                               // run-time cores
};

// Pick R under the given budgets; 1 means no row lifting.
int64_t selectRowsPerProgram(int64_t grid, int64_t maxDotRows,
                             int64_t maxTensorBytes, int64_t peakLiveBytes,
                             const RowCountPolicy &policy = RowCountPolicy{});

// Input description of one row-lift candidate.
struct RowLiftCandidate {
  triton::FuncOp function;     // function to rewrite
  Operation *anchor = nullptr; // anchor that fixes the loop bound
  int64_t rowsPerProgram = 1;  // lifted rows R
  bool runtimeGrid = false;    // grid extent known only at run time
  int64_t grid = 0;            // static grid size
  RowLiftFeatures features;    // enabled features
  int64_t splitValueBudgetBytes = kMaxLiftedValueBytes; // split budget
};

// Reject when enabled is false; when true the expandability check still runs.
bool canExpandTensorPointer(triton::MakeTensorPtrOp makePtr, bool enabled);

// Whether the condition and results of an scf.if are row-invariant.
bool liftsRowInvariantIf(scf::IfOp ifOp, Value pid);

// Whether a wide accumulator can be halved; budgetBytes is the half budget.
bool canSplitWideAccumulators(scf::ForOp loop, int64_t rowsPerProgram,
                              bool enabled,
                              int64_t budgetBytes = kMaxLiftedValueBytes);

class RowLiftBuilder {
public:
  // rewriter is the IR rewriter; candidate carries the target and budgets.
  RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate);

  // Lift the rows of fn; false means give up and keep the IR unchanged.
  bool run(triton::FuncOp fn);

private:
  // Expanded block pointer: pointer, per-dim shape, offsets and block shape.
  struct LiftedDescriptor {
    Value pointer;
    SmallVector<Value> shapes;
    SmallVector<Value> offsets;
    SmallVector<int64_t> tileShape;
  };

  // Rewrite one op; returns whether it was rewritten.
  bool rewrite(Operation *op);

  bool rewriteRowLoad(triton::LoadOp load);

  // Masked write-back: write only the valid rows to stay in bounds.
  bool rewriteMaskedStore(triton::StoreOp store);

  bool rewriteReduce(triton::ReduceOp reduce);

  bool rewriteTensorPointer(triton::MakeTensorPtrOp makePtr);

  // Expand makePtr into a pointer tensor; lastExtent is the last row range,
  // lastOffsetOverride replaces the last row's offset when non-null.
  bool expandTensorPointer(triton::MakeTensorPtrOp makePtr, int64_t lastExtent,
                           Value lastOffsetOverride,
                           LiftedDescriptor &descriptor);

  bool rewriteDescriptorLoad(triton::LoadOp load,
                             const LiftedDescriptor &descriptor);

  bool emitDescriptorLoad(triton::LoadOp load,
                          const LiftedDescriptor &descriptor, Value &result);

  bool rewriteDescriptorStore(triton::StoreOp store,
                              const LiftedDescriptor &descriptor);

  Value broadcastFullValue(Value v, RankedTensorType target);

  // Build the [0, extent) row indices along axis and broadcast to target.
  Value expandTileRange(Location loc, int64_t axis, int64_t extent,
                        RankedTensorType target);

  // Combine the descriptor with the per-dim bounds test (boundaryCheck).
  Value buildDescriptorMask(const LiftedDescriptor &descriptor,
                            ArrayRef<int32_t> boundaryCheck,
                            RankedTensorType target);

  bool rewriteIfOp(scf::IfOp old);

  // split marks values halved per row; flat means compute is flattened.
  bool rewriteIfBody(Region &from, Region &to, ArrayRef<bool> split, bool flat);

  // Merge the pointer mask with an extra mask; ok=false when unsafe.
  Value combineMemoryMask(Value pointer, Value originalMask, Value extraMask,
                          bool &ok);

  // Merge several masks; ok=false when they cannot be merged.
  Value mergeMasks(Value pointer, Value base, ArrayRef<Value> parts, bool &ok);

  // Build the per-iteration valid-row mask so invalid rows skip access.
  Value createLoopActiveMask(scf::ForOp old, scf::ForOp loop);

  // The two halves of a wide value split along its width.
  struct SplitValue {
    Value first;
    Value second;
  };

  bool planWideAccumulatorSplits(scf::ForOp loop);

  // Block-pointer description after a split.
  struct SplitDescriptor {
    LiftedDescriptor first;  // first half
    LiftedDescriptor second; // second half
    Value firstPointer;
    Value secondPointer;
    bool expanded = false;
  };

  // Take the row vector of old and halve it; false when it cannot.
  bool ensureRowHalves(Value old, SplitValue &halves);

  // Rebuild a full-width operand as a half-width tensor.
  Value rebuildHalfOperand(Value full, RankedTensorType halfType);

  bool rewriteSplitTensorPointer(triton::MakeTensorPtrOp makePtr,
                                 SplitDescriptor &descriptor);

  bool rewriteSplitDescriptorLoad(triton::LoadOp load,
                                  SplitDescriptor &descriptor);

  bool rewriteSplitOp(Operation *op);

  bool rewriteSplitDot(triton::DotOp dot);

  bool rewriteSplitElementwise(Operation *op);

  bool rewriteSplitStore(triton::StoreOp store);

  bool rewriteLoop(scf::ForOp old);

  // Lifted type: row-dependent values gain an [R] axis; flat merges [R,M,..]
  // into [R*M,..].
  RankedTensorType getLiftedType(Type old, bool flat = false);

  Value getMappedValue(Value old);

  bool hasRowDimension(Value old);

  Value getRowValue(Value old, bool flat = false);

  Value reshape(Value v, ArrayRef<int64_t> shape);

  Value flattenDotRows(Value v);

  Value broadcastRowValue(Value v, RankedTensorType target);

  void planFlatCompute(scf::ForOp loop);

  Type flatType(Type type) const;

  Value view(Value value, Type type);

  // Flattened compute loop
  scf::ForOp computeLoop;
  // Rewritten loop, used for write-back and teardown
  scf::ForOp rewrittenLoop;
  int64_t dotRows = 0;
  // Ops that take part in the flattened compute
  DenseSet<Operation *> flatOperations;
  DenseMap<std::pair<Value, Type>, Value> boundaryViews;

  IRRewriter &rewriter;
  RowLiftCandidate candidate;
  RowLiftFeatures features;
  // Old value -> lifted value mapping
  IRMapping valueMapping;
  DenseMap<Value, LiftedDescriptor> liftedDescriptors;
  DenseMap<Value, SplitValue> splitValues;
  // Last-row-only mapping used for the loop-bound evidence
  IRMapping lastRowMapping;
  Value validRows;
  Value rowActiveMask;
  SmallVector<int64_t> plannedSplitIndices;
  DenseMap<Value, int64_t> plannedHalfWidths;
  DenseMap<Value, SplitDescriptor> splitDescriptors;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif
