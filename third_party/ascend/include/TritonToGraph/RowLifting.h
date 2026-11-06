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

bool dependsOnValue(Value v, Value pid);

bool isPidInvariant(Value value, Value pid);

std::optional<int64_t> getConstantIntValue(Value v);

int64_t getElementStorageBytes(Type element);

struct IntegerBounds {
  int64_t lo;
  int64_t hi;
};

std::optional<IntegerBounds> inferMonotonicBounds(Value value, Value pid,
                                                  int64_t grid);

bool canUseLastRowLoopBound(scf::ForOp loop, Value pid, int64_t grid);

constexpr int64_t kMaxLiftedValueBytes = 128 * 1024;

struct RowCountPolicy {
  int64_t targetDotRows = 0;
  int64_t maxLiftedValueBytes = kMaxLiftedValueBytes;
  uint64_t ubBudgetBytes = 0;
  int64_t maxL0ABytes = 0;
  int64_t maxL0CBytes = 0;
  uint64_t l0ABudgetBytes = 0;
  uint64_t l0CBudgetBytes = 0;
  uint64_t coreNum = 0;
};

int64_t selectRowsPerProgram(int64_t grid, int64_t maxDotRows,
                             int64_t maxTensorBytes, int64_t peakLiveBytes,
                             const RowCountPolicy &policy = RowCountPolicy{});

struct RowLiftCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  int64_t rowsPerProgram = 1;
  bool runtimeGrid = false;
  int64_t grid = 0;
};

class RowLiftBuilder {
public:
  RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate);

  bool run(triton::FuncOp fn);

private:

  bool rewrite(Operation *op);
  Value combineMemoryMask(Value pointer, Value originalMask, Value extraMask,
                          bool &ok);
  Value mergeMasks(Value pointer, Value base, ArrayRef<Value> parts, bool &ok);
  Value createLoopActiveMask(scf::ForOp old, scf::ForOp loop);

  bool rewriteLoop(scf::ForOp old);
  RankedTensorType getLiftedType(Type old);
  Value getMappedValue(Value old);
  bool hasRowDimension(Value old);
  Value ensureRowDimension(Value old);
  Value reshape(Value v, ArrayRef<int64_t> shape);
  Value flattenDotRows(Value v);
  Value broadcastRowValue(Value v, RankedTensorType target);

  IRRewriter &rewriter;
  RowLiftCandidate candidate;

  IRMapping valueMapping;
  IRMapping lastRowMapping;
  Value validRows;
  Value rowActiveMask;
};

}
}
}

#endif
