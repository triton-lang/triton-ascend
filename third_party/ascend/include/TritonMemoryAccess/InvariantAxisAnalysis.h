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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_INVARIANT_AXIS_ANALYSIS_H
#define TRITON_ASCEND_MEMORY_ACCESS_INVARIANT_AXIS_ANALYSIS_H

#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::ptr {

/// Prove which tensor dimensions do not contribute to an integer or pointer
/// value. Straight-line expressions delegate to CommonPtrRules; scf.for
/// iter_args and scf.if results extend those same facts through control flow.
/// A true mask bit means the axis is invariant. Scalars have an empty mask.
///
/// The analysis may append pure helper operations through CommonPtrRules. The
/// caller owns those operations and must keep the analyzed IR stable until the
/// analysis session is destroyed.
class InvariantAxisAnalysis {
public:
  using AxisMask = SmallVector<bool>;

  explicit InvariantAxisAnalysis(OpBuilder &builder) : builder(builder) {}

  FailureOr<AxisMask> analyze(Value value);

private:
  FailureOr<PtrInfo> analyzePtr(Value value);
  FailureOr<PtrInfo> analyzeForResult(scf::ForOp forOp, unsigned resultIndex);
  FailureOr<PtrInfo> analyzeForIterArg(scf::ForOp forOp, unsigned index);
  FailureOr<PtrInfo> analyzeIfResult(scf::IfOp ifOp, unsigned resultIndex);
  FailureOr<PtrInfo> joinAxes(Value result, const PtrInfo &lhs,
                              const PtrInfo &rhs);
  static AxisMask getAxisMask(const PtrInfo &info, unsigned rank);

  OpBuilder &builder;
  CommonPtrRules rules;
  DenseMap<Value, PtrInfo> cache;
  DenseMap<Value, PtrInfo> bindings;
  SmallPtrSet<Value, 16> active;
};

} // namespace mlir::triton::ptr

#endif // TRITON_ASCEND_MEMORY_ACCESS_INVARIANT_AXIS_ANALYSIS_H
