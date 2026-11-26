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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_H
#define TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_H

#include "TritonMemoryAccess/PtrAnalysisInfo.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::arith {
class FPToSIOp;
}
namespace mlir::scf {
class ForOp;
class WhileOp;
class IfOp;
} // namespace mlir::scf
namespace mlir::tensor {
class ExtractSliceOp;
class InsertOp;
class InsertSliceOp;
class CastOp;
} // namespace mlir::tensor
namespace mlir::triton {
class SplatOp;
class BroadcastOp;
class ExpandDimsOp;
class ReshapeOp;
class MakeRangeOp;
class LoadOp;
class ReduceOp;
class AddPtrOp;
class BitcastOp;
} // namespace mlir::triton

namespace mlir::triton::ptr {

/// Check actual dependencies at the caller's current insertion point.
LogicalResult checkInsertionPoint(OpBuilder &builder,
                                  ArrayRef<Value> dependencies);

/// Integer and pointer facts for one stable source-IR session. Queries may add
/// pure numeric helpers next to their dependencies. End the session before
/// rewriting source IR or running CSE/canonicalization.
class PtrAnalysis {
public:
  explicit PtrAnalysis(OpBuilder &builder) : builder(builder) {}
  PtrAnalysis(const PtrAnalysis &) = delete;
  PtrAnalysis &operator=(const PtrAnalysis &) = delete;
  OpBuilder &getBuilder() const { return builder; }

  FailureOr<PtrInfo> analyze(Value value);

  /// Borrow until the next query, setInfo/reset, or session destruction.
  /// Copy before recursive queries: growing a DenseMap moves its entries.
  const PtrInfo *findCached(Value value) const;

  /// Import equivalent facts for the same live SSA between top-level queries.
  /// Other valid cached facts remain available, without precision refresh.
  /// Referenced SSA fields must dominate all uses of the analyzed value.
  LogicalResult setInfo(Value value, const PtrInfo &info);

  /// Validate and copy all inputs before replacing the cache. A repeated key
  /// takes its last entry. Invalid input leaves the session unchanged.
  /// Use reset when new facts should refine dependent results; reset() clears
  /// everything. This does not erase helper IR or repair rewritten source IR.
  LogicalResult reset(ArrayRef<std::pair<Value, PtrInfo>> known = {});

  /// Compare components in a fixed descriptor layout. For a loop, reference
  /// is its current argument and initial is the corresponding initial value;
  /// forwarding either proves preservation. Without initial, compare against
  /// reference itself (e.g. two branches). Each call has its own bool memo.
  /// The initial value supplies the comparison layout, never unproved current
  /// arg values. Other loop arguments reuse fields only after checking their
  /// initial values and all backedges for invariance.
  /// Numeric uniform/residual reconstruction and SCF rewriting are separate
  /// from this preservation query. Unsupported relations conservatively change.
  FailureOr<ChangeInfo> analyzeChanges(Value value, Value reference,
                                       Value initial = {});

private:
  using CacheMap = llvm::DenseMap<Value, PtrInfo>;
  Operation *getCacheOwner(Value value) const;
  CacheMap &getCacheMap(Operation *owner);
  void popLoopCache(Operation *loop);
  ChangeInfo computeChanges(Value value, Value reference, Value initial,
                            const PtrInfo &layout,
                            llvm::DenseMap<Value, ChangeInfo> &cache);
  void computeLoopChanges(Operation *loop, Value reference, Value initial,
                          const PtrInfo &layout,
                          llvm::DenseMap<Value, ChangeInfo> &cache);

  FailureOr<PtrInfo> parseOperation(Value value);
  static bool isControlFlowValue(Value value);
  FailureOr<PtrInfo> parseControlFlow(Value value);
  FailureOr<PtrInfo> parseFor(Value value, scf::ForOp loop);
  FailureOr<PtrInfo> parseWhile(Value value, scf::WhileOp loop);
  FailureOr<PtrInfo> parseIf(OpResult value, scf::IfOp branch);

  FailureOr<PtrInfo> parseSplat(triton::SplatOp op);
  FailureOr<PtrInfo> parseBroadcast(triton::BroadcastOp op);
  FailureOr<PtrInfo> parseExpandDims(triton::ExpandDimsOp op);
  FailureOr<PtrInfo> parseReshape(triton::ReshapeOp op);
  FailureOr<PtrInfo> parseExtractSlice(tensor::ExtractSliceOp op);
  FailureOr<PtrInfo> parseInsert(tensor::InsertOp op);
  FailureOr<PtrInfo> parseInsertSlice(tensor::InsertSliceOp op);
  FailureOr<PtrInfo> parseTensorCast(tensor::CastOp op);
  FailureOr<PtrInfo> parseSelect(arith::SelectOp op);

  PtrInfo parseMakeRange(triton::MakeRangeOp op);
  FailureOr<PtrInfo> parseBinaryArithmetic(Value value);
  FailureOr<PtrInfo> parseCast(Value value, Value input, bool unsignedCast);
  FailureOr<PtrInfo> parseFloatToInt(arith::FPToSIOp op);
  FailureOr<PtrInfo> parseLoad(triton::LoadOp op);
  FailureOr<PtrInfo> parseReduce(Value value, triton::ReduceOp op);

  FailureOr<PtrInfo> parseAddPtr(triton::AddPtrOp op);
  FailureOr<PtrInfo> parseBitcast(triton::BitcastOp op);
  FailureOr<PtrInfo> parseMakeTensorPtr(triton::MakeTensorPtrOp op);
  FailureOr<PtrInfo> parseAdvance(triton::AdvanceOp op);

  OpBuilder &builder;
  CacheMap rootCache;
  // Entering through an inner body can trigger outer-loop analysis, which
  // solves the same inner loop again under the outer hypotheses. Only the
  // newest map for a loop belongs to that invocation; older maps are suspended.
  llvm::DenseMap<Operation *, SmallVector<CacheMap, 1>> loopCaches;
};

} // namespace mlir::triton::ptr
#endif
