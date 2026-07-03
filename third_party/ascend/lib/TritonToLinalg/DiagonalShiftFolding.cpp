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

#include "TritonToLinalg/DiagonalShiftFolding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallVector.h"

namespace DiagonalShiftFolding {

using namespace mlir;
using namespace triton;

namespace {

static bool getConstInt(Value v, int64_t &out) {
  APInt ap;
  if (matchPattern(v, m_ConstantInt(&ap))) {
    out = ap.getSExtValue();
    return true;
  }
  DenseElementsAttr dea;
  if (matchPattern(v, m_Constant(&dea)) && dea.isSplat() &&
      isa<IntegerType>(dea.getElementType())) {
    out = dea.getSplatValue<APInt>().getSExtValue();
    return true;
  }
  return false;
}

// Check whether a scan/reduce combine region is a single addf + return.
static bool isAddfCombine(Region &region) {
  if (region.empty())
    return false;
  Block &body = region.front();
  if (body.getNumArguments() != 2)
    return false;
  auto ops = body.without_terminator();
  if (std::distance(ops.begin(), ops.end()) != 1)
    return false;
  if (!isa<arith::AddFOp>(*ops.begin()))
    return false;
  auto *term = body.getTerminator();
  if (term->getNumOperands() != 1 ||
      term->getOperand(0) != (*ops.begin()).getResult(0))
    return false;
  return true;
}

static bool isZeroFloat(Value v) {
  DenseElementsAttr dea;
  if (matchPattern(v, m_Constant(&dea)) && dea.isSplat() &&
      isa<FloatType>(dea.getElementType())) {
    return dea.getSplatValue<APFloat>().isZero();
  }
  return false;
}

// Try to match:
//   maybeRow = broadcast(expand_dims(make_range, axis=1))   [row indices]
//   maybeCol = broadcast(op(expand_dims(make_range, axis=0), K))  [col ± K]
// Returns K such that the mask represents (row == col + K), or 0 if no match.
// K = +1 → sub-diagonal (left shift), K = -1 → super-diagonal (right shift).
static int64_t tryMatchDiagonal(Value maybeRow, Value maybeCol) {
  auto bcastRow = maybeRow.getDefiningOp<triton::BroadcastOp>();
  if (!bcastRow)
    return 0;
  auto expdRow = bcastRow.getSrc().getDefiningOp<triton::ExpandDimsOp>();
  if (!expdRow || expdRow.getAxis() != 1)
    return 0;
  if (!expdRow.getSrc().getDefiningOp<triton::MakeRangeOp>())
    return 0;

  auto bcastCol = maybeCol.getDefiningOp<triton::BroadcastOp>();
  if (!bcastCol)
    return 0;

  Value colSrc = bcastCol.getSrc();
  int64_t K = 0;

  if (auto addi = colSrc.getDefiningOp<arith::AddIOp>()) {
    Value rangeV, constV;
    if (addi.getLhs().getDefiningOp<triton::ExpandDimsOp>()) {
      rangeV = addi.getLhs();
      constV = addi.getRhs();
    } else if (addi.getRhs().getDefiningOp<triton::ExpandDimsOp>()) {
      rangeV = addi.getRhs();
      constV = addi.getLhs();
    } else {
      return 0;
    }
    auto expdCol = rangeV.getDefiningOp<triton::ExpandDimsOp>();
    if (!expdCol || expdCol.getAxis() != 0)
      return 0;
    if (!expdCol.getSrc().getDefiningOp<triton::MakeRangeOp>())
      return 0;
    if (!getConstInt(constV, K))
      return 0;
  } else if (auto subi = colSrc.getDefiningOp<arith::SubIOp>()) {
    auto expdCol = subi.getLhs().getDefiningOp<triton::ExpandDimsOp>();
    if (!expdCol || expdCol.getAxis() != 0)
      return 0;
    if (!expdCol.getSrc().getDefiningOp<triton::MakeRangeOp>())
      return 0;
    int64_t v;
    if (!getConstInt(subi.getRhs(), v))
      return 0;
    K = -v;
  } else {
    return 0;
  }

  if (std::abs(K) != 1)
    return 0;
  return K;
}

}  // namespace

void rewriteDiagonalShiftFold(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());

  SmallVector<triton::ReduceOp> candidates;
  moduleOp.walk([&](triton::ReduceOp red) { candidates.push_back(red); });

  for (auto red : candidates) {
    // 1. Reduce must be axis=1, single result, addf combine (row-sum).
    if (red.getAxis() != 1 || red->getNumResults() != 1)
      continue;
    if (!isAddfCombine(red.getCombineOp()))
      continue;

    // 2. Input must be select(mask, broadcast(...), zero_constant).
    if (red.getNumOperands() != 1)
      continue;
    auto sel = red.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!sel)
      continue;
    if (!isZeroFloat(sel.getFalseValue()))
      continue;

    // 3. True value: broadcast(expand_dims(scan_result, axis=0)) → NxN.
    auto bcast = sel.getTrueValue().getDefiningOp<triton::BroadcastOp>();
    if (!bcast)
      continue;
    auto bcastTy = dyn_cast<RankedTensorType>(bcast.getType());
    if (!bcastTy || bcastTy.getRank() != 2 ||
        bcastTy.getShape()[0] != bcastTy.getShape()[1])
      continue;  // must be square NxN
    auto expd = bcast.getSrc().getDefiningOp<triton::ExpandDimsOp>();
    if (!expd || expd.getAxis() != 0)
      continue;
    Value vec = expd.getSrc();

    // 4. vec must be the single result of a cumulative-sum scan (axis=0, addf).
    auto scan = vec.getDefiningOp<triton::ScanOp>();
    if (!scan)
      continue;
    if (scan->getNumResults() != 1 || scan.getNumOperands() != 1)
      continue;
    if (scan.getAxis() != 0)
      continue;
    if (!isAddfCombine(scan.getCombineOp()))
      continue;

    // 5. Condition must be a unit-diagonal mask: cmpi eq(row, col ± 1).
    auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
      continue;

    int64_t K = tryMatchDiagonal(cmp.getLhs(), cmp.getRhs());
    if (K == 0)
      K = tryMatchDiagonal(cmp.getRhs(), cmp.getLhs());
    if (K == 0)
      continue;

    // 6. Scan direction must match shift direction:
    //    K > 0 (sub-diagonal → left shift) requires forward scan (reverse=false)
    //    K < 0 (super-diagonal → right shift) requires reverse scan (reverse=true)
    if ((K > 0) == scan.getReverse())
      continue;

    // All conditions met. Replace:
    //   reduce_result = subf(scan_result, scan_input)
    Value scanInput = scan.getOperand(0);
    rw.setInsertionPoint(red);
    Value replacement =
        rw.create<arith::SubFOp>(red.getLoc(), vec, scanInput);
    rw.replaceAllUsesWith(red->getResult(0), replacement);

    // Clean up the now-dead chain in reverse dependency order.
    rw.eraseOp(red);
    if (sel->use_empty())
      rw.eraseOp(sel);
    if (bcast->use_empty())
      rw.eraseOp(bcast);
    if (expd->use_empty())
      rw.eraseOp(expd);
  }
}

}  // namespace DiagonalShiftFolding
