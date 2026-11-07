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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// SelectedBlockScanRule (#2686, rule bit 65536): the rewriter.
//
// Emits the sequential scan on the caller's private sandbox clone; the live
// function is only touched by apply(), after the joint row-lift gate and the
// budget gate both accepted the sandbox. Sections: [0] scaffolding, [1]
// prologue, [2] weights, [3] online-softmax body, [4] additive body, [5] entry,
// [6] budget gate.

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/ResourceCostModel.h"
#include "TritonToGraph/RowLifting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>

using namespace mlir;
using namespace triton;
using namespace cfg;

// The joint scan + row-lift gate is documented in SelectedBlockScanRule.cpp;
// the rewriter only has to leave the sandbox in a state that gate can accept.

#include "TritonToGraph/SelectedBlockScanFacts.h"

namespace {

// ---- [0] Shared scaffolding ----
//
// Three constraints are hard: safe_m subtracts from BOTH the rescale factor and
// the scores (fwd only); the per-token weight multiplies p in f32 BEFORE the
// bf16 truncation (never the -inf mask, where -inf * 0 is NaN); BN comes from
// IR constants, never a new kernel argument.

// Widen a rank-1 row vector to a rank-2 tile: expand it at `axis` (the axis it
// is broadcast along) and broadcast the result to `type`. Both templates do
// this for every row-shaped operand, and doing it in one place keeps the
// expand/broadcast pair from being spelled eight slightly different ways.
static Value broadcastRow(IRRewriter &rewriter, Location loc, Value v,
                          Type type, int64_t axis) {
  auto source = cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> shape(source.getShape());
  shape.insert(shape.begin() + axis, 1);
  Value expanded = rewriter.create<triton::ExpandDimsOp>(
      loc, RankedTensorType::get(shape, source.getElementType()), v, axis);
  return rewriter.create<triton::BroadcastOp>(loc, type, expanded);
}

// A ranked tensor type: the one type constructor the emitters need.
static RankedTensorType tensorTy(ArrayRef<int64_t> shape, Type element) {
  return RankedTensorType::get(shape, element);
}

// A splat constant tensor. Every neutral operand of the scan and the tile
// "other" value is one of these.
static Value splatConstant(IRRewriter &rewriter, Location loc,
                           ArrayRef<int64_t> shape, Attribute value) {
  Type element = cast<TypedAttr>(value).getType();
  return rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(tensorTy(shape, element), value));
}

// The f32 matrix `v` cast to the element type its dot takes, in the direction
// the matcher recorded. Both templates replay one such cast.
static Value castMatrix(IRRewriter &rewriter, Location loc, Type elementType,
                        Value v, bool isExtend) {
  Type type =
      tensorTy(cast<RankedTensorType>(v.getType()).getShape(), elementType);
  return isExtend ? rewriter.create<arith::ExtFOp>(loc, type, v).getResult()
                  : rewriter.create<arith::TruncFOp>(loc, type, v).getResult();
}

// Build "tt.reduce" with a two-argument combine region. The result type is
// inferred from the input and the axis.

template <typename CombineOp>
static Value buildReduce(IRRewriter &rewriter, Location loc, Value input,
                         int32_t axis) {
  auto tensor = cast<RankedTensorType>(input.getType());
  auto reduce = rewriter.create<triton::ReduceOp>(loc, ValueRange{input},
                                                  static_cast<int>(axis));
  Block &combine = reduce.getCombineOp().emplaceBlock();
  combine.addArguments({tensor.getElementType(), tensor.getElementType()},
                       {loc, loc});
  rewriter.setInsertionPointToStart(&combine);
  Value combined = rewriter.create<CombineOp>(loc, combine.getArgument(0),
                                              combine.getArgument(1));
  rewriter.create<triton::ReduceReturnOp>(loc, ValueRange{combined});
  rewriter.setInsertionPointAfter(reduce.getOperation());
  return reduce->getResult(0);
}

// Build a tt.dot, carrying over the original dot's attributes (input
// precision and friends). The generic creation form avoids depending on a
// specific builder overload for defaulted attributes.
static Value buildDot(IRRewriter &rewriter, Location loc, Type resultType,
                      Value a, Value b, Value c, Operation *original) {
  SmallVector<Value> operands{a, b, c};
  Operation *dot =
      rewriter.create(loc, original->getName().getIdentifier(), operands,
                      TypeRange{resultType}, original->getAttrs());
  return dot->getResult(0);
}

// The prologue both templates share: the table is read once as a vector,
// starts = ids*BS vectorizes the old i_s, slot_valid gates the stale [NS, S)
// tail of that read, and the scan bound is cdiv(i_t + 1, BN).
// ---- [1] Prologue: the index-table vector read and the lifted rows ----

struct ScanPrologue {
  int64_t tableWidth = 0; // TW = next_pow2(S), the padded slot axis
  Value startsRow;        // [1, TW] i32
  Value slotRow;          // [1, TW] i1
  Value tokensRange;      // [BN] i32, [0, BN)
  Value pidRow;           // [BN] i32, splat(pid_x)
  Value bsConstant;       // i32 BS
  Value zeroTokens;       // [BN] f32 zero
  Value zeroRows;         // [G] f32 zero, used only by the fwd template
  Value negInfRows;       // [G] f32 -inf, used only by the fwd template
  Value scanUpperBound;   // i32, the new trip count
  Value bound64;          // i64 K/V dynamic bound (online-softmax only)
  Value c0, c1, cBN;      // i32 constants of the new loop
};

static std::optional<ScanPrologue>
buildScanPrologue(IRRewriter &rewriter, ScanCandidate &candidate, Type i32,
                  Type i64, Type f32, int64_t S, int64_t BS, int64_t BN,
                  int64_t G) {
  Location loc = candidate.loop.forOp.getLoc();
  // The slot axis is padded to TW = next_pow2(S), because tt.make_range is
  // specified for power-of-two extents while S is any positive integer. Padded
  // lanes read a clamped, in-row slot and are neutralized by slot_valid below.
  int64_t TW = 1;
  while (TW < S)
    TW <<= 1;
  Value offs =
      rewriter.create<triton::MakeRangeOp>(loc, tensorTy({TW}, i32), 0, TW);
  // The lane index only has to be clamped to the last valid slot when padding
  // lanes exist (TW > S).  With TW == S the clamp is the identity, and
  // keeping it turns the table-read column index into min(range, constant),
  // which the downstream offset analysis cannot treat as structured and
  // forces the whole table read onto indirect_load.
  // Clamp the lane to the last valid slot only when padded lanes exist
  // (TW > S).  For TW == S the clamp is an identity, and keeping it turns the
  // slot index into min(range, const) - which the offset analysis classifies as
  // unstructured and forces onto the indirect-load path.
  Value slotIdx = offs;
  if (TW > S) {
    Value lastSlot = rewriter.create<arith::ConstantIntOp>(loc, S - 1, 32);
    Value lastSlotSplat =
        rewriter.create<triton::SplatOp>(loc, tensorTy({TW}, i32), lastSlot);
    slotIdx = rewriter.create<arith::MinSIOp>(loc, offs, lastSlotSplat);
  }
  // Spelling (b) of the index address (tt.addptr %root, (addi %rowOff, %iv))
  // has no row-base addptr to reuse: build it here, next to the vectorized read
  // that consumes it and outside the loop, so its dominance is the prologue's.
  Value tableBase = candidate.table.base;
  if (!tableBase)
    tableBase = rewriter.create<triton::AddPtrOp>(
        loc, candidate.table.root.getType(), candidate.table.root,
        candidate.table.rowOffset);
  Type tablePtrTensor = tensorTy({TW}, tableBase.getType());
  Value tablePtrs =
      rewriter.create<triton::SplatOp>(loc, tablePtrTensor, tableBase);
  Value tablePtrVec = rewriter.create<triton::AddPtrOp>(loc, tablePtrTensor,
                                                        tablePtrs, slotIdx);
  Value ids = rewriter.create<triton::LoadOp>(
      loc, tablePtrVec, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, false);
  // A wider table entry is truncated to i32. The matcher accepts any width
  // >= 32, so this assumes block numbers stay below 2^31, which S <= 4096
  // (kMaxTableLength) makes safe in practice.
  Value idsI32 = ids;
  if (cast<RankedTensorType>(ids.getType()).getElementType() != i32)
    idsI32 = rewriter.create<arith::TruncIOp>(loc, tensorTy({TW}, i32), ids);

  // starts = ids * BS, the vectorized old i_s.
  Value bsConstant = rewriter.create<arith::ConstantIntOp>(loc, BS, 32);
  Value bsSplat =
      rewriter.create<triton::SplatOp>(loc, tensorTy({TW}, i32), bsConstant);
  Value starts = rewriter.create<arith::MulIOp>(loc, idsI32, bsSplat);

  // slot_valid = offs < NS, NS being the old trip count. It must survive here
  // when it is a block_counts load, and it is clamped into [0, S] in its OWN
  // width first: an NS > S would make the gate true for every slot, and an i64
  // NS beyond i32 would truncate negative and yield an empty result.
  Value nsI32;
  if (auto nsConstant = getConstantIntValue(candidate.scanBound)) {
    nsI32 = rewriter.create<arith::ConstantIntOp>(loc, *nsConstant, 32);
  } else {
    Type nsType = candidate.scanBound.getType();
    if (!nsType.isInteger(32) && !nsType.isInteger(64))
      return std::nullopt;
    Value nsZero =
        rewriter.create<arith::ConstantIntOp>(loc, nsType, int64_t{0});
    Value nsMax = rewriter.create<arith::ConstantIntOp>(loc, nsType, S);
    Value nsLow =
        rewriter.create<arith::MaxSIOp>(loc, candidate.scanBound, nsZero);
    Value nsClamped = rewriter.create<arith::MinSIOp>(loc, nsLow, nsMax);
    nsI32 = nsType.isInteger(32)
                ? nsClamped
                : rewriter.create<arith::TruncIOp>(loc, i32, nsClamped);
  }
  Value nsSplat =
      rewriter.create<triton::SplatOp>(loc, tensorTy({TW}, i32), nsI32);
  Value slotValid = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, offs, nsSplat);

  // ---- Scan bound: (row + BN) / BN. ----
  if (BN < 2 || BN > std::numeric_limits<int32_t>::max())
    return std::nullopt;
  Value rowPosition = candidate.rowPosition;
  if (rowPosition.getType() != i32) {
    unsigned width = rowPosition.getType().getIntOrFloatBitWidth();
    if (width > 32)
      rowPosition = rewriter.create<arith::TruncIOp>(loc, i32, rowPosition);
    else if (width < 32)
      rowPosition = rewriter.create<arith::ExtSIOp>(loc, i32, rowPosition);
    else
      return std::nullopt;
  }
  Value c0 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value c1 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value cBN = rewriter.create<arith::ConstantIntOp>(loc, BN, 32);

  Value scanUpperBound;
  auto pid = rowPosition.getDefiningOp<triton::GetProgramIdOp>();
  if (pid && pid.getAxisAsInt() == 0)
    scanUpperBound = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::DivSIOp>(loc, rowPosition, cBN), c1);
  else
    scanUpperBound = rewriter.create<arith::DivSIOp>(
        loc, rewriter.create<arith::AddIOp>(loc, rowPosition, cBN), cBN);

  // Re-create the K/V dynamic bound outside the loop, unfolded. Only the
  // online-softmax template's make_tensor_ptr rebuild consumes it.
  const MptrTiles *mptr = std::get_if<MptrTiles>(&candidate.tiles);
  Value bound64 = mptr ? mptr->dynamicBoundResult : Value();
  if (mptr && mptr->dynamicBoundInput)
    bound64 =
        rewriter.create<arith::ExtSIOp>(loc, i64, mptr->dynamicBoundInput);

  // Loop-invariant rows of the weight matrix (hoisted ahead of the loop).
  Value startsRow = rewriter.create<triton::ExpandDimsOp>(
      loc, tensorTy({1, TW}, i32), starts, 0);
  Value slotRow = rewriter.create<triton::ExpandDimsOp>(
      loc, tensorTy({1, TW}, rewriter.getI1Type()), slotValid, 0);
  Value tokensRange =
      rewriter.create<triton::MakeRangeOp>(loc, tensorTy({BN}, i32), 0, BN);
  Value pidRow =
      rewriter.create<triton::SplatOp>(loc, tensorTy({BN}, i32), rowPosition);

  // Fresh neutral constants; later canonicalization/CSE folds them into the
  // surviving ones. Keeping them fresh makes the builder independent of the
  // original constants' shapes when BN != BS.
  Value zeroTokens =
      splatConstant(rewriter, loc, {BN}, rewriter.getFloatAttr(f32, 0.0));
  // The two neutral row vectors belong to the online-softmax template only, so
  // the additive form does not build them at all instead of leaving them for
  // the cleanup pass to erase.
  Value zeroRows;
  Value negInfRows;
  if (std::holds_alternative<OnlineSoftmaxRecurrence>(candidate.recurrence)) {
    zeroRows =
        splatConstant(rewriter, loc, {G}, rewriter.getFloatAttr(f32, 0.0));
    negInfRows =
        splatConstant(rewriter, loc, {G},
                      rewriter.getFloatAttr(
                          f32, APFloat::getInf(APFloat::IEEEsingle(), true)));
  }

  return ScanPrologue{
      TW,         startsRow,  slotRow,  tokensRange, pidRow,
      bsConstant, zeroTokens, zeroRows, negInfRows,  scanUpperBound,
      bound64,    c0,         c1,       cBN};
}

// ---- [2] The per-token weight both templates share ----
//
// The number of times the token's block was selected. BN == BS collapses the
// block vector to the tile start; R = BN/BS > 1 spreads each block's count over
// its BS tokens. valid = (weight > 0) & (token <= i_t) merges selection with
// the causal predicate, which is what lets the rewrite replace the scf.if with
// a zero.
struct ScanWeights {
  Value tokenCounts; // [BN] f32
  Value tokens;      // [BN] i32, token_start + [0, BN)
  Value valid;       // [BN] i1
};

// Per-token selection weights: every tile slot becomes a weight in the score
// domain, so the guarded region can run unconditionally. The geometry (BS, BN,
// R) and the two element types are read from the candidate and the rewriter,
// which is why only the loop values and the prologue are passed here.
static ScanWeights buildScanWeights(IRRewriter &rewriter, Location loc,
                                    Value iv, Value tokenStart,
                                    Value tokenBound,
                                    const ScanPrologue &prologue,
                                    const ScanCandidate &candidate) {
  const Type i32 = rewriter.getI32Type();
  const Type f32 = rewriter.getF32Type();
  const int64_t TW = prologue.tableWidth;
  const int64_t BS = tileBlockSize(candidate.tiles);
  const bool additive =
      std::holds_alternative<AdditiveRecurrence>(candidate.recurrence);
  const int64_t BN = scanTileWidth(BS, additive);
  const int64_t R = blocksPerTile(BS, additive);
  Value startsRow = prologue.startsRow;
  Value slotRow = prologue.slotRow;
  Value bsConstant = prologue.bsConstant;
  Value matchesEq;
  Value slotMatrix = slotRow;
  if (R == 1) {
    Value blockStartRow = rewriter.create<triton::SplatOp>(
        loc, tensorTy({1, TW}, i32), tokenStart);
    matchesEq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               startsRow, blockStartRow);
  } else {
    Value cR = rewriter.create<arith::ConstantIntOp>(loc, R, 32);
    Value blocksBase = rewriter.create<arith::MulIOp>(loc, iv, cR);
    Value blocks = rewriter.create<arith::AddIOp>(
        loc,
        rewriter.create<triton::SplatOp>(loc, tensorTy({R}, i32), blocksBase),
        rewriter.create<triton::MakeRangeOp>(loc, tensorTy({R}, i32), 0, R));
    Value blockStarts = rewriter.create<arith::MulIOp>(
        loc, blocks,
        rewriter.create<triton::SplatOp>(loc, tensorTy({R}, i32), bsConstant));
    Value startsMatrix = rewriter.create<triton::BroadcastOp>(
        loc, tensorTy({R, TW}, i32), startsRow);
    Value blockMatrix = rewriter.create<triton::BroadcastOp>(
        loc, tensorTy({R, TW}, i32),
        rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({R, 1}, i32),
                                              blockStarts, 1));
    matchesEq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               startsMatrix, blockMatrix);
    slotMatrix = rewriter.create<triton::BroadcastOp>(
        loc, tensorTy({R, TW}, rewriter.getI1Type()), slotRow);
  }
  Value matches = rewriter.create<arith::AndIOp>(loc, matchesEq, slotMatrix);
  SmallVector<int64_t> matrixShape{R, TW};
  Value countsI =
      rewriter.create<arith::ExtUIOp>(loc, tensorTy(matrixShape, i32), matches);
  Value counts = buildReduce<arith::AddIOp>(rewriter, loc, countsI, 1);
  Value countsF =
      rewriter.create<arith::SIToFPOp>(loc, tensorTy({R}, f32), counts);
  Value tokenCounts2D =
      broadcastRow(rewriter, loc, countsF, tensorTy({R, BS}, f32), 1);
  Value tokenCounts = rewriter.create<triton::ReshapeOp>(
      loc, tensorTy({BN}, f32), tokenCounts2D, false);

  // valid = (weight > 0) & (token <= i_t): selection plus causality. A -1
  // padding sentinel never matches a tile start, so it needs no special case;
  // blocks past i_t are masked here and the scan bound skips their tiles
  // entirely. The token positions are recomputed as token_start + [0, BN), so
  // the matched causal offset must be exactly zero.
  Value tokens = rewriter.create<arith::AddIOp>(
      loc,
      rewriter.create<triton::SplatOp>(loc, tensorTy({BN}, i32), tokenStart),
      prologue.tokensRange);
  Value validWeight = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, tokenCounts, prologue.zeroTokens);
  Value validCausal = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, tokens, prologue.pidRow);
  Value valid = rewriter.create<arith::AndIOp>(loc, validWeight, validCausal);
  // tokens < TK: redundant while the causal term is present (the row position
  // is at most TK - 1, so zero-padded K/V past the sequence end is already
  // masked), but it keeps the tile from accumulating padding if a future
  // variant weakens the causal mask. Only emitted when the K/V view carries a
  // dynamic bound.
  if (tokenBound) {
    Value bound = tokenBound;
    if (bound.getType() != i32) {
      unsigned width = bound.getType().getIntOrFloatBitWidth();
      bound =
          width > 32
              ? rewriter.create<arith::TruncIOp>(loc, i32, bound).getResult()
              : rewriter.create<arith::ExtSIOp>(loc, i32, bound).getResult();
    }
    Value boundSplat =
        rewriter.create<triton::SplatOp>(loc, tensorTy({BN}, i32), bound);
    Value validTokenBound = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, tokens, boundSplat);
    valid = rewriter.create<arith::AndIOp>(loc, valid, validTokenBound);
  }

  return ScanWeights{tokenCounts, tokens, valid};
}

// The form-specific template bodies, called from the top-level builder.
// Everything the two template bodies share: the rewritten loop, the tile
// geometry, the two tiles, the per-token weights and the score tile. The bodies
// differ only in how they turn those into the carried state, so only that
// difference is spelled in their signatures.
struct ScanBodyInputs {
  scf::ForOp loop;
  int64_t G = 0;    // query rows
  int64_t BN = 0;   // scan tile width
  Value bK;         // the K tile at this scan tile's token start
  Value bV;         // the V tile
  Value scores;     // [G, BN] f32, dot(q, k, 0)
  Value zeroScores; // [G, BN] f32 zero, the neutral operand of both dots
  ScanWeights weights;
  const ScanPrologue *prologue = nullptr;
};

// ---- [3] The online-softmax (fwd) scan body ----

// The fwd template's tile read: rebuild the two make_tensor_ptr views at
// the scan tile's token start and load them unconditionally.
static void buildMakeTensorPtrTile(IRRewriter &rewriter, Location loc,
                                   MptrTiles &mptr, int64_t BN,
                                   Value tokenStart,
                                   const ScanPrologue &prologue, Value &bK,
                                   Value &bV) {
  // The re-created K/V dynamic bound is built inside the prologue so that the
  // forward template's op order is unchanged; the tile views read it through
  // this mapping.
  auto mapAddressOperand = [&](Value v) -> Value {
    return v == mptr.dynamicBoundResult ? prologue.bound64 : v;
  };
  // The same views and strides, token offset = token_start, loads
  // unconditional with the original boundary behaviour: the tile is always
  // fully read and selection happens in the score domain.
  //
  // Because the tile is read unconditionally, every element now takes part in
  // arithmetic the baseline skipped for unselected blocks. K is safe: the
  // score goes through select(valid, s, -inf) and exp(masked - safe_m), so the
  // -inf wins. V is NOT: the weight multiplies b_p in f32 and p_exp is 0 for
  // an unselected token, so the product is 0 * V and V == +-Inf yields NaN.
  // Every V element that can be multiplied by a zero weight must be finite.
  auto rebuildTilePtr = [&](triton::MakeTensorPtrOp ptr,
                            unsigned tokenAxis) -> Value {
    auto ptrType = cast<triton::PointerType>(ptr.getType());
    auto block = cast<RankedTensorType>(ptrType.getPointeeType());
    SmallVector<int64_t> blockShape(block.getShape());
    blockShape[tokenAxis] = BN;
    Type newPtrType = triton::PointerType::get(
        RankedTensorType::get(blockShape, block.getElementType()),
        ptrType.getAddressSpace());
    SmallVector<Value> shape;
    for (Value v : ptr.getShape())
      shape.push_back(mapAddressOperand(v));
    SmallVector<Value> strides;
    for (Value v : ptr.getStrides())
      strides.push_back(mapAddressOperand(v));
    SmallVector<Value> offsets;
    for (unsigned axis = 0; axis < ptr.getOffsets().size(); ++axis)
      offsets.push_back(axis == tokenAxis
                            ? tokenStart
                            : mapAddressOperand(ptr.getOffsets()[axis]));
    return rewriter.create<triton::MakeTensorPtrOp>(
        loc, newPtrType, mapAddressOperand(ptr.getBase()), shape, strides,
        offsets, ptr.getOrder());
  };
  Value kTilePtr = rebuildTilePtr(mptr.kPtr, mptr.kTokenAxis);
  Value vTilePtr = rebuildTilePtr(mptr.vPtr, mptr.vTokenAxis);
  bK = rewriter.create<triton::LoadOp>(
      loc, kTilePtr, mptr.kLoad.getBoundaryCheck(), mptr.kLoad.getPadding(),
      mptr.kLoad.getCache(), mptr.kLoad.getEvict(), mptr.kLoad.getIsVolatile());
  bV = rewriter.create<triton::LoadOp>(
      loc, vTilePtr, mptr.vLoad.getBoundaryCheck(), mptr.vLoad.getPadding(),
      mptr.vLoad.getCache(), mptr.vLoad.getEvict(), mptr.vLoad.getIsVolatile());
}

// The fwd template's loop body: safe_m online softmax over the scores,
// then the (o, m, l) yields.
static void buildOnlineSoftmaxBody(IRRewriter &rewriter, Location loc,
                                   OnlineSoftmaxRecurrence &online,
                                   const ScanBodyInputs &in, BlockArgument oArg,
                                   BlockArgument mArg, BlockArgument lArg,
                                   Type oType) {
  const Type f32 = rewriter.getF32Type();
  scf::ForOp scan = in.loop;
  const int64_t G = in.G;
  const int64_t BN = in.BN;
  const Value scores = in.scores;
  const ScanPrologue &prologue = *in.prologue;
  Value maskMatrix = broadcastRow(rewriter, loc, in.weights.valid,
                                  tensorTy({G, BN}, rewriter.getI1Type()), 0);
  Value negInfScores = splatConstant(
      rewriter, loc, {G, BN},
      rewriter.getFloatAttr(f32, APFloat::getInf(APFloat::IEEEsingle(), true)));
  Value maskedScores =
      rewriter.create<arith::SelectOp>(loc, maskMatrix, scores, negInfScores);

  // Online softmax with the safe_m guard. The guard replaces the baseline's
  // scf.if: a fully masked tile with m == -inf would otherwise compute
  // exp(-inf - -inf) = NaN and permanently poison o and l.
  Value tileMax = buildReduce<arith::MaxNumFOp>(rewriter, loc, maskedScores, 1);
  Value mNew = rewriter.create<arith::MaxNumFOp>(loc, mArg, tileMax);
  Value needsGuard = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, mNew, prologue.negInfRows);
  Value safeM = rewriter.create<arith::SelectOp>(loc, needsGuard,
                                                 prologue.zeroRows, mNew);
  // Hard constraint 1: safe_m subtracts from BOTH the rescale factor and
  // the scores. Yielding the TRUE max keeps the epilogue's lse identical.
  Value rescale = rewriter.create<math::ExpOp>(
      loc, rewriter.create<arith::SubFOp>(loc, mArg, safeM));
  Value safeMatrix =
      broadcastRow(rewriter, loc, safeM, tensorTy({G, BN}, f32), 1);
  Value pExp = rewriter.create<math::ExpOp>(
      loc, rewriter.create<arith::SubFOp>(loc, maskedScores, safeMatrix));
  // Hard constraint 2: the weight multiplies b_p, in f32, before the bf16
  // truncation: never the -inf mask (-inf * 0 = NaN) and never after the
  // cast (truncf(p*w) != truncf(p)*w).
  Value weightMatrix = broadcastRow(rewriter, loc, in.weights.tokenCounts,
                                    tensorTy({G, BN}, f32), 0);
  Value pWeighted = rewriter.create<arith::MulFOp>(loc, pExp, weightMatrix);

  // l = l * rescale + sum(p)
  Value lScaled = rewriter.create<arith::MulFOp>(loc, lArg, rescale);
  Value pSum = buildReduce<arith::AddFOp>(rewriter, loc, pWeighted, 1);
  Value lNew = rewriter.create<arith::AddFOp>(loc, lScaled, pSum);

  // o = o * rescale + dot(p, V)
  Value rescaleMatrix = broadcastRow(rewriter, loc, rescale, oType, 1);
  Value oScaled = rewriter.create<arith::MulFOp>(loc, oArg, rescaleMatrix);
  Type valueLhsElem =
      cast<RankedTensorType>(online.valueDot.getA().getType()).getElementType();
  Value valueLhs = valueLhsElem == f32
                       ? pWeighted
                       : castMatrix(rewriter, loc, valueLhsElem, pWeighted,
                                    online.pCastIsExtend);
  Value oNew =
      buildDot(rewriter, loc, oType, valueLhs, in.bV, oScaled, online.valueDot);

  rewriter.setInsertionPointToEnd(scan.getBody());
  SmallVector<Value> yielded{Value(), Value(), Value()};
  yielded[online.oArg] = oNew;
  yielded[online.mArg] = mNew;
  yielded[online.lArg] = lNew;
  rewriter.create<scf::YieldOp>(loc, yielded);
}

// ---- [4] The additive (bwd dq) scan body ----

// One additive-form tile read at the scan tile's token start: the baseline's
// own two-level chain, with the mask rebuilt from the captured bound. BN is
// passed in rather than read back from the vector's type, so the width has one
// source.
static Value buildAddPtrTileLoad(IRRewriter &rewriter, Location loc,
                                 const AddPtrTile &tile,
                                 triton::LoadOp original, Value tokensI32,
                                 int64_t BN) {
  SmallVector<int64_t, 1> tokenShape{BN};
  const int64_t F = tile.freeExtent;
  Type intType = rewriter.getIntegerType(tile.tokenWidth);
  Type freeElem =
      cast<RankedTensorType>(tile.freeVec.getType()).getElementType();
  Type baseType = tile.base.getType();
  Type elemType = cast<triton::PointerType>(baseType).getPointeeType();
  Type i1 = rewriter.getI1Type();

  // Preserve the addressing width of the baseline: the bwd form does its
  // token arithmetic in the index table's width while the free axis stays
  // i32, so the tile's token term is rebuilt at that width.
  Type tokenTensor = tensorTy(tokenShape, intType);
  Value tokens = tokensI32;
  if (tokens.getType() != tokenTensor)
    tokens = rewriter.create<arith::ExtSIOp>(loc, tokenTensor, tokensI32);

  // base + free[:, None], broadcast over the tokens.
  Type freeOne = tensorTy({F, 1}, freeElem);
  Type ptrOne = tensorTy({F, 1}, baseType);
  Type ptrType = tensorTy({F, BN}, baseType);
  auto intOne = tensorTy({1, BN}, intType);
  Value freeOff =
      rewriter.create<triton::ExpandDimsOp>(loc, freeOne, tile.freeVec, 1);
  Value innerPtr = rewriter.create<triton::AddPtrOp>(
      loc, ptrOne, rewriter.create<triton::SplatOp>(loc, ptrOne, tile.base),
      freeOff);
  Value tokenOff = rewriter.create<arith::MulIOp>(
      loc, rewriter.create<triton::ExpandDimsOp>(loc, intOne, tokens, 0),
      rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(intOne, rewriter.getIntegerAttr(
                                                  intType, tile.tokenStride))));
  Value tilePtr = rewriter.create<triton::AddPtrOp>(
      loc, ptrType,
      rewriter.create<triton::BroadcastOp>(loc, ptrType, innerPtr),
      rewriter.create<triton::BroadcastOp>(loc, tensorTy({F, BN}, intType),
                                           tokenOff));

  // mask = (free < F) & (tokens < T), with the bound captured by the
  // matcher: the token term is the baseline's own range guard, not a
  // boundary check the scan form could synthesize.
  Type maskType = tensorTy({F, BN}, i1);
  Value freeBound = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, freeOff, tile.freeBoundRhs);
  Value tokenMask = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, tokens,
      rewriter.create<triton::SplatOp>(loc, tokenTensor, tile.tokenBound));
  Value mask = rewriter.create<arith::AndIOp>(
      loc, rewriter.create<triton::BroadcastOp>(loc, maskType, freeBound),
      broadcastRow(rewriter, loc, tokenMask, maskType, 0));
  // other = 0, the baseline's padding value for masked-off elements. Pick the
  // attribute kind first: getFloatAttr() on an integer element type is a hard
  // error in MLIR, so it must not be evaluated speculatively.
  Attribute zero;
  if (elemType.isIntOrIndex())
    zero = rewriter.getIntegerAttr(elemType, 0);
  else
    zero = rewriter.getFloatAttr(elemType, 0.0);
  Value other = splatConstant(rewriter, loc, {F, BN}, zero);
  return rewriter.create<triton::LoadOp>(
      loc, tilePtr, mask, other, original.getCache(), original.getEvict(),
      original.getIsVolatile());
}

// The bwd template's loop body: WHERE-then-weight scores, dp, the
// accumulation dot into the carried state and the single yield.
static void buildAdditiveBody(IRRewriter &rewriter, Location loc,
                              AdditiveRecurrence &additive,
                              const ScanBodyInputs &in, BlockArgument accArg,
                              Type stateType) {
  const Type f32 = rewriter.getF32Type();
  scf::ForOp scan = in.loop;
  const int64_t G = in.G;
  const int64_t BN = in.BN;
  const Value zeroScores = in.zeroScores;
  const Value bK = in.bK;
  // Additive state update (bwd dq). The carried state is a single accumulator
  // with neutral 0 and no rescale, so a tile contributes through the dot's C
  // operand.
  //
  // The WHERE comes BEFORE the weight multiply. An empty query row has
  // lse = -inf, hence exp(s - lse) is +inf on every token; multiplying that by
  // the zero weight first would turn it into inf * 0 = NaN and poison the
  // whole dq row. Selecting to 0 first keeps every invalid lane an exact
  // zero. This is why the forward template's ordering must not be copied.
  Value lseMatrix =
      broadcastRow(rewriter, loc, additive.lseVec, tensorTy({G, BN}, f32), 1);
  Value pExp = rewriter.create<math::ExpOp>(
      loc, rewriter.create<arith::SubFOp>(loc, in.scores, lseMatrix));
  Value validMatrix = broadcastRow(rewriter, loc, in.weights.valid,
                                   tensorTy({G, BN}, rewriter.getI1Type()), 0);
  Value pMasked =
      rewriter.create<arith::SelectOp>(loc, validMatrix, pExp, zeroScores);
  Value weightMatrix = broadcastRow(rewriter, loc, in.weights.tokenCounts,
                                    tensorTy({G, BN}, f32), 0);
  Value pWeighted = rewriter.create<arith::MulFOp>(loc, pMasked, weightMatrix);

  // dp = dot(dO, v): a fresh matmul, not an accumulation into state.
  Value dp =
      buildDot(rewriter, loc, tensorTy({G, BN}, f32), additive.valueDot.getA(),
               in.bV, zeroScores, additive.valueDot);
  Value deltaMatrix =
      broadcastRow(rewriter, loc, additive.deltaVec, tensorTy({G, BN}, f32), 1);
  Value ds = rewriter.create<arith::MulFOp>(
      loc, pWeighted, rewriter.create<arith::SubFOp>(loc, dp, deltaMatrix));

  // ds enters the accumulation dot in its input type; the baseline
  // truncates in the same place (one rounding of p*w*(dp-delta)).
  Type dsElem =
      cast<RankedTensorType>(additive.accDot.getA().getType()).getElementType();
  Value dsCast = dsElem == f32 ? ds
                               : castMatrix(rewriter, loc, dsElem, ds,
                                            additive.dsCastIsExtend);

  // k^T with the baseline's own order: the position of the token axis is
  // semantic, so a hard-coded [1, 0] would transpose the wrong axis for an
  // equivalent spelling.
  auto kTileType = cast<RankedTensorType>(bK.getType());
  SmallVector<int64_t> transShape;
  for (int32_t axis : additive.transpose.getOrder())
    transShape.push_back(kTileType.getShape()[axis]);
  Value kT = rewriter.create<triton::TransOp>(
      loc, RankedTensorType::get(transShape, kTileType.getElementType()), bK,
      additive.transpose.getOrder());

  Value accNew =
      buildDot(rewriter, loc, stateType, dsCast, kT, accArg, additive.accDot);
  rewriter.setInsertionPointToEnd(scan.getBody());
  rewriter.create<scf::YieldOp>(loc, accNew);
}

} // namespace

namespace mlir {
namespace triton {
namespace cfg {

// ---- [5] Entry: emit the whole scan form ----

// The top-level builder: shared prologue and new loop, per-token weights,
// the scores dot, the template dispatch and the commit.
bool buildSelectedBlockScan(IRRewriter &rewriter, ScanCandidate &candidate) {
  scf::ForOp oldLoop = candidate.loop.forOp;
  Location loc = oldLoop.getLoc();
  Type i32 = rewriter.getI32Type();
  Type i64 = rewriter.getI64Type();
  Type f32 = rewriter.getF32Type();

  const int64_t S = candidate.table.length;
  const int64_t BS = tileBlockSize(candidate.tiles);
  const bool additive =
      std::holds_alternative<AdditiveRecurrence>(candidate.recurrence);
  const int64_t BN = scanTileWidth(BS, additive);
  // The variant has two alternatives, so this is set exactly when the form is
  // not additive. The tablegen accessors are not const-qualified in this tree,
  // so the handle is mutable and no field is written through it.
  OnlineSoftmaxRecurrence *online =
      std::get_if<OnlineSoftmaxRecurrence>(&candidate.recurrence);

  // The carried state keeps its original types and initial values. G is the
  // row count: the online-softmax max vector and the additive accumulator
  // share their first dimension with the query rows. The additive accumulator
  // is always iter arg 0; the matcher records the online-softmax (o, m, l).
  RankedTensorType stateType = cast<RankedTensorType>(
      oldLoop.getInitArgs()[additive ? 0 : online->mArg].getType());
  const int64_t G = stateType.getShape()[0];
  RankedTensorType oType =
      additive ? stateType
               : cast<RankedTensorType>(
                     oldLoop.getInitArgs()[online->oArg].getType());

  SmallVector<Value> inits(oldLoop.getInitArgs());

  rewriter.setInsertionPoint(oldLoop);

  std::optional<ScanPrologue> built =
      buildScanPrologue(rewriter, candidate, i32, i64, f32, S, BS, BN, G);
  if (!built)
    return false;
  ScanPrologue &prologue = *built;

  // The new loop: i32 induction variable, the original iter args and initial
  // values, running to the folded scan bound.
  scf::ForOp scan = rewriter.create<scf::ForOp>(
      loc, prologue.c0, prologue.scanUpperBound, prologue.c1, inits);
  // A loop with no iter args has no body block, and then there is no terminator
  // to drop either.  (In this tree the freshly built scf.for body is empty, so
  // Block::getTerminator() may only be called when it is not.)
  if (!scan.getBody()->empty())
    rewriter.eraseOp(scan.getBody()->getTerminator());
  rewriter.setInsertionPointToStart(scan.getBody());
  Value iv = scan.getInductionVar();
  BlockArgument accArg = scan.getRegionIterArgs()[0];

  Value tokenStart = rewriter.create<arith::MulIOp>(loc, iv, prologue.cBN);

  // ---- K/V tiles. The two templates read the same two tiles per
  // iteration; only the addressing differs. ----
  Value bK, bV;
  MptrTiles *mptr = std::get_if<MptrTiles>(&candidate.tiles);
  if (mptr) {
    // The online-softmax form always reads through make_tensor_ptr; the
    // additive form does too when the baseline spelled the tiles that way
    // (production nsa_bwd_dq), and only the addptr spelling needs the rebuilt
    // two-level chain below.
    buildMakeTensorPtrTile(rewriter, loc, *mptr, BN, tokenStart, prologue, bK,
                           bV);
  }

  ScanWeights weights = buildScanWeights(
      rewriter, loc, iv, tokenStart, mptr ? mptr->dynamicBoundInput : Value(),
      prologue, candidate);
  Value tokens = weights.tokens;

  if (const AddPtrTiles *addptr = std::get_if<AddPtrTiles>(&candidate.tiles)) {
    // Additive tiles, addptr spelling: the baseline's own two-level addptr at
    // the scan tile's token start, with the token-range mask preserved.
    bK = buildAddPtrTileLoad(rewriter, loc, addptr->kTile, addptr->kLoad,
                             tokens, BN);
    bV = buildAddPtrTileLoad(rewriter, loc, addptr->vTile, addptr->vLoad,
                             tokens, BN);
  }

  // The scores both templates start from: dot(q, k, 0) over the K tile.
  triton::DotOp scoreDot = scoreDotOf(candidate.recurrence);
  Value zeroScores =
      splatConstant(rewriter, loc, {G, BN}, rewriter.getFloatAttr(f32, 0.0));
  Value scores = buildDot(rewriter, loc, tensorTy({G, BN}, f32),
                          scoreDot.getA(), bK, zeroScores, scoreDot);
  ScanBodyInputs bodyInputs{scan,   G,          BN,      bK,       bV,
                            scores, zeroScores, weights, &prologue};
  if (AdditiveRecurrence *additive =
          std::get_if<AdditiveRecurrence>(&candidate.recurrence))
    buildAdditiveBody(rewriter, loc, *additive, bodyInputs, accArg, stateType);
  else
    buildOnlineSoftmaxBody(rewriter, loc, *online, bodyInputs,
                           scan.getRegionIterArgs()[online->oArg],
                           scan.getRegionIterArgs()[online->mArg],
                           scan.getRegionIterArgs()[online->lArg], oType);

  // ---- Commit within the sandbox: the scan loop replaces the gather loop
  // wholesale; the epilogue keeps reading the same (o, m, l) results. ----
  rewriter.replaceOp(oldLoop, scan->getResults());
  return true;
}

// The plan: a privately prepared and verified sandbox plus a commit-only
// apply().

// ---- [6] On-chip budget gate ----
//
// The rewrite grows the loop body (whole-table vector read, unconditional
// K/V loads per iteration, scan mask), so it measures the rewritten
// sandbox: UB via getSafeUBBudget + estimatePeakLiveBytes (the row-lift
// accounting) and L0 via exceedsCubeBudgets from #2841.  A zero capacity
// disables the corresponding query.  Going over budget warns on the live
// function and returns true, so the caller declines the rewrite
// (apply() itself must not fail).
bool budgetOverflow(triton::FuncOp sandboxFunction,
                    const ResourceSnapshot &resources, llvm::StringRef form,
                    triton::FuncOp liveFunction) {
  if (std::optional<uint64_t> safeUB = resources.getSafeUBBudget()) {
    LiveByteEstimate estimate = estimatePeakLiveBytes(sandboxFunction);
    // The static peak counts one copy of each UB tensor, but the backend
    // multi-buffers loop-carried UB buffers, so the check scales it by the
    // replication factor (see the measured data in kUBBufferReplication).
    // The static peak counts one copy of each UB tensor while the backend
    // multi-buffers loop-carried UB buffers, so the predicate budgets the
    // replicated demand (see the measured numbers on kUBBufferReplication).
    // Fail closed and divide, like QueryRowBatchingRule's fitsQueryRowBudgets:
    // an unknown or overflowing estimate is not proof that the rewrite fits,
    // and the division form cannot wrap the product.
    const bool fits = estimate.known && !estimate.overflow &&
                      estimate.peakLiveBytes <= *safeUB / kUBBufferReplication;
    const uint64_t demand =
        estimate.known ? estimate.peakLiveBytes * kUBBufferReplication : 0;
    if (::getenv("TRITON_GRAPH_UB_DEBUG"))
      llvm::errs() << "graph-ub[rule]: form=" << form
                   << " liftedPeak=" << estimate.peakLiveBytes
                   << " known=" << estimate.known
                   << " overflow=" << estimate.overflow << " demand=" << demand
                   << " ubBudget=" << *safeUB << "\n";
    if (!fits) {
      liveFunction.emitWarning()
          << "selected-block-scan: " << form << " needs " << demand
          << " bytes of UB, budget " << *safeUB << "; declining";
      return true;
    }
  }
  if (exceedsCubeBudgets(sandboxFunction, resources)) {
    liveFunction.emitWarning()
        << "selected-block-scan: " << form
        << " exceeds the L0A/L0B/L0C cube budgets; declining";
    return true;
  }
  return false;
}

// Tidy the private sandbox before the joint consult, so the row lift judges
// exactly the body apply() commits. `make_ttir` in backend/compiler.py places
// add_graph_optimize last, with no canonicalize/CSE after it, so without this
// cleanup the rewrite would leave splat(constant) and a few dead ops behind
// for the downstream lowering.
LogicalResult runSandboxStructuralCleanup(ModuleOp sandbox) {
  MLIRContext *context = sandbox.getContext();
  PassManager cleanup(context, ModuleOp::getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return cleanup.run(sandbox);
}

} // namespace cfg
} // namespace triton
} // namespace mlir
