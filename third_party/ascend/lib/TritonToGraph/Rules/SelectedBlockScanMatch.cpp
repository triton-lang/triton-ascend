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
// SelectedBlockScanRule (#2686, rule bit 65536): the matcher.
//
// This translation unit only decides whether the gather loop is the selected-
// block scan pattern and what its parameters are; it never mutates IR. A miss
// is std::nullopt, silently and without diagnostics.
//
// Sections: [0] IR primitives, [1] facts, [2] six parsers, [3] validateScan,
// [4] assembly. The contract and the pipeline are documented once, in
// SelectedBlockScanFacts.h; this file follows the same order.

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/RowLifting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>
#include <variant>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

// The joint gate lives in SelectedBlockScanRule.cpp: the scan rewrite never
// consults a read-only row-lift matcher, it materializes the row lift's rewrite
// inside its own sandbox, so neither rewrite can land without the other.

#include "TritonToGraph/SelectedBlockScanFacts.h"

namespace {

// ===========================================================================
// [0] IR primitives
// ===========================================================================

// ---- [0.1] Constants ----

static bool isPowerOfTwo(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

// Scalar value of a splat integer tensor constant. The bwd form spells its
// token stride and free-axis bound as tensors rather than as scalars.
static std::optional<int64_t> getSplatIntTensorValue(Value v) {
  if (auto def = v.getDefiningOp<arith::ConstantOp>())
    if (auto dense = dyn_cast<DenseIntElementsAttr>(def.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}

// The value of a float constant, scalar or splat tensor. Every neutral element
// the pattern accepts (the -inf mask of the online-softmax form, the 0 of the
// additive form) is read through this one extractor, so "is it that constant"
// is a comparison and not a second pattern.
static std::optional<APFloat> getFloatConstantValue(Value v) {
  auto def = v.getDefiningOp<arith::ConstantOp>();
  if (!def)
    return std::nullopt;
  if (auto dense = dyn_cast<DenseFPElementsAttr>(def.getValue())) {
    if (dense.isSplat())
      return dense.getSplatValue<APFloat>();
    return std::nullopt;
  }
  if (auto scalar = dyn_cast<FloatAttr>(def.getValue()))
    return scalar.getValue();
  return std::nullopt;
}

// The neutral element a masked tile or a skipped block contributes: an integer
// zero is only ever spelled on the tile masks, a float zero on the values.
static bool isZeroSplatConstant(Value v) {
  if (auto value = getFloatConstantValue(v))
    return value->isZero();
  return getSplatIntTensorValue(v) == std::optional<int64_t>(0);
}

// The online-softmax form masks unselected positions with -inf, which is what
// makes exp() vanish and the scan's arithmetic exact.
static bool isNegativeInfSplatConstant(Value v) {
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    v = splat.getSrc();
  auto value = getFloatConstantValue(v);
  return value && value->isInfinity() && value->isNegative();
}

// ---- [0.2] Def-use reachability ----
//
//   dependsOn(v, pivot, rules)  is v computed from pivot?
//   reachesLoad(v, loop)        does v's chain cross a load (inside `loop`
//   only)?
//
// Both are memoized and depth-bounded (a shared subexpression tree is otherwise
// exponential); the depth cap declines the match rather than accepting it.
//
// They are hand-written rather than read from ProgramAxisDependenceAnalysis,
// which the other axis-aware rules (PersistentTaskStripMining,
// IndependentAxisTensorize, RowCoalescing) declare and use as a precondition:
// this matcher also runs on the plan's sandbox clone, and the context's
// analyses hold live-IR Operation* and are invalidated after every commit, so
// they cannot answer for a clone. The walks are self-contained and re-derive on
// whichever module they are handed.
constexpr unsigned kMaxDependenceDepth = 4096;

// What a dependence walk treats as a root.
struct Dependence {
  // An operation cut out of the walk. The scan rewrite cuts the per-iteration
  // index load: it is the mechanism the rewrite removes, so nothing downstream
  // depends on the program id *through* it.
  Operation *cut = nullptr;
  // When set, a scalar load outside that loop is an opaque per-program
  // constant. That is the varlen spelling: the in-sequence row position and the
  // sequence bases are read before the loop, so every value "depends" on the
  // program id only through those load addresses.
  scf::ForOp opaqueLoads;
};

static bool dependsOnImpl(Value v, Value pivot, const Dependence &rules,
                          llvm::DenseSet<Value> &visited, unsigned depth) {
  if (v == pivot)
    return true;
  if (depth > kMaxDependenceDepth)
    return true; // too deep: assume dependence
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (op == rules.cut)
    return false;
  if (rules.opaqueLoads && isa<triton::LoadOp>(op) &&
      !rules.opaqueLoads->isAncestor(op) && !isa<RankedTensorType>(v.getType()))
    return false; // loop-invariant scalar load: a per-program constant
  if (!visited.insert(v).second)
    return false;
  return llvm::any_of(op->getOperands(), [&](Value a) {
    return dependsOnImpl(a, pivot, rules, visited, depth + 1);
  });
}

static bool dependsOn(Value v, Value pivot, const Dependence &rules = {}) {
  llvm::DenseSet<Value> visited;
  return dependsOnImpl(v, pivot, rules, visited, 0);
}

static bool reachesLoadImpl(Value v, scf::ForOp onlyInside,
                            llvm::DenseSet<Value> &visited, unsigned depth) {
  if (depth > kMaxDependenceDepth)
    return true; // too deep: assume it reaches a load
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (isa<triton::LoadOp>(op))
    return !onlyInside || onlyInside->isAncestor(op);
  if (!visited.insert(v).second)
    return false;
  return llvm::any_of(op->getOperands(), [&](Value a) {
    return reachesLoadImpl(a, onlyInside, visited, depth + 1);
  });
}

// True when the value's defining chain crosses any load. Forbidden in the K/V
// address chains: the varlen dispatch computes the K/V base and bound from
// loads.
static bool reachesLoad(Value v) {
  llvm::DenseSet<Value> visited;
  return reachesLoadImpl(v, scf::ForOp(), visited, 0);
}

// True when the chain crosses a load executed *inside* the loop. Loads defined
// outside the loop are per-program constants (the varlen dispatch reads
// token_indices_q / cu_seqlens before the loop) and are allowed.
static bool reachesLoopInteriorLoad(Value v, scf::ForOp loop) {
  llvm::DenseSet<Value> visited;
  return reachesLoadImpl(v, loop, visited, 0);
}

static bool isDefinedOutsideLoop(Value v, scf::ForOp loop) {
  // A block argument (the induction variable or a loop-carried value) has no
  // defining op but is still produced by the loop, so it is not invariant.
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *owner = arg.getOwner()->getParentOp();
    return owner != loop.getOperation() && !loop->isAncestor(owner);
  }
  return !loop->isAncestor(v.getDefiningOp());
}

// ---- [0.3] Peeling ----

// Strip the integer width casts the front end may wrap around the loaded index.
static Value stripIntegerCasts(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!isa<arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(op))
      break;
    v = op->getOperand(0);
  }
  return v;
}

// Peel the rank-expanding broadcasts down to a row operand: splat, expand_dims
// and broadcast all reintroduce a [G] row as a [G, BS] matrix. This is the
// peeling of *values*: the row vector the arithmetic is really about.
static Value peelBroadcasts(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp>(op))
      break;
    v = op->getOperand(0);
  }
  return v;
}

// Peel the broadcasts above a comparison down to the comparison itself. A splat
// is *not* peeled here: it would turn a constant matrix into a scalar and
// change what the comparison is made of, which is exactly the misreading the
// causal check must not make.
static Value peelConditionBroadcasts(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!isa<triton::BroadcastOp, triton::ExpandDimsOp>(op))
      break;
    v = op->getOperand(0);
  }
  return v;
}

// The comparison predicate with its operands exchanged.
static arith::CmpIPredicate flippedPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  default:
    return pred;
  }
}

// Peel the float width cast around v and report its direction. Both templates
// accept one cast around a value they feed to a dot, and the rewriter re-emits
// the same cast, so the direction is a fact and the op is never carried.
static Value peelFloatCast(Value v, bool &isExtend) {
  Operation *cast = v.getDefiningOp();
  if (!isa_and_nonnull<arith::TruncFOp, arith::ExtFOp>(cast))
    return v;
  isExtend = isa<arith::ExtFOp>(cast);
  return cast->getOperand(0);
}

// A row-broadcast chain down to the pivot.
static bool isBroadcastOf(Value v, Value pivot) {
  return peelBroadcasts(v) == pivot;
}

// Exactly one tt.broadcast, which is what a mask row keeps after lifting: the
// mask of the addptr spelling is andi(broadcast(row), broadcast(row)), and
// peeling further would accept a mask that is not a pair of rows.
static Value peelSingleBroadcast(Value v) {
  if (auto broadcast = v.getDefiningOp<triton::BroadcastOp>())
    return broadcast.getSrc();
  return Value();
}

// ---- [0.4] Shapes ----

// The only two load shapes both templates accept: unmasked (the scalar index
// read and the mptr tiles, whose boundary check the tile parser verifies), or
// an explicit mask with other = 0 (the addptr tile spelling). Volatile is never
// re-emitted.
enum class LoadShape { Unmasked, MaskedZeroOther };

static bool hasLoadShape(triton::LoadOp load, LoadShape shape) {
  if (load.getIsVolatile())
    return false;
  switch (shape) {
  case LoadShape::Unmasked:
    return !load.getMask() && !load.getOther();
  case LoadShape::MaskedZeroOther:
    return load.getMask() && load.getOther() != Value() &&
           load.getBoundaryCheck().empty() && !load.getPadding() &&
           isZeroSplatConstant(load.getOther());
  }
  return false;
}

// A widened mptr tile is only safe when the view itself zero-fills past the
// tensor extent: the extra lanes then read 0, which the weights neutralize.
// PAD_NAN would poison the accumulator instead (0 * NaN is NaN), and a token
// axis outside the boundary check would read past the row instead of padding.
static bool isZeroPaddedTokenBoundary(MptrTiles &tiles) {
  triton::LoadOp loads[2] = {tiles.kLoad, tiles.vLoad};
  unsigned axes[2] = {tiles.kTokenAxis, tiles.vTokenAxis};
  for (unsigned i = 0; i < 2; ++i) {
    triton::LoadOp load = loads[i];
    if (load.getPadding() == triton::PaddingOption::PAD_NAN)
      return false;
    if (!llvm::is_contained(load.getBoundaryCheck(),
                            static_cast<int32_t>(axes[i])))
      return false;
  }
  return true;
}

// A fresh matmul of the guarded region: a dot whose accumulator is zero, i.e. a
// product rather than an accumulation into the carried state.
static triton::DotOp matchFreshDot(Value v, Region &thenRegion) {
  auto dot = v.getDefiningOp<triton::DotOp>();
  if (!dot || dot->getParentRegion() != &thenRegion ||
      !isZeroSplatConstant(dot.getC()))
    return nullptr;
  return dot;
}

// Recognize idx * BS by value: muli(x, BS) or shli(x, log2(BS)), with x a cast
// chain of the raw index load. Matching by value keeps both spellings and
// non-CSE'd duplicates working.
static std::optional<int64_t> matchIdxTimesBS(Value offset, Value rawIndex) {
  Operation *op = offset.getDefiningOp();
  if (!op)
    return std::nullopt;
  Value other;
  int64_t factor = 0;
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    if (auto c = getConstantIntValue(mul.getRhs())) {
      other = mul.getLhs();
      factor = *c;
    } else if (auto c = getConstantIntValue(mul.getLhs())) {
      other = mul.getRhs();
      factor = *c;
    } else {
      return std::nullopt;
    }
  } else if (auto shift = dyn_cast<arith::ShLIOp>(op)) {
    auto c = getConstantIntValue(shift.getRhs());
    if (!c || *c < 0 || *c > 62)
      return std::nullopt;
    other = shift.getLhs();
    factor = int64_t{1} << *c;
  } else {
    return std::nullopt;
  }
  if (stripIntegerCasts(other) != rawIndex)
    return std::nullopt;
  return factor;
}

// The block's token positions: splat(start) + make_range{0, extent}. Both tile
// spellings spell them exactly this way, so the *shape* has one judgement; what
// `start` must be is the caller's business, because make_tensor_ptr pins it to
// the tile's own offset value while the addptr spelling can only compare the
// product it re-multiplies.
struct TokenRange {
  Value start;
  int64_t extent = 0;
};

// The two addends in the order splat, range. The range may be wrapped in the
// integer width casts the bwd form uses; the builder recomputes the token
// positions, so the cast chain is not carried over.
static std::optional<TokenRange> matchTokenRangeAddends(Value maybeSplat,
                                                        Value maybeRange) {
  auto splat = maybeSplat.getDefiningOp<triton::SplatOp>();
  auto range =
      stripIntegerCasts(maybeRange).getDefiningOp<triton::MakeRangeOp>();
  if (!splat || !range)
    return std::nullopt;
  auto rangeType = dyn_cast<RankedTensorType>(range.getType());
  if (!rangeType || rangeType.getRank() != 1 ||
      !rangeType.getElementType().isInteger(32))
    return std::nullopt;
  if (range.getStart() != 0 || range.getEnd() != rangeType.getShape()[0])
    return std::nullopt;
  return TokenRange{splat.getOperand(), range.getEnd()};
}

// A token vector: the add_i both spellings build it with, in either operand
// order.
static std::optional<TokenRange> matchTokenRange(Value tokenVector) {
  auto add = tokenVector.getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  if (auto range = matchTokenRangeAddends(add.getLhs(), add.getRhs()))
    return range;
  return matchTokenRangeAddends(add.getRhs(), add.getLhs());
}

// The extent of a rank-2 tile on one axis. Both spellings state the tile shape
// the same way - as the type of the loaded block - so the token extent and the
// free extent are one judgement, parameterized by axis.
static std::optional<int64_t> tileExtent(Type type, unsigned axis) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked || axis >= ranked.getRank())
    return std::nullopt;
  return ranked.getShape()[axis];
}

// The rewrite materializes a vector read of S table entries, so S is bounded.
// 4096 is a generous ceiling that still bounds the cost.
constexpr int64_t kMaxTableLength = 4096;

// ===========================================================================
// [1] Facts
// ===========================================================================
//
// One struct per parser. The builder-facing facts are the ones declared in
// SelectedBlockScanFacts.h; the structs here are either matcher-only facts or
// a parser's whole result (its published fact plus the anchors the next parser
// needs). Nothing is shared mutable state: every struct is returned by value
// and read by the parsers that follow.

// The table address as the index load spells it: the pointer the row is offset
// from, the row base when it survived as its own value, and the loop-invariant
// row offset.
struct TableAddress {
  Value root;
  Value rowBase; // null for the spelling that folds the row advance into iv
  Value rowOffset;
};

// Which row of block_indices this program reads: the axis-0 program id the row
// offset is computed from, and the i_h*S slot term next to it (null when the
// row base carries no slot addend).
struct TableRowPosition {
  Value pid;
  Value slotTerm;
};

// The scaled-index guard of the single scf.if.
struct GuardFact {
  scf::IfOp guard;
  Value causalOperand;             // the row side of the causal comparison
  SmallVector<int64_t, 2> factors; // BS, once per guard predicate
};

// The row-position fact. Two sides of one knot: the causal row the block guard
// compares against, and the table row the program reads. Both are derived from
// the same axis-0 program id, so they are resolved here together and every
// later use of the row position is an identity check against this value.
struct RowFact {
  triton::LoadOp indexLoad; // the scalar load the rewrite removes
  Value rawIndex;           // its result: what the guard scales by BS
  Value pid;                // the axis-0 program id
  Value rowPosition;        // pid, or pid + a loop-invariant decode offset
  bool rowFromLoad = false; // the varlen spelling: rowPosition is itself loaded
  scf::IfOp guard;          // the single scf.if the block body hangs from
  SmallVector<int64_t, 2> factors;
  TableAddress address; // the table row this position is crossed with
  Value slotTerm;       // the i_h*S addend of that row, null when absent
};

// The causal comparison. Which side is the row is resolved here, because only
// one side can be a token range; whether the row is on the greater side, and
// whether it is the same value the guard used, are tie-outs and live in
// validateScan.
struct MaskFact {
  arith::CmpIPredicate predicate = arith::CmpIPredicate::eq;
  bool pivotIsLhs = false;
  Value pivot; // the row-position side, peeled
  TokenRange tokens;
};

// The recurrence parser's result: the published variant, plus the causal select
// both templates build their neutral path on. The select is matcher-only (the
// rewriter re-derives the causal comparison from the row position), so it does
// not travel in the fact.
struct RecurrenceMatch {
  RecurrenceFact fact;
  arith::SelectOp maskSelect;
};

// The tile parser's result: the published variant, plus the token start each
// tile is indexed from. The causal mask must be built from that same start;
// again the value stays in the matcher, because the rewriter recomputes the
// token positions itself.
struct TilesMatch {
  TilesFact fact;
  Value kTokenStart;
  Value vTokenStart;
};

// ===========================================================================
// [2] Parsers
// ===========================================================================

// ---- [2.1] The loop ----
//
// One scf.for at the function body's top level, all-tensor iter args, lb 0 and
// step 1: the row lift requires the same shape and the rewrite assumes no other
// loop nest. The carried arity selects the recurrence template (3 fwd, 1 bwd).
static std::optional<LoopFact> parseLoop(triton::FuncOp fn) {
  LoopFact fact;
  unsigned forOps = 0;
  fn.walk([&](scf::ForOp forOp) {
    ++forOps;
    fact.forOp = forOp;
  });
  if (forOps != 1 || !fact.forOp ||
      fact.forOp->getBlock() != &fn.getBody().front())
    return std::nullopt;
  bool foreignControlFlow = false;
  fn.walk([&](Operation *op) {
    if (isa<scf::WhileOp, scf::ParallelOp>(op))
      foreignControlFlow = true;
  });
  fact.carried = fact.forOp.getNumResults();
  if (foreignControlFlow || (fact.carried != 3 && fact.carried != 1))
    return std::nullopt;
  for (Value init : fact.forOp.getInitArgs())
    if (!isa<RankedTensorType>(init.getType()))
      return std::nullopt;
  if (getConstantIntValue(fact.forOp.getLowerBound()) !=
          std::optional<int64_t>(0) ||
      getConstantIntValue(fact.forOp.getStep()) != std::optional<int64_t>(1))
    return std::nullopt;

  // Scalar loads outside the loop: at most one on the dense production path
  // (the dyn-NS block_counts read) and six on a varlen dispatch, seven when
  // both combine. They are loop-invariant and the rewrite keeps them, so this
  // is only a cheap upper bound; the loop-interior budget is the load census.
  constexpr unsigned kMaxOutsideScalarLoads = 7;
  unsigned outsideScalarLoads = 0;
  fn.walk([&](triton::LoadOp load) {
    if (!load->getParentOfType<scf::ForOp>() &&
        !isa<RankedTensorType>(load.getType()))
      ++outsideScalarLoads;
  });
  if (outsideScalarLoads > kMaxOutsideScalarLoads)
    return std::nullopt;
  return fact;
}

// ---- [2.2] The row position ----
//
// The index load, the table address it reads, the axis-0 program id that
// address is built from and the block guard are one quantity, so they are
// resolved together: the table row and the guard row must be the same row.

// The gather load: exactly one scalar load inside the loop, plus exactly two
// tensor loads (the K/V tiles). Anything else means the loop does more than
// gather block indices and load one K/V tile pair for this scan.
static std::optional<triton::LoadOp> findIndexLoad(scf::ForOp loop) {
  triton::LoadOp indexLoad;
  unsigned inLoopLoads = 0;
  unsigned inLoopScalarLoads = 0;
  loop->walk([&](triton::LoadOp load) {
    ++inLoopLoads;
    if (!isa<RankedTensorType>(load.getType())) {
      ++inLoopScalarLoads;
      indexLoad = load;
    }
  });
  if (inLoopLoads != 3 || inLoopScalarLoads != 1)
    return std::nullopt;
  return indexLoad;
}

// The table address, read off the load's pointer operand. Two spellings occur:
//   (a) tt.addptr %rowBase, %iv, with %rowBase = tt.addptr %root, %rowOff
//   (b) tt.addptr %root, arith.addi(%rowOff, %iv)  -- the row advance folded
//   into
//       the in-loop offset (the dyn-NS path, i64 induction variable)
// In both, the per-iteration advance must BE the induction variable: the
// rewrite reads table[0, S) once and indexes it by the iteration number.
static std::optional<TableAddress> parseTableAddress(triton::LoadOp indexLoad,
                                                     scf::ForOp loop) {
  auto addPtr = indexLoad.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!addPtr)
    return std::nullopt;
  Value base = addPtr.getPtr();
  Value advance = addPtr.getOffset();
  Value iv = loop.getInductionVar();
  TableAddress address;
  if (auto rowBase = base.getDefiningOp<triton::AddPtrOp>()) {
    // (a) the row base kept its own addptr
    if (advance != iv)
      return std::nullopt;
    address.root = rowBase.getPtr();
    address.rowBase = base;
    address.rowOffset = rowBase.getOffset();
  } else {
    // (b) the row offset is folded into the in-loop offset: exactly one addend
    // is the induction variable, the other is the loop-invariant row offset.
    auto offsetAdd = advance.getDefiningOp<arith::AddIOp>();
    if (!offsetAdd)
      return std::nullopt;
    Value lhs = offsetAdd.getLhs();
    Value rhs = offsetAdd.getRhs();
    if ((lhs == iv) == (rhs == iv))
      return std::nullopt;
    address.root = base;
    address.rowBase = Value();
    address.rowOffset = lhs == iv ? rhs : lhs;
  }
  // Both halves of the address are per-program constants; only iv moves.
  if (!isDefinedOutsideLoop(address.root, loop) ||
      !isDefinedOutsideLoop(address.rowOffset, loop))
    return std::nullopt;
  return address;
}

// Which row of block_indices this program reads, and the i_h*S slot next to it.
// The query addend carries exactly one axis-0 program id: none means it is not
// a query row, two means the row, the causal predicate and the mask can drift
// apart. Both are refused rather than guessed.
static std::optional<TableRowPosition>
parseTableRowPosition(Value rowOffset, triton::FuncOp fn) {
  SmallVector<Value, 4> axis0Pids;
  fn.walk([&](triton::GetProgramIdOp op) {
    if (op.getAxisAsInt() == 0)
      axis0Pids.push_back(op.getResult());
  });
  // The single axis-0 program id v is built from, or nullopt when it is built
  // from none or from more than one.
  auto uniquePidReaching = [&](Value v) -> std::optional<Value> {
    Value found;
    for (Value pid : axis0Pids) {
      if (!dependsOn(v, pid))
        continue;
      if (found)
        return std::nullopt;
      found = pid;
    }
    if (!found)
      return std::nullopt; // none reaches: not a query-position dependency
    return found;
  };
  if (auto offsetAdd = rowOffset.getDefiningOp<arith::AddIOp>()) {
    std::optional<Value> lhsPid = uniquePidReaching(offsetAdd.getLhs());
    std::optional<Value> rhsPid = uniquePidReaching(offsetAdd.getRhs());
    // Exactly one addend carries the query position; the other is the i_h*S
    // term of the table layout. Ambiguous on either side is refused as well.
    if (lhsPid.has_value() == rhsPid.has_value())
      return std::nullopt;
    return TableRowPosition{lhsPid ? *lhsPid : *rhsPid,
                            lhsPid ? offsetAdd.getRhs() : offsetAdd.getLhs()};
  }
  std::optional<Value> pid = uniquePidReaching(rowOffset);
  if (!pid)
    return std::nullopt;
  return TableRowPosition{*pid, Value()};
}

// The causal query position: the axis-0 program id, or the program id advanced
// by a loop-invariant decode offset (Q_OFFSET = TK - TQ). The varlen dispatch
// loads the in-sequence position instead, so any loop-invariant integer is
// accepted; the three uses must still be the very same SSA value, which
// validateScan checks.
static Value matchRowPosition(Value v, Value pid, scf::ForOp loop) {
  Value stripped = stripIntegerCasts(v);
  if (stripped == pid)
    return pid;
  if (auto add = stripped.getDefiningOp<arith::AddIOp>()) {
    Value lhs = stripIntegerCasts(add.getLhs());
    Value rhs = stripIntegerCasts(add.getRhs());
    Value offset;
    if (lhs == pid)
      offset = rhs;
    else if (rhs == pid)
      offset = lhs;
    if (offset) {
      // The offset may itself be load-derived as long as it is loop-invariant:
      // the varlen dispatch derives Q_OFFSET = TK - TQ from the cu_seqlens
      // loads before the loop, and the rewrite only needs ONE loop-invariant
      // scalar shared by the causal mask, the table row and the scan bound.
      // The addend being invariant is not enough: the add itself has to be,
      // otherwise the rewriter hoists a loop-interior value into the prologue.
      if (!isDefinedOutsideLoop(offset, loop) ||
          !isDefinedOutsideLoop(stripped, loop))
        return Value();
      return stripped;
    }
  }
  if (!isa<IntegerType>(stripped.getType()) ||
      !isDefinedOutsideLoop(stripped, loop))
    return Value();
  return stripped;
}

// The block guard: the single scf.if of the loop body, whose condition is
// andi(causal, range) over the SCALED index. Direction flips and operand order
// are free; comparisons on the unscaled index do not match. The causal operand
// is recorded, not resolved: the row position it names is the program id's
// business, and that is the cross parseRowPosition performs.
static std::optional<GuardFact> parseGuard(scf::ForOp loop, Value rawIndex) {
  GuardFact fact;
  unsigned guards = 0;
  for (Operation &op : loop.getBody()->without_terminator())
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      fact.guard = ifOp;
      ++guards;
    }
  if (guards != 1)
    return std::nullopt;
  auto predicateAnd = fact.guard.getCondition().getDefiningOp<arith::AndIOp>();
  if (!predicateAnd)
    return std::nullopt;
  // One judgement per comparison: bring the scaled index to the left, then the
  // two accepted predicates read the same way whatever order the front end
  // wrote them in. The factor is recorded on the way, because every block
  // predicate has to scale the index by the tile's own BS.
  bool causalFound = false;
  bool rangeFound = false;
  for (Value operand : {predicateAnd.getLhs(), predicateAnd.getRhs()}) {
    auto cmp = operand.getDefiningOp<arith::CmpIOp>();
    if (!cmp || isa<RankedTensorType>(cmp.getLhs().getType()))
      return std::nullopt;
    std::optional<int64_t> factor = matchIdxTimesBS(cmp.getLhs(), rawIndex);
    Value other = cmp.getRhs();
    arith::CmpIPredicate pred = cmp.getPredicate();
    if (!factor) {
      factor = matchIdxTimesBS(cmp.getRhs(), rawIndex);
      other = cmp.getLhs();
      pred = flippedPredicate(cmp.getPredicate());
    }
    if (!factor || *factor <= 0)
      return std::nullopt;
    fact.factors.push_back(*factor);
    if (pred == arith::CmpIPredicate::sle) { // i_s <= row
      if (causalFound)
        return std::nullopt;
      fact.causalOperand = other;
      causalFound = true;
      continue;
    }
    std::optional<int64_t> zero = getConstantIntValue(other);
    if (pred == arith::CmpIPredicate::sge && zero && *zero == 0) { // 0 <= i_s
      rangeFound = true;
      continue;
    }
    return std::nullopt;
  }
  if (!causalFound || !rangeFound)
    return std::nullopt;
  return fact;
}

static std::optional<RowFact> parseRowPosition(triton::FuncOp fn,
                                               const LoopFact &loop) {
  scf::ForOp forOp = loop.forOp;
  std::optional<triton::LoadOp> indexLoad = findIndexLoad(forOp);
  if (!indexLoad)
    return std::nullopt;
  std::optional<TableAddress> address = parseTableAddress(*indexLoad, forOp);
  if (!address)
    return std::nullopt;
  // The table side of the cross: which program's row this is.
  std::optional<TableRowPosition> position =
      parseTableRowPosition(address->rowOffset, fn);
  if (!position)
    return std::nullopt;
  Value rawIndex = indexLoad->getResult();
  std::optional<GuardFact> guard = parseGuard(forOp, rawIndex);
  if (!guard)
    return std::nullopt;
  // The guard side of the cross: the row the causal predicate advances.
  Value rowPosition =
      matchRowPosition(guard->causalOperand, position->pid, forOp);
  if (!rowPosition)
    return std::nullopt;
  RowFact fact;
  fact.indexLoad = *indexLoad;
  fact.rawIndex = rawIndex;
  fact.pid = position->pid;
  fact.rowPosition = rowPosition;
  // The varlen dispatch loads the in-sequence row position; the dot-operand
  // split then has to treat loop-invariant loads as per-program constants.
  // Recorded once here, so no later stage re-derives it.
  fact.rowFromLoad = reachesLoad(rowPosition);
  fact.guard = guard->guard;
  fact.factors = std::move(guard->factors);
  fact.address = *address;
  fact.slotTerm = position->slotTerm;
  return fact;
}

// ---- [2.3] The index table ----
//
// The row length S is a host-side shape, so it comes from the i_h*S slot addend
// of the row base, or from a constant trip count (the case NS == S), or - only
// under H == 1 - from the row stride multiplier. loopCarriesHeadSplit says
// which.

// True when the function still computes a head index `i_bh % H` with H > 1. The
// remainder survives constant folding exactly when the divisor is greater than
// one, so its absence is the layout's own statement that H == 1.
static bool loopCarriesHeadSplit(triton::FuncOp fn) {
  bool found = false;
  fn.walk([&](Operation *op) {
    if (found || !isa<arith::RemUIOp, arith::RemSIOp>(op))
      return;
    if (std::optional<int64_t> divisor = getConstantIntValue(op->getOperand(1));
        divisor && *divisor > 1)
      found = true;
  });
  return found;
}

// The constant multiplier of v: muli(x, C) or muli(C, x). The front end may
// widen the product (extsi(muli(i_h, S))), so the integer width casts are
// peeled first.
static std::optional<int64_t> matchTimesConstant(Value v) {
  auto mul = stripIntegerCasts(v).getDefiningOp<arith::MulIOp>();
  if (!mul)
    return std::nullopt;
  if (auto rhs = getConstantIntValue(mul.getRhs()))
    return rhs;
  return getConstantIntValue(mul.getLhs());
}

static std::optional<int64_t> deriveTableLength(scf::ForOp loop,
                                                triton::FuncOp fn,
                                                Value rowOffset,
                                                Value slotTerm) {
  if (slotTerm) {
    // S is the STATIC row length of the index table: the host passes
    // block_indices as [i_h, S], so the i_h*S multiplier of the row base IS S.
    return matchTimesConstant(slotTerm);
  }
  // No i_h*S addend: the layout is ((bos + i_t) * H * S) + i_h * S and the slot
  // addend folds away exactly when H == 1, so the row stride H*S becomes S.
  // With H > 1 the same constant is H*S and a read of that width would run past
  // the row, so S is only read from it when the layout carries no head split.
  if (auto trip = getConstantIntValue(loop.getUpperBound()))
    return *trip;
  if (loopCarriesHeadSplit(fn))
    return std::nullopt;
  return matchTimesConstant(rowOffset);
}

static std::optional<TableFact>
parseIndexTable(triton::FuncOp fn, const LoopFact &loop, const RowFact &row) {
  scf::ForOp forOp = loop.forOp;
  Value root = row.address.root;
  // The table entries must be wide enough for the i32 token arithmetic below;
  // production index tables are i64.
  auto tablePtrType = dyn_cast<triton::PointerType>(root.getType());
  if (!tablePtrType)
    return std::nullopt;
  auto tableElem = dyn_cast<IntegerType>(tablePtrType.getPointeeType());
  if (!tableElem || tableElem.getWidth() < 32)
    return std::nullopt;
  // The rewrite merges the per-iteration loads into one vector read, so a
  // masked, "other"-padded or volatile entry load is not this pattern.
  if (!hasLoadShape(row.indexLoad, LoadShape::Unmasked))
    return std::nullopt;
  std::optional<int64_t> length =
      deriveTableLength(forOp, fn, row.address.rowOffset, row.slotTerm);
  if (!length)
    return std::nullopt;
  // S drives a vector read of S table entries, so an unbounded S would
  // materialize an arbitrarily wide tensor read.
  if (*length <= 0 || *length > kMaxTableLength)
    return std::nullopt;
  TableFact fact;
  fact.base = row.address.rowBase;
  fact.root = root;
  fact.rowOffset = row.address.rowOffset;
  fact.length = *length;
  return fact;
}

// ---- [2.4] The recurrence ----
//
// The shape of the recurrence selects the variant the rewriter emits (online
// softmax: o/m/l driven by maxnumf; additive: one accumulator, neutral 0). The
// else region yields the iter args unchanged, which is what lets the rewrite
// replace the branch with a zero weight.

// Facts the online-softmax walk discovers phase by phase. Phase 1 finds the
// masked score tile and its row max, phase 2 turns them into the rescale factor
// and the exponentials, phase 3 rebuilds the value dot from those.
struct SoftmaxWalk {
  Value maskedScores;         // [G, BS] f32: select(mask, dot(q, k), -inf)
  arith::SelectOp maskSelect; // that select
  triton::DotOp scoreDot;     // dot(q, k, 0)
  Value alpha;                // exp(m_old - m_new)
  Value pExp;                 // exp(masked_scores - m_new)
  triton::DotOp valueDot;     // dot(cast(p), V, o * alpha)
};

// The row reduction of the guarded region, with the combiner the template
// requires - a max over the scores, a sum over the exponentials. Any other
// reduction (another axis, another combiner) is not this pattern.
static triton::ReduceOp matchRowReduce(Value v, Region &thenRegion,
                                       llvm::StringRef combinerName) {
  auto reduce = v.getDefiningOp<triton::ReduceOp>();
  if (!reduce || reduce.getAxis() != 1 ||
      reduce->getParentRegion() != &thenRegion)
    return nullptr;
  Operation *combiner = reduce.getSingleCombiner();
  if (!combiner || combiner->getName().getStringRef() != combinerName)
    return nullptr;
  return reduce;
}

// m_new = maxnumf(m_old, tile_max); tile_max = reduce{maxnumf}(masked scores);
// masked scores = select(mask, dot(q, k), -inf).
static bool parseSoftmaxMax(Region &thenRegion, BlockArgument mArg, Value mNew,
                            SoftmaxWalk &walk) {
  auto mUpdate = mNew.getDefiningOp<arith::MaxNumFOp>();
  if (!mUpdate)
    return false;
  Value tileMax;
  if (mUpdate.getLhs() == mArg)
    tileMax = mUpdate.getRhs();
  else if (mUpdate.getRhs() == mArg)
    tileMax = mUpdate.getLhs();
  else
    return false;
  triton::ReduceOp tileReduce =
      matchRowReduce(tileMax, thenRegion, arith::MaxNumFOp::getOperationName());
  if (!tileReduce)
    return false;
  walk.maskedScores = tileReduce->getOperand(0);
  walk.maskSelect = walk.maskedScores.getDefiningOp<arith::SelectOp>();
  if (!walk.maskSelect ||
      !isNegativeInfSplatConstant(walk.maskSelect.getFalseValue()))
    return false;
  walk.scoreDot = matchFreshDot(walk.maskSelect.getTrueValue(), thenRegion);
  return static_cast<bool>(walk.scoreDot);
}

// alpha = exp(m_old - m_new); l = l * alpha + sum(p); p = exp(masked - m_new).
// The subtraction order is semantic, everything else is checked structurally.
static bool parseSoftmaxNorm(Region &thenRegion, BlockArgument mArg,
                             BlockArgument lArg, Value lNew, Value mNew,
                             SoftmaxWalk &walk) {
  auto lUpdate = lNew.getDefiningOp<arith::AddFOp>();
  if (!lUpdate)
    return false;
  auto mulWithOperand = [](Value v, Value operand, Value &other) {
    auto mul = v.getDefiningOp<arith::MulFOp>();
    if (!mul)
      return false;
    if (mul.getLhs() == operand) {
      other = mul.getRhs();
      return true;
    }
    if (mul.getRhs() == operand) {
      other = mul.getLhs();
      return true;
    }
    return false;
  };
  Value alpha;
  Value lSum;
  if (mulWithOperand(lUpdate.getLhs(), lArg, alpha))
    lSum = lUpdate.getRhs();
  else if (mulWithOperand(lUpdate.getRhs(), lArg, alpha))
    lSum = lUpdate.getLhs();
  else
    return false;
  auto alphaExp = alpha.getDefiningOp<math::ExpOp>();
  if (!alphaExp)
    return false;
  auto alphaSub = alphaExp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!alphaSub || alphaSub.getLhs() != mArg || alphaSub.getRhs() != mNew)
    return false;
  triton::ReduceOp sumReduce =
      matchRowReduce(lSum, thenRegion, arith::AddFOp::getOperationName());
  if (!sumReduce)
    return false;
  Value pExp = sumReduce->getOperand(0);
  auto pExpOp = pExp.getDefiningOp<math::ExpOp>();
  if (!pExpOp)
    return false;
  auto pSub = pExpOp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!pSub || pSub.getLhs() != walk.maskedScores)
    return false;
  if (!isBroadcastOf(pSub.getRhs(), mNew))
    return false;
  walk.alpha = alpha;
  walk.pExp = pExp;
  return true;
}

// o = dot(cast(p), V, o * alpha). The cast direction is recorded, not
// re-derived: the rewriter re-emits the same cast.
static bool parseSoftmaxValue(Region &thenRegion, BlockArgument oArg,
                              Value oNew, SoftmaxWalk &walk,
                              bool &pCastIsExtend) {
  auto valueDot = oNew.getDefiningOp<triton::DotOp>();
  if (!valueDot || valueDot->getParentRegion() != &thenRegion)
    return false;
  bool isExtend = false;
  if (Value p = peelFloatCast(valueDot.getA(), isExtend); p != walk.pExp)
    return false;
  pCastIsExtend = isExtend;
  auto oScale = valueDot.getC().getDefiningOp<arith::MulFOp>();
  if (!oScale)
    return false;
  if (!((oScale.getLhs() == oArg &&
         isBroadcastOf(oScale.getRhs(), walk.alpha)) ||
        (oScale.getRhs() == oArg &&
         isBroadcastOf(oScale.getLhs(), walk.alpha))))
    return false;
  walk.valueDot = valueDot;
  return true;
}

// The (o, m, l) roles are recognised by the op that produces them, so a kernel
// that carries them in another order still matches; the roles are then replayed
// in iter-argument order.
static std::optional<RecurrenceMatch>
parseOnlineSoftmaxRecurrence(scf::ForOp loop, scf::IfOp branch,
                             scf::YieldOp thenYield) {
  constexpr unsigned kAbsent = 3;
  unsigned oArg = kAbsent;
  unsigned mArg = kAbsent;
  unsigned lArg = kAbsent;
  for (unsigned i = 0; i < 3; ++i) {
    Value yielded = thenYield->getOperand(i);
    if (yielded.getDefiningOp<arith::MaxNumFOp>() && mArg == kAbsent)
      mArg = i;
    else if (yielded.getDefiningOp<arith::AddFOp>() && lArg == kAbsent)
      lArg = i;
    else if (yielded.getDefiningOp<triton::DotOp>() && oArg == kAbsent)
      oArg = i;
  }
  if (mArg == kAbsent || lArg == kAbsent || oArg == kAbsent)
    return std::nullopt;
  BlockArgument oIterArg = loop.getRegionIterArgs()[oArg];
  BlockArgument mIterArg = loop.getRegionIterArgs()[mArg];
  BlockArgument lIterArg = loop.getRegionIterArgs()[lArg];
  Region &thenRegion = branch.getThenRegion();
  SoftmaxWalk walk;
  if (!parseSoftmaxMax(thenRegion, mIterArg, thenYield->getOperand(mArg), walk))
    return std::nullopt;
  if (!parseSoftmaxNorm(thenRegion, mIterArg, lIterArg,
                        thenYield->getOperand(lArg),
                        thenYield->getOperand(mArg), walk))
    return std::nullopt;
  OnlineSoftmaxRecurrence fact;
  fact.oArg = oArg;
  fact.mArg = mArg;
  fact.lArg = lArg;
  if (!parseSoftmaxValue(thenRegion, oIterArg, thenYield->getOperand(oArg),
                         walk, fact.pCastIsExtend))
    return std::nullopt;
  fact.scoreDot = walk.scoreDot;
  fact.valueDot = walk.valueDot;
  return RecurrenceMatch{fact, walk.maskSelect};
}

// acc' = dot(ds, k^T, acc): the contribution lands in the carried accumulator
// itself. The recurrence is read backwards from that dot.
static std::optional<RecurrenceMatch>
parseAdditiveRecurrence(scf::ForOp loop, Region &thenRegion,
                        BlockArgument accArg, Value yielded) {
  triton::DotOp accDot = yielded.getDefiningOp<triton::DotOp>();
  if (!accDot || accDot->getParentRegion() != &thenRegion ||
      accDot.getC() != accArg)
    return std::nullopt;
  // k^T: a transpose of the K tile. Its order decides where the token axis
  // lands, so it is stored and replayed verbatim.
  auto transpose = accDot.getB().getDefiningOp<triton::TransOp>();
  if (!transpose || transpose->getParentRegion() != &thenRegion)
    return std::nullopt;
  auto kLoad = transpose.getSrc().getDefiningOp<triton::LoadOp>();
  if (!kLoad || kLoad->getParentRegion() != &thenRegion)
    return std::nullopt;
  // ds = [cast](select(causal, p, 0) * (dp - delta)): the causal select times
  // the delta-corrected value gradient.
  AdditiveRecurrence fact;
  Value ds = peelFloatCast(accDot.getA(), fact.dsCastIsExtend);
  auto dsMul = ds.getDefiningOp<arith::MulFOp>();
  if (!dsMul)
    return std::nullopt;
  auto lhsSelect = dsMul.getLhs().getDefiningOp<arith::SelectOp>();
  auto rhsSelect = dsMul.getRhs().getDefiningOp<arith::SelectOp>();
  if (static_cast<bool>(lhsSelect) == static_cast<bool>(rhsSelect))
    return std::nullopt;
  arith::SelectOp maskSelect = lhsSelect ? lhsSelect : rhsSelect;
  auto dsSub = (lhsSelect ? dsMul.getRhs() : dsMul.getLhs())
                   .getDefiningOp<arith::SubFOp>();
  if (!dsSub)
    return std::nullopt;
  // The neutral path is select(causal, p, 0): the additive form's neutral
  // element is zero (a -inf mask belongs to the online-softmax form).
  if (!isZeroSplatConstant(maskSelect.getFalseValue()))
    return std::nullopt;
  // p = exp(s - lse), lse a loop-invariant [G] row vector read outside.
  auto pExp = maskSelect.getTrueValue().getDefiningOp<math::ExpOp>();
  if (!pExp)
    return std::nullopt;
  auto scoreSub = pExp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!scoreSub)
    return std::nullopt;
  triton::DotOp scoreDot = matchFreshDot(scoreSub.getLhs(), thenRegion);
  if (!scoreDot)
    return std::nullopt;
  Value lseVec = peelBroadcasts(scoreSub.getRhs());
  if (!isDefinedOutsideLoop(lseVec, loop))
    return std::nullopt;
  // dp = dot(dO, v, 0), and delta the same kind of loop-invariant row vector.
  triton::DotOp valueDot = matchFreshDot(dsSub.getLhs(), thenRegion);
  if (!valueDot)
    return std::nullopt;
  Value deltaVec = peelBroadcasts(dsSub.getRhs());
  if (!isDefinedOutsideLoop(deltaVec, loop))
    return std::nullopt;
  // The score dot's rhs IS the transposed tile; the value dot reads the other.
  auto vLoad = valueDot.getB().getDefiningOp<triton::LoadOp>();
  if (scoreDot.getB() != transpose.getSrc() || !vLoad ||
      vLoad->getParentRegion() != &thenRegion)
    return std::nullopt;
  // Both dots' lhs are loop-invariant operands (Q*scale and dO), and the two
  // row vectors have the accumulator's row count.
  if (!isDefinedOutsideLoop(scoreDot.getA(), loop) ||
      !isDefinedOutsideLoop(valueDot.getA(), loop))
    return std::nullopt;
  auto accType = cast<RankedTensorType>(accArg.getType());
  auto isRowVector = [&](Value v) {
    auto type = dyn_cast<RankedTensorType>(v.getType());
    return type && type.getRank() == 1 && type.getElementType().isF32() &&
           type.getShape()[0] == accType.getShape()[0];
  };
  if (accType.getRank() != 2 || !isRowVector(lseVec) || !isRowVector(deltaVec))
    return std::nullopt;
  fact.scoreDot = scoreDot;
  fact.valueDot = valueDot;
  fact.accDot = accDot;
  fact.transpose = transpose;
  fact.lseVec = lseVec;
  fact.deltaVec = deltaVec;
  return RecurrenceMatch{fact, maskSelect};
}

static std::optional<RecurrenceMatch> parseRecurrence(const LoopFact &loop,
                                                      const RowFact &row) {
  scf::ForOp forOp = loop.forOp;
  scf::IfOp branch = row.guard;
  const unsigned numCarried = loop.carried;
  if (branch.getElseRegion().empty() ||
      !llvm::hasSingleElement(branch.getElseRegion()))
    return std::nullopt;
  auto elseYield =
      dyn_cast<scf::YieldOp>(branch.getElseRegion().front().getTerminator());
  if (!elseYield || elseYield->getNumOperands() != numCarried)
    return std::nullopt;
  for (unsigned i = 0; i < numCarried; ++i)
    if (elseYield->getOperand(i) != forOp.getRegionIterArgs()[i])
      return std::nullopt;
  if (branch->getNumResults() != numCarried)
    return std::nullopt;
  auto latchYield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!latchYield || latchYield->getNumOperands() != numCarried)
    return std::nullopt;
  for (unsigned i = 0; i < numCarried; ++i)
    if (latchYield->getOperand(i) != branch->getResult(i))
      return std::nullopt;
  if (branch.getThenRegion().empty() ||
      !llvm::hasSingleElement(branch.getThenRegion()))
    return std::nullopt;
  Block &thenBlock = branch.getThenRegion().front();
  auto thenYield = dyn_cast<scf::YieldOp>(thenBlock.getTerminator());
  if (!thenYield || thenYield->getNumOperands() != numCarried)
    return std::nullopt;
  if (numCarried == 3)
    return parseOnlineSoftmaxRecurrence(forOp, branch, thenYield);
  // The recurrence is read backwards from the dot that accumulates into the
  // carried state.
  return parseAdditiveRecurrence(forOp, branch.getThenRegion(),
                                 forOp.getRegionIterArgs()[0],
                                 thenYield->getOperand(0));
}

// ---- [2.5] The tiles ----
//
// One K tile and one V tile per iteration, in one of two spellings:
//   make_tensor_ptr  the token axis is an offset of the view, boundary-checked
//   addptr           two nested tt.addptr's, explicit mask and other = 0
// The additive form may arrive in either (production nsa_bwd_dq uses the
// first).

// Finds the single mptr offset that scales the block index and records the
// block size it implies; blockSize is only compared once it is known.
static bool identifyTileAxis(triton::MakeTensorPtrOp ptr, Value rawIndex,
                             int64_t &blockSize, unsigned &tokenAxis,
                             Value &tokenOffset) {
  unsigned found = 0;
  for (unsigned axis = 0; axis < ptr.getOffsets().size(); ++axis) {
    Value offset = ptr.getOffsets()[axis];
    if (!dependsOn(offset, rawIndex))
      continue;
    ++found;
    tokenAxis = axis;
    auto factor = matchIdxTimesBS(offset, rawIndex);
    if (!factor || *factor <= 0)
      return false;
    if (blockSize && *factor != blockSize)
      return false;
    blockSize = *factor;
    tokenOffset = offset;
  }
  return found == 1;
}

// No load may feed a *loop-interior* address expression. A loop-invariant
// operand may be load-derived: the varlen dispatch folds the sequence base
// (bos_k) into the K/V pointer before the loop, and the rewrite reuses that
// value verbatim. The only loop-carried address computation accepted is the
// sign extension of the dynamic K/V bound, which is loop-invariant and
// re-created outside the loop.
static bool checkAddressOperand(Value v, scf::ForOp loop,
                                Value &dynamicBoundResult,
                                Value &dynamicBoundInput) {
  Operation *def = v.getDefiningOp();
  if (!def || !loop->isAncestor(def))
    return true; // defined outside: reused verbatim, even if load-derived
  if (reachesLoopInteriorLoad(v, loop))
    return false;
  auto ext = dyn_cast<arith::ExtSIOp>(def);
  if (!ext || !isDefinedOutsideLoop(ext.getIn(), loop))
    return false;
  if (dynamicBoundResult && dynamicBoundResult != ext.getResult())
    return false;
  dynamicBoundResult = ext.getResult();
  dynamicBoundInput = ext.getIn();
  return true;
}

// Every operand of the view's address chain except the token-axis offset, which
// is the one that has to move with the block index.
static bool checkTileAddress(triton::MakeTensorPtrOp ptr, unsigned tokenAxis,
                             scf::ForOp loop, Value &dynamicBoundResult,
                             Value &dynamicBoundInput) {
  if (!checkAddressOperand(ptr.getBase(), loop, dynamicBoundResult,
                           dynamicBoundInput))
    return false;
  for (Value v : llvm::concat<Value>(ptr.getShape(), ptr.getStrides()))
    if (!checkAddressOperand(v, loop, dynamicBoundResult, dynamicBoundInput))
      return false;
  for (unsigned axis = 0; axis < ptr.getOffsets().size(); ++axis)
    if (axis != tokenAxis &&
        !checkAddressOperand(ptr.getOffsets()[axis], loop, dynamicBoundResult,
                             dynamicBoundInput))
      return false;
  return true;
}

static std::optional<TilesMatch> parseMptrTiles(scf::ForOp loop, Value rawIndex,
                                                triton::LoadOp kLoad,
                                                triton::LoadOp vLoad) {
  // The rewrite loads each tile unconditionally, so a per-element mask, an
  // "other" padding value or a volatile qualifier on the tile loads is out of
  // scope.
  if (!hasLoadShape(kLoad, LoadShape::Unmasked) ||
      !hasLoadShape(vLoad, LoadShape::Unmasked))
    return std::nullopt;
  MptrTiles fact;
  fact.kLoad = kLoad;
  fact.vLoad = vLoad;
  fact.kPtr = kLoad.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
  fact.vPtr = vLoad.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
  if (!fact.kPtr || !fact.vPtr)
    return std::nullopt;
  Value kTokenOffset;
  Value vTokenOffset;
  if (!identifyTileAxis(fact.kPtr, rawIndex, fact.blockSize, fact.kTokenAxis,
                        kTokenOffset) ||
      !identifyTileAxis(fact.vPtr, rawIndex, fact.blockSize, fact.vTokenAxis,
                        vTokenOffset))
    return std::nullopt;
  // The tile extent on the token axis must equal BS: the rewrite pins the
  // tile's token extent to BN, so a narrower or wider tile would change how
  // many tokens a selected block accumulates, and a dynamic extent cannot be
  // proven equal to BN at all.
  std::optional<int64_t> kExtent = tileExtent(kLoad.getType(), fact.kTokenAxis);
  std::optional<int64_t> vExtent = tileExtent(vLoad.getType(), fact.vTokenAxis);
  if (!kExtent || *kExtent != fact.blockSize || !vExtent ||
      *vExtent != fact.blockSize)
    return std::nullopt;
  // mptr offsets are i32, so the token offsets are i32 by construction.
  if (!kTokenOffset.getType().isInteger(32) ||
      !vTokenOffset.getType().isInteger(32))
    return std::nullopt;
  if (!checkTileAddress(fact.kPtr, fact.kTokenAxis, loop,
                        fact.dynamicBoundResult, fact.dynamicBoundInput) ||
      !checkTileAddress(fact.vPtr, fact.vTokenAxis, loop,
                        fact.dynamicBoundResult, fact.dynamicBoundInput))
    return std::nullopt;
  return TilesMatch{fact, kTokenOffset, vTokenOffset};
}

// A value the loop cannot change and that reads no memory: the additive tile's
// base, its free axis and its two mask bounds are all of this kind.
static bool isLoopInvariantValue(Value v, scf::ForOp loop) {
  return isDefinedOutsideLoop(v, loop) && !reachesLoad(v);
}

// The upper bound of `pivot < bound`, or of the same comparison written the
// other way round; a null value when the pivot is not the bounded side.
static Value matchUpperBound(arith::CmpIOp cmp, Value pivot) {
  if (cmp.getLhs() == pivot && cmp.getPredicate() == arith::CmpIPredicate::slt)
    return cmp.getRhs();
  if (cmp.getRhs() == pivot && cmp.getPredicate() == arith::CmpIPredicate::sgt)
    return cmp.getLhs();
  return Value();
}

// One additive-form tile, decoded from the two vectors the baseline built both
// halves from:
//
//   address  addptr(broadcast(addptr(splat(base), expand_dims(free, 1))),
//                   broadcast(expand_dims(tokens, 0) * stride))
//   mask     (free < F) & (tokens < T), one broadcast row each
//
// The token-range term of the mask is the only guard that keeps the read inside
// the row's token count, so it is carried, not rebuilt.
static bool parseAddPtrTile(triton::LoadOp load, Value rawIndex,
                            scf::ForOp loop, AddPtrTile &tile) {
  // The gather version reads exactly one block, so this spelling always carries
  // an explicit mask with other = 0 and never an mptr boundary check: the
  // rewriter reproduces the mask, not the boundary check.
  if (!hasLoadShape(load, LoadShape::MaskedZeroOther))
    return false;
  auto block = dyn_cast<RankedTensorType>(load.getType());
  if (!block || block.getRank() != 2)
    return false;

  // ---- the address ----
  auto add = load.getPtr().getDefiningOp<triton::AddPtrOp>();
  auto ptrBroadcast = add ? add.getPtr().getDefiningOp<triton::BroadcastOp>()
                          : triton::BroadcastOp();
  auto offBroadcast = add ? add.getOffset().getDefiningOp<triton::BroadcastOp>()
                          : triton::BroadcastOp();
  if (!ptrBroadcast || !offBroadcast)
    return false;
  // Token term: expand_dims(tokens, 0) * stride, the stride one constant.
  auto tokenMul = offBroadcast.getSrc().getDefiningOp<arith::MulIOp>();
  if (!tokenMul)
    return false;
  std::optional<int64_t> stride = getSplatIntTensorValue(tokenMul.getRhs());
  Value tokenTerm = tokenMul.getLhs();
  if (!stride) {
    stride = getSplatIntTensorValue(tokenMul.getLhs());
    tokenTerm = tokenMul.getRhs();
  }
  if (!stride || *stride <= 0)
    return false;
  auto tokenExpanded = tokenTerm.getDefiningOp<triton::ExpandDimsOp>();
  if (!tokenExpanded || tokenExpanded.getAxis() != 0)
    return false;
  Value tokens = tokenExpanded.getSrc();
  auto tokenVector = dyn_cast<RankedTensorType>(tokens.getType());
  if (!tokenVector || tokenVector.getRank() != 1)
    return false;
  auto tokenInt = dyn_cast<IntegerType>(tokenVector.getElementType());
  if (!tokenInt || tokenInt.getWidth() < 32)
    return false;
  // tokens = splat(idx*BS) + range{0, BS}: the block's token start and the
  // tile's token extent are the product the block guard scales by.
  std::optional<TokenRange> range = matchTokenRange(tokens);
  std::optional<int64_t> factor =
      range ? matchIdxTimesBS(range->start, rawIndex) : std::nullopt;
  if (!factor || *factor <= 0 || *factor != range->extent ||
      block.getShape()[1] != *factor)
    return false;
  tile.blockStart = range->start;
  tile.blockSize = *factor;
  // Free term: addptr(splat(base), expand_dims(free, 1)).
  auto inner = ptrBroadcast.getSrc().getDefiningOp<triton::AddPtrOp>();
  auto freeExpanded =
      inner ? inner.getOffset().getDefiningOp<triton::ExpandDimsOp>()
            : triton::ExpandDimsOp();
  auto baseSplat = inner ? inner.getPtr().getDefiningOp<triton::SplatOp>()
                         : triton::SplatOp();
  if (!freeExpanded || freeExpanded.getAxis() != 1 || !baseSplat ||
      !isa<triton::PointerType>(baseSplat.getSrc().getType()) ||
      !isLoopInvariantValue(baseSplat.getSrc(), loop) ||
      !isLoopInvariantValue(freeExpanded.getSrc(), loop))
    return false;
  tile.base = baseSplat.getSrc();
  tile.freeVec = freeExpanded.getSrc();
  tile.freeExtent = block.getShape()[0];
  tile.tokenStride = *stride;
  tile.tokenWidth = tokenInt.getWidth();

  // ---- the mask ----
  auto maskAnd = load.getMask().getDefiningOp<arith::AndIOp>();
  if (!maskAnd)
    return false;
  Value maskLhs = peelSingleBroadcast(maskAnd.getLhs());
  Value maskRhs = peelSingleBroadcast(maskAnd.getRhs());
  if (!maskLhs || !maskRhs)
    return false;
  // Splitting the two rows by "does it depend on the token block start" keeps
  // the token bound next to the address it protects.
  bool lhsIsToken = dependsOn(maskLhs, tile.blockStart);
  if (lhsIsToken == dependsOn(maskRhs, tile.blockStart))
    return false;
  Value tokenRow = lhsIsToken ? maskLhs : maskRhs;
  Value freeRow = lhsIsToken ? maskRhs : maskLhs;
  // The token row lifts the address's own token vector and bounds it by a
  // loop-invariant scalar.
  auto tokenLift = tokenRow.getDefiningOp<triton::ExpandDimsOp>();
  auto tokenCmp = tokenLift ? tokenLift.getSrc().getDefiningOp<arith::CmpIOp>()
                            : arith::CmpIOp();
  if (!tokenLift || tokenLift.getAxis() != 0 || !tokenCmp)
    return false;
  Value tokenBound = matchUpperBound(tokenCmp, tokens);
  auto boundSplat = tokenBound ? tokenBound.getDefiningOp<triton::SplatOp>()
                               : triton::SplatOp();
  if (!boundSplat || !isLoopInvariantValue(boundSplat.getSrc(), loop))
    return false;
  tile.tokenBound = boundSplat.getSrc();
  // The free row compares the free offset the address adds in: a shifted or
  // rescaled bound would silently drop the padding rows.
  auto freeCmp = freeRow.getDefiningOp<arith::CmpIOp>();
  if (!freeCmp)
    return false;
  tile.freeBoundRhs = matchUpperBound(freeCmp, freeExpanded.getResult());
  return tile.freeBoundRhs && isLoopInvariantValue(tile.freeBoundRhs, loop);
}

static std::optional<TilesMatch> parseAddPtrTiles(scf::ForOp loop,
                                                  Value rawIndex,
                                                  triton::LoadOp kLoad,
                                                  triton::LoadOp vLoad) {
  AddPtrTiles fact;
  fact.kLoad = kLoad;
  fact.vLoad = vLoad;
  if (!parseAddPtrTile(kLoad, rawIndex, loop, fact.kTile) ||
      !parseAddPtrTile(vLoad, rawIndex, loop, fact.vTile))
    return std::nullopt;
  fact.blockSize = fact.kTile.blockSize;
  return TilesMatch{fact, fact.kTile.blockStart, fact.vTile.blockStart};
}

static std::optional<TilesMatch> parseTiles(const LoopFact &loop,
                                            const RowFact &row,
                                            const RecurrenceMatch &recurrence) {
  scf::ForOp forOp = loop.forOp;
  scf::IfOp branch = row.guard;
  const bool additive =
      std::holds_alternative<AdditiveRecurrence>(recurrence.fact);
  triton::DotOp scoreDot = scoreDotOf(recurrence.fact);
  triton::DotOp valueDot = valueDotOf(recurrence.fact);
  triton::LoadOp kLoad = scoreDot.getB().getDefiningOp<triton::LoadOp>();
  triton::LoadOp vLoad = valueDot.getB().getDefiningOp<triton::LoadOp>();
  if (!kLoad || !vLoad || kLoad == vLoad)
    return std::nullopt;
  if (kLoad->getParentRegion() != &branch.getThenRegion() ||
      vLoad->getParentRegion() != &branch.getThenRegion())
    return std::nullopt;
  std::optional<TilesMatch> tiles;
  if (additive)
    tiles = parseAddPtrTiles(forOp, row.rawIndex, kLoad, vLoad);
  if (!tiles)
    tiles = parseMptrTiles(forOp, row.rawIndex, kLoad, vLoad);
  if (!tiles)
    return std::nullopt;
  // The online-softmax template widens the tile to BN = max(64, BS) and reads
  // it unconditionally, so every lane past the baseline's BS-wide window exists
  // only because the view masks it. The additive template keeps BN == BS and
  // re-reads the baseline's own addresses, so only the widened form needs this.
  if (!additive) {
    MptrTiles *mptr = std::get_if<MptrTiles>(&tiles->fact);
    if (!mptr || !isZeroPaddedTokenBoundary(*mptr))
      return std::nullopt;
  }
  // The tile width contract: BS must be a power of two so a block number owns
  // BS consecutive tokens, and BN = max(64, BS) then divides BS by construction
  // (64 % BS == 0 for BS <= 64, BS % BS == 0 above it), so there is nothing
  // left to check. BN comes from IR constants, never a new kernel argument.
  if (!isPowerOfTwo(tileBlockSize(tiles->fact)))
    return std::nullopt;
  return tiles;
}

// ---- [2.6] The causal mask ----
//
// The condition compares the row position against the block's token positions:
// row >= splat(token_start) + make_range{0, BS}. Which side is the row is
// resolved here; the identity tie-outs live in validateScan.
static std::optional<MaskFact>
parseCausalMask(const RecurrenceMatch &recurrence) {
  arith::SelectOp select = recurrence.maskSelect;
  // Only broadcasts above the comparison are peeled.
  Value condition = peelConditionBroadcasts(select.getCondition());
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare || !isa<RankedTensorType>(compare.getLhs().getType()))
    return std::nullopt;
  // The token side must be exactly splat(X) + make_range{0, BS}. A diagonal
  // shift, a +-1 nudge or a rescaled index would be dropped silently by the
  // rewrite, so those spellings are rejected instead of normalized.
  Value lhs = peelConditionBroadcasts(compare.getLhs());
  Value rhs = peelConditionBroadcasts(compare.getRhs());
  MaskFact fact;
  fact.predicate = compare.getPredicate();
  if (std::optional<TokenRange> tokens = matchTokenRange(lhs)) {
    fact.pivot = rhs;
    fact.pivotIsLhs = false;
    fact.tokens = *tokens;
  } else if ((tokens = matchTokenRange(rhs))) {
    fact.pivot = lhs;
    fact.pivotIsLhs = true;
    fact.tokens = *tokens;
  } else {
    return std::nullopt;
  }
  return fact;
}

// ---- [3] validateScan ----
//
// Every judgement that needs two facts belongs here and nowhere else. The
// parsers above validate one anchor each; what compares facts produced by
// different parsers is collected in this one function, so the set of agreements
// a match depends on can be read in one place.

// Both dots must produce acc's row count by BS, in f32.
static bool isScoreTile(Type type, RankedTensorType accType,
                        int64_t blockSize) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  return ranked && ranked.getRank() == 2 && ranked.getElementType().isF32() &&
         ranked.getShape()[0] == accType.getShape()[0] &&
         ranked.getShape()[1] == blockSize;
}

// Everything inside the gather loop must be either rebuilt or safely deletable,
// so no unrecognized op may carry effects: a store, atomic, call or device-side
// tt.assert would be silently dropped. tt.trans is only rebuilt by the additive
// template, so it stays unrecognized for the online-softmax form.
static bool hasSupportedVocabulary(scf::ForOp loop, bool additive) {
  bool supported = true;
  loop->walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::IfOp, scf::YieldOp>(op))
      return;
    if (isa<triton::AddPtrOp, triton::LoadOp, triton::MakeTensorPtrOp,
            triton::DotOp, triton::MakeRangeOp, triton::SplatOp,
            triton::ExpandDimsOp, triton::BroadcastOp, triton::ReduceOp,
            triton::ReduceReturnOp>(op))
      return;
    if (additive && isa<triton::TransOp>(op))
      return;
    StringRef ns = op->getName().getDialectNamespace();
    if (ns == "arith" || ns == "math")
      return;
    supported = false;
  });
  return supported;
}

static bool validateScan(const LoopFact &loop, const RowFact &row,
                         const TableFact &table,
                         const RecurrenceMatch &recurrence,
                         const TilesMatch &tiles, const MaskFact &mask) {
  scf::ForOp forOp = loop.forOp;
  const TilesFact &tileFact = tiles.fact;
  const int64_t blockSize = tileBlockSize(tileFact);
  const bool additive =
      std::holds_alternative<AdditiveRecurrence>(recurrence.fact);
  triton::DotOp scoreDot = scoreDotOf(recurrence.fact);
  triton::DotOp valueDot = valueDotOf(recurrence.fact);

  // [3.2] A constant trip count is the special case NS == S: the two
  // derivations of the table width must agree, or the vector read and the scan
  // bound disagree about how many slots exist.
  if (auto trip = getConstantIntValue(forOp.getUpperBound());
      trip && *trip != table.length)
    return false;

  // [3.3] Every block predicate must scale the index by the same BS as the tile
  // offsets; otherwise "one index entry == BS tokens" has two meanings.
  for (int64_t factor : row.factors)
    if (factor != blockSize)
      return false;

  // [3.4] The causal mask: the same row position as the guard, on the greater
  // side, over the tiles' own token vector.
  if (stripIntegerCasts(peelBroadcasts(mask.pivot)) != row.rowPosition)
    return false;
  bool pivotGreater =
      (mask.predicate == arith::CmpIPredicate::sge && mask.pivotIsLhs) ||
      (mask.predicate == arith::CmpIPredicate::sle && !mask.pivotIsLhs);
  if (!pivotGreater || mask.tokens.extent != blockSize)
    return false;
  // The range must start at one of the tiles' token starts. make_tensor_ptr
  // pins the start by identity - the same SSA value the view's token axis is
  // offset by. The addptr spelling re-multiplies that product inside its
  // address chain, so there the comparison is on the product, which is the same
  // number.
  bool startIsTileStart = mask.tokens.start == tiles.kTokenStart ||
                          mask.tokens.start == tiles.vTokenStart;
  if (!startIsTileStart && std::holds_alternative<AddPtrTiles>(tileFact)) {
    auto factor = matchIdxTimesBS(mask.tokens.start, row.rawIndex);
    startIsTileStart = factor && *factor == blockSize;
  }
  if (!startIsTileStart)
    return false;

  // [3.5] The dots: every query-side operand depends on the query row, every
  // tile-side operand does not, with the removed index load cut out of the
  // walk. The varlen dispatch loads the row position, so there those two
  // assertions are dropped and the loads count as per-program constants; the
  // causal identity check above still pins the row position.
  Dependence rules;
  rules.cut = row.indexLoad;
  if (row.rowFromLoad)
    rules.opaqueLoads = forOp;
  auto depends = [&](Value v) { return dependsOn(v, row.pid, rules); };
  if (!row.rowFromLoad) {
    if (!depends(scoreDot.getA()) || !depends(valueDot.getA()))
      return false;
  }
  if (depends(scoreDot.getB()) || depends(valueDot.getB()))
    return false;
  // The score dot's lhs is Q, loaded and scaled outside the loop, so the
  // rewrite can reuse the operand verbatim.
  if (!isDefinedOutsideLoop(scoreDot.getA(), forOp))
    return false;

  // [3.6] The loop body's vocabulary.
  if (!hasSupportedVocabulary(forOp, additive))
    return false;

  // The online-softmax template reuses the carried state's types verbatim
  // (o/m/l) and hard-codes the score tile to [G, BN] f32, so those are
  // preconditions of the rewrite, not only of the additive form.
  if (!additive) {
    const auto &online = std::get<OnlineSoftmaxRecurrence>(recurrence.fact);
    auto stateOf = [&](unsigned arg) {
      return dyn_cast<RankedTensorType>(forOp.getInitArgs()[arg].getType());
    };
    RankedTensorType mType = stateOf(online.mArg);
    RankedTensorType lType = stateOf(online.lArg);
    RankedTensorType oType = stateOf(online.oArg);
    if (!mType || !lType || !oType || mType.getRank() != 1 ||
        lType.getRank() != 1 || oType.getRank() != 2 ||
        !mType.getElementType().isF32() || !lType.getElementType().isF32() ||
        !oType.getElementType().isF32())
      return false;
    // G is the m row count and the rewriter derives every [G, ...] tile from
    // it, so the o/l row counts have to agree with it as well.
    if (lType.getShape()[0] != mType.getShape()[0] ||
        oType.getShape()[0] != mType.getShape()[0])
      return false;
    auto scoreType = dyn_cast<RankedTensorType>(scoreDot.getType());
    return scoreType && scoreType.getRank() == 2 &&
           scoreType.getElementType().isF32() &&
           scoreType.getShape()[0] == mType.getShape()[0] &&
           scoreType.getShape()[1] == blockSize;
  }

  // [3.7] The additive form is spelling independent but its two tiles must
  // describe the same token block: the builder re-emits one token-range bound
  // for both.
  int64_t kFreeExtent = 0;
  int64_t vFreeExtent = 0;
  if (const auto *addPtrTiles = std::get_if<AddPtrTiles>(&tileFact)) {
    if (addPtrTiles->kTile.blockSize != addPtrTiles->vTile.blockSize ||
        addPtrTiles->kTile.tokenBound != addPtrTiles->vTile.tokenBound)
      return false;
    kFreeExtent = addPtrTiles->kTile.freeExtent;
    vFreeExtent = addPtrTiles->vTile.freeExtent;
  } else {
    MptrTiles mptrTiles = std::get<MptrTiles>(tileFact);
    std::optional<int64_t> kExtent =
        tileExtent(mptrTiles.kLoad.getType(), 1 - mptrTiles.kTokenAxis);
    std::optional<int64_t> vExtent =
        tileExtent(mptrTiles.vLoad.getType(), 1 - mptrTiles.vTokenAxis);
    if (!kExtent || !vExtent)
      return false;
    kFreeExtent = *kExtent;
    vFreeExtent = *vExtent;
  }
  // dot(ds[G, BS], k^T[BS, F]) -> acc[G, F]: both free axes must be the
  // accumulator's column count, and both dots must produce [G, BS].
  auto accType = cast<RankedTensorType>(forOp.getInitArgs()[0].getType());
  if (accType.getRank() != 2)
    return false;
  return isScoreTile(scoreDot.getType(), accType, blockSize) &&
         isScoreTile(valueDot.getType(), accType, blockSize) &&
         kFreeExtent == accType.getShape()[1] &&
         vFreeExtent == accType.getShape()[1];
}

// ===========================================================================
// [4] Assembly
// ===========================================================================

// The candidate is the composition of the facts, field for field: no parser
// computes anything twice here, and nothing is derived that the facts do not
// already state.
static ScanCandidate assemble(triton::FuncOp fn, const LoopFact &loop,
                              const RowFact &row, const TableFact &table,
                              const RecurrenceMatch &recurrence,
                              const TilesMatch &tiles) {
  scf::ForOp forOp = loop.forOp;
  ScanCandidate candidate;
  candidate.function = fn;
  candidate.loop = loop;
  candidate.table = table;
  candidate.rowPosition = row.rowPosition;
  candidate.recurrence = recurrence.fact;
  candidate.tiles = tiles.fact;
  // The old trip count IS NS, the runtime number of selected blocks: the
  // rewrite clamps it into [0, S] before using it as the slot-validity gate.
  candidate.scanBound = forOp.getUpperBound();
  return candidate;
}

} // namespace

namespace mlir {
namespace triton {
namespace cfg {

std::optional<ScanCandidate> matchSelectedBlockScan(triton::FuncOp function) {
  // A decline is silent on the wire and explained under
  // -debug-only=graph-optimize, so a refusal is attributable without a
  // debugger.
  auto declined = [](const char *stage) {
    LLVM_DEBUG(llvm::dbgs()
               << "selected-block-scan: stage " << stage << " declined\n");
    return std::nullopt;
  };

  std::optional<LoopFact> loop = parseLoop(function);
  if (!loop)
    return declined("loop");
  std::optional<RowFact> row = parseRowPosition(function, *loop);
  if (!row)
    return declined("row-position");
  std::optional<TableFact> table = parseIndexTable(function, *loop, *row);
  if (!table)
    return declined("index-table");
  std::optional<RecurrenceMatch> recurrence = parseRecurrence(*loop, *row);
  if (!recurrence)
    return declined("recurrence");
  std::optional<TilesMatch> tiles = parseTiles(*loop, *row, *recurrence);
  if (!tiles)
    return declined("tiles");
  std::optional<MaskFact> mask = parseCausalMask(*recurrence);
  if (!mask)
    return declined("causal-mask");
  if (!validateScan(*loop, *row, *table, *recurrence, *tiles, *mask))
    return declined("agreements");
  return assemble(function, *loop, *row, *table, *recurrence, *tiles);
}

} // namespace cfg
} // namespace triton
} // namespace mlir
