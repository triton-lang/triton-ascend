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

// The facts of SelectedBlockScanRule (#2686, rule bit 65536): what the scan
// rewrite matches on, the candidate that aggregates the facts, and the four
// entry points its three translation units share.
//
// Naming used across these files: "the scan rewrite" is this rule (rule bit
// 65536, SelectedBlockScanRule); "the row lift" is QueryRowBatchingRule (rule
// bit 131072), which the scan rewrite always materializes in its own sandbox.
// RFC #2686 calls these two rules A and B; this tree says SelectedBlockScanRule
// (the scan rewrite) and QueryRowBatchingRule (the row lift).
//
// No logic and no IR walking live here. The matcher, the rewriter and the
// driver shell are separate translation units and each documents its own
// sections; ScanCandidate is the only value that crosses from one to the next.
//
// The matcher is six parsers plus validateScan, one fact each, with no shared
// mutable state:
//
//   parseLoop        (fn)                    -> LoopFact
//   parseRowPosition (fn, loop)              -> RowFact          (matcher only)
//   parseIndexTable  (fn, loop, row)         -> TableFact
//   parseRecurrence  (loop, row)             -> recurrence fact  (variant)
//   parseTiles       (loop, row, recurrence) -> tiles fact       (variant)
//   parseCausalMask  (recurrence)            -> MaskFact         (matcher only)
//   validateScan     (all facts)             -> every tie-out between them
//
// The order is data dependence, and the signatures say so: a parser takes the
// facts it reads and returns the one it produces. Two knots keep both of their
// sides inside one parser instead of being cross-checked later: the causal row
// position and the table row (the same program id, so one fact), and the table
// address with its row length (one address chain). The candidate is the
// composition of the builder-facing facts; the matcher-only ones never leave
// the matcher.

#ifndef TRITON_TO_GRAPH_SELECTED_BLOCK_SCAN_FACTS_H
#define TRITON_TO_GRAPH_SELECTED_BLOCK_SCAN_FACTS_H

#include "TritonToGraph/GraphOptimizationRule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <variant>

namespace mlir {
namespace triton {
namespace cfg {

// ---------------------------------------------------------------------------
// [Fact] The gather loop.
// ---------------------------------------------------------------------------
struct LoopFact {
  scf::ForOp forOp;     // the only scf.for, at the function body's top level
  unsigned carried = 0; // loop-carried tensor values: 3 (fwd) or 1 (bwd)
};

// ---------------------------------------------------------------------------
// [Fact] The index table: where this program's row is, and how long it is.
// ---------------------------------------------------------------------------
struct TableFact {
  // The row-base pointer. Null when the front end folded the row advance into
  // the in-loop address; the rewriter then re-creates the base from
  // root + rowOffset before the vectorized read.
  Value base;
  Value root;         // the pointer the row base is offset from
  Value rowOffset;    // loop-invariant row offset: program row + head slot
  int64_t length = 0; // S, the row length in slots
};

// ---------------------------------------------------------------------------
// [Fact] The additive (bwd dq) tile address: everything the rewriter needs to
// re-emit one masked `addptr` tile load at a different token start.
// ---------------------------------------------------------------------------
struct AddPtrTile {
  Value base;               // scalar pointer, loop-invariant and load-free
  Value freeVec;            // [F] integer vector, loop-invariant
  Value freeBoundRhs;       // [F,1] tensor the free-axis mask compares against
  Value tokenBound;         // scalar bound of the "tokens < T" mask
  Value blockStart;         // idx*BS: the value the causal mask also uses
  unsigned tokenWidth = 32; // width of the token arithmetic: 32 or 64
  int64_t tokenStride = 0;  // stride of one token along the address
  int64_t freeExtent = 0;   // F, the tile's free-axis extent
  int64_t blockSize = 0;    // BS, the tile's token extent
};

// ---------------------------------------------------------------------------
// [Fact] The carried-state recurrence. The variant IS the rewrite template:
// nothing else records the choice, so the two can never disagree.
// ---------------------------------------------------------------------------
// Online softmax (fwd): three iter args (o, m, l); m is updated with maxnumf
// and the state is rescaled by exp(m_old - m_new).
struct OnlineSoftmaxRecurrence {
  triton::DotOp scoreDot; // dot(q, k, 0)
  triton::DotOp valueDot; // dot(cast(p), v, o * alpha)
  unsigned oArg = 0;      // carried-state roles, in iter-argument order
  unsigned mArg = 0;
  unsigned lArg = 0;
  // The float width cast accepted around p, recorded as a direction rather
  // than as the original op: the rewriter re-emits the same cast.
  bool pCastIsExtend = false;
};

// Additive (bwd dq): one carried accumulator; the contribution lands in it
// through the dot's C operand and the neutral element is 0. It is always iter
// argument 0 (LoopFact::carried == 1 for this form), so no role index is
// recorded.
struct AdditiveRecurrence {
  triton::DotOp scoreDot;      // dot(q, k, 0)
  triton::DotOp valueDot;      // dot(dO, v, 0)
  triton::DotOp accDot;        // dot(ds, k^T, acc)
  triton::TransOp transpose;   // k^T, order replayed verbatim
  Value lseVec;                // [G] f32, loop-invariant
  Value deltaVec;              // [G] f32, loop-invariant
  bool dsCastIsExtend = false; // ds was widened, not narrowed
};

// The variant is a deliberate deviation from the rest of Rules/, where a form
// is an `enum class` field of one merged struct (IndependentAxisTensorizeRule's
// TensorizeFormMatch is the clearest example). A tag and a merged struct leave
// the other alternative's fields readable, and a rewriter that reads one still
// compiles; a variant makes exactly one alternative exist at a time, so the
// dispatch is exhaustive by construction and no field of the form not being
// emitted can be touched.
using RecurrenceFact =
    std::variant<OnlineSoftmaxRecurrence, AdditiveRecurrence>;

// ---- Accessors for the two dots both templates build on. ----

inline triton::DotOp scoreDotOf(const RecurrenceFact &r) {
  return std::visit([](const auto &f) { return f.scoreDot; }, r);
}

inline triton::DotOp valueDotOf(const RecurrenceFact &r) {
  return std::visit([](const auto &f) { return f.valueDot; }, r);
}

// ---------------------------------------------------------------------------
// [Fact] How the K/V tiles are addressed. The variant IS the spelling: the
// rewriter has one emitter per spelling and dispatches on it.
// ---------------------------------------------------------------------------
// tt.make_tensor_ptr: the token axis is one of the view's offsets.
struct MptrTiles {
  triton::LoadOp kLoad;
  triton::LoadOp vLoad;
  triton::MakeTensorPtrOp kPtr;
  triton::MakeTensorPtrOp vPtr;
  unsigned kTokenAxis = 0;
  unsigned vTokenAxis = 0;
  int64_t blockSize = 0; // BS, the idx multiplier
  // The only loop-carried address computation, re-created outside the loop.
  // Filled for every make_tensor_ptr spelling, not only for the online-softmax
  // recurrence: the rebuild is keyed on the spelling, so an additive candidate
  // spelled this way reads them too.
  Value dynamicBoundResult; // null when the bound is already outside
  Value dynamicBoundInput;  // null together with dynamicBoundResult
};

// Two nested tt.addptr's plus an explicit mask and other = 0.
struct AddPtrTiles {
  triton::LoadOp kLoad;
  triton::LoadOp vLoad;
  AddPtrTile kTile;
  AddPtrTile vTile;
  int64_t blockSize = 0; // BS, the idx multiplier
};

using TilesFact = std::variant<MptrTiles, AddPtrTiles>;

// ---- Accessors over the two variants. The rewriter's two spelling emitters
// ---- take their variant directly (each needs several fields, and dispatching
// ---- once is clearer than dispatching per field); these are for the callers
// ---- that need one field without caring which spelling produced it: the
// ---- shared weight builder and the driver's identity check. ----

inline triton::LoadOp kTileLoad(const TilesFact &tiles) {
  return std::visit([](const auto &t) { return t.kLoad; }, tiles);
}

inline triton::LoadOp vTileLoad(const TilesFact &tiles) {
  return std::visit([](const auto &t) { return t.vLoad; }, tiles);
}

inline int64_t tileBlockSize(const TilesFact &tiles) {
  return std::visit([](const auto &t) { return t.blockSize; }, tiles);
}

// The tile width contract, one source for both translation units:
// BN = max(64, BS) and R = BN / BS. BS is a power of two (checked by the
// matcher), so the division is exact by construction.  The additive dQ form
// keeps BN = BS: its three dots retain K for a transposed use, so widening the
// scan would double both the score temporaries and the shared K/V tiles.
inline int64_t scanTileWidth(int64_t blockSize, bool additive) {
  return additive ? blockSize : std::max<int64_t>(64, blockSize);
}

inline int64_t blocksPerTile(int64_t blockSize, bool additive) {
  return scanTileWidth(blockSize, additive) / blockSize;
}

// ---------------------------------------------------------------------------
// [Fact] The scan candidate: the composition of the builder-facing facts.
// Every field comes from exactly one parser and is read by the rewriter, or by
// the driver for revalidation. Nothing here is derivable from another field.
// ---------------------------------------------------------------------------
struct ScanCandidate {
  triton::FuncOp function;
  LoopFact loop;
  TableFact table;
  // The causal row position (pid, or pid + a loop-invariant decode offset):
  // the scan bound and every causal comparison are keyed by this value. The
  // matcher always sets it, so the rewriter may read it unconditionally.
  Value rowPosition;
  RecurrenceFact recurrence;
  TilesFact tiles;
  Value scanBound; // the old trip count == NS
};

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

// The matcher: nullopt means "not this pattern", no diagnostics and no IR
// mutation. Implemented in SelectedBlockScanMatch.cpp.
std::optional<ScanCandidate> matchSelectedBlockScan(triton::FuncOp function);

// The rewriter: emits the sequential scan for `candidate` into the caller's
// private sandbox clone. A false result leaves the sandbox partially modified
// and must be treated as "decline"; implemented in SelectedBlockScanBuild.cpp.
bool buildSelectedBlockScan(IRRewriter &rewriter, ScanCandidate &candidate);

// The UB / L0 admission gate on the rebuilt sandbox. Warns on the live
// function, never mutates it; implemented in SelectedBlockScanBuild.cpp.
bool budgetOverflow(triton::FuncOp sandboxFunction,
                    const ResourceSnapshot &resources, llvm::StringRef form,
                    triton::FuncOp liveFunction);

// Canonicalize/CSE/LICM pass over the private sandbox, so the joint row-lift
// consult judges exactly the body that apply() commits. Implemented in
// SelectedBlockScanBuild.cpp.
LogicalResult runSandboxStructuralCleanup(ModuleOp sandbox);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_SELECTED_BLOCK_SCAN_FACTS_H
