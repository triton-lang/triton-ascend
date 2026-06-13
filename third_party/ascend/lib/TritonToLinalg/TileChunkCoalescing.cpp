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

#include "TritonToLinalg/TileChunkCoalescing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>
#include <optional>

namespace TileChunkCoalescing {

using namespace mlir;
using namespace triton;

namespace {

// Tunables.
constexpr int64_t kMinContigBytes = 512;  // floor: smallest merged-block size

// Upper bound on the number of tiles coalesced per program. We now derive it
// from a UB (Unified Buffer) footprint budget instead of a single flat cap:
//
//   maxH = clamp(kUBBytesBudget / per-program-footprint(H=1), 2, kMaxCoalesceTilesCeil)
//
// This lets kernels with a small working set (pure 1-D load->store) coalesce far
// more aggressively -- bigger contiguous DMAs, fewer persistent-loop trips --
// while kernels carrying a large on-chip tensor (e.g. a 16x16 segsum matrix)
// automatically back off so the lifted footprint still fits UB.
//
// The footprint is the SUM of every lifted region tensor's bytes at H=1. That
// over-counts (the backend reuses buffers by liveness, so true peak live set is
// smaller), which is the safe direction: we may pick a smaller H than strictly
// necessary, but never one that overflows UB (runtime error 341). If on-board
// profiling shows UB headroom, raise kUBBytesBudget; if it overflows, lower it.
//
// UB is ~256 KB/core on this arch (dav-c310, __UB_SIZE = 262144). The budget is
// a conservative slice that leaves room for double-buffering and alignment.
constexpr int64_t kUBBytesBudget = 96 * 1024;
// Absolute ceiling regardless of UB headroom: a very large H shrinks the launch
// grid (the host launcher divides grid[axis] by H) and can starve AI cores, so
// we never coalesce more than this many tiles per program.
constexpr int64_t kMaxCoalesceTilesCeil = 16;

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
// The grid axis (= program_id axis) whose launch dimension the host launcher
// divides by the factor. The full TA path owns the grid division: bishengir no
// longer interprets either attr (see driver.py / compiler.py).
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";

struct TileSeed {
  triton::GetProgramIdOp pid;  // the (outermost) tile-index program id
  int32_t axis = 0;            // pid axis == the grid dim the launcher divides
  int64_t tileLen = 0;         // T  (constexpr tile length / CHUNK_SIZE)
  int64_t bound = 0;           // BOUND (constexpr problem length / SEQLEN)
  Value mask;                  // tile OOB mask (cmpi result). It is provably
                               // all-true (we require bound % tileLen == 0) and
                               // is dropped from the coalesced load/store.
};

// Read a constant integer from a scalar i32 or a splat-of-constant tensor.
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

// The set of ops we know how to lift (prepend an H lane). Anything else in the
// load->store subgraph makes the pass bail.
//
// arith/math ops are all side-effect-free, elementwise (or scalar) and carry no
// regions, so the generic "prepend H + rebuild" handles them uniformly -- we
// whitelist those dialects wholesale rather than enumerating ops (so math.exp,
// math.log, math.sqrt, ... all work). The remaining cases are the triton ops
// that need shape/axis-aware handling.
static bool isLiftable(Operation *op) {
  if (auto *d = op->getDialect()) {
    StringRef ns = d->getNamespace();
    if (ns == arith::ArithDialect::getDialectNamespace() ||
        ns == math::MathDialect::getDialectNamespace())
      return true;
  }
  return isa<triton::SplatOp, triton::AddPtrOp, triton::BroadcastOp,
             triton::ExpandDimsOp, triton::LoadOp, triton::StoreOp,
             triton::ScanOp, triton::ReduceOp>(op);
}

// Detect the tile-index signature:
//     blk  = muli(program_id_max, C)          (C constexpr, the tile length T)
//     offs = splat(blk) + make_range[0, C)
// The boundary mask `cmpi slt(offs, BOUND)` is required: it gives the pass a
// static tile count and lets the launcher shrink grid[axis] by an exact divisor.
// Unmasked kernels carry no tile count in the IR; coalescing them would rely on
// runtime grid[axis] being both >= H and divisible by H, which is not guaranteed.
// When the mask is present we use it to recover BOUND (and to drop the
// provably-all-true mask later); when it is a real partial-tile mask
// (BOUND % C != 0) we must NOT coalesce, so we bail.
static std::optional<TileSeed> findSeed(ModuleOp moduleOp) {
  // The coalesced axis must be the OUTERMOST grid axis: the host launcher
  // divides grid[axis] by coalesce_factor, and bishengir reconstructs the
  // outermost (highest-index) program id as the most-significant digit of the
  // linear block id, so only that axis stays consistent. Coalescing an inner
  // axis would mis-assign work.
  int32_t maxAxis = -1;
  moduleOp.walk([&](triton::GetProgramIdOp pid) {
    maxAxis = std::max<int32_t>(maxAxis, pid.getAxisAsInt());
  });

  // Only one program-id op may read the coalesced axis: we replace the seed
  // pid by the per-lane tile vector, but a second (non-CSE'd) pid of the same
  // axis would keep returning the now-divided block id -> its tile indices
  // would silently address the wrong tiles. Bail unless unique.
  int32_t maxAxisPids = 0;
  moduleOp.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() == maxAxis)
      ++maxAxisPids;
  });
  if (maxAxisPids != 1)
    return std::nullopt;

  // Full TA path: the launcher divides grid[maxAxis] by H, so the kernel-visible
  // num_programs(maxAxis) becomes grid/H instead of grid. If the kernel reads it
  // (e.g. for its own bound arithmetic) coalescing would silently change that
  // value -> wrong results. Refuse to coalesce such kernels (they keep the
  // original, uncoalesced -- but correct -- path).
  bool readsMaxAxisNumPrograms = false;
  moduleOp.walk([&](triton::GetNumProgramsOp np) {
    if (np.getAxisAsInt() == maxAxis)
      readsMaxAxisNumPrograms = true;
  });
  if (readsMaxAxisNumPrograms)
    return std::nullopt;

  std::optional<TileSeed> result;
  moduleOp.walk([&](triton::GetProgramIdOp pid) {
    if (result || pid.getAxisAsInt() != maxAxis)
      return;
    for (Operation *u : pid.getResult().getUsers()) {
      auto mul = dyn_cast<arith::MulIOp>(u);
      if (!mul)
        continue;
      Value other = (mul.getLhs() == pid.getResult()) ? mul.getRhs()
                                                       : mul.getLhs();
      int64_t T = 0;
      if (!getConstInt(other, T) || T <= 1)
        continue;  // non-const multiplier (e.g. segsum chunk stride) -> skip

      // muli -> splat -> addi(splat, make_range[0,T))
      for (Operation *mu : mul.getResult().getUsers()) {
        auto sp = dyn_cast<triton::SplatOp>(mu);
        if (!sp)
          continue;
        for (Operation *su : sp.getResult().getUsers()) {
          auto add = dyn_cast<arith::AddIOp>(su);
          if (!add)
            continue;
          Value rangeV =
              (add.getLhs() == sp.getResult()) ? add.getRhs() : add.getLhs();
          auto range = rangeV.getDefiningOp<triton::MakeRangeOp>();
          if (!range || range.getStart() != 0 || range.getEnd() != T)
            continue;

          // Boundary-safety scan: taint everything pid-derived (through
          // elementwise arith and shape ops) and require that NO tainted value
          // feeds boundary handling, except the one canonical all-true tile
          // mask `cmpi slt(offs, BOUND)` directly on offs with constant
          // BOUND >= T and BOUND % T == 0 (provably all-true, droppable).
          //
          // Everything else is real boundary logic on the chunk axis and must
          // block coalescing:
          //   * cmpi with a runtime BOUND (kernel arg): a partial last tile may
          //     exist; the lifted mask `(pid*H+h)*T + t < BOUND` is
          //     non-separable (MaskAnalysis bails, load stays unconverted) and
          //     the launcher's grid/H drops remainder tiles -> wrong results.
          //   * cmpi reached through expand_dims/broadcast/addi/... or with
          //     other predicates/operand order: same non-separability.
          //   * min/max clamps and div/rem wraparound on pid-derived offsets:
          //     the lifted addressing is non-affine across tiles (PtrAnalysis
          //     bails).
          int64_t bound = 0;
          Value mask;
          bool unsafe = false;
          DenseSet<Value> taint;
          SmallVector<Value> twl;
          taint.insert(pid.getResult());
          twl.push_back(pid.getResult());
          while (!twl.empty() && !unsafe) {
            Value cur = twl.pop_back_val();
            for (Operation *tu : cur.getUsers()) {
              if (auto cmp = dyn_cast<arith::CmpIOp>(tu)) {
                if (mask && cmp.getResult() == mask)
                  continue;
                int64_t b = 0;
                if (!mask && cur == add.getResult() && cmp.getLhs() == cur &&
                    cmp.getPredicate() == arith::CmpIPredicate::slt &&
                    getConstInt(cmp.getRhs(), b) && b >= T && b % T == 0) {
                  bound = b;
                  mask = cmp.getResult();
                  continue;
                }
                unsafe = true;
                break;
              }
              if (isa<arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp,
                      arith::MaxUIOp, arith::RemSIOp, arith::RemUIOp,
                      arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp,
                      arith::FloorDivSIOp>(tu)) {
                unsafe = true;
                break;
              }
              bool propagates = isa<triton::SplatOp, triton::ExpandDimsOp,
                                    triton::BroadcastOp, triton::AddPtrOp>(tu);
              if (auto *d = tu->getDialect())
                propagates |= d->getNamespace() ==
                              arith::ArithDialect::getDialectNamespace();
              if (!propagates)
                continue;  // load/store/scan/... results carry data, not offsets
              for (Value r : tu->getResults())
                if (taint.insert(r).second)
                  twl.push_back(r);
            }
          }
          if (unsafe)
            return;  // real boundary handling on the chunk axis; abandon pid
          if (!mask)
            return;  // unmasked kernel: runtime tile count is absent from IR

          result = TileSeed{pid, maxAxis, T, bound, mask};
          return;
        }
      }
    }
  });
  return result;
}

// Forward slice from the tile pid down to (and including) the stores. Fills
// `region` (all ops) and `ordered` (region ops in IR order). Returns false and
// leaves the IR untouched if the slice contains an unliftable op, if a region
// value escapes the slice, or if there are no store sinks.
static bool collectRegion(TileSeed &seed, ModuleOp moduleOp,
                          DenseSet<Operation *> &region,
                          SmallVectorImpl<Operation *> &ordered) {
  SmallVector<Operation *> wl;
  DenseSet<Operation *> visited;
  bool hasStore = false;
  for (Operation *u : seed.pid.getResult().getUsers())
    wl.push_back(u);
  while (!wl.empty()) {
    Operation *op = wl.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    if (!isLiftable(op))
      return false;  // bail: unknown op in the chain (e.g. tt.dot)
    if (isa<triton::StoreOp>(op))
      hasStore = true;
    region.insert(op);
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        wl.push_back(u);
  }
  if (!hasStore)
    return false;

  // No region value (other than store effects) may escape the slice.
  for (Operation *op : region) {
    if (isa<triton::StoreOp>(op))
      continue;
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        if (!region.count(u))
          return false;
  }

  // Topological (IR) order, restricted to the region.
  moduleOp.walk([&](Operation *op) {
    if (region.count(op))
      ordered.push_back(op);
  });
  return true;
}

// Pick H (number of tiles coalesced per program). The floor `hMin` is the
// smallest factor whose merged block reaches kMinContigBytes -- the minimum that
// fixes the tiny-DMA pathology. `maxH` is the UB-footprint-derived ceiling
// (see kUBBytesBudget); it has already been clamped to kMaxCoalesceTilesCeil.
//
// H must divide the statically known tile count so the launcher's
// `grid[axis] / H` is exact. Pick the largest divisor in [hMin, maxH] so the
// already-contiguous transfers get bigger DMAs / fewer loop iterations.
static int64_t chooseH(int64_t numTiles, int64_t tileLen, int64_t elemBytes,
                       int64_t maxH) {
  int64_t blockBytes = tileLen * elemBytes;
  int64_t hMin = (kMinContigBytes + blockBytes - 1) / blockBytes;
  if (hMin < 2)
    hMin = 2;
  if (maxH < hMin)
    return 0;  // UB budget too tight to even reach the contiguity floor

  if (numTiles <= 0)
    return 0;

  int64_t H = 0;
  for (int64_t c = hMin; c <= numTiles; ++c)
    if (numTiles % c == 0) {
      H = c;
      break;
    }
  if (H == 0)
    return 0;

  for (int64_t c = numTiles; c > H; --c)
    if (numTiles % c == 0 && c <= maxH) {
      H = c;
      break;
    }
  return H;
}

static void rewriteModule(ModuleOp moduleOp, IRRewriter &rw) {
  // Yield to the higher-priority StridedAxisCoalescing: the launcher divides
  // grid[axis] by a single (hacc.coalesce_factor, hacc.coalesce_axis) pair, so
  // at most one pass may own it per module. If StridedAxisCoalescing (which runs
  // first) already claimed it, coalescing again here would lift a second set of
  // tiles while only one (factor, axis) survives -> wrong block count. Bail and
  // leave our tiles to the strided dispatch.
  if (moduleOp->hasAttr(kCoalesceFactorAttr))
    return;

  auto seed = findSeed(moduleOp);
  if (!seed)
    return;

  DenseSet<Operation *> region;
  SmallVector<Operation *> ordered;
  if (!collectRegion(*seed, moduleOp, region, ordered))
    return;

  // Element size from the first region load.
  int64_t elemBytes = 0;
  for (Operation *op : ordered)
    if (auto ld = dyn_cast<triton::LoadOp>(op)) {
      auto rt = dyn_cast<RankedTensorType>(ld.getResult().getType());
      if (rt)
        elemBytes = rt.getElementTypeBitWidth() / 8;
      break;
    }
  if (elemBytes == 0)
    return;

  // Per-program UB footprint at H=1: sum of the bytes of the tensors that
  // actually live in UB. Each lifts to H copies, so footprint(H) = H *
  // footprintUnit. We count only DMA'd data (load results, store values) and
  // floating-point compute results; pointer / integer-offset / i1-mask tensors
  // are address arithmetic that folds into memref strides and never occupies a
  // UB data buffer. Summing (vs peak-live) over-counts -- the safe direction:
  // we may pick a smaller H, never one that overflows UB (runtime error 341).
  auto tensorBytes = [](Type t) -> int64_t {
    auto rt = dyn_cast<RankedTensorType>(t);
    if (!rt)
      return 0;
    Type et = rt.getElementType();
    if (!et.isIntOrFloat())
      return 0;
    return rt.getNumElements() * ((et.getIntOrFloatBitWidth() + 7) / 8);
  };
  int64_t footprintUnit = 0;
  for (Operation *op : ordered) {
    if (auto ld = dyn_cast<triton::LoadOp>(op))
      footprintUnit += tensorBytes(ld.getResult().getType());
    else if (auto st = dyn_cast<triton::StoreOp>(op))
      footprintUnit += tensorBytes(st.getValue().getType());
    else
      for (Type t : op->getResultTypes())
        if (auto rt = dyn_cast<RankedTensorType>(t))
          if (isa<FloatType>(rt.getElementType()))
            footprintUnit += tensorBytes(t);
  }
  // maxH = clamp(budget / footprintUnit, 2, ceil).
  int64_t maxH = kMaxCoalesceTilesCeil;
  if (footprintUnit > 0)
    maxH = std::min<int64_t>(maxH, kUBBytesBudget / footprintUnit);
  if (maxH < 2)
    maxH = 2;

  int64_t numTiles = (seed->bound + seed->tileLen - 1) / seed->tileLen;
  int64_t H = chooseH(numTiles, seed->tileLen, elemBytes, maxH);
  if (H <= 1)
    return;

  Value pidVal = seed->pid.getResult();
  Location ploc = seed->pid.getLoc();
  Value seedMask = seed->mask;
  Operation *seedMaskOp = seedMask ? seedMask.getDefiningOp() : nullptr;

  // The all-true tile mask is dropped from the loads/stores it guards and is
  // never rebuilt (skipped in the loop below). That is only valid if it feeds
  // *nothing else*: any other consumer (a broadcast before the load mask, a
  // tt.where, ...) would try to lift the skipped mask and reference a null
  // value -> invalid IR. Bail in that case (kernel keeps the correct,
  // uncoalesced path) rather than risk a miscompile.
  if (seedMask) {
    for (Operation *u : seedMask.getUsers()) {
      auto ld = dyn_cast<triton::LoadOp>(u);
      auto st = dyn_cast<triton::StoreOp>(u);
      bool okAsLoadMask = ld && ld.getMask() == seedMask;
      bool okAsStoreMask = st && st.getMask() == seedMask;
      if (!okAsLoadMask && !okAsStoreMask)
        return;
    }
  }

  // Preflight: from here on we create ops; every op of the region must be
  // provably rebuildable in lifted form, otherwise the new op would fail the
  // verifier AFTER the IR is already mutated -- a hard compile error instead
  // of a silent fallback. Anything we cannot prove safe bails here, while the
  // IR is still untouched.
  Block *pidBlock = seed->pid->getBlock();
  for (Operation *op : ordered) {
    // Straight-line code only: a region op nested in scf.for/scf.if has
    // loop-carried/branch semantics the elementwise lift does not model.
    if (op->getBlock() != pidBlock)
      return;
    // Zero-operand ops (constants, make_range) reach the generic rebuild with
    // a lifted result type but unchanged attributes (e.g. a DenseElementsAttr
    // of the OLD shape) -> verifier failure.
    if (!isa<triton::LoadOp, triton::StoreOp>(op) && op->getNumOperands() == 0)
      return;
    // arith.select: the lift turns a scalar condition into tensor<Hxi1>, which
    // no longer matches the lifted tensor operands -> invalid op.
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      bool condTensor = isa<RankedTensorType>(sel.getCondition().getType());
      bool valTensor = isa<RankedTensorType>(sel.getTrueValue().getType());
      if (condTensor != valTensor)
        return;
    }
    // Lifting bumps the boundary-check axes; only block-pointer accesses carry
    // them and those never reach here, so any non-empty set means an IR shape
    // we did not anticipate.
    if (auto ld = dyn_cast<triton::LoadOp>(op))
      if (!ld.getBoundaryCheck().empty())
        return;
    if (auto st = dyn_cast<triton::StoreOp>(op))
      if (!st.getBoundaryCheck().empty())
        return;
  }

  auto liftTy = [&](Type t) -> RankedTensorType {
    if (auto rt = dyn_cast<RankedTensorType>(t)) {
      SmallVector<int64_t> s;
      s.push_back(H);
      s.append(rt.getShape().begin(), rt.getShape().end());
      return RankedTensorType::get(s, rt.getElementType());
    }
    return RankedTensorType::get({H}, t);
  };

  // tileVec = splat(pid * H) + arange(0, H)  : tensor<Hxi32>, the per-lane
  // global tile index covered by this (now coalesced) program.
  rw.setInsertionPointAfter(seed->pid);
  Value cH = rw.create<arith::ConstantIntOp>(ploc, H, 32);
  Value pidH = rw.create<arith::MulIOp>(ploc, pidVal, cH);
  auto i32 = rw.getI32Type();
  auto hVecTy = RankedTensorType::get({H}, i32);
  Value rangeH = rw.create<triton::MakeRangeOp>(ploc, hVecTy, 0, H);
  Value splatPidH = rw.create<triton::SplatOp>(ploc, hVecTy, pidH);
  Value tileVec = rw.create<arith::AddIOp>(ploc, splatPidH, rangeH);

  DenseMap<Value, Value> vmap;
  std::function<Value(Value)> lift = [&](Value v) -> Value {
    if (v == pidVal)
      return tileVec;
    auto it = vmap.find(v);
    if (it != vmap.end())
      return it->second;
    Operation *def = v.getDefiningOp();
    if (def && region.count(def)) {
      // Rebuilt earlier (IR order guarantees dominance). The only region op we
      // never rebuild is the all-true tile mask, and the precondition above
      // guarantees it is not used here, so the lookup must succeed.
      auto rebuilt = vmap.find(v);
      assert(rebuilt != vmap.end() && "region value used before its rebuild");
      return rebuilt->second;
    }
    if (!isa<RankedTensorType>(v.getType()))
      return v;  // region-invariant scalar; caller splats if needed
    // region-invariant tensor: prepend a size-1 dim then broadcast over H.
    Value e = rw.create<triton::ExpandDimsOp>(v.getLoc(), v, 0);
    Value b = rw.create<triton::BroadcastOp>(v.getLoc(), liftTy(v.getType()), e);
    vmap[v] = b;
    return b;
  };
  auto liftOpd = [&](Value v) -> Value {
    Value lv = lift(v);
    if (!isa<RankedTensorType>(lv.getType()))
      lv = rw.create<triton::SplatOp>(lv.getLoc(), liftTy(v.getType()), lv);
    return lv;
  };

  auto liftOpdOrNull = [&](Value v) -> Value {
    return v ? liftOpd(v) : Value();
  };
  auto bumpBoundary = [&](ArrayRef<int32_t> bc) {
    return llvm::map_to_vector(bc, [](int32_t i) { return i + 1; });
  };
  // Copy over any extra attrs the typed builder did not already set.
  auto copyAttrs = [&](Operation *from, Operation *to) {
    for (NamedAttribute a : from->getAttrs())
      if (!to->hasAttr(a.getName()))
        to->setAttr(a.getName(), a.getValue());
  };

  for (Operation *op : ordered) {
    // The tile mask is provably all-true (bound % tileLen == 0) and is dropped
    // from every load/store below, so its (non-separable) lifted form is never
    // needed -- skip rebuilding it.
    if (op == seedMaskOp)
      continue;

    rw.setInsertionPoint(op);
    Location loc = op->getLoc();

    // tt.splat needs a scalar source; if the source became a per-lane tensor
    // ([H]) we instead expand+broadcast it across the splatted shape (mirrors
    // PropagateUnrealizedCastDown::rewriteSplat).
    if (auto sp = dyn_cast<triton::SplatOp>(op)) {
      Value lin = lift(sp.getSrc());
      if (!isa<RankedTensorType>(lin.getType())) {
        vmap[sp.getResult()] =
            rw.create<triton::SplatOp>(loc, liftTy(sp.getType()), lin);
      } else {
        int addDims = cast<RankedTensorType>(sp.getType()).getRank();
        Value cur = lin;  // tensor<Hx...>
        for (int k = 0; k < addDims; ++k)
          cur = rw.create<triton::ExpandDimsOp>(
              loc, cur, cast<RankedTensorType>(cur.getType()).getRank());
        vmap[sp.getResult()] =
            rw.create<triton::BroadcastOp>(loc, liftTy(sp.getType()), cur);
      }
      continue;
    }
    if (auto ed = dyn_cast<triton::ExpandDimsOp>(op)) {
      vmap[ed.getResult()] = rw.create<triton::ExpandDimsOp>(
          loc, liftOpd(ed.getSrc()), ed.getAxis() + 1);
      continue;
    }
    if (auto bc = dyn_cast<triton::BroadcastOp>(op)) {
      vmap[bc.getResult()] = rw.create<triton::BroadcastOp>(
          loc, liftTy(bc.getType()), liftOpd(bc.getSrc()));
      continue;
    }
    if (auto scan = dyn_cast<triton::ScanOp>(op)) {
      SmallVector<Value> srcs = llvm::map_to_vector(
          scan.getSrcs(), [&](Value s) { return liftOpd(s); });
      auto nu = rw.create<triton::ScanOp>(loc, srcs, scan.getAxis() + 1,
                                          scan.getReverse());
      rw.cloneRegionBefore(scan.getCombineOp(), nu.getCombineOp(),
                           nu.getCombineOp().end());
      copyAttrs(scan, nu);
      for (auto [o, n] : llvm::zip(scan.getResults(), nu.getResults()))
        vmap[o] = n;
      continue;
    }
    if (auto red = dyn_cast<triton::ReduceOp>(op)) {
      SmallVector<Value> srcs = llvm::map_to_vector(
          red.getSrcs(), [&](Value s) { return liftOpd(s); });
      auto nu = rw.create<triton::ReduceOp>(loc, srcs, red.getAxis() + 1);
      rw.cloneRegionBefore(red.getCombineOp(), nu.getCombineOp(),
                           nu.getCombineOp().end());
      copyAttrs(red, nu);
      for (auto [o, n] : llvm::zip(red.getResults(), nu.getResults()))
        vmap[o] = n;
      continue;
    }
    if (auto ld = dyn_cast<triton::LoadOp>(op)) {
      // Drop the all-true tile mask (and its `other`) so the coalesced load is
      // an unmasked contiguous block that PtrAnalysis can fully convert.
      bool dropMask = ld.getMask() == seedMask;
      Value m = dropMask ? Value() : liftOpdOrNull(ld.getMask());
      Value o = dropMask ? Value() : liftOpdOrNull(ld.getOther());
      auto bc = bumpBoundary(ld.getBoundaryCheck());
      auto nu = rw.create<triton::LoadOp>(loc, liftOpd(ld.getPtr()), m, o, bc,
                                          ld.getPadding(), ld.getCache(),
                                          ld.getEvict(), ld.getIsVolatile());
      copyAttrs(ld, nu);
      vmap[ld.getResult()] = nu.getResult();
      continue;
    }
    if (auto st = dyn_cast<triton::StoreOp>(op)) {
      bool dropMask = st.getMask() == seedMask;
      Value m = dropMask ? Value() : liftOpdOrNull(st.getMask());
      auto bc = bumpBoundary(st.getBoundaryCheck());
      rw.create<triton::StoreOp>(loc, liftOpd(st.getPtr()),
                                 liftOpd(st.getValue()), m, bc, st.getCache(),
                                 st.getEvict());
      continue;
    }

    // Elementwise / address arithmetic (arith.*, tt.addptr, ...): generic
    // rebuild with lifted operands and prepended result type (mirrors
    // PropagateUnrealizedCastDown::rewriteGeneraleOp).
    SmallVector<Value> operands =
        llvm::map_to_vector(op->getOperands(), [&](Value o) { return liftOpd(o); });
    SmallVector<Type> resTypes = llvm::map_to_vector(
        op->getResultTypes(), [&](Type t) -> Type { return liftTy(t); });
    Operation *nu = rw.create(loc, op->getName().getIdentifier(), operands,
                              resTypes, op->getAttrs());
    for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
      vmap[oldR] = newR;
  }

  // Drop the original chain (reverse IR order: uses before defs).
  for (auto it = ordered.rbegin(); it != ordered.rend(); ++it)
    rw.eraseOp(*it);

  // Record (factor, axis) for the host launcher. compiler.py reads + strips both
  // attrs; the launcher then divides grid[axis] by H. bishengir is not involved.
  auto i32Ty = IntegerType::get(moduleOp.getContext(), 32);
  moduleOp->setAttr(kCoalesceFactorAttr, IntegerAttr::get(i32Ty, H));
  moduleOp->setAttr(kCoalesceAxisAttr, IntegerAttr::get(i32Ty, seed->axis));
}

}  // namespace

void rewriteTileChunkCoalesce(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  rewriteModule(moduleOp, rw);
}

}  // namespace TileChunkCoalescing
