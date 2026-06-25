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

// This pass folds a launch-grid split of a short contiguous logical axis back
// into the per-program tile. Kernels in this family originally launch one
// program per logical lane, e.g. `pid -> (batch, h, col_tile)`, then each program
// executes the same scalar-heavy compute for a single `h`. Coalescing rewrites
// such rank-1 load/compute/store chains to rank-2 `[tile, h]` chains and records
// `hacc.coalesce_factor`/`hacc.coalesce_axis` so the launcher divides that grid
// dimension by the folded factor.
//
// Supported patterns are intentionally local MLIR rewrite patterns:
//   * BlockPtrPattern is rooted at `tt.load` over
//     make_tensor_ptr(base + pid % S, stride S).
//   * AddPtrPattern is rooted at the vector `tt.addptr` address produced by
//     kernels that compute `output_row = pid / grid_dim`, `h = output_row % S`,
//     and vector columns from `pid % grid_dim`.
//
// The shared rewrite only handles the common load->store dataflow contract:
// every seed in one rewrite must have the same factor, tile size, and launch
// axis; every intermediate op must be lane-independent; every sink store must
// match the same concrete pattern as the root load.

#include "TritonToLinalg/StridedAxisCoalescing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <functional>

namespace StridedAxisCoalescing {

using namespace mlir;
using namespace triton;

// Lift a rank-1 tensor type tensor<BTxe> to rank-2. Block-pointer kernels keep
// the folded H axis inner ([BT,S]); addptr jagged kernels keep the column tile
// inner ([S,BT]) so GM copies remain contiguous on the column dimension.
static Type lift2D(Type t, int64_t S, bool laneInner) {
    auto rt = dyn_cast<RankedTensorType>(t);
    if (!rt || rt.getRank() != 1) return t;
    int64_t BT = rt.getShape()[0];
    SmallVector<int64_t, 2> shape =
        laneInner ? SmallVector<int64_t, 2>{BT, S}
                  : SmallVector<int64_t, 2>{S, BT};
    return RankedTensorType::get(shape, rt.getElementType());
}

// An op is safe to 2D-ify (lane-parallel over the folded H axis) iff every
// lane s computes independently: a pure elementwise arith/math op, a cast, a
// splat, or a scan/reduce ALONG THE T axis (axis 0). On the 2D tile [BT,S] a
// T-axis reduce is per-lane; on [S,BT] the rewrite remaps that axis to 1. In
// both layouts the S lanes stay independent (output [S]) and a T-axis scan does
// likewise, so both lift directly -- no need to pre-collapse the reverse-cumsum
// idiom into a single scan. Ops that mix or move the lane (transpose, tt.dot,
// reshape, reduce/scan along the lane) are NOT here, so the caller bails and
// keeps the original (indirect) path.
static bool is2DSafe(Operation *op) {
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::CmpFOp, arith::SelectOp,
            arith::ExtFOp, arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp,
            arith::FPToSIOp, arith::FPToUIOp, arith::CmpIOp, arith::AndIOp,
            arith::OrIOp>(op))
        return true;
    // Every math dialect op (exp/log/sqrt/tanh/erf/...) is a per-lane scalar
    // elementwise map -> 2D-safe. This covers gate activations expressed as
    // tensor math (softplus = log(1+exp), silu, gelu, ...) without enumerating
    // each op, so such cumsum kernels coalesce without per-idiom pattern match.
    if (isa<math::MathDialect>(op->getDialect()))
        return true;
    if (isa<triton::SplatOp>(op)) return true;
    if (auto scan = dyn_cast<triton::ScanOp>(op))
        return scan.getAxis() == 0 && scan->getNumResults() == 1;
    if (auto reduce = dyn_cast<triton::ReduceOp>(op))
        return reduce.getAxis() == 0 && reduce->getNumResults() == 1;
    return false;
}

using LiftValueFn = std::function<Value(Value)>;

template <typename PatternT>
static LogicalResult rewriteCoalescedRegion(ModuleOp moduleOp,
                                            const PatternT &pattern,
                                            const typename PatternT::Seed &rootSeed,
                                            PatternRewriter &rw) {
    if (moduleOp->hasAttr("hacc.coalesce_factor"))
        return failure();
    constexpr bool laneInner = PatternT::kLaneInner;

    // Collect the strided ih-base 1D loads (seeds). All must share one stride S
    // (the folded H axis); BT is the per-chunk tile length.
    SmallVector<typename PatternT::Seed> seeds{rootSeed};
    int64_t S = rootSeed.S, BT = rootSeed.BT;
    int32_t coalesceAxis = rootSeed.coalesceAxis;
    if (S <= 1 || BT <= 0 || coalesceAxis < 0)
        return failure();

    moduleOp.walk([&](triton::LoadOp load) {
        if (load == rootSeed.load) return;
        typename PatternT::Seed seed;
        if (!pattern.matchLoad(load, seed)) return;
        if (seed.S != S || seed.BT != BT ||
            seed.coalesceAxis != coalesceAxis)
            return;
        seeds.push_back(seed);
    });

    // The grid axis the launcher will divide by S (the pid feeding `pid % S`).
    // Full TA path: the launcher divides grid[coalesceAxis] by S, so the
    // kernel-visible num_programs(coalesceAxis) becomes grid/S. If the kernel
    // reads it, coalescing would change that value -> wrong results. Bail (the
    // kernel keeps its original, correct, uncoalesced path).
    bool readsAxisNumPrograms = false;
    moduleOp.walk([&](triton::GetNumProgramsOp np) {
        if (np.getAxisAsInt() == coalesceAxis) readsAxisNumPrograms = true;
    });
    if (readsAxisNumPrograms)
        return failure();

    // The H axis is folded into the coalesced tile, so every per-head value
    // must be expanded across the S lanes. Pattern-specific collectors find
    // scalar loads that are safe to lane-expand; they bail when an H-dependent
    // value reaches something that cannot be represented as independent lanes.
    SmallVector<triton::LoadOp> headLoads;
    SmallVector<triton::LoadOp> flatDenseLoads;
    DenseMap<Operation *, Value> flatDenseBases;
    DenseMap<Operation *, unsigned> flatDenseSeedIndex;
    DenseSet<Operation *> seedOps;
    for (auto &seed : seeds)
        seedOps.insert(seed.load.getOperation());
    DenseSet<Operation *> seenFlatLoads;
    for (unsigned i = 0; i < seeds.size(); ++i) {
        if (failed(pattern.collectExtraLoads(moduleOp, seeds, i, seedOps,
                                             headLoads, flatDenseLoads,
                                             flatDenseBases,
                                             flatDenseSeedIndex,
                                             seenFlatLoads)))
            return failure();
    }

    // Discover the load->store subgraph by forward reachability from the seeds.
    // Every op on the way must be 2D-safe (elementwise / cast / splat / T-axis
    // scan or reduce); stores are the sinks. Any unsafe op (or a value escaping
    // to one) aborts the whole rewrite.
    DenseSet<Operation *> region;
    SmallVector<triton::StoreOp> sinks;
    DenseSet<Operation *> visited;
    SmallVector<Operation *> wl;
    for (auto s : seeds)
        for (Operation *u : s.load.getResult().getUsers()) wl.push_back(u);
    while (!wl.empty()) {
        Operation *op = wl.pop_back_val();
        if (!visited.insert(op).second) continue;
        if (auto st = dyn_cast<triton::StoreOp>(op)) { sinks.push_back(st); continue; }
        if (!is2DSafe(op)) return failure();
        region.insert(op);
        for (Value r : op->getResults())
            for (Operation *u : r.getUsers()) wl.push_back(u);
    }
    if (sinks.empty()) return failure();

    // Every sink store must also match the same concrete addressing family as
    // the root load: block-ptr stride-S, or addptr jagged/dense row addressing.
    SmallVector<typename PatternT::Store> coalescedStores;
    for (auto st : sinks) {
        typename PatternT::Store store;
        if (!pattern.matchStore(st, seeds, S, store))
            return failure();
        coalescedStores.push_back(store);
    }

    // Map each 1D value to its 2D counterpart, materializing splats/constants
    // on demand. Returns null to signal an un-liftable operand (bail).
    DenseMap<Value, Value> vmap;
    std::function<Value(Value)> get2D = [&](Value v) -> Value {
        auto it = vmap.find(v);
        if (it != vmap.end()) return it->second;
        if (!isa<RankedTensorType>(v.getType())) {
            // Scalar: if it derives from a per-head load (whose [S] lane vector is
            // already in vmap), lift the elementwise scalar chain to [S] so each
            // lane gets its own value. i_h-independent scalars stay scalar (they
            // splat uniformly across lanes, which is correct).
            Operation *def = v.getDefiningOp();
            bool liftable = def && (isa<math::MathDialect>(def->getDialect()) ||
                isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                    arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
                    arith::MaxNumFOp, arith::MinNumFOp, arith::ExtFOp,
                    arith::TruncFOp>(def));
            if (liftable) {
                SmallVector<Value> ops2;
                bool anyLane = false;
                for (Value o : def->getOperands()) {
                    Value n = get2D(o);
                    if (!n) return Value();
                    if (isa<RankedTensorType>(n.getType())) anyLane = true;
                    ops2.push_back(n);
                }
                if (anyLane) {
                    OpBuilder::InsertionGuard g(rw);
                    rw.setInsertionPointAfter(def);
                    for (Value &o : ops2)
                        if (!isa<RankedTensorType>(o.getType()))
                            o = rw.create<triton::SplatOp>(
                                def->getLoc(), RankedTensorType::get({S}, o.getType()), o);
                    OperationState st(def->getLoc(), def->getName());
                    st.addOperands(ops2);
                    st.addAttributes(def->getAttrs());
                    for (Value r : def->getResults())
                        st.addTypes(RankedTensorType::get({S}, r.getType()));
                    Operation *nu = rw.create(st);
                    vmap[v] = nu->getResult(0);
                    return nu->getResult(0);
                }
            }
            return v;  // i_h-independent scalar, splats uniformly
        }
        // Save/restore the insertion point: the materializers below move it next
        // to the original splat/constant (which may sit at the top of the func).
        // Without this the caller's `rw.setInsertionPoint(op)` would be clobbered
        // and the rebuilt op emitted before its operands -> dominance violation.
        OpBuilder::InsertionGuard guard(rw);
        if (auto sp = v.getDefiningOp<triton::SplatOp>()) {
            Value src2 = get2D(sp.getSrc());
            if (!src2) return Value();
            rw.setInsertionPointAfter(sp);
            Value n;
            if (isa<RankedTensorType>(src2.getType())) {
                // src2 is the per-lane reduce result [S] (the 1D scalar source
                // became a vector once the reduce was 2D-ified). Broadcast it
                // across T: [S] -> [1,S] -> [BT,S] for block-ptr, or
                // [S] -> [S,1] -> [S,BT] for addptr.
                Value ex = rw.create<triton::ExpandDimsOp>(
                    sp.getLoc(), src2, laneInner ? 0 : 1);
                n = rw.create<triton::BroadcastOp>(sp.getLoc(),
                                                   lift2D(sp.getType(), S,
                                                          laneInner),
                                                   ex);
            } else {
                // True scalar splat: lift to a 2D splat over the coalesced tile.
                n = rw.create<triton::SplatOp>(sp.getLoc(),
                                               lift2D(sp.getType(), S,
                                                      laneInner),
                                               src2);
            }
            vmap[v] = n;
            return n;
        }
        if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
            if (auto dea = dyn_cast<DenseElementsAttr>(c.getValue())) {
                if (dea.isSplat()) {
                    auto nt = cast<RankedTensorType>(
                        lift2D(c.getType(), S, laneInner));
                    rw.setInsertionPointAfter(c);
                    Value n = rw.create<arith::ConstantOp>(
                        c.getLoc(), nt,
                        DenseElementsAttr::get(nt, dea.getSplatValue<Attribute>()));
                    vmap[v] = n;
                    return n;
                }
            }
        }
        if (Operation *def = v.getDefiningOp()) {
            if (is2DSafe(def) && def->getNumResults() == 1) {
                SmallVector<Value> operands;
                bool any2D = false;
                for (Value o : def->getOperands()) {
                    Value n = get2D(o);
                    if (!n) return Value();
                    if (isa<RankedTensorType>(n.getType())) any2D = true;
                    operands.push_back(n);
                }
                if (any2D) {
                    rw.setInsertionPointAfter(def);
                    OperationState st(def->getLoc(), def->getName());
                    st.addOperands(operands);
                    st.addAttributes(def->getAttrs());
                    st.addTypes(lift2D(def->getResult(0).getType(), S,
                                       laneInner));
                    Operation *nu = rw.create(st);
                    vmap[v] = nu->getResult(0);
                    return nu->getResult(0);
                }
            }
        }
        return Value();
    };

    // Build 2D loads for the seeds.
    for (auto seed : seeds) {
        Value load2D = pattern.buildLoad2D(rw, seed, get2D);
        if (!load2D) return failure();
        vmap[seed.load.getResult()] = load2D;
    }

    // Lift each per-head scalar load to an [S] per-lane vector load: base[0:S]
    // (lane s = base[s], matching the folded i_h = s). The offset must be
    // exactly i_h = pid % S so that lane s maps to base[s]; otherwise bail.
    for (auto ld : headLoads) {
        auto ap = ld.getPtr().getDefiningOp<triton::AddPtrOp>();
        if (!ap) return failure();
        Value off = ap.getOffset();
        while (auto e = off.getDefiningOp<arith::ExtSIOp>()) off = e.getIn();
        while (auto t = off.getDefiningOp<arith::TruncIOp>()) off = t.getIn();
        if (!off.getDefiningOp<arith::RemSIOp>()) return failure();
        Value base = ap.getPtr();
        rw.setInsertionPoint(ld);
        auto loc = ld.getLoc();
        Value cS = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(S));
        Value c1 = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(1));
        Value c0 = rw.create<arith::ConstantOp>(loc, rw.getI32IntegerAttr(0));
        SmallVector<Value, 1> shape{cS}, strides{c1}, offsets{c0};
        SmallVector<int32_t, 1> blockShape{static_cast<int32_t>(S)}, order{0};
        auto p = rw.create<triton::MakeTensorPtrOp>(loc, base, shape, strides,
                                                    offsets, blockShape, order);
        auto vl = rw.create<triton::LoadOp>(
            loc, p.getResult(), ArrayRef<int32_t>{0}, triton::PaddingOption::PAD_ZERO,
            triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
        vmap[ld.getResult()] = vl.getResult();
    }

    // AddPtrPattern dense row loads use output_row in the original IR. Folded
    // form loads all H lanes for one batch as a vector [S].
    for (auto ld : flatDenseLoads) {
        auto baseIt = flatDenseBases.find(ld.getOperation());
        auto seedIt = flatDenseSeedIndex.find(ld.getOperation());
        if (baseIt == flatDenseBases.end() ||
            seedIt == flatDenseSeedIndex.end())
            return failure();
        vmap[ld.getResult()] =
            pattern.buildDenseLoad2D(rw, ld, seeds[seedIt->second],
                                     baseIt->second);
    }

    // Rebuild the region ops in IR (topological) order as 2D.
    SmallVector<Operation *> ordered;
    moduleOp.walk([&](Operation *op) { if (region.count(op)) ordered.push_back(op); });
    for (Operation *op : ordered) {
        rw.setInsertionPoint(op);
        if (auto scan = dyn_cast<triton::ScanOp>(op)) {
            Value in = get2D(scan.getOperand(0));
            if (!in) return failure();
            int axis = laneInner ? scan.getAxis() : scan.getAxis() + 1;
            auto ns = rw.create<triton::ScanOp>(scan.getLoc(), ValueRange{in},
                                                axis, scan.getReverse());
            rw.cloneRegionBefore(scan.getCombineOp(), ns.getCombineOp(),
                                 ns.getCombineOp().end());
            vmap[scan->getResult(0)] = ns->getResult(0);
            continue;
        }
        if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
            Value in = get2D(reduce.getOperand(0));
            if (!in) return failure();
            // T-axis reduce on the 2D tile -> [S]: one independent reduction
            // per lane (the S lanes do not mix). The 1D result was a scalar;
            // its 2D counterpart is the per-lane vector [S], which a downstream
            // splat turns into an expand_dims+broadcast (see get2D).
            int axis = laneInner ? reduce.getAxis() : reduce.getAxis() + 1;
            auto nr = rw.create<triton::ReduceOp>(reduce.getLoc(), ValueRange{in},
                                                  axis);
            rw.cloneRegionBefore(reduce.getCombineOp(), nr.getCombineOp(),
                                 nr.getCombineOp().end());
            vmap[reduce->getResult(0)] = nr->getResult(0);
            continue;
        }
        // Splats are materialized on demand by get2D (which lifts a scalar splat
        // to a 2D splat, and a per-lane reduce result [S] to expand_dims +
        // broadcast). Skip here so it is not rebuilt as an invalid 2D splat.
        if (isa<triton::SplatOp>(op)) continue;
        SmallVector<Value> operands;
        for (Value o : op->getOperands()) {
            Value n = get2D(o);
            if (!n) return failure();
            operands.push_back(n);
        }
        OperationState st(op->getLoc(), op->getName());
        st.addOperands(operands);
        st.addAttributes(op->getAttrs());
        for (Value r : op->getResults())
            st.addTypes(lift2D(r.getType(), S, laneInner));
        Operation *nu = rw.create(st);
        for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
            vmap[oldR] = newR;
    }

    // Build the 2D stores.
    for (const auto &store : coalescedStores) {
        triton::StoreOp st = store.store;
        Value val = get2D(st.getValue());
        if (!val) return failure();
        if (!pattern.buildStore2D(rw, store, val, S, BT))
            return failure();
    }

    // i_b = divsi(get_program_id, S): with the H axis folded into the inner
    // tile the per-instance i_b becomes the raw program id. Redirect it when
    // the concrete pattern has such a folded batch id.
    pattern.rewriteFoldedBatchId(rw, seeds.front(), S);

    // Erase the original chain (sinks, then region in reverse order, then seeds).
    for (auto st : sinks) rw.eraseOp(st);
    for (auto it = ordered.rbegin(); it != ordered.rend(); ++it) rw.eraseOp(*it);
    for (auto seed : seeds) rw.eraseOp(seed.load);

    auto i32t = IntegerType::get(moduleOp.getContext(), 32);
    moduleOp->setAttr("hacc.coalesce_factor", IntegerAttr::get(i32t, S));
    moduleOp->setAttr("hacc.coalesce_axis", IntegerAttr::get(i32t, coalesceAxis));
    return success();
}

// Coalesces block-pointer kernels that split a contiguous H axis onto the launch
// grid:
//   before: tt.load make_tensor_ptr(base + (pid % S), ..., stride S), tile [BT]
//   after:  tt.load make_tensor_ptr(base, ..., shape [BT,S], stride inner 1)
// The rewrite also redirects `pid / S` users to `pid`, because the launcher will
// divide that grid dimension by S after `hacc.coalesce_factor` is set.
struct BlockPtrPattern final : public OpRewritePattern<triton::LoadOp> {
    using OpRewritePattern<triton::LoadOp>::OpRewritePattern;
    static constexpr bool kLaneInner = true;

    struct Seed {
        triton::LoadOp load;
        int64_t S = 0;
        int64_t BT = 0;
        int32_t coalesceAxis = -1;
        triton::MakeTensorPtrOp blockPtr;
    };
    struct Store {
        triton::StoreOp store;
        triton::MakeTensorPtrOp blockPtr;
    };

    LogicalResult matchAndRewrite(triton::LoadOp load,
                                  PatternRewriter &rewriter) const override {
        ModuleOp moduleOp = load->getParentOfType<ModuleOp>();
        if (!moduleOp || moduleOp->hasAttr("hacc.coalesce_factor"))
            return failure();

        Seed seed;
        if (!matchLoad(load, seed))
            return failure();
        return rewriteCoalescedRegion(moduleOp, *this, seed, rewriter);
    }

    bool matchLoad(triton::LoadOp load, Seed &seed) const {
        auto m = load.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        if (!m) return false;
        auto rt = dyn_cast<RankedTensorType>(load.getResult().getType());
        if (!rt || rt.getRank() != 1) return false;
        auto strides = m.getStrides();
        if (strides.empty()) return false;
        APInt sC;
        if (!matchPattern(strides.back(), m_ConstantInt(&sC))) return false;
        int64_t s = std::abs(sC.getSExtValue());
        if (s <= 1 || !findIhAddPtr(m.getBase(), s)) return false;

        seed.load = load;
        seed.S = s;
        seed.BT = rt.getShape()[0];
        seed.coalesceAxis = findIhAxis(m.getBase(), s);
        seed.blockPtr = m;
        return seed.coalesceAxis >= 0;
    }

    bool matchStore(triton::StoreOp store, ArrayRef<Seed> seeds, int64_t S,
                    Store &sink) const {
        auto m = store.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        if (!m || !findIhAddPtr(m.getBase(), S)) return false;
        auto strides = m.getStrides();
        if (strides.empty()) return false;
        APInt sC;
        if (!matchPattern(strides.back(), m_ConstantInt(&sC)) ||
            std::abs(sC.getSExtValue()) != S)
            return false;

        sink.store = store;
        sink.blockPtr = m;
        return true;
    }

    LogicalResult collectExtraLoads(ModuleOp, ArrayRef<Seed> seeds,
                                    unsigned seedIndex,
                                    DenseSet<Operation *> &,
                                    SmallVectorImpl<triton::LoadOp> &headLoads,
                                    SmallVectorImpl<triton::LoadOp> &,
                                    DenseMap<Operation *, Value> &,
                                    DenseMap<Operation *, unsigned> &,
                                    DenseSet<Operation *> &) const {
        const Seed &seed = seeds[seedIndex];
        triton::MakeTensorPtrOp blockPtr = seed.blockPtr;
        Value ihRem = findIhRem(blockPtr.getBase(), seed.S);
        if (!ihRem) return success();

        SmallVector<Operation *> worklist(ihRem.getUsers().begin(),
                                          ihRem.getUsers().end());
        DenseSet<Operation *> seen;
        while (!worklist.empty()) {
            Operation *user = worklist.pop_back_val();
            if (!seen.insert(user).second) continue;
            if (isa<triton::MakeTensorPtrOp>(user)) continue;
            if (auto load = dyn_cast<triton::LoadOp>(user)) {
                // Per-head scalar load: must be scalar and must not itself feed
                // an address (indirect gather). Then it is liftable to [S].
                if (isa<RankedTensorType>(load.getResult().getType()))
                    return failure();
                for (Operation *resultUser : load.getResult().getUsers())
                    if (isa<triton::AddPtrOp>(resultUser))
                        return failure();
                headLoads.push_back(load);
                continue;
            }
            if (isa<triton::AddPtrOp, arith::ExtSIOp, arith::TruncIOp,
                    arith::RemSIOp, arith::AddIOp, arith::MulIOp>(user))
                for (Operation *nextUser : user->getResult(0).getUsers())
                    worklist.push_back(nextUser);
        }
        return success();
    }

    Value buildLoad2D(RewriterBase &rw, const Seed &seed, LiftValueFn &) const {
        triton::LoadOp load = seed.load;
        Value ptr = build2DBlockPtr(rw, seed.blockPtr, seed.S, seed.BT);
        if (!ptr) return Value();
        rw.setInsertionPoint(load);
        return rw.create<triton::LoadOp>(
                     load.getLoc(), ptr, ArrayRef<int32_t>{0, 1},
                     load.getPadding(), load.getCache(), load.getEvict(),
                     load.getIsVolatile())
            .getResult();
    }

    Value buildDenseLoad2D(RewriterBase &, triton::LoadOp, const Seed &,
                           Value) const {
        return Value();
    }

    bool buildStore2D(RewriterBase &rw, const Store &sink, Value value,
                      int64_t S, int64_t BT) const {
        triton::StoreOp store = sink.store;
        Value ptr = build2DBlockPtr(rw, sink.blockPtr, S, BT);
        if (!ptr) return false;
        rw.setInsertionPoint(store);
        rw.create<triton::StoreOp>(store.getLoc(), ptr, value,
                                   ArrayRef<int32_t>{0, 1}, store.getCache(),
                                   store.getEvict());
        return true;
    }

    void rewriteFoldedBatchId(RewriterBase &rw, const Seed &seed,
                              int64_t S) const {
        triton::MakeTensorPtrOp blockPtr = seed.blockPtr;
        triton::AddPtrOp ih = findIhAddPtr(blockPtr.getBase(), S);
        if (!ih) return;
        auto rem = ih.getOffset().getDefiningOp<arith::RemSIOp>();
        if (!rem) return;
        Value lhs = rem.getLhs();
        while (true) {
            if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) { lhs = e.getIn(); continue; }
            if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) { lhs = t.getIn(); continue; }
            break;
        }
        SmallVector<arith::DivSIOp, 2> divs;
        for (Operation *user : lhs.getUsers())
            if (auto div = dyn_cast<arith::DivSIOp>(user)) {
                APInt dC;
                if (div.getLhs() == lhs &&
                    matchPattern(div.getRhs(), m_ConstantInt(&dC)) &&
                    std::abs(dC.getSExtValue()) == S)
                    divs.push_back(div);
            }
        for (auto div : divs)
            rw.replaceAllUsesWith(div.getResult(), lhs);
    }

private:
    // Detects the FLA per-head strided base `base + (pid % S)` produced by
    // splitting the H axis (the contiguous axis folded onto the grid). Returns the
    // matching AddPtrOp, or a null AddPtrOp if `base` is not such an ih-split ptr.
    static triton::AddPtrOp findIhAddPtr(Value base, int64_t S) {
        Value src = base;
        while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
            if (isa<RankedTensorType>(addptr.getPtr().getType())) break;
            if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
                APInt cC;
                if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
                    std::abs(cC.getSExtValue()) == S) {
                    Value lhs = rem.getLhs();
                    while (true) {
                        if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) { lhs = e.getIn(); continue; }
                        if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) { lhs = t.getIn(); continue; }
                        break;
                    }
                    if (lhs.getDefiningOp<triton::GetProgramIdOp>())
                        return addptr;
                }
            }
            src = addptr.getPtr();
        }
        return triton::AddPtrOp();
    }

    // Mirror of findIhAddPtr that returns the program_id axis driving the ih split
    // (i.e. the grid dim the host launcher must divide by S), or -1 if `base` is
    // not such an ih-split ptr. Whenever findIhAddPtr succeeds this does too.
    static int32_t findIhAxis(Value base, int64_t S) {
        Value src = base;
        while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
            if (isa<RankedTensorType>(addptr.getPtr().getType())) break;
            if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
                APInt cC;
                if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
                    std::abs(cC.getSExtValue()) == S) {
                    Value lhs = rem.getLhs();
                    while (true) {
                        if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) { lhs = e.getIn(); continue; }
                        if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) { lhs = t.getIn(); continue; }
                        break;
                    }
                    if (auto pid = lhs.getDefiningOp<triton::GetProgramIdOp>())
                        return pid.getAxisAsInt();
                }
            }
            src = addptr.getPtr();
        }
        return -1;
    }

    // Returns the i_h value `pid % S` (the per-head index feeding the ih split), or
    // null. Mirror of findIhAddPtr but yields the RemSIOp result itself, used to
    // check whether i_h also feeds a per-head scalar load (which coalescing cannot
    // lane-expand -- see the correctness guard in rewriteStridedAxisCoalesce).
    static Value findIhRem(Value base, int64_t S) {
        Value src = base;
        while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
            if (isa<RankedTensorType>(addptr.getPtr().getType())) break;
            if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
                APInt cC;
                if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
                    std::abs(cC.getSExtValue()) == S)
                    return rem.getResult();
            }
            src = addptr.getPtr();
        }
        return Value();
    }

    static Value build2DBlockPtr(RewriterBase &rw,
                                 triton::MakeTensorPtrOp m1d,
                                 int64_t S, int64_t BT) {
        triton::AddPtrOp ih = findIhAddPtr(m1d.getBase(), S);
        if (!ih) return Value();
        auto loc = m1d.getLoc();
        rw.setInsertionPoint(m1d);
        Value newBase = ih.getPtr();
        Value cH = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(S));
        Value c1 = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(1));
        Value c0 = rw.create<arith::ConstantOp>(loc, rw.getI32IntegerAttr(0));
        SmallVector<Value, 2> shape{m1d.getShape()[0], cH};
        SmallVector<Value, 2> strides{m1d.getStrides()[0], c1};
        SmallVector<Value, 2> offsets{m1d.getOffsets()[0], c0};
        SmallVector<int32_t, 2> blockShape{static_cast<int32_t>(BT),
                                           static_cast<int32_t>(S)};
        SmallVector<int32_t, 2> order{1, 0};
        auto p = rw.create<triton::MakeTensorPtrOp>(
            loc, newBase, shape, strides, offsets, blockShape, order);
        return p.getResult();
    }
};

// Coalesces addptr-addressed jagged/dense kernels where the launch grid flattens
// `(batch, h, column-tile)` into one pid:
//   before: output_row = pid / cdiv(D, BT), h = output_row % S,
//           ptr = splat(base + begin + h * D) + (range(0, BT) + group * BT)
//   after:  materialize [S,BT] offsets so one program computes every h lane for
//           the same batch and column tile while keeping columns contiguous.
//           Per-row dense scalar loads become [S] vector loads and are broadcast
//           through the shared 2D rewrite.
struct AddPtrPattern final : public OpRewritePattern<triton::AddPtrOp> {
    using OpRewritePattern<triton::AddPtrOp>::OpRewritePattern;
    static constexpr bool kLaneInner = false;

    struct Access {
        enum class RowKind { Jagged, Dense };

        Value outputRow;
        Value batch;
        Value lane;
        Value d;
        Value cols;
        Value colMask;
        Value rowMask;
        Value begin;
        Value base;
        RowKind rowKind = RowKind::Jagged;
    };
    struct Seed {
        triton::LoadOp load;
        int64_t S = 0;
        int64_t BT = 0;
        int32_t coalesceAxis = -1;
        Access access;
    };
    struct Store {
        triton::StoreOp store;
        Access access;
        Value base;
        Value baseOffset;
        bool includeBatch = true;
    };

    LogicalResult matchAndRewrite(triton::AddPtrOp addPtr,
                                  PatternRewriter &rewriter) const override {
        ModuleOp moduleOp = addPtr->getParentOfType<ModuleOp>();
        if (!moduleOp || moduleOp->hasAttr("hacc.coalesce_factor"))
            return failure();

        Seed seed;
        if (!matchLoadAddress(addPtr, seed))
            return failure();
        return rewriteCoalescedRegion(moduleOp, *this, seed, rewriter);
    }

    bool matchLoad(triton::LoadOp load, Seed &seed) const {
        auto addPtr = load.getPtr().getDefiningOp<triton::AddPtrOp>();
        if (!addPtr) return false;
        return matchLoadAddress(addPtr, load, seed);
    }

    bool matchLoadAddress(triton::AddPtrOp addPtr, Seed &seed) const {
        bool found = false;
        for (Operation *user : addPtr.getResult().getUsers()) {
            auto load = dyn_cast<triton::LoadOp>(user);
            if (!load || load.getPtr() != addPtr.getResult())
                return false;

            Seed candidate;
            if (!matchLoadAddress(addPtr, load, candidate))
                return false;
            if (!found) {
                seed = candidate;
                found = true;
            }
        }
        return found;
    }

    bool matchLoadAddress(triton::AddPtrOp addPtr, triton::LoadOp load,
                          Seed &seed) const {
        return matchStridedLoad(addPtr, load, seed);
    }

    bool matchStore(triton::StoreOp store, ArrayRef<Seed> seeds, int64_t,
                    Store &sink) const {
        const Access *access = nullptr;
        Value base;
        for (const Seed &seed : seeds) {
            auto addPtr = store.getPtr().getDefiningOp<triton::AddPtrOp>();
            if (!addPtr || addPtr.getOffset() != seed.access.cols) continue;
            Value rowPtr;
            if (auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>())
                rowPtr = splat.getSrc();
            if (!rowPtr) continue;
            if (store.getMask() != seed.access.colMask) continue;
            Value baseOffset;
            bool includeBatch = true;
            if (!matchDenseRowPtr(rowPtr, seed.access.outputRow,
                                  seed.access.d, base)) {
                if (!seed.access.begin ||
                    !matchJaggedRowPtr(rowPtr, seed.access.begin,
                                       seed.access.lane, seed.access.d, base))
                    continue;
                baseOffset = seed.access.begin;
                includeBatch = false;
            }
            access = &seed.access;
            sink.baseOffset = baseOffset;
            sink.includeBatch = includeBatch;
            break;
        }
        if (!access) return false;

        sink.store = store;
        sink.access = *access;
        sink.base = base;
        return true;
    }

    LogicalResult collectExtraLoads(ModuleOp moduleOp, ArrayRef<Seed> seeds,
                                    unsigned seedIndex,
                                    DenseSet<Operation *> &seedOps,
                                    SmallVectorImpl<triton::LoadOp> &,
                                    SmallVectorImpl<triton::LoadOp> &denseLoads,
                                    DenseMap<Operation *, Value> &denseBases,
                                    DenseMap<Operation *, unsigned> &denseSeedIndex,
                                    DenseSet<Operation *> &seenDenseLoads) const {
        const Seed &seed = seeds[seedIndex];
        // Dense side inputs indexed by output_row are scalar in the original
        // kernel. After folding H into the tile they become [S] lane loads.
        moduleOp.walk([&](triton::LoadOp load) {
            if (seedOps.contains(load.getOperation())) return;
            if (isa<RankedTensorType>(load.getType())) return;
            Value ptr = load.getPtr();
            if (auto addPtr = ptr.getDefiningOp<triton::AddPtrOp>()) {
                auto zero = getConstantIntValue(addPtr.getOffset());
                if (zero && *zero == 0)
                    ptr = addPtr.getPtr();
            }
            auto addPtr = ptr.getDefiningOp<triton::AddPtrOp>();
            if (!addPtr || addPtr.getOffset() != seed.access.outputRow) return;
            if (load.getMask() && seed.access.rowMask &&
                load.getMask() != seed.access.rowMask)
                return;
            if (!seenDenseLoads.insert(load.getOperation()).second) return;
            denseLoads.push_back(load);
            denseBases[load.getOperation()] = addPtr.getPtr();
            denseSeedIndex[load.getOperation()] = seedIndex;
        });
        return success();
    }

    Value buildLoad2D(RewriterBase &rw, const Seed &seed,
                      LiftValueFn &get2D) const {
        triton::LoadOp load = seed.load;
        rw.setInsertionPoint(load);
        Value other = load.getOther() ? get2D(load.getOther()) : Value();
        if (load.getOther() && !other) return Value();
        const Access &access = seed.access;
        Location loc = load.getLoc();
        auto resultType = cast<RankedTensorType>(load.getType());
        auto tileType = cast<RankedTensorType>(
            lift2D(resultType, seed.S, kLaneInner));
        auto tilePtrType =
            RankedTensorType::get({seed.S, seed.BT}, access.base.getType());
        auto tileI1Ty =
            RankedTensorType::get({seed.S, seed.BT}, rw.getI1Type());

        // Row offsets supply the coalesced H dimension, cols supplies the
        // column tile. Broadcast both into [S,BT] before rebuilding addptr.
        Value offsets = buildOffsets2D(
            rw, loc, access, seed.S, seed.BT,
            access.rowKind == Access::RowKind::Jagged
                                 ? access.begin
                                 : Value(),
            access.rowKind == Access::RowKind::Dense);
        Value base = rw.create<triton::SplatOp>(loc, tilePtrType, access.base);
        Value ptrs =
            rw.create<triton::AddPtrOp>(loc, tilePtrType, base, offsets);

        Value expandedColMask =
            rw.create<triton::ExpandDimsOp>(loc, access.colMask, 0);
        Value colMask2d =
            rw.create<triton::BroadcastOp>(loc, tileI1Ty, expandedColMask);
        Value mask = colMask2d;
        if (access.rowMask) {
            Value rowMask2d = rw.create<triton::SplatOp>(
                loc, RankedTensorType::get({seed.S, seed.BT}, rw.getI1Type()),
                access.rowMask);
            mask = rw.create<arith::AndIOp>(loc, colMask2d, rowMask2d);
        }
        if (!other) {
            Attribute zero = rw.getZeroAttr(tileType.getElementType());
            other = rw.create<arith::ConstantOp>(
                loc, DenseElementsAttr::get(tileType, zero));
        }
        return rw.create<triton::LoadOp>(
                     loc, ptrs, mask, other, ArrayRef<int32_t>{}, nullptr,
                     load.getCache(), load.getEvict(), load.getIsVolatile())
            .getResult();
    }

    Value buildDenseLoad2D(RewriterBase &rw, triton::LoadOp load,
                           const Seed &seed, Value base) const {
        rw.setInsertionPoint(load);
        const Access &access = seed.access;
        Location loc = load.getLoc();
        Type elemTy = load.getType();
        Type i32Ty = rw.getI32Type();
        auto hI32Ty = RankedTensorType::get({seed.S}, i32Ty);
        auto hElemTy = RankedTensorType::get({seed.S}, elemTy);
        auto hPtrTy = RankedTensorType::get({seed.S}, base.getType());

        // Dense row load: original base[output_row] becomes base[batch*S + h].
        Value cS = rw.create<arith::ConstantIntOp>(loc, seed.S, 32);
        Value hs = rw.create<triton::MakeRangeOp>(loc, hI32Ty, 0, seed.S);
        Value batchS = rw.create<arith::MulIOp>(loc, access.batch, cS);
        Value batchSplat = rw.create<triton::SplatOp>(
            loc, RankedTensorType::get({seed.S}, i32Ty), batchS);
        Value offsets = rw.create<arith::AddIOp>(loc, batchSplat, hs);
        Value baseSplat = rw.create<triton::SplatOp>(loc, hPtrTy, base);
        Value ptrs =
            rw.create<triton::AddPtrOp>(loc, hPtrTy, baseSplat, offsets);
        Value mask = Value();
        if (access.rowMask)
            mask = rw.create<triton::SplatOp>(
                loc, RankedTensorType::get({seed.S}, rw.getI1Type()),
                access.rowMask);
        Attribute zero = rw.getZeroAttr(hElemTy.getElementType());
        Value other = rw.create<arith::ConstantOp>(
            loc, DenseElementsAttr::get(hElemTy, zero));
        return rw.create<triton::LoadOp>(
                     loc, ptrs, mask, other, ArrayRef<int32_t>{}, nullptr,
                     load.getCache(), load.getEvict(), load.getIsVolatile())
            .getResult();
    }

    bool buildStore2D(RewriterBase &rw, const Store &sink, Value value,
                      int64_t S, int64_t BT) const {
        triton::StoreOp store = sink.store;
        rw.setInsertionPoint(store);
        const Access &access = sink.access;
        Location loc = store.getLoc();
        auto tilePtrType =
            RankedTensorType::get({S, BT}, sink.base.getType());
        auto tileI1Ty =
            RankedTensorType::get({S, BT}, rw.getI1Type());
        // Stores may use either dense output_row rows or the same jagged row
        // form as loads; the matched store carries that choice.
        Value offsets = buildOffsets2D(rw, loc, access, S, BT, sink.baseOffset,
                                       sink.includeBatch);
        Value baseSplat =
            rw.create<triton::SplatOp>(loc, tilePtrType, sink.base);
        Value ptrs =
            rw.create<triton::AddPtrOp>(loc, tilePtrType, baseSplat, offsets);
        Value expandedMask =
            rw.create<triton::ExpandDimsOp>(loc, access.colMask, 0);
        Value mask =
            rw.create<triton::BroadcastOp>(loc, tileI1Ty, expandedMask);
        rw.create<triton::StoreOp>(loc, ptrs, value, mask);
        return true;
    }

    void rewriteFoldedBatchId(RewriterBase &rw, const Seed &seed,
                              int64_t) const {
        // Before coalescing:
        //   output_row = pid / grid_dim
        //   batch      = output_row / S
        //   lane       = output_row % S
        // After the launcher divides the pid axis by S, output_row already is
        // the folded batch id. Leaving `batch = output_row / S` would make the
        // rewritten kernel read/write only one out of every S batches.
        rw.replaceAllUsesWith(seed.access.batch, seed.access.outputRow);
    }

private:
    // Recognizes the launch-grid flattening:
    // pid -> output_row = pid / grid_dim, group = pid % grid_dim,
    // output_row -> batch = output_row / S, optional lane = output_row % S.
    // Dense row addresses only need the batch split; jagged row addresses check
    // the lane value when matching their row pointer.
    bool matchFlattenedRowAndLane(triton::GetProgramIdOp pidOp,
                                  Value &gridDim, Value &outputRow,
                                  Value &group, Value &batch, Value &lane,
                                  int64_t &S) const {
        for (Operation *user : pidOp.getResult().getUsers()) {
            auto div = dyn_cast<arith::DivSIOp>(user);
            if (!div || div.getLhs() != pidOp.getResult()) continue;

            Value candidateGridDim = div.getRhs();
            Value candidateGroup;
            for (Operation *maybeRemUser : pidOp.getResult().getUsers()) {
                auto rem = dyn_cast<arith::RemSIOp>(maybeRemUser);
                if (rem && rem.getLhs() == pidOp.getResult() &&
                    rem.getRhs() == candidateGridDim) {
                    candidateGroup = rem.getResult();
                    break;
                }
            }
            if (!candidateGroup) continue;

            for (Operation *rowUser : div.getResult().getUsers()) {
                auto batchDiv = dyn_cast<arith::DivSIOp>(rowUser);
                if (!batchDiv || batchDiv.getLhs() != div.getResult())
                    continue;
                auto candidateS = getConstantIntValue(batchDiv.getRhs());
                if (!candidateS || *candidateS <= 1) continue;

                Value candidateLane;
                for (Operation *maybeLaneUser : div.getResult().getUsers()) {
                    auto rem = dyn_cast<arith::RemSIOp>(maybeLaneUser);
                    if (rem && rem.getLhs() == div.getResult() &&
                        getConstantIntValue(rem.getRhs()) &&
                        *getConstantIntValue(rem.getRhs()) == *candidateS) {
                        candidateLane = rem.getResult();
                        break;
                    }
                }

                gridDim = candidateGridDim;
                outputRow = div.getResult();
                group = candidateGroup;
                batch = batchDiv.getResult();
                lane = candidateLane;
                S = *candidateS;
                return true;
            }
        }
        return false;
    }

    // Matches `range(0, BT) + group * BT` and the companion `cols < D` mask.
    bool matchColumnOffsets(Value cols, Value group, Value &d,
                            Value &colMask, int64_t &BT) const {
        auto add = cols.getDefiningOp<arith::AddIOp>();
        if (!add) return false;

        triton::MakeRangeOp range;
        Value colBase;
        if ((range = add.getLhs().getDefiningOp<triton::MakeRangeOp>())) {
            if (auto splat = add.getRhs().getDefiningOp<triton::SplatOp>())
                colBase = splat.getSrc();
        } else if ((range = add.getRhs().getDefiningOp<triton::MakeRangeOp>())) {
            if (auto splat = add.getLhs().getDefiningOp<triton::SplatOp>())
                colBase = splat.getSrc();
        }
        if (!range || range.getStart() != 0 || !colBase) return false;

        BT = range.getEnd();
        auto colBaseMul = colBase.getDefiningOp<arith::MulIOp>();
        if (BT <= 1 || !colBaseMul)
            return false;
        auto lhsConst = getConstantIntValue(colBaseMul.getLhs());
        auto rhsConst = getConstantIntValue(colBaseMul.getRhs());
        if (!((colBaseMul.getLhs() == group && rhsConst && *rhsConst == BT) ||
              (colBaseMul.getRhs() == group && lhsConst && *lhsConst == BT)))
            return false;

        for (Operation *user : cols.getUsers()) {
            auto cmp = dyn_cast<arith::CmpIOp>(user);
            if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
                cmp.getLhs() != cols)
                continue;
            auto splat = cmp.getRhs().getDefiningOp<triton::SplatOp>();
            if (!splat) continue;
            d = splat.getSrc();
            colMask = cmp.getResult();
            return true;
        }
        return false;
    }

    // Finds the row pointer table accesses `begin = offsets[batch]` and
    // `end = offsets[batch + 1]`; begin is needed for the jagged data address.
    bool matchBeginEnd(Value batch, Value &begin) const {
        for (Operation *user : batch.getUsers()) {
            auto beginPtr = dyn_cast<triton::AddPtrOp>(user);
            if (!beginPtr || beginPtr.getOffset() != batch) continue;

            triton::LoadOp beginLoad;
            for (Operation *ptrUser : beginPtr.getResult().getUsers()) {
                beginLoad = dyn_cast<triton::LoadOp>(ptrUser);
                if (beginLoad) break;
            }
            if (!beginLoad) continue;

            Value base = beginPtr.getPtr();
            for (Operation *batchUser : batch.getUsers()) {
                auto add = dyn_cast<arith::AddIOp>(batchUser);
                if (!add ||
                    !((add.getLhs() == batch &&
                       getConstantIntValue(add.getRhs()) &&
                       *getConstantIntValue(add.getRhs()) == 1) ||
                      (add.getRhs() == batch &&
                       getConstantIntValue(add.getLhs()) &&
                       *getConstantIntValue(add.getLhs()) == 1)))
                    continue;
                for (Operation *nextUser : add.getResult().getUsers()) {
                    auto endPtr = dyn_cast<triton::AddPtrOp>(nextUser);
                    if (!endPtr || endPtr.getPtr() != base ||
                        endPtr.getOffset() != add.getResult())
                        continue;
                    for (Operation *ptrUser : endPtr.getResult().getUsers()) {
                        auto endLoad = dyn_cast<triton::LoadOp>(ptrUser);
                        if (!endLoad) continue;
                        begin = beginLoad.getResult();
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // Matches the jagged row base `base + begin + lane * D`.
    bool matchJaggedRowPtr(Value rowPtr, Value begin, Value lane, Value d,
                           Value &base) const {
        auto addPtr = rowPtr.getDefiningOp<triton::AddPtrOp>();
        if (!addPtr) return false;
        auto add = addPtr.getOffset().getDefiningOp<arith::AddIOp>();
        if (!add) return false;

        Value laneOffset = add.getLhs() == begin ? add.getRhs() : add.getLhs();
        while (true) {
            if (auto e = laneOffset.getDefiningOp<arith::ExtSIOp>()) { laneOffset = e.getIn(); continue; }
            if (auto t = laneOffset.getDefiningOp<arith::TruncIOp>()) { laneOffset = t.getIn(); continue; }
            break;
        }
        auto laneMul = laneOffset.getDefiningOp<arith::MulIOp>();
        if (!((add.getLhs() == begin || add.getRhs() == begin) && laneMul &&
              ((laneMul.getLhs() == lane && laneMul.getRhs() == d) ||
               (laneMul.getLhs() == d && laneMul.getRhs() == lane))))
            return false;

        base = addPtr.getPtr();
        return true;
    }

    // Matches a dense row base `base + output_row * D`.
    bool matchDenseRowPtr(Value rowPtr, Value outputRow, Value d,
                          Value &base) const {
        auto addPtr = rowPtr.getDefiningOp<triton::AddPtrOp>();
        auto mul = addPtr ? addPtr.getOffset().getDefiningOp<arith::MulIOp>()
                          : arith::MulIOp();
        if (!addPtr || !mul ||
            !((mul.getLhs() == outputRow && mul.getRhs() == d) ||
              (mul.getLhs() == d && mul.getRhs() == outputRow)))
            return false;
        base = addPtr.getPtr();
        return true;
    }

    // Splits the load mask into the required column mask and an optional row
    // mask. The row mask is kept scalar and later splatted across the tile.
    bool matchLoadMask(Value mask, Value colMask, Value &rowMask) const {
        if (mask == colMask) return true;
        auto andOp = mask.getDefiningOp<arith::AndIOp>();
        if (!andOp) return false;
        if (andOp.getLhs() == colMask) {
            if (auto splat = andOp.getRhs().getDefiningOp<triton::SplatOp>())
                rowMask = splat.getSrc();
            return static_cast<bool>(rowMask);
        }
        if (andOp.getRhs() == colMask) {
            if (auto splat = andOp.getLhs().getDefiningOp<triton::SplatOp>())
                rowMask = splat.getSrc();
            return static_cast<bool>(rowMask);
        }
        return false;
    }

    // Combines the local checks above into one addptr seed. This deliberately
    // accepts any matching pid in the entry block, so the pattern is tied to
    // address structure rather than a specific kernel name or argument order.
    bool matchStridedLoad(triton::AddPtrOp addPtr, triton::LoadOp load,
                          Seed &seed) const {
        auto vectorType = dyn_cast<RankedTensorType>(load.getType());
        if (!vectorType || vectorType.getRank() != 1) return false;
        Value rowPtr;
        if (auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>())
            rowPtr = splat.getSrc();
        if (!rowPtr) return false;

        auto func = load->getParentOfType<triton::FuncOp>();
        if (!func) return false;
        Block &body = func.getBody().front();
        for (Operation &op : body) {
            auto pid = dyn_cast<triton::GetProgramIdOp>(op);
            if (!pid) continue;

            Access candidate;
            Value gridDim;
            Value group;
            int64_t S = 0;
            int64_t BT = 0;
            if (!matchFlattenedRowAndLane(pid, gridDim, candidate.outputRow,
                                          group, candidate.batch,
                                          candidate.lane, S))
                continue;
            if (!matchColumnOffsets(addPtr.getOffset(), group, candidate.d,
                                    candidate.colMask, BT))
                continue;
            auto gridDiv = gridDim.getDefiningOp<arith::DivSIOp>();
            auto gridBiasAdd = gridDiv
                                   ? gridDiv.getLhs().getDefiningOp<arith::AddIOp>()
                                   : arith::AddIOp();
            int64_t bias = BT - 1;
            bool isCdiv = false;
            if (gridDiv && gridBiasAdd) {
                auto gridDivisor = getConstantIntValue(gridDiv.getRhs());
                auto lhsBias = getConstantIntValue(gridBiasAdd.getRhs());
                auto rhsBias = getConstantIntValue(gridBiasAdd.getLhs());
                isCdiv = gridDivisor && *gridDivisor == BT &&
                         ((gridBiasAdd.getLhs() == candidate.d && lhsBias &&
                           *lhsBias == bias) ||
                          (gridBiasAdd.getRhs() == candidate.d && rhsBias &&
                           *rhsBias == bias));
            }
            if (vectorType.getShape()[0] != BT || !isCdiv)
                continue;
            if (!load.getMask() ||
                !matchLoadMask(load.getMask(), candidate.colMask,
                               candidate.rowMask))
                continue;
            if (matchDenseRowPtr(rowPtr, candidate.outputRow, candidate.d,
                                 candidate.base)) {
                candidate.rowKind = Access::RowKind::Dense;
            } else {
                if (!matchBeginEnd(candidate.batch, candidate.begin)) continue;
                if (!matchJaggedRowPtr(rowPtr, candidate.begin, candidate.lane,
                                       candidate.d, candidate.base))
                    continue;
                candidate.rowKind = Access::RowKind::Jagged;
            }

            candidate.cols = addPtr.getOffset();
            seed.load = load;
            seed.S = S;
            seed.BT = BT;
            seed.coalesceAxis = pid.getAxisAsInt();
            seed.access = candidate;
            return true;
        }
        return false;
    }

    // Constructs the [S,BT] byte/element offsets shared by addptr loads and
    // stores. `baseOffset` is jagged begin for input loads; `includeBatch`
    // switches stores to dense output_row addressing.
    Value buildOffsets2D(RewriterBase &rw, Location loc, const Access &access,
                         int64_t S, int64_t BT, Value baseOffset,
                         bool includeBatch) const {
        Type i32Ty = rw.getI32Type();
        Type i64Ty = rw.getI64Type();
        auto hI32Ty = RankedTensorType::get({S}, i32Ty);
        auto hI64Ty = RankedTensorType::get({S}, i64Ty);
        auto tileI64Ty = RankedTensorType::get({S, BT}, i64Ty);

        Value cS = rw.create<arith::ConstantIntOp>(loc, S, 32);
        Value hs = rw.create<triton::MakeRangeOp>(loc, hI32Ty, 0, S);
        Value rowBase = hs;
        if (includeBatch) {
            Value batchS = rw.create<arith::MulIOp>(loc, access.batch, cS);
            Value batchSplat = rw.create<triton::SplatOp>(
                loc, RankedTensorType::get({S}, i32Ty), batchS);
            rowBase = rw.create<arith::AddIOp>(loc, batchSplat, hs);
        }
        Value dRows = rw.create<triton::SplatOp>(
            loc, RankedTensorType::get({S}, i32Ty), access.d);
        Value rowTimesD = rw.create<arith::MulIOp>(loc, rowBase, dRows);
        Value rowTimesD64 = rw.create<arith::ExtSIOp>(loc, hI64Ty, rowTimesD);
        Value expandedRows =
            rw.create<triton::ExpandDimsOp>(loc, rowTimesD64, 1);
        Value rowOffsets2d =
            rw.create<triton::BroadcastOp>(loc, tileI64Ty, expandedRows);

        auto colI64Ty = RankedTensorType::get({BT}, i64Ty);
        Value cols64 = rw.create<arith::ExtSIOp>(loc, colI64Ty, access.cols);
        Value expandedCols = rw.create<triton::ExpandDimsOp>(loc, cols64, 0);
        Value cols2d =
            rw.create<triton::BroadcastOp>(loc, tileI64Ty, expandedCols);
        Value offsets = rw.create<arith::AddIOp>(loc, rowOffsets2d, cols2d);
        if (baseOffset) {
            Value base2d = rw.create<triton::SplatOp>(
                loc, RankedTensorType::get({S, BT}, i64Ty),
                baseOffset);
            offsets = rw.create<arith::AddIOp>(loc, base2d, offsets);
        }
        return offsets;
    }

};

void rewriteStridedAxisCoalesce(ModuleOp moduleOp) {
    RewritePatternSet patterns(moduleOp.getContext());
    patterns.add<BlockPtrPattern, AddPtrPattern>(moduleOp.getContext());
    (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

}  // namespace StridedAxisCoalescing
