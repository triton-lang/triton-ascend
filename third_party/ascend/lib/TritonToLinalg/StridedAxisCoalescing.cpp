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

#include "TritonToLinalg/StridedAxisCoalescing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include <functional>

namespace StridedAxisCoalescing {

using namespace mlir;
using namespace triton;

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
                if (lhs.getDefiningOp<triton::GetProgramIdOp>()) return addptr;
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

static Value build2DBlockPtr(IRRewriter &rw, triton::MakeTensorPtrOp m1d,
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
    SmallVector<int32_t, 2> blockShape{static_cast<int32_t>(BT), static_cast<int32_t>(S)};
    SmallVector<int32_t, 2> order{1, 0};
    auto p = rw.create<triton::MakeTensorPtrOp>(loc, newBase, shape, strides,
                                                offsets, blockShape, order);
    return p.getResult();
}

// Returns true if `op` (a tt.scan or tt.reduce) combines its two block args
// with a single floating-point add (i.e. it is a sum / cumsum).
static bool isPlusCombiner(Operation *op) {
    if (op->getNumRegions() != 1) return false;
    Region &r = op->getRegion(0);
    if (!r.hasOneBlock()) return false;
    Block &b = r.front();
    if (b.getNumArguments() != 2) return false;
    Operation *term = b.getTerminator();
    if (!term || term->getNumOperands() != 1) return false;
    auto add = term->getOperand(0).getDefiningOp<arith::AddFOp>();
    if (!add) return false;
    Value a0 = b.getArgument(0), a1 = b.getArgument(1);
    return (add.getLhs() == a0 && add.getRhs() == a1) ||
           (add.getLhs() == a1 && add.getRhs() == a0);
}

// Detect the FLA reverse-cumsum idiom built on top of a forward cumsum:
//   b_o = cumsum(v);  b_z = sum(v);  store(-b_o + b_z + v)
// which is algebraically the inclusive reverse cumsum of v. `v` is the f32
// value the scan operates on (the load, or extf(load) for fp16). On success
// returns the forward ScanOp, sets `storeOut`, and fills `eraseList` with the
// intermediate ops to remove (in use-before-def order).
static triton::ScanOp matchReverseCumsum(Value v, triton::StoreOp &storeOut,
                                         SmallVectorImpl<Operation *> &eraseList) {
    triton::ScanOp scan;
    triton::ReduceOp reduce;
    arith::AddFOp a2;
    for (Operation *u : v.getUsers()) {
        if (auto s = dyn_cast<triton::ScanOp>(u)) { if (scan) return {}; scan = s; }
        else if (auto rd = dyn_cast<triton::ReduceOp>(u)) { if (reduce) return {}; reduce = rd; }
        else if (auto a = dyn_cast<arith::AddFOp>(u)) { if (a2) return {}; a2 = a; }
        else return {};
    }
    if (!scan || !reduce || !a2) return {};
    if (scan.getReverse() || scan->getNumResults() != 1 || reduce->getNumResults() != 1)
        return {};
    if (!isPlusCombiner(scan) || !isPlusCombiner(reduce)) return {};

    // scan -> subf(0, scan) -> a1
    Value scanRes = scan->getResult(0);
    if (!scanRes.hasOneUse()) return {};
    auto sub = dyn_cast<arith::SubFOp>(*scanRes.user_begin());
    if (!sub || sub.getRhs() != scanRes) return {};
    DenseElementsAttr zeroAttr;
    if (!matchPattern(sub.getLhs(), m_Constant(&zeroAttr)) || !zeroAttr.isSplat() ||
        !zeroAttr.getSplatValue<APFloat>().isZero())
        return {};
    if (!sub.getResult().hasOneUse()) return {};
    auto a1 = dyn_cast<arith::AddFOp>(*sub.getResult().user_begin());
    if (!a1) return {};

    // reduce -> splat -> a1
    Value redRes = reduce->getResult(0);
    if (!redRes.hasOneUse()) return {};
    auto splat = dyn_cast<triton::SplatOp>(*redRes.user_begin());
    if (!splat || !splat.getResult().hasOneUse()) return {};
    if (*splat.getResult().user_begin() != a1.getOperation()) return {};

    bool a1ok = (a1.getLhs() == sub.getResult() && a1.getRhs() == splat.getResult()) ||
                (a1.getRhs() == sub.getResult() && a1.getLhs() == splat.getResult());
    if (!a1ok || !a1.getResult().hasOneUse()) return {};
    if (*a1.getResult().user_begin() != a2.getOperation()) return {};

    bool a2ok = (a2.getLhs() == a1.getResult() && a2.getRhs() == v) ||
                (a2.getRhs() == a1.getResult() && a2.getLhs() == v);
    if (!a2ok) return {};

    // a2 (the idiom output, = reverse cumsum of v) may feed further elementwise
    // ops (e.g. * scale) before the store, so do NOT require it to feed a store
    // directly. The caller RAUWs a2's uses to a single reverse scan; multiple
    // uses are fine. Only a2 itself is erased after the RAUW.
    storeOut = nullptr;
    eraseList.assign({a2.getOperation(), a1.getOperation(), sub.getOperation(),
                      splat.getOperation(), reduce.getOperation(), scan.getOperation()});
    return scan;
}

// Lift a rank-1 tensor type tensor<BTxe> to tensor<BTxSxe> (append the folded
// H axis as the inner lane). Scalars / non-rank-1 types pass through unchanged.
static Type lift2D(Type t, int64_t S) {
    auto rt = dyn_cast<RankedTensorType>(t);
    if (!rt || rt.getRank() != 1) return t;
    return RankedTensorType::get({rt.getShape()[0], S}, rt.getElementType());
}

// An op is safe to 2D-ify (lane-parallel over the appended H axis) iff it is a
// pure elementwise arith op, a cast, a splat, or a scan along the T axis. Ops
// that mix or move the lane (tt.reduce, transpose, tt.dot, reshape, ...) are
// NOT here, so the caller bails and keeps the original (indirect) path.
static bool is2DSafe(Operation *op) {
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::CmpFOp, arith::SelectOp,
            arith::ExtFOp, arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp,
            arith::FPToSIOp, arith::FPToUIOp>(op))
        return true;
    if (isa<triton::SplatOp>(op)) return true;
    if (auto scan = dyn_cast<triton::ScanOp>(op))
        return scan.getAxis() == 0 && scan->getNumResults() == 1;
    return false;
}

// Phase 0: collapse the FLA reverse-cumsum idiom (forward scan + reduce + the
// `-b_o + b_z + b_s` fixup, see matchReverseCumsum) into a single 1D reverse
// scan. This removes the only shape-changing op (tt.reduce) so the remaining
// load->store subgraph is pure elementwise/scan and Phase 1 can lift it
// uniformly. Operates on 1D IR only.
static void simplifyReverseCumsum1D(ModuleOp moduleOp, int64_t S) {
    IRRewriter rw(moduleOp.getContext());
    SmallVector<triton::LoadOp> loads;
    moduleOp.walk([&](triton::LoadOp l) { loads.push_back(l); });
    for (triton::LoadOp loadOp : loads) {
        auto m1d = loadOp.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        if (!m1d) continue;
        auto rt = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
        if (!rt || rt.getRank() != 1) continue;
        auto strides = m1d.getStrides();
        if (strides.empty()) continue;
        APInt sC;
        if (!matchPattern(strides.back(), m_ConstantInt(&sC))) continue;
        if (std::abs(sC.getSExtValue()) != S) continue;
        if (!findIhAddPtr(m1d.getBase(), S)) continue;

        Value v = loadOp.getResult();
        if (v.hasOneUse())
            if (auto e = dyn_cast<arith::ExtFOp>(*v.user_begin())) v = e.getResult();

        triton::StoreOp st;
        SmallVector<Operation *> el;  // {a2, a1, sub, splat, reduce, scan}
        triton::ScanOp fwd = matchReverseCumsum(v, st, el);
        if (!fwd) continue;

        rw.setInsertionPoint(fwd);
        auto rev = rw.create<triton::ScanOp>(fwd.getLoc(), ValueRange{v},
                                             static_cast<int>(fwd.getAxis()),
                                             /*reverse=*/true);
        rw.cloneRegionBefore(fwd.getCombineOp(), rev.getCombineOp(),
                             rev.getCombineOp().end());
        rw.replaceAllUsesWith(el.front()->getResult(0), rev->getResult(0));
        for (Operation *o : el) {
            assert(o->use_empty() && "reverse simplify: op still has uses");
            rw.eraseOp(o);
        }
    }
}

void rewriteStridedAxisCoalesce(ModuleOp moduleOp) {
    IRRewriter rw(moduleOp.getContext());

    // This pass has priority over TileChunkCoalescing for the single module-level
    // hacc.coalesce_factor: it runs first and unconditionally; TileChunk yields
    // if we claim the factor here.
    //
    // Full TA path: we record (hacc.coalesce_factor=S, hacc.coalesce_axis) and the
    // host launcher divides grid[axis] by S. bishengir does NOT interpret these.

    // Collect the strided ih-base 1D loads (seeds). All must share one stride S
    // (the folded H axis); BT is the per-chunk tile length.
    SmallVector<triton::LoadOp> seeds;
    int64_t S = 0, BT = 0;
    moduleOp.walk([&](triton::LoadOp l) {
        auto m = l.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        if (!m) return;
        auto rt = dyn_cast<RankedTensorType>(l.getResult().getType());
        if (!rt || rt.getRank() != 1) return;
        auto strides = m.getStrides();
        if (strides.empty()) return;
        APInt sC;
        if (!matchPattern(strides.back(), m_ConstantInt(&sC))) return;
        int64_t s = std::abs(sC.getSExtValue());
        if (s <= 1) return;
        if (!findIhAddPtr(m.getBase(), s)) return;
        if (S == 0) { S = s; BT = rt.getShape()[0]; }
        if (s != S) return;
        seeds.push_back(l);
    });
    if (seeds.empty()) return;

    // The grid axis the launcher will divide by S (the pid feeding `pid % S`).
    int32_t coalesceAxis = -1;
    if (auto m0 = seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>())
        coalesceAxis = findIhAxis(m0.getBase(), S);
    if (coalesceAxis < 0) return;  // cannot identify the axis -> do not coalesce

    // Full TA path: the launcher divides grid[coalesceAxis] by S, so the
    // kernel-visible num_programs(coalesceAxis) becomes grid/S. If the kernel
    // reads it, coalescing would change that value -> wrong results. Bail (the
    // kernel keeps its original, correct, uncoalesced path).
    bool readsAxisNumPrograms = false;
    moduleOp.walk([&](triton::GetNumProgramsOp np) {
        if (np.getAxisAsInt() == coalesceAxis) readsAxisNumPrograms = true;
    });
    if (readsAxisNumPrograms) return;

    // Phase 0: turn any reverse-cumsum idiom into a single reverse scan.
    simplifyReverseCumsum1D(moduleOp, S);

    // Phase 1: discover the load->store subgraph by forward reachability from
    // the seeds. Every op on the way must be 2D-safe; stores are the sinks.
    // Any unsafe op (or a value escaping to one) aborts the whole rewrite.
    DenseSet<Operation *> region;
    SmallVector<triton::StoreOp> sinks;
    DenseSet<Operation *> visited;
    SmallVector<Operation *> wl;
    for (auto s : seeds)
        for (Operation *u : s.getResult().getUsers()) wl.push_back(u);
    while (!wl.empty()) {
        Operation *op = wl.pop_back_val();
        if (!visited.insert(op).second) continue;
        if (auto st = dyn_cast<triton::StoreOp>(op)) { sinks.push_back(st); continue; }
        if (!is2DSafe(op)) return;  // bail: unsafe consumer in the chain
        region.insert(op);
        for (Value r : op->getResults())
            for (Operation *u : r.getUsers()) wl.push_back(u);
    }
    if (sinks.empty()) return;

    // Every sink store must also be a matching 1D stride-S ih-base block ptr.
    for (auto st : sinks) {
        auto m = st.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        if (!m || !findIhAddPtr(m.getBase(), S)) return;
        auto os = m.getStrides();
        if (os.empty()) return;
        APInt soC;
        if (!matchPattern(os.back(), m_ConstantInt(&soC)) ||
            std::abs(soC.getSExtValue()) != S)
            return;
    }

    // Map each 1D value to its 2D counterpart, materializing splats/constants
    // on demand. Returns null to signal an un-liftable operand (bail).
    DenseMap<Value, Value> vmap;
    std::function<Value(Value)> get2D = [&](Value v) -> Value {
        auto it = vmap.find(v);
        if (it != vmap.end()) return it->second;
        if (!isa<RankedTensorType>(v.getType())) return v;  // scalar, unchanged
        if (auto sp = v.getDefiningOp<triton::SplatOp>()) {
            rw.setInsertionPointAfter(sp);
            Value n = rw.create<triton::SplatOp>(sp.getLoc(),
                                                 lift2D(sp.getType(), S), sp.getSrc());
            vmap[v] = n;
            return n;
        }
        if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
            if (auto dea = dyn_cast<DenseElementsAttr>(c.getValue())) {
                if (dea.isSplat()) {
                    auto nt = cast<RankedTensorType>(lift2D(c.getType(), S));
                    rw.setInsertionPointAfter(c);
                    Value n = rw.create<arith::ConstantOp>(
                        c.getLoc(), nt,
                        DenseElementsAttr::get(nt, dea.getSplatValue<Attribute>()));
                    vmap[v] = n;
                    return n;
                }
            }
        }
        return Value();
    };

    // Build 2D loads for the seeds.
    for (auto l : seeds) {
        auto m = l.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        Value p2 = build2DBlockPtr(rw, m, S, BT);
        if (!p2) return;
        rw.setInsertionPoint(l);
        auto nl = rw.create<triton::LoadOp>(
            l.getLoc(), p2, ArrayRef<int32_t>{0, 1}, l.getPadding(),
            l.getCache(), l.getEvict(), l.getIsVolatile());
        vmap[l.getResult()] = nl.getResult();
    }

    // Rebuild the region ops in IR (topological) order as 2D.
    SmallVector<Operation *> ordered;
    moduleOp.walk([&](Operation *op) { if (region.count(op)) ordered.push_back(op); });
    for (Operation *op : ordered) {
        rw.setInsertionPoint(op);
        if (auto scan = dyn_cast<triton::ScanOp>(op)) {
            Value in = get2D(scan.getOperand(0));
            if (!in) return;
            auto ns = rw.create<triton::ScanOp>(scan.getLoc(), ValueRange{in},
                                                static_cast<int>(scan.getAxis()),
                                                scan.getReverse());
            rw.cloneRegionBefore(scan.getCombineOp(), ns.getCombineOp(),
                                 ns.getCombineOp().end());
            vmap[scan->getResult(0)] = ns->getResult(0);
            continue;
        }
        SmallVector<Value> operands;
        for (Value o : op->getOperands()) {
            Value n = get2D(o);
            if (!n) return;
            operands.push_back(n);
        }
        OperationState st(op->getLoc(), op->getName());
        st.addOperands(operands);
        st.addAttributes(op->getAttrs());
        for (Value r : op->getResults()) st.addTypes(lift2D(r.getType(), S));
        Operation *nu = rw.create(st);
        for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
            vmap[oldR] = newR;
    }

    // Build the 2D stores.
    for (auto st : sinks) {
        Value val = get2D(st.getValue());
        if (!val) return;
        auto m = st.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        Value p2 = build2DBlockPtr(rw, m, S, BT);
        if (!p2) return;
        rw.setInsertionPoint(st);
        rw.create<triton::StoreOp>(st.getLoc(), p2, val, ArrayRef<int32_t>{0, 1},
                                   st.getCache(), st.getEvict());
    }

    // i_b = divsi(get_program_id, S): with the H axis folded into the inner
    // tile the per-instance i_b becomes the raw program id. Redirect it (this
    // also fixes the new 2D block ptr bases, which reuse i_b).
    if (auto m0 = seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>()) {
        if (triton::AddPtrOp ihA = findIhAddPtr(m0.getBase(), S)) {
            if (auto rem = ihA.getOffset().getDefiningOp<arith::RemSIOp>()) {
                Value lhs = rem.getLhs();
                while (true) {
                    if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) { lhs = e.getIn(); continue; }
                    if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) { lhs = t.getIn(); continue; }
                    break;
                }
                SmallVector<arith::DivSIOp, 2> divs;
                for (Operation *u : lhs.getUsers())
                    if (auto dv = dyn_cast<arith::DivSIOp>(u)) {
                        APInt dC;
                        if (dv.getLhs() == lhs &&
                            matchPattern(dv.getRhs(), m_ConstantInt(&dC)) &&
                            std::abs(dC.getSExtValue()) == S)
                            divs.push_back(dv);
                    }
                for (auto dv : divs) rw.replaceAllUsesWith(dv.getResult(), lhs);
            }
        }
    }

    // Erase the original chain (sinks, then region in reverse order, then seeds).
    for (auto st : sinks) rw.eraseOp(st);
    for (auto it = ordered.rbegin(); it != ordered.rend(); ++it) rw.eraseOp(*it);
    for (auto l : seeds) rw.eraseOp(l);

    auto i32t = IntegerType::get(moduleOp.getContext(), 32);
    moduleOp->setAttr("hacc.coalesce_factor", IntegerAttr::get(i32t, S));
    moduleOp->setAttr("hacc.coalesce_axis", IntegerAttr::get(i32t, coalesceAxis));
}

}  // namespace StridedAxisCoalescing
