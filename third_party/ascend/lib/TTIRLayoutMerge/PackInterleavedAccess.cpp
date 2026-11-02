/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to do so, subject to the
 * following conditions:
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

#include "ascend/include/TTIRLayoutMerge/PackInterleavedAccess.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <functional>
#include <optional>
using namespace mlir;
using namespace triton;
namespace {
struct Address {
  Value rows;
  int64_t field = 0;
};

std::optional<int64_t> constant(Value value) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)) ||
      integer.getBitWidth() > 64)
    return std::nullopt;
  return integer.getSExtValue();
}

Address address(Value pointer) {
  Address result{pointer, 0};
  while (auto add = result.rows.getDefiningOp<triton::AddPtrOp>()) {
    auto offset = constant(add.getOffset());
    // Bound the arithmetic here; arbitrary integer overflow is not a proof.
    if (!offset || *offset < 0 || *offset > 65536 || result.field > 65536)
      break;
    result.field += *offset;
    result.rows = add.getPtr();
  }
  return result;
}

int64_t rowStride(Value rows) {
  auto add = rows.getDefiningOp<triton::AddPtrOp>();
  if (!add || !add.getPtr().getDefiningOp<triton::SplatOp>())
    return 0;
  auto multiply = add.getOffset().getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return 0;
  auto stride = constant(multiply.getRhs());
  if (!stride)
    stride = constant(multiply.getLhs());
  // Power-of-two strides preserve residues even when integer offsets wrap.
  return stride && *stride >= 2 && *stride <= 65536 &&
                 llvm::isPowerOf2_64(*stride)
             ? *stride
             : 0;
}

Value ptr(Operation *op) {
  if (auto l = dyn_cast<triton::LoadOp>(op))
    return l.getPtr();
  if (auto s = dyn_cast<triton::StoreOp>(op))
    return s.getPtr();
  return {};
}
// Profitability budget for the SIMT backend, not a hardware transfer limit.
// Wider tiles can distribute fields over lanes and require communication to
// reconstruct the original rank-one consumers. Keep each group at most 8 bytes.
// The rule depends on element width and row stride, never on an operator name
// or a particular batch/channel shape.
int64_t packingWidth(Type element, int64_t stride) {
  if ((!element.isF16() && !element.isF32()) || stride < 2 || stride > 8)
    return 0;
  return std::min<int64_t>(stride, 64 / element.getIntOrFloatBitWidth());
}
Value mask(Operation *op) {
  if (auto l = dyn_cast<triton::LoadOp>(op))
    return l.getMask();
  return cast<triton::StoreOp>(op).getMask();
}
struct Group {
  SmallVector<Operation *> ops;
  Value rows;
  int64_t width = 0;
  bool sinkLoads = false;
};
std::optional<Group> group(Operation *first) {
  if (!isa<triton::LoadOp>(first))
    return std::nullopt;
  auto t = dyn_cast<RankedTensorType>(ptr(first).getType());
  if (!t || t.getRank() != 1 || t.getEncoding() || t.isDynamicDim(0) ||
      t.getDimSize(0) <= 0 || t.getDimSize(0) > 8192)
    return std::nullopt;
  Address a = address(ptr(first));
  int64_t w = rowStride(a.rows);
  // Compile-time layout choice, independent of B/D and runtime pointers.
  // Half precision may mask the missing first field of the first group.
  // Neither choice expands the set of active addresses.
  Type element = cast<triton::PointerType>(t.getElementType()).getPointeeType();
  bool half = element.isF16();
  int64_t width = packingWidth(element, w);
  if (!width || a.field >= w ||
      (a.field % width != 0 && !(half && a.field == 1)))
    return std::nullopt;
  if (auto l = dyn_cast<triton::LoadOp>(first); l && l.getIsVolatile())
    return std::nullopt;
  int64_t endField = half && a.field == 1 ? width : a.field + width;
  Group g{{first}, a.rows, width};
  int64_t next = a.field + 1;
  for (Operation *op = first->getNextNode(); op; op = op->getNextNode()) {
    if (isa<triton::LoadOp, triton::StoreOp>(op)) {
      if (auto l = dyn_cast<triton::LoadOp>(op); l && l.getIsVolatile())
        return std::nullopt;
      Address b = address(ptr(op));
      bool same = b.rows == a.rows;
      bool sameKind = isa<triton::LoadOp>(op);
      if (same && sameKind && b.field == next && mask(first) == mask(op) &&
          ptr(first).getType() == ptr(op).getType() &&
          first->getAttrDictionary() == op->getAttrDictionary()) {
        if (cast<triton::LoadOp>(first).getOther() !=
            cast<triton::LoadOp>(op).getOther())
          return std::nullopt;
        g.ops.push_back(op);
        ++next;
        if (next == endField)
          return g.ops.size() > 1 ? std::optional<Group>(g) : std::nullopt;
        continue;
      }
      if (isa<triton::LoadOp>(op))
        continue;
      // Load hoisting crosses only writes outside remaining source columns.
      if (same && b.field < w && b.field < next)
        continue;
      // Unknown pointer relationships are a compile-time barrier.
      // No runtime guard or speculative noalias contract is introduced.
      return std::nullopt;
    }
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      return std::nullopt;
  }
  return std::nullopt;
}
// Prove alignment from existing IR contracts and modular integer arithmetic.
// No noalias attribute is consulted or synthesized.
bool multipleOf(Value v, int64_t modulus, unsigned depth = 0) {
  if (depth > 24)
    return false;
  if (auto c = constant(v))
    return *c % modulus == 0;
  if (auto op = v.getDefiningOp<arith::MulIOp>())
    return multipleOf(op.getLhs(), modulus, depth + 1) ||
           multipleOf(op.getRhs(), modulus, depth + 1);
  if (auto op = v.getDefiningOp<arith::AddIOp>())
    return multipleOf(op.getLhs(), modulus, depth + 1) &&
           multipleOf(op.getRhs(), modulus, depth + 1);
  if (auto op = v.getDefiningOp<arith::ExtSIOp>())
    return multipleOf(op.getIn(), modulus, depth + 1);
  if (auto op = v.getDefiningOp<arith::ExtUIOp>())
    return multipleOf(op.getIn(), modulus, depth + 1);
  if (auto op = v.getDefiningOp<triton::SplatOp>())
    return multipleOf(op.getSrc(), modulus, depth + 1);
  return false;
}
bool alignedPointer(Value v, int64_t elements, unsigned depth = 0) {
  if (depth > 24)
    return false;
  if (auto op = v.getDefiningOp<triton::SplatOp>())
    return alignedPointer(op.getSrc(), elements, depth + 1);
  if (auto op = v.getDefiningOp<triton::AddPtrOp>())
    return multipleOf(op.getOffset(), elements) &&
           alignedPointer(op.getPtr(), elements, depth + 1);
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || !isa<triton::PointerType>(v.getType()))
    return false;
  auto func = dyn_cast_or_null<triton::FuncOp>(arg.getOwner()->getParentOp());
  if (!func || arg.getOwner() != &func.getBody().front())
    return false;
  auto attr =
      func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "tt.divisibility");
  Type elem = cast<triton::PointerType>(v.getType()).getPointeeType();
  if (!elem.isF16() && !elem.isF32())
    return false;
  int64_t bytes = elements * (elem.getIntOrFloatBitWidth() / 8);
  return attr && attr.getInt() > 0 && attr.getInt() % bytes == 0;
}
// Moving a read later also requires moving its pure consumers. Reject any
// consumer which would change a memory operation, control flow, or side effect.
std::optional<SmallVector<Operation *>> sinkUsers(const Group &g) {
  Operation *last = g.ops.back();
  llvm::SmallPtrSet<Operation *, 32> needed;
  bool valid = true;
  std::function<void(Operation *)> visit = [&](Operation *op) {
    for (Operation *user : op->getUsers()) {
      if (user->getBlock() != last->getBlock()) {
        valid = false;
        continue;
      }
      if (!user->isBeforeInBlock(last))
        continue;
      if (user->getNumRegions() || !isMemoryEffectFree(user)) {
        valid = false;
        continue;
      }
      if (needed.insert(user).second)
        visit(user);
    }
  };
  for (Operation *op : g.ops)
    visit(op);
  if (!valid)
    return std::nullopt;
  SmallVector<Operation *> ordered;
  for (Operation *op = g.ops.front(); op != last; op = op->getNextNode())
    if (needed.contains(op))
      ordered.push_back(op);
  return ordered;
}
std::optional<Group> sinkingGroup(Operation *first) {
  auto load = dyn_cast<triton::LoadOp>(first);
  if (!load || load.getIsVolatile())
    return std::nullopt;
  auto t = dyn_cast<RankedTensorType>(load.getPtr().getType());
  if (!t || t.getRank() != 1 || t.getEncoding() || t.getDimSize(0) <= 0 ||
      t.getDimSize(0) > 8192)
    return std::nullopt;
  Address a = address(load.getPtr());
  int64_t w = rowStride(a.rows);
  Type element = cast<triton::PointerType>(t.getElementType()).getPointeeType();
  int64_t width = packingWidth(element, w);
  if (!width || a.field < 0 || a.field + width > w || a.field % width != 0 ||
      !alignedPointer(a.rows, w))
    return std::nullopt;
  Group g{{first}, a.rows, width, true};
  int64_t next = a.field + 1;
  for (Operation *op = first->getNextNode(); op; op = op->getNextNode()) {
    if (auto l = dyn_cast<triton::LoadOp>(op)) {
      if (l.getIsVolatile())
        return std::nullopt;
      Address b = address(l.getPtr());
      if (b.rows == a.rows && b.field == next &&
          l.getPtr().getType() == load.getPtr().getType() &&
          l.getMask() == load.getMask() && l.getOther() == load.getOther() &&
          op->getAttrDictionary() == first->getAttrDictionary()) {
        g.ops.push_back(op);
        if (++next == a.field + width)
          return sinkUsers(g) ? std::optional<Group>(g) : std::nullopt;
      }
      continue;
    }
    if (auto st = dyn_cast<triton::StoreOp>(op)) {
      Address b = address(st.getPtr());
      // Equal element types and aligned row bases prove field residues.
      // All reads already collected occupy [a.field,next); this store is
      // outside.
      if (st.getPtr().getType() == load.getPtr().getType() &&
          (b.field < a.field || b.field >= next) && b.field < w &&
          rowStride(b.rows) == w && alignedPointer(b.rows, w))
        continue;
      return std::nullopt;
    }
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      return std::nullopt;
  }
  return std::nullopt;
}
// Restrict the profitability heuristic to fields consumed by floating-point
// arithmetic, directly or through an FP extension. Pure copies/reordering can
// regress despite fewer loads; this pass does not model their store layout.
// This is a data-flow condition, not an operator or shape whitelist.
bool hasArithmeticConsumer(Operation *load) {
  auto arithmetic = [](Operation *op) {
    return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(op);
  };
  for (Operation *user : load->getUsers()) {
    if (arithmetic(user))
      return true;
    if (isa<arith::ExtFOp>(user))
      for (Operation *consumer : user->getUsers())
        if (arithmetic(consumer))
          return true;
  }
  return false;
}
std::optional<Group> candidateGroup(Operation *op) {
  if (isa<triton::StoreOp>(op))
    return std::nullopt;
  auto g = group(op);
  if (!g)
    g = sinkingGroup(op);
  if (g && llvm::all_of(g->ops, hasArithmeticConsumer))
    return g;
  return std::nullopt;
}

Value reshape(IRRewriter &r, Location l, Value v, ArrayRef<int64_t> shape) {
  return r.create<triton::ReshapeOp>(
      l, cast<RankedTensorType>(v.getType()).clone(shape), v);
}
SmallVector<Value> splitColumns(IRRewriter &r, Location l, Value v, int64_t n,
                                int64_t w) {
  if (w == 1)
    return {reshape(r, l, v, {n})};
  if (w > 2)
    v = reshape(r, l, v, {n, w / 2, 2});
  auto s = r.create<triton::SplitOp>(l, v);
  auto even = splitColumns(r, l, s.getOutLHS(), n, w / 2),
       odd = splitColumns(r, l, s.getOutRHS(), n, w / 2);
  SmallVector<Value> result;
  for (int64_t i = 0; i < w / 2; ++i) {
    result.push_back(even[i]);
    result.push_back(odd[i]);
  }
  return result;
}
LogicalResult rewriteGroup(IRRewriter &r, Operation *anchor) {
  auto candidate = candidateGroup(anchor);
  if (!candidate)
    return failure();
  auto g = *candidate;
  auto l = anchor->getLoc();
  int64_t n = cast<RankedTensorType>(ptr(anchor).getType()).getDimSize(0);
  if (g.sinkLoads) {
    auto users = sinkUsers(g);
    if (!users)
      return failure();
    Operation *after = g.ops.back();
    for (Operation *user : *users) {
      user->moveAfter(after);
      after = user;
    }
  }
  r.setInsertionPoint(g.sinkLoads ? g.ops.back() : anchor);
  auto broadcast = [&](Value v) -> Value {
    if (!v)
      return {};
    auto t = cast<RankedTensorType>(v.getType());
    Value e = r.create<triton::ExpandDimsOp>(l, v, 1);
    return r.create<triton::BroadcastOp>(l, t.clone({n, g.width}), e);
  };
  Value range = r.create<triton::MakeRangeOp>(
      l, RankedTensorType::get({g.width}, r.getI32Type()), 0, g.width);
  Value cols = r.create<triton::ExpandDimsOp>(l, range, 0);
  cols = r.create<triton::BroadcastOp>(
      l, RankedTensorType::get({n, g.width}, r.getI32Type()), cols);
  int64_t firstField = address(ptr(anchor)).field;
  bool maskedPrefix = firstField == 1;
  if (firstField && !maskedPrefix) {
    auto ct = cast<RankedTensorType>(cols.getType());
    Value shift = r.create<arith::ConstantOp>(
        l, ct, DenseElementsAttr::get(ct, r.getI32IntegerAttr(firstField)));
    cols = r.create<arith::AddIOp>(l, cols, shift);
  }
  Value bases = broadcast(g.rows);
  Value ps = r.create<triton::AddPtrOp>(l, bases.getType(), bases, cols);
  Value m = broadcast(mask(anchor));
  if (maskedPrefix) {
    auto ct = cast<RankedTensorType>(cols.getType());
    Value zero = r.create<arith::ConstantOp>(
        l, ct, DenseElementsAttr::get(ct, r.getI32IntegerAttr(0)));
    Value active =
        r.create<arith::CmpIOp>(l, arith::CmpIPredicate::ne, cols, zero);
    m = m ? Value(r.create<arith::AndIOp>(l, m, active)) : active;
  }
  auto old = cast<triton::LoadOp>(anchor);
  Value v = r.create<triton::LoadOp>(l, ps, m, broadcast(old.getOther()),
                                     old.getCache(), old.getEvict(), false);
  auto values = splitColumns(r, l, v, n, g.width);
  if (cast<RankedTensorType>(v.getType()).getElementType().isF16()) {
    SmallVector<SmallVector<arith::ExtFOp>> casts(g.ops.size());
    bool any = false;
    for (unsigned i = 0; i < g.ops.size(); ++i)
      for (Operation *u : g.ops[i]->getUsers())
        if (auto ext = dyn_cast<arith::ExtFOp>(u))
          if (cast<RankedTensorType>(ext.getType()).getElementType().isF32()) {
            casts[i].push_back(ext);
            any = true;
          }
    if (any) {
      Value f = r.create<arith::ExtFOp>(
          l, cast<RankedTensorType>(v.getType()).clone(r.getF32Type()), v);
      auto floats = splitColumns(r, l, f, n, g.width);
      for (unsigned i = 0; i < g.ops.size(); ++i)
        for (auto ext : casts[i])
          r.replaceOp(ext, floats[i + (maskedPrefix ? 1 : 0)]);
    }
  }
  for (unsigned i = 0; i < g.ops.size(); ++i)
    r.replaceOp(g.ops[i], values[i + (maskedPrefix ? 1 : 0)]);
  return success();
}

struct PackInterleavedAccessPass
    : PassWrapper<PackInterleavedAccessPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackInterleavedAccessPass)
  StringRef getArgument() const final {
    return "triton-pack-interleaved-access";
  }
  StringRef getDescription() const final {
    return "Pack proven interleaved scalar-field accesses into contiguous "
           "tiles";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, arith::ArithDialect>();
  }
  void runOnOperation() override {
    mlir::triton::packInterleavedAccesses(getOperation());
  }
};
} // namespace

void mlir::triton::packInterleavedAccesses(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  // A rewrite replaces rank-one groups with rank-two memory operations and
  // strictly decreases their count. Restart after mutation to avoid stale
  // plans.
  while (true) {
    Operation *anchor = nullptr;
    module.walk([&](Operation *op) {
      if (!anchor && candidateGroup(op))
        anchor = op;
    });
    if (!anchor)
      return;
    if (failed(rewriteGroup(rewriter, anchor)))
      return;
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPackInterleavedAccessPass() {
  return std::make_unique<PackInterleavedAccessPass>();
}

// Also expose the pass to standalone triton-opt for focused regression tests.
static PassRegistration<PackInterleavedAccessPass> registration;
