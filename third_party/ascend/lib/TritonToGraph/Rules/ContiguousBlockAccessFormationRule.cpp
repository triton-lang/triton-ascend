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

#include "TritonToGraph/GraphOptimizationRule.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"
#include <optional>
using namespace mlir;
using namespace triton;
using namespace cfg;
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
Value mask(Operation *op) {
  if (auto l = dyn_cast<triton::LoadOp>(op))
    return l.getMask();
  return cast<triton::StoreOp>(op).getMask();
}
struct Group {
  SmallVector<Operation *> ops;
  Value rows;
  int64_t width = 0;
};
std::optional<Group> group(Operation *first) {
  if (!isa<triton::LoadOp>(first))
    return std::nullopt;
  auto t = dyn_cast<RankedTensorType>(ptr(first).getType());
  if (!t || t.getRank() != 1 || t.getEncoding() || t.isDynamicDim(0) ||
      t.getDimSize(0) <= 0 || t.getDimSize(0) > 4096)
    return std::nullopt;
  Type element = cast<triton::PointerType>(t.getElementType()).getPointeeType();
  if (!element.isF16() && !element.isF32())
    return std::nullopt;
  Address a = address(ptr(first));
  int64_t w = rowStride(a.rows);
  // Bounded resource cost. Preserve missing leading field with an explicit
  // mask.
  if (w < 2 || w > 8 || a.field > 1)
    return std::nullopt;
  if (auto l = dyn_cast<triton::LoadOp>(first); l && l.getIsVolatile())
    return std::nullopt;
  Group g{{first}, a.rows, w};
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
        if (next == w)
          return g.ops.size() > 1 ? std::optional<Group>(g) : std::nullopt;
        continue;
      }
      // Reads commute, including reads through possibly aliasing pointers.
      if (isa<triton::LoadOp>(op))
        continue;
      // Only move a later read across writes proven disjoint from all
      // remaining columns. Same rows and power-of-two stride establish
      // distinct residues, including across lanes. Unknown aliases stop us.
      if (same && b.field < w && b.field < next)
        continue;
      return std::nullopt;
    }
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      return std::nullopt;
  }
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
class PackedPlan final : public RewritePlan {
  Operation *anchor;
  unsigned epoch;

public:
  PackedPlan(Operation *a, unsigned e) : anchor(a), epoch(e) {}
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::ContiguousBlockAccessFormation;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return anchor; }
  unsigned getCreationEpoch() const override { return epoch; }
  LogicalResult revalidate(GraphOptimizationContext &c) const override {
    return success(anchor->getParentOfType<triton::FuncOp>() ==
                       c.getFunction() &&
                   group(anchor).has_value());
  }
  LogicalResult apply(IRRewriter &r) override {
    auto candidate = group(anchor);
    if (!candidate)
      return failure();
    auto g = *candidate;
    auto l = anchor->getLoc();
    int64_t n = cast<RankedTensorType>(ptr(anchor).getType()).getDimSize(0);
    r.setInsertionPoint(anchor);
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
    Value bases = broadcast(g.rows);
    Value ps = r.create<triton::AddPtrOp>(l, bases.getType(), bases, cols);
    Value m = broadcast(mask(anchor));
    int64_t start = address(ptr(anchor)).field;
    if (start) {
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
    for (unsigned i = 0; i < g.ops.size(); ++i)
      r.replaceOp(g.ops[i], values[i + start]);
    return success();
  }
};
class PackedRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::ContiguousBlockAccessFormation;
  }
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }
  LogicalResult findCandidates(
      GraphOptimizationContext &c,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    c.getFunction().walk([&](Operation *op) {
      if (group(op))
        plans.push_back(std::make_unique<PackedPlan>(op, c.getEpoch()));
    });
    return success();
  }
};
} // namespace
std::unique_ptr<GraphOptimizationRule>
cfg::createContiguousBlockAccessFormationRule(
    const ContiguousBlockAccessFormationRuleOptions &) {
  return std::make_unique<PackedRule>();
}
