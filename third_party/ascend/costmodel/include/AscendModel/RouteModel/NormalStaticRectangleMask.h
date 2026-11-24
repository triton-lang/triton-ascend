// Bounded static tensor evaluator for a literal 8x32 store predicate, not
// runtime data.
#ifndef ASCENDMODEL_NORMAL_STATIC_RECTANGLE_MASK_H
#define ASCENDMODEL_NORMAL_STATIC_RECTANGLE_MASK_H
#include "AscendModel/RouteModel/NormalMemoryAuxPartition.h"
namespace mlir::ascend::normal_static_mask {
using namespace specialized_segments_ownership;
struct Tensor {
  SmallVector<int64_t> shape;
  std::vector<int64_t> values;
};
inline std::optional<Tensor> layout(Value v) {
  Tensor x;
  if (auto t = dyn_cast<RankedTensorType>(v.getType())) {
    if (!t.hasStaticShape() || t.getEncoding() ||
        !isa<IntegerType>(t.getElementType()) || t.getRank() > 2)
      return std::nullopt;
    x.shape.assign(t.getShape().begin(), t.getShape().end());
  } else if (!isa<IntegerType>(v.getType()))
    return std::nullopt;
  int64_t count = 1;
  for (int64_t d : x.shape)
    if (d <= 0 || d > 256 || count > 256 / d)
      return std::nullopt;
    else
      count *= d;
  x.values.resize(count);
  return x;
}
inline int64_t number(const llvm::APInt &x) {
  return x.getBitWidth() == 1 ? x.getZExtValue() : x.getSExtValue();
}
inline std::optional<Tensor> evaluate(Value v, unsigned depth = 0) {
  if (depth > 24)
    return std::nullopt;
  auto out = layout(v);
  Operation *op = def(v);
  if (!out || !op || op->getNumRegions())
    return std::nullopt;
  if (auto c = dyn_cast<arith::ConstantOp>(op)) {
    if (auto a = dyn_cast<IntegerAttr>(c.getValue())) {
      if (a.getValue().getBitWidth() > 64 || out->values.size() != 1)
        return std::nullopt;
      out->values[0] = number(a.getValue());
      return out;
    }
    auto a = dyn_cast<DenseIntElementsAttr>(c.getValue());
    if (!a || a.getNumElements() != int64_t(out->values.size()) ||
        !isa<IntegerType>(a.getElementType()) ||
        cast<IntegerType>(a.getElementType()).getWidth() > 64)
      return std::nullopt;
    unsigned i = 0;
    for (auto z : a.getValues<llvm::APInt>())
      out->values[i++] = number(z);
    return out;
  }
  if (named(op, "tt.make_range")) {
    auto start = op->getAttrOfType<IntegerAttr>("start"),
         end = op->getAttrOfType<IntegerAttr>("end");
    if (!start || !end || start.getValue().getBitWidth() > 64 ||
        end.getValue().getBitWidth() > 64 || out->shape.size() != 1 ||
        end.getInt() - start.getInt() != int64_t(out->values.size()))
      return std::nullopt;
    for (unsigned i = 0; i < out->values.size(); ++i)
      out->values[i] = start.getInt() + i;
    return out;
  }
  if ((named(op, "tt.splat") || named(op, "tt.expand_dims") ||
       named(op, "tt.broadcast")) &&
      op->getNumOperands() == 1) {
    auto a = evaluate(op->getOperand(0), depth + 1);
    if (!a)
      return std::nullopt;
    if (named(op, "tt.splat")) {
      if (a->values.size() != 1)
        return std::nullopt;
      std::fill(out->values.begin(), out->values.end(), a->values[0]);
      return out;
    }
    if (named(op, "tt.expand_dims")) {
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (!axis || axis.getValue().getBitWidth() > 64 || axis.getInt() < 0 ||
          axis.getInt() > int64_t(a->shape.size()) ||
          out->values.size() != a->values.size())
        return std::nullopt;
      SmallVector<int64_t> expected = a->shape;
      expected.insert(expected.begin() + axis.getInt(), 1);
      if (expected != out->shape)
        return std::nullopt;
      out->values = a->values;
      return out;
    }
    if (a->shape.size() != out->shape.size())
      return std::nullopt;
    for (unsigned d = 0; d < a->shape.size(); ++d)
      if (a->shape[d] != 1 && a->shape[d] != out->shape[d])
        return std::nullopt;
    for (size_t flat = 0; flat < out->values.size(); ++flat) {
      size_t input = 0, stride = 1, remaining = flat;
      for (int d = int(out->shape.size()) - 1; d >= 0; --d) {
        size_t coordinate = remaining % out->shape[d];
        remaining /= out->shape[d];
        input += (a->shape[d] == 1 ? 0 : coordinate) * stride;
        stride *= a->shape[d];
      }
      out->values[flat] = a->values[input];
    }
    return out;
  }
  if ((named(op, "arith.andi") || named(op, "arith.cmpi")) &&
      op->getNumOperands() == 2) {
    auto a = evaluate(op->getOperand(0), depth + 1),
         b = evaluate(op->getOperand(1), depth + 1);
    if (!a || !b || a->shape != out->shape || b->shape != out->shape)
      return std::nullopt;
    for (unsigned i = 0; i < out->values.size(); ++i) {
      if (named(op, "arith.andi"))
        out->values[i] = a->values[i] & b->values[i];
      else {
        auto cmp = dyn_cast<arith::CmpIOp>(op);
        if (!cmp)
          return std::nullopt;
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::slt:
          out->values[i] = a->values[i] < b->values[i];
          break;
        case arith::CmpIPredicate::ult:
          out->values[i] = uint64_t(a->values[i]) < uint64_t(b->values[i]);
          break;
        case arith::CmpIPredicate::eq:
          out->values[i] = a->values[i] == b->values[i];
          break;
        default:
          return std::nullopt;
        }
      }
    }
    return out;
  }
  return std::nullopt;
}
inline std::optional<std::pair<int64_t, int64_t>> rectangle(Value mask) {
  auto x = evaluate(mask);
  if (!x || x->shape != SmallVector<int64_t>({8, 32}))
    return std::nullopt;
  int64_t h = 0, k = 0;
  for (int64_t r = 0; r < 8; ++r)
    for (int64_t c = 0; c < 32; ++c)
      if (x->values[r * 32 + c]) {
        h = std::max(h, r + 1);
        k = std::max(k, c + 1);
      }
  if (!h || !k)
    return std::nullopt;
  for (int64_t r = 0; r < 8; ++r)
    for (int64_t c = 0; c < 32; ++c)
      if (bool(x->values[r * 32 + c]) != (r < h && c < k))
        return std::nullopt;
  return std::pair<int64_t, int64_t>(h, k);
}
} // namespace mlir::ascend::normal_static_mask
#endif
