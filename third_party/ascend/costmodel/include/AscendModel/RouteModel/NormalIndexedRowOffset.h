// Symbolic loaded-row offsets, independent of runtime Index contents.
#ifndef ASCENDMODEL_NORMAL_INDEXED_ROW_OFFSET_H
#define ASCENDMODEL_NORMAL_INDEXED_ROW_OFFSET_H
#include "AscendModel/RouteModel/NormalStaticRectangleMask.h"
namespace mlir::ascend::normal_indexed_offset {
using namespace specialized_segments_ownership;
struct Entry {
  int row = -1;
  int64_t scale = 0, bias = 0;
};
struct Tensor {
  SmallVector<int64_t> shape;
  std::vector<Entry> values;
};
inline std::optional<Tensor> evaluate(Value v, Operation *index,
                                      unsigned depth = 0) {
  if (depth > 24)
    return std::nullopt;
  auto l = normal_static_mask::layout(v);
  if (!l)
    return std::nullopt;
  Tensor out;
  out.shape = l->shape;
  out.values.resize(l->values.size());
  if (auto x = normal_static_mask::evaluate(v)) {
    for (size_t i = 0; i < x->values.size(); ++i)
      out.values[i].bias = x->values[i];
    return out;
  }
  Operation *op = def(v);
  if (!op || op->getNumRegions())
    return std::nullopt;
  if (op == index) {
    if (out.shape != SmallVector<int64_t>({8}) || !named(op, "tt.load"))
      return std::nullopt;
    for (int i = 0; i < 8; ++i)
      out.values[i] = {i, 1, 0};
    return out;
  }
  if ((named(op, "tt.expand_dims") || named(op, "tt.broadcast")) &&
      op->getNumOperands() == 1) {
    auto a = evaluate(op->getOperand(0), index, depth + 1);
    if (!a)
      return std::nullopt;
    if (named(op, "tt.expand_dims")) {
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (!axis || axis.getValue().getBitWidth() > 64 || axis.getInt() < 0 ||
          axis.getInt() > int64_t(a->shape.size()) ||
          out.values.size() != a->values.size())
        return std::nullopt;
      auto expected = a->shape;
      expected.insert(expected.begin() + axis.getInt(), 1);
      if (expected != out.shape)
        return std::nullopt;
      out.values = a->values;
      return out;
    }
    if (a->shape.size() != out.shape.size())
      return std::nullopt;
    for (unsigned d = 0; d < a->shape.size(); ++d)
      if (a->shape[d] != 1 && a->shape[d] != out.shape[d])
        return std::nullopt;
    for (size_t f = 0; f < out.values.size(); ++f) {
      size_t p = 0, stride = 1, rem = f;
      for (int d = int(out.shape.size()) - 1; d >= 0; --d) {
        size_t coordinate = rem % out.shape[d];
        rem /= out.shape[d];
        p += (a->shape[d] == 1 ? 0 : coordinate) * stride;
        stride *= a->shape[d];
      }
      out.values[f] = a->values[p];
    }
    return out;
  }
  if ((named(op, "arith.muli") || named(op, "arith.addi")) &&
      op->getNumOperands() == 2) {
    auto a = evaluate(op->getOperand(0), index, depth + 1),
         b = evaluate(op->getOperand(1), index, depth + 1);
    if (!a || !b || a->shape != out.shape || b->shape != out.shape)
      return std::nullopt;
    for (size_t i = 0; i < out.values.size(); ++i) {
      auto x = a->values[i], y = b->values[i];
      Entry z;
      if (named(op, "arith.muli")) {
        if (x.scale && y.scale)
          return std::nullopt;
        if (y.scale)
          std::swap(x, y);
        if (y.bias < -32768 || y.bias > 32768 || x.scale < -32768 ||
            x.scale > 32768 || x.bias < -1048576 || x.bias > 1048576)
          return std::nullopt;
        z = {x.row, x.scale * y.bias, x.bias * y.bias};
      } else {
        if (x.scale && y.scale && x.row != y.row)
          return std::nullopt;
        if (x.bias < -1048576 || x.bias > 1048576 || y.bias < -1048576 ||
            y.bias > 1048576 || x.scale < -32768 || x.scale > 32768 ||
            y.scale < -32768 || y.scale > 32768)
          return std::nullopt;
        z = {x.scale ? x.row : y.row, x.scale + y.scale, x.bias + y.bias};
      }
      if (!z.scale)
        z.row = -1;
      out.values[i] = z;
    }
    return out;
  }
  return std::nullopt;
}
inline bool target(Value offset, Operation *index) {
  auto x = evaluate(offset, index);
  if (!x || x->shape != SmallVector<int64_t>({8, 32}))
    return false;
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 32; ++c) {
      auto e = x->values[r * 32 + c];
      if (e.row != r || e.scale != 32768 || e.bias != c)
        return false;
    }
  return true;
}
inline bool constantRamp(Value offset, int64_t size) {
  auto x = evaluate(offset, nullptr);
  if (!x || int64_t(x->values.size()) != size)
    return false;
  for (int64_t i = 0; i < size; ++i)
    if (x->values[i].row != -1 || x->values[i].scale != 0 ||
        x->values[i].bias != i)
      return false;
  return true;
}
} // namespace mlir::ascend::normal_indexed_offset
#endif
