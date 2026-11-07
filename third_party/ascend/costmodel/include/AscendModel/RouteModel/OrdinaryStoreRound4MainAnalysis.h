// Restricted post-transform TTIR facts for main TA 70f770c.
// No external shape/pitch/axis metadata is trusted by this analysis.
#ifndef ORDINARY_STORE_ROUND4_MAIN_ANALYSIS_H
#define ORDINARY_STORE_ROUND4_MAIN_ANALYSIS_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <optional>

namespace mlir::ascend::ordinary_store_main {
// Each axis is either one constant-affine coordinate, or one loaded index
// scaled by a compile-time constant. A mixture on the same axis is rejected.
struct Axis {
  int64_t stride = 0;
  int64_t loadedScale = 0;
  Value loaded;
  bool valid = true;
};
inline SmallVector<Axis> unknown(Value value) {
  auto t = dyn_cast<RankedTensorType>(value.getType());
  return t ? SmallVector<Axis>(t.getRank(), Axis{0, 0, {}, false})
           : SmallVector<Axis>{};
}
inline std::optional<int64_t> constant(Value value, unsigned depth = 0) {
  if (depth > 48) return std::nullopt;
  auto *op = value.getDefiningOp();
  if (!op) return std::nullopt;
  if ((op->getName().getStringRef() == "tt.splat" ||
       op->getName().getStringRef() == "tt.broadcast") && op->getNumOperands() == 1)
    return constant(op->getOperand(0), depth + 1);
  if (op->getName().getStringRef() != "arith.constant") return std::nullopt;
  if (auto a = op->getAttrOfType<IntegerAttr>("value")) {
    if (a.getValue().getBitWidth() > 64) return std::nullopt;
    return a.getInt();
  }
  if (auto a = op->getAttrOfType<DenseIntElementsAttr>("value"))
    if (a.isSplat() && a.getElementType().getIntOrFloatBitWidth() <= 64)
      return a.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}
inline SmallVector<Axis> axesImpl(Value value, unsigned depth, unsigned &remaining) {
  auto result = unknown(value);
  if (depth > 48 || !remaining) return result;
  --remaining;
  auto *op = value.getDefiningOp();
  if (!op) return result;
  const auto name = op->getName().getStringRef();
  const size_t rank = result.size();
  if (name == "tt.splat" && op->getNumOperands() == 1 &&
      !isa<RankedTensorType>(op->getOperand(0).getType())) {
    // A scalar pointer/program offset is uniform. It never determines an axis.
    for (auto &a : result) a.valid = true;
    return result;
  }
  if (constant(value)) {
    for (auto &a : result) a.valid = true;
    return result;
  }
  if (name == "tt.make_range" && rank == 1) {
    result[0] = Axis{1, 0, {}, true}; return result;
  }
  if (name == "tt.load") {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    // Only integer one-dimensional/separable indices have the measured D
    // contract. Arbitrary multi-axis loaded tensors are not separable geometry.
    if (!type || !type.getElementType().isInteger(32) ||
        llvm::count_if(type.getShape(), [](int64_t n) { return n > 1; }) != 1)
      return result;
    for (size_t i = 0; i < rank; ++i) {
      result[i] = Axis{};
      if (type.getDimSize(i) > 1) {
        result[i].loaded = value; result[i].loadedScale = 1;
      }
    }
    return result;
  }
  if (name == "tt.expand_dims" && op->getNumOperands() == 1) {
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    auto src = axesImpl(op->getOperand(0), depth + 1, remaining);
    if (!axis || axis.getInt() < 0 || static_cast<size_t>(axis.getInt()) >= rank ||
        src.size() + 1 != rank) return result;
    for (size_t i = 0, j = 0; i < rank; ++i)
      result[i] = i == static_cast<size_t>(axis.getInt()) ? Axis{} : src[j++];
    return result;
  }
  if (name == "tt.broadcast" && op->getNumOperands() == 1) {
    auto srcType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto dstType = dyn_cast<RankedTensorType>(value.getType());
    auto src = axesImpl(op->getOperand(0), depth + 1, remaining);
    if (!srcType || !dstType || src.size() != rank) return result;
    for (size_t i = 0; i < rank; ++i) {
      if (srcType.getDimSize(i) == 1) result[i] = Axis{};
      else if (srcType.getDimSize(i) == dstType.getDimSize(i)) result[i] = src[i];
    }
    return result;
  }
  // Signed widening preserves loaded integer values; truncation, zero
  // extension and reshape do not receive an implicit geometry equivalence.
  if (name == "arith.extsi" && op->getNumOperands() == 1) {
    auto src = axesImpl(op->getOperand(0), depth + 1, remaining);
    return src.size() == rank ? src : result;
  }
  if (op->getNumOperands() != 2 ||
      (name != "tt.addptr" && name != "arith.addi" &&
       name != "arith.subi" && name != "arith.muli")) return result;
  auto lhs = axesImpl(op->getOperand(0), depth + 1, remaining);
  auto rhs = axesImpl(op->getOperand(1), depth + 1, remaining);
  if (lhs.size() != rank || rhs.size() != rank) return result;
  auto lconst = constant(op->getOperand(0)), rconst = constant(op->getOperand(1));
  for (size_t i = 0; i < rank; ++i) {
    if (!lhs[i].valid || !rhs[i].valid) continue;
    Axis a;
    if (name == "arith.muli") {
      if (!lconst && !rconst) continue; // Dynamic Host/device pitch is unsupported.
      const auto scale = lconst ? *lconst : *rconst;
      const auto &source = lconst ? rhs[i] : lhs[i];
      if (__builtin_mul_overflow(source.stride, scale, &a.stride) ||
          __builtin_mul_overflow(source.loadedScale, scale, &a.loadedScale)) continue;
      a.loaded = source.loaded;
    } else {
      const int64_t sign = name == "arith.subi" ? -1 : 1;
      int64_t stride, loadedScale;
      if (__builtin_mul_overflow(rhs[i].stride, sign, &stride) ||
          __builtin_add_overflow(lhs[i].stride, stride, &a.stride) ||
          __builtin_mul_overflow(rhs[i].loadedScale, sign, &loadedScale) ||
          __builtin_add_overflow(lhs[i].loadedScale, loadedScale, &a.loadedScale)) continue;
      if (lhs[i].loaded && rhs[i].loaded && lhs[i].loaded != rhs[i].loaded) continue;
      a.loaded = lhs[i].loaded ? lhs[i].loaded : rhs[i].loaded;
    }
    if (a.loaded && a.stride) continue; // No guessed mixed term on one logical axis.
    result[i] = a;
  }
  return result;
}
inline SmallVector<Axis> axes(Value value) {
  unsigned remaining = 2048;
  return axesImpl(value, 0, remaining);
}

// Evaluate only constant/range mask graphs, never loaded or runtime values.
struct Tensor {
  SmallVector<int64_t> shape, values;
};
class StaticMask {
  llvm::DenseMap<Value, std::optional<Tensor>> memo;
  unsigned remainingNodes = 512;

public:
  std::optional<Tensor> evaluate(Value value) {
    if (auto it = memo.find(value); it != memo.end())
      return it->second;
    if (!remainingNodes) return std::nullopt;
    --remainingNodes;
    auto result = compute(value);
    memo[value] = result;
    return result;
  }

private:
  std::optional<Tensor> compute(Value value) {
    auto *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    Tensor result;
    int64_t count = 1;
    Type scalarType = value.getType();
    if (auto t = dyn_cast<RankedTensorType>(value.getType())) {
      if (!t.hasStaticShape() || t.getNumElements() > 8192)
        return std::nullopt;
      result.shape.assign(t.getShape().begin(), t.getShape().end());
      count = t.getNumElements();
      scalarType = t.getElementType();
    }
    if (auto integer = dyn_cast<IntegerType>(scalarType)) {
      if (integer.getWidth() > 64)
        return std::nullopt;
    } else if (!isa<IndexType>(scalarType))
      return std::nullopt;
    auto name = op->getName().getStringRef();
    if (name == "arith.constant") {
      if (auto a = op->getAttrOfType<IntegerAttr>("value"))
        result.values.assign(count, a.getValue().isOne() ? 1 : a.getInt());
      else if (auto a = op->getAttrOfType<DenseIntElementsAttr>("value"))
        for (const auto &v : a.getValues<APInt>())
          result.values.push_back(v.isOne() ? 1 : v.getSExtValue());
      else
        return std::nullopt;
      return result;
    }
    if (name == "tt.make_range") {
      auto start = op->getAttrOfType<IntegerAttr>("start");
      if (!start)
        return std::nullopt;
      for (int64_t i = 0; i < count; ++i)
        result.values.push_back(start.getInt() + i);
      return result;
    }
    if (op->getNumOperands() == 1 &&
        llvm::is_contained({StringRef("tt.splat"), StringRef("tt.broadcast"),
                            StringRef("tt.expand_dims"),
                            StringRef("tt.reshape"), StringRef("arith.extsi"),
                            StringRef("arith.extui"),
                            StringRef("arith.index_cast")},
                           name)) {
      auto src = evaluate(op->getOperand(0));
      if (!src)
        return std::nullopt;
      // Values are represented as signed int64_t, except i1 predicates use
      // 0/1. Do not treat zero extension of negative values, or sign extension
      // of true i1, as an identity: that can prove a false rectangle.
      if (name == "arith.extui" &&
          llvm::any_of(src->values, [](int64_t v) { return v < 0; }))
        return std::nullopt;
      Type sourceType = op->getOperand(0).getType();
      if (auto tensor = dyn_cast<RankedTensorType>(sourceType))
        sourceType = tensor.getElementType();
      if (name == "arith.extsi" && sourceType.isInteger(1) &&
          llvm::any_of(src->values, [](int64_t v) { return v != 0; }))
        return std::nullopt;
      if (name != "tt.broadcast" && name != "tt.splat") {
        if (int64_t(src->values.size()) != count)
          return std::nullopt;
        result.values = src->values;
      } else {
        for (int64_t t = 0; t < count; ++t) {
          int64_t flat = t, from = 0, stride = 1;
          for (size_t i = result.shape.size(); i-- > 0;) {
            int64_t j = flat % result.shape[i];
            flat /= result.shape[i];
            if (i < src->shape.size()) {
              if (src->shape[i] != 1 && src->shape[i] != result.shape[i])
                return std::nullopt;
              from += (src->shape[i] == 1 ? 0 : j) * stride;
              stride *= src->shape[i];
            }
          }
          result.values.push_back(src->values[from]);
        }
      }
      return result;
    }
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto a = evaluate(op->getOperand(0)), b = evaluate(op->getOperand(1));
    if (!a || !b || a->values.size() != size_t(count) ||
        b->values.size() != size_t(count))
      return std::nullopt;
    for (int64_t i = 0; i < count; ++i) {
      const int64_t x = a->values[i], y = b->values[i];
      int64_t z = 0;
      if (name == "arith.addi") {
        if (__builtin_add_overflow(x, y, &z))
          return std::nullopt;
      } else if (name == "arith.subi") {
        if (__builtin_sub_overflow(x, y, &z))
          return std::nullopt;
      } else if (name == "arith.muli") {
        if (__builtin_mul_overflow(x, y, &z))
          return std::nullopt;
      } else if (name == "arith.andi")
        z = x & y;
      else if (name == "arith.ori")
        z = x | y;
      else if (name == "arith.divsi" || name == "arith.remsi" ||
               name == "arith.divui" || name == "arith.remui") {
        if (y <= 0 || x < 0)
          return std::nullopt;
        z = name.contains("div") ? x / y : x % y;
      } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          z = x == y;
          break;
        case arith::CmpIPredicate::ne:
          z = x != y;
          break;
        case arith::CmpIPredicate::slt:
          z = x < y;
          break;
        case arith::CmpIPredicate::sle:
          z = x <= y;
          break;
        case arith::CmpIPredicate::sgt:
          z = x > y;
          break;
        case arith::CmpIPredicate::sge:
          z = x >= y;
          break;
        case arith::CmpIPredicate::ult:
          z = uint64_t(x) < uint64_t(y);
          break;
        case arith::CmpIPredicate::ule:
          z = uint64_t(x) <= uint64_t(y);
          break;
        case arith::CmpIPredicate::ugt:
          z = uint64_t(x) > uint64_t(y);
          break;
        case arith::CmpIPredicate::uge:
          z = uint64_t(x) >= uint64_t(y);
          break;
        }
      } else
        return std::nullopt;
      Type scalarType = value.getType();
      if (auto tensor = dyn_cast<RankedTensorType>(scalarType))
        scalarType = tensor.getElementType();
      if (auto integer = dyn_cast<IntegerType>(scalarType)) {
        const unsigned bits = integer.getWidth();
        if (bits > 64 || (bits > 1 && bits < 64 &&
                          (z < -(int64_t(1) << (bits - 1)) ||
                           z > (int64_t(1) << (bits - 1)) - 1)))
          return std::nullopt;
      }
      result.values.push_back(z);
    }
    return result;
  }
};

} // namespace mlir::ascend::ordinary_store_main
#endif
