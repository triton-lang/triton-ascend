// Shared conservative TTIR evaluators; no runtime execution.
#ifndef PARTIAL_STRUCTURED_STATIC_ANALYSIS_H
#define PARTIAL_STRUCTURED_STATIC_ANALYSIS_H
namespace {
// Evaluate only constant/range mask graphs, never loaded or runtime values.
struct Tensor {
  SmallVector<int64_t> shape, values;
};
class StaticMask {
  llvm::DenseMap<Value, std::optional<Tensor>> memo;

public:
  std::optional<Tensor> evaluate(Value value) {
    if (auto it = memo.find(value); it != memo.end())
      return it->second;
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
      if (!t.hasStaticShape() || t.getNumElements() > 2048)
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

// Extract the explicit additive program_id term. A loaded index remains an
// opaque residual even if its table address itself contains program_id.
struct AffineProgram {
  int64_t constant = 0, coefficient = 0;
  bool opaque = false, valid = true;
};
AffineProgram programTerm(Value v, unsigned depth = 0) {
  if (depth > 64)
    return {0, 0, true, false};
  auto *op = v.getDefiningOp();
  if (!op)
    return {0, 0, true, true};
  auto name = op->getName().getStringRef();
  if (name == "tt.load")
    return {0, 0, true, true};
  if (name == "tt.get_program_id") {
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    return {0, axis && axis.getInt() == 0 ? 1 : 0, false,
            axis && axis.getInt() == 0};
  }
  if (name == "arith.constant") {
    if (auto a = op->getAttrOfType<IntegerAttr>("value")) {
      if (a.getValue().getBitWidth() > 64)
        return {0, 0, true, false};
      return {a.getInt(), 0, false, true};
    }
    if (auto a = op->getAttrOfType<DenseIntElementsAttr>("value"))
      if (a.isSplat()) {
        auto value = a.getSplatValue<APInt>();
        if (value.getBitWidth() > 64)
          return {0, 0, true, false};
        return {value.getSExtValue(), 0, false, true};
      }
    return {0, 0, true, true};
  }
  if (name == "tt.bitcast" && op->getNumOperands() == 1) {
    auto src = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto dst = dyn_cast<RankedTensorType>(v.getType());
    if (src && dst && src.getShape() == dst.getShape()) {
      auto sp = dyn_cast<triton::PointerType>(src.getElementType());
      auto dp = dyn_cast<triton::PointerType>(dst.getElementType());
      if (sp && dp && sp.getPointeeType().isInteger(1) &&
          dp.getPointeeType().isInteger(8) &&
          sp.getAddressSpace() == dp.getAddressSpace())
        return programTerm(op->getOperand(0), depth + 1);
    }
  }
  if (op->getNumOperands() == 1 &&
      llvm::is_contained({StringRef("tt.splat"), StringRef("tt.broadcast"),
                          StringRef("tt.expand_dims"), StringRef("tt.reshape"),
                          StringRef("arith.extsi"), StringRef("arith.extui"),
                          StringRef("arith.index_cast")},
                         name))
    return programTerm(op->getOperand(0), depth + 1);
  if (op->getNumOperands() == 2) {
    auto a = programTerm(op->getOperand(0), depth + 1);
    auto b = programTerm(op->getOperand(1), depth + 1);
    if (!a.valid || !b.valid)
      return {0, 0, true, false};
    if (name == "arith.addi" || name == "tt.addptr" || name == "arith.subi") {
      int64_t k, c;
      bool fail =
          name == "arith.subi"
              ? __builtin_sub_overflow(a.coefficient, b.coefficient, &k) ||
                    __builtin_sub_overflow(a.constant, b.constant, &c)
              : __builtin_add_overflow(a.coefficient, b.coefficient, &k) ||
                    __builtin_add_overflow(a.constant, b.constant, &c);
      return fail ? AffineProgram{0, 0, true, false}
                  : AffineProgram{c, k, a.opaque || b.opaque, true};
    }
    if (name == "arith.muli" && ((!a.opaque && a.coefficient == 0) ||
                                 (!b.opaque && b.coefficient == 0))) {
      if (a.opaque || a.coefficient != 0)
        std::swap(a, b);
      int64_t k, c;
      if (__builtin_mul_overflow(a.constant, b.coefficient, &k) ||
          __builtin_mul_overflow(a.constant, b.constant, &c))
        return {0, 0, true, false};
      return {c, k, b.opaque, true};
    }
    if (a.coefficient || b.coefficient)
      return {0, 0, true, false};
  }
  // Do not silently discard a program term inside an unsupported operation.
  for (Value operand : op->getOperands()) {
    auto a = programTerm(operand, depth + 1);
    if (!a.valid || a.coefficient)
      return {0, 0, true, false};
  }
  return {0, 0, true, true};
}
} // namespace

#endif
