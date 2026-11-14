// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/MemoryModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>
#include <utility>
namespace mlir::triton::cfg {
namespace {
Type scalarType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}
} // namespace
bool hasBatchDynamicContiguousCopy(ModuleOp module) {
  auto owner = module->getAttrOfType<StringAttr>("hacc.block_mapping_owner");
  if (!owner || owner.getValue() != "block_batching")
    return false;

  // A dynamic outer prefix with complete, densely packed inner dimensions
  // lowers to a variable-length 1D transfer. A padded row, strided element
  // or dynamic inner dimension does not establish this property.
  auto isDynamicContiguous = [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    if (!memref || memref.getRank() == 0 || !memref.isDynamicDim(0))
      return false;
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(memref.getStridesAndOffset(strides, offset)))
      return false;
    int64_t expected = 1;
    for (int64_t dim = memref.getRank() - 1; dim >= 0; --dim) {
      if (strides[dim] != expected)
        return false;
      if (dim == 0)
        break;
      int64_t size = memref.getDimSize(dim);
      if (size <= 0 || expected > std::numeric_limits<int64_t>::max() / size)
        return false;
      expected *= size;
    }
    return true;
  };

  bool found = false;
  module.walk([&](Operation *op) {
    if (auto copy = dyn_cast<memref::CopyOp>(op))
      found |= isDynamicContiguous(copy.getSource().getType()) &&
               isDynamicContiguous(copy.getTarget().getType());
    else if (auto store =
                 dyn_cast<bufferization::MaterializeInDestinationOp>(op))
      found |= isDynamicContiguous(store.getDest().getType());
  });
  return found;
}

bool isBatchBoolToByteCast(Operation *op) {
  auto cast = dyn_cast_or_null<triton::BitcastOp>(op);
  if (!cast)
    return false;
  auto source =
      dyn_cast<triton::PointerType>(scalarType(cast.getSrc().getType()));
  auto result = dyn_cast<triton::PointerType>(scalarType(cast.getType()));
  return source && result && source.getAddressSpace() == 1 &&
         result.getAddressSpace() == 1 &&
         source.getPointeeType().isInteger(1) &&
         result.getPointeeType().isInteger(8);
}
bool isBatchNormalizedBoolLoad(Operation *op) {
  auto load = dyn_cast_or_null<triton::LoadOp>(op);
  if (!load)
    return false;
  auto marker = load->getAttrOfType<BoolAttr>("was_bool_to_int8");
  return marker && marker.getValue() &&
         isBatchBoolToByteCast(load.getPtr().getDefiningOp());
}
namespace {
unsigned zeroBits(Value value, unsigned depth = 0) {
  if (depth > 32)
    return 0;
  Operation *op = value.getDefiningOp();
  if (!op)
    return 0;
  if (auto c = dyn_cast<arith::ConstantOp>(op)) {
    if (auto integer = dyn_cast<IntegerAttr>(c.getValue()))
      return integer.getValue().countTrailingZeros();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue());
        dense && dense.isSplat())
      return dense.getSplatValue<APInt>().countTrailingZeros();
    return 0;
  }
  if (auto splat = dyn_cast<triton::SplatOp>(op))
    return zeroBits(splat.getSrc(), depth + 1);
  auto name = op->getName().getStringRef();
  if (name == "arith.muli")
    return std::min(64u, zeroBits(op->getOperand(0), depth + 1) +
                             zeroBits(op->getOperand(1), depth + 1));
  if (name == "arith.addi" || name == "arith.subi")
    return std::min(zeroBits(op->getOperand(0), depth + 1),
                    zeroBits(op->getOperand(1), depth + 1));
  if (name == "arith.extsi" || name == "arith.extui" ||
      name == "arith.trunci") {
    Type type = op->getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    return std::min(cast<IntegerType>(type).getWidth(),
                    zeroBits(op->getOperand(0), depth + 1));
  }
  return 0;
}
std::optional<BatchPointerPlan> match(Value value, unsigned bytes,
                                      unsigned depth) {
  if (depth > 32)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto cast = dyn_cast<triton::PtrToIntOp>(op))
    return BatchPointerPlan{cast.getSrc(), {}};
  if (auto splat = dyn_cast<triton::SplatOp>(op))
    return match(splat.getSrc(), bytes, depth + 1);
  auto name = op->getName().getStringRef();
  if (name != "arith.addi" && name != "arith.subi")
    return std::nullopt;
  auto lhs = match(op->getOperand(0), bytes, depth + 1);
  auto rhs = match(op->getOperand(1), bytes, depth + 1);
  if (static_cast<bool>(lhs) == static_cast<bool>(rhs) ||
      (rhs && name == "arith.subi"))
    return std::nullopt;
  bool left = static_cast<bool>(lhs);
  Value delta = op->getOperand(left ? 1 : 0);
  unsigned required = bytes == 8 ? 3 : bytes == 4 ? 2 : 1;
  if (zeroBits(delta) < required)
    return std::nullopt;
  auto result = left ? std::move(*lhs) : std::move(*rhs);
  result.byteOffsets.emplace_back(delta, name == "arith.subi");
  return result;
}
} // namespace
std::optional<BatchPointerPlan> getBatchPointerPlan(Value address,
                                                    unsigned bytes) {
  if (bytes != 2 && bytes != 4 && bytes != 8)
    return std::nullopt;
  return match(address, bytes, 0);
}
} // namespace mlir::triton::cfg
