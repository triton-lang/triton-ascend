//===- TritonToTensorView.cpp - TensorView pointer legalization
//------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ascend/include/TritonToTensorView/Passes.h"
#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOTENSORVIEW
#include "ascend/include/TritonToTensorView/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
static Type getAddressType(Type type, OpBuilder &builder) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (isa<triton::PointerType>(tensor.getElementType()))
      return tensor.clone(builder.getI64Type());
  } else if (isa<triton::PointerType>(type)) {
    return builder.getI64Type();
  }
  return type;
}

static FailureOr<Value>
convertPointer(Value pointer, OpBuilder &builder,
               llvm::DenseMap<Value, Value> &addresses) {
  if (auto it = addresses.find(pointer); it != addresses.end())
    return it->second;
  Type type = getAddressType(pointer.getType(), builder);
  if (type == pointer.getType())
    return pointer;
  Location loc = pointer.getLoc();
  if (isa<triton::PointerType>(pointer.getType())) {
    Value address = builder.create<triton::PtrToIntOp>(loc, type, pointer);
    addresses[pointer] = address;
    return address;
  }
  Operation *producer = pointer.getDefiningOp();
  if (!producer)
    return emitError(loc, "pointer tensors must be inlined before TensorView "
                          "legalization"),
           failure();
  if (auto add = dyn_cast<triton::AddPtrOp>(producer)) {
    auto base = convertPointer(add.getPtr(), builder, addresses);
    if (failed(base))
      return failure();
    Value offset = add.getOffset();
    auto offsetType = cast<RankedTensorType>(offset.getType());
    Type wideType = offsetType.clone(builder.getI64Type());
    if (offsetType.getElementType().isIndex())
      offset = builder.create<arith::IndexCastOp>(loc, wideType, offset);
    else if (offsetType.getElementType().getIntOrFloatBitWidth() < 64)
      offset = builder.create<arith::ExtSIOp>(loc, wideType, offset);
    Type elementType =
        cast<triton::PointerType>(
            cast<RankedTensorType>(pointer.getType()).getElementType())
            .getPointeeType();
    if (!elementType.isIntOrFloat())
      return producer->emitError("unsupported pointer element type"), failure();
    int64_t bytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;
    Value scale = builder.create<arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(cast<RankedTensorType>(wideType),
                                       llvm::APInt(64, bytes)));
    Value scaled = builder.create<arith::MulIOp>(loc, offset, scale);
    Value result = builder.create<arith::AddIOp>(loc, *base, scaled);
    addresses[pointer] = result;
    return result;
  }
  if (auto cast = dyn_cast<triton::IntToPtrOp>(producer)) {
    if (cast.getSrc().getType() != type)
      return cast.emitOpError("TensorView addresses must be i64"), failure();
    addresses[pointer] = cast.getSrc();
    return cast.getSrc();
  }
  if (auto cast = dyn_cast<triton::BitcastOp>(producer)) {
    auto address = convertPointer(cast.getSrc(), builder, addresses);
    if (failed(address))
      return failure();
    addresses[pointer] = *address;
    return *address;
  }
  bool shapeOp = llvm::StringSwitch<bool>(producer->getName().getStringRef())
                     .Cases("tt.splat", "tt.broadcast", "tt.expand_dims", true)
                     .Cases("tt.trans", "tt.reshape", "tt.cat", true)
                     .Cases("tt.join", "tt.split", "tt.gather", true)
                     .Case("arith.select", true)
                     .Default(false);
  if (!shapeOp)
    return producer->emitError("unsupported TensorView pointer producer"),
           failure();
  IRMapping mapping;
  for (Value operand : producer->getOperands()) {
    auto address = convertPointer(operand, builder, addresses);
    if (failed(address))
      return failure();
    mapping.map(operand, *address);
  }
  Operation *copy = builder.clone(*producer, mapping);
  for (auto [oldResult, newResult] :
       llvm::zip(producer->getResults(), copy->getResults())) {
    newResult.setType(getAddressType(oldResult.getType(), builder));
    addresses[oldResult] = newResult;
  }
  return addresses.lookup(pointer);
}

struct TritonToTensorViewPass
    : triton::impl::TritonToTensorViewBase<TritonToTensorViewPass> {
  void runOnOperation() override {
    SmallVector<triton::PtrToIntOp> conversions;
    getOperation().walk([&](triton::PtrToIntOp op) {
      if (!isa<RankedTensorType>(op.getSrc().getType()))
        return;
      if (llvm::any_of(op->getUsers(), [](Operation *user) {
            return isa<tv::PtrLoadOp, tv::PtrStoreOp>(user);
          }))
        conversions.push_back(op);
    });
    for (triton::PtrToIntOp op : conversions) {
      OpBuilder builder(op);
      llvm::DenseMap<Value, Value> addresses;
      auto address = convertPointer(op.getSrc(), builder, addresses);
      if (failed(address))
        return signalPassFailure();
      op.getResult().replaceAllUsesWith(*address);
      op.erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToTensorViewPass() {
  return std::make_unique<TritonToTensorViewPass>();
}
