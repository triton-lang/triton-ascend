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

#include "TritonMemoryAccess/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <optional>

namespace mlir {
namespace {

Value createConstIndexValueOp(const Location &loc, OpBuilder &builder,
                              int64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(value))
      .getResult();
}

} // namespace

std::optional<int64_t> getConstantOfAttr(const OpFoldResult &arg) {
  if (isa<Attribute>(arg))
    return getConstantIntValue(arg);
  return std::nullopt;
}

OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() + rhsInt.value());

  if (!lhsInt && rhsInt && rhsInt.value() == 0)
    return lhs;
  if (!rhsInt && lhsInt && lhsInt.value() == 0)
    return rhs;

  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() - rhsInt.value());

  if (!lhsInt && rhsInt && rhsInt.value() == 0)
    return lhs;

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::SubIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() * rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
    if (lhsInt.value() == 1)
      return rhs;
  }
  if (rhsInt) {
    if (rhsInt.value() == 0)
      return rhs;
    if (rhsInt.value() == 1)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::MulIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot div 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() / rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  if (rhsInt) {
    if (rhsInt.value() == 1)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::DivSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot remainder by 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() % rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::RemSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::min(lhsInt.value(), rhsInt.value()));

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::MinSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::max(lhsInt.value(), rhsInt.value()));

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    rhsValue = convertToIndexIfNeeded(rhsValue, loc, b);
    assert(isa<IndexType>(rhsValue.getType()));
  }

  return b.create<arith::MaxSIOp>(loc, lhsValue, rhsValue).getResult();
}
std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  Attribute attr;
  if (auto val = dyn_cast<Value>(ofr)) {
    if (!val.getDefiningOp<arith::ConstantOp>())
      return std::nullopt;
    attr = cast<IntegerAttr>(val.getDefiningOp<arith::ConstantOp>().getValue());
  } else {
    attr = dyn_cast<Attribute>(ofr);
  }
  if (attr && isa<IntegerAttr>(attr))
    return dyn_cast<IntegerAttr>(attr).getInt();
  return std::nullopt;
}

Value materializeValue(OpBuilder &builder, Location loc, OpFoldResult ofr) {
  if (auto val = ofr.dyn_cast<Value>()) {
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(intVal.value()));
  }
  assert(intVal.has_value());
  return Value();

  // return builder.create<arith::ConstantIndexOp>(
  //     loc, dyn_cast<IntegerAttr>(attr).getInt());
}

bool isZero(const OpFoldResult ofr) {
  auto staticOfr = getIntAttr(ofr);
  return staticOfr.has_value() && staticOfr.value() == 0;
}

bool isOne(const OpFoldResult ofr) {
  auto staticOfr = getIntAttr(ofr);
  return staticOfr.has_value() && staticOfr.value() == 1;
}

Value convertToIndexIfNeeded(Value input, const Location &loc, OpBuilder &b) {
  auto inputType = input.getType();
  if (auto intType = dyn_cast<IntegerType>(inputType)) {
    if (intType.isInteger(32) || intType.isInteger(64)) {
      return b.create<arith::IndexCastOp>(loc, b.getIndexType(), input);
    }
  }
  return input;
}

RankedTensorType getExtractSlicedType(ArrayRef<OpFoldResult> shape,
                                      const llvm::SmallBitVector &droppedDims,
                                      Type elemType) {
  SmallVector<int64_t> targetShape;
  for (auto [idx, dimOfr] : llvm::enumerate(shape)) {
    if (!droppedDims[idx]) {
      if (auto dim = getConstantIntValue(dimOfr)) {
        targetShape.push_back(dim.value());
      } else {
        targetShape.push_back(ShapedType::kDynamic);
      }
    }
  }
  return RankedTensorType::get(targetShape, elemType);
}

// Fold layout constant info to attr, otherwise convert to index type value.
OpFoldResult getOpFoldResultOfLayoutInfo(Value value, OpBuilder &builder) {
  OpFoldResult constantFold = getAsOpFoldResult(value);
  if (llvm::isa<Attribute>(constantFold)) {
    assert(isa<IntegerAttr>(cast<Attribute>(constantFold)));
    return constantFold;
  }

  if (!isa<IntegerType>(value.getType()))
    llvm_unreachable("Illegal data type when parse block data layout info");

  if (!isa<IndexType>(value.getType())) {
    if (value.getType().isInteger(/*width*/ 1))
      value = builder.create<arith::IndexCastUIOp>(
          value.getLoc(), builder.getIndexType(), value);
    else
      value = builder.create<arith::IndexCastOp>(value.getLoc(),
                                                 builder.getIndexType(), value);
  }

  return value;
}

} // namespace mlir
