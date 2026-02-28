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

#include "ascend/include/TritonToLinalg/ArgMinMaxConverter.h"
#include <cstdint>
#include <limits>

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

// ArgMinConverter functions
LogicalResult ArgMinConverter::matchComparisonResult(
    Value currValue, Value currIndex, Value reduceValue, Value reduceIndex,
    mlir::Block::iterator &it, Value &comparisonResult) {
  LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");

  auto cmpOp = dyn_cast<arith::CmpFOp>(*it);
  auto cmpIOp = dyn_cast<arith::CmpIOp>(*it++);
  if (!cmpOp && !cmpIOp)
    return failure();

  if (cmpOp) {
    if (cmpOp.getPredicate() != arith::CmpFPredicate::OLT ||
        currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpOp;
  }

  if (cmpIOp) {
    if ((cmpIOp.getPredicate() != arith::CmpIPredicate::slt &&
         cmpIOp.getPredicate() != arith::CmpIPredicate::ult) ||
        currValue != cmpIOp.getLhs() || reduceValue != cmpIOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpIOp;
  }

  return success();
}

float ArgMinConverter::getBaseReductionValue() {
  return std::numeric_limits<float>::infinity();
}

int8_t ArgMinConverter::getBaseReductionIntValue() {
  return std::numeric_limits<int8_t>::max();
}
uint8_t ArgMinConverter::getBaseReductionUIntValue() {
  return std::numeric_limits<uint8_t>::max();
}

// ArgMaxConverter functions
LogicalResult ArgMaxConverter::matchComparisonResult(
    Value currValue, Value currIndex, Value reduceValue, Value reduceIndex,
    mlir::Block::iterator &it, Value &comparisonResult) {
  auto cmpOp = dyn_cast<arith::CmpFOp>(*it);
  auto cmpIOp = dyn_cast<arith::CmpIOp>(*it++);
  if (!cmpOp && !cmpIOp)
    return failure();

  if (cmpOp) {
    if (cmpOp.getPredicate() != arith::CmpFPredicate::OGT ||
        currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpOp;
  }

  if (cmpIOp) {
    if ((cmpIOp.getPredicate() != arith::CmpIPredicate::sgt &&
         cmpIOp.getPredicate() != arith::CmpIPredicate::ugt) ||
        currValue != cmpIOp.getLhs() || reduceValue != cmpIOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpIOp;
  }

  return success();
}

float ArgMaxConverter::getBaseReductionValue() {
  return -std::numeric_limits<float>::infinity();
}

int8_t ArgMaxConverter::getBaseReductionIntValue() {
  return std::numeric_limits<int8_t>::min();
}
uint8_t ArgMaxConverter::getBaseReductionUIntValue() {
  return std::numeric_limits<uint8_t>::min();
}

} // namespace TTOpConverters
