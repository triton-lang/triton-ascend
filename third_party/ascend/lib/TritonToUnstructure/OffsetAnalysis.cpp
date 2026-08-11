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

#include "TritonToUnstructure/OffsetAnalysis.h"
#include "Utils/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-offset-analysis"

namespace mlir {
namespace triton {

PtrOffsetInfo::PtrOffsetInfo() : ptr(nullptr), offset(nullptr) {}

PtrOffsetInfo::PtrOffsetInfo(const PtrOffsetInfo &other) { *this = other; }

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr) : ptr(ptr) { setZeroOffset(); }

PtrOffsetInfo::PtrOffsetInfo(ArrayRef<AxisInfo> structured)
    : ptr(nullptr), offset(nullptr) {
  setStructured(structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, AxisInfo structured) : ptr(ptr) {
  setZeroOffset();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, ArrayRef<AxisInfo> structured)
    : ptr(ptr) {
  setStructured(structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, const Value &offset,
                             AxisInfo structured)
    : ptr(ptr), offset(offset) {
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, const Value &offset,
                             ArrayRef<AxisInfo> structured)
    : ptr(ptr), offset(offset) {
  setStructured(structured);
}

PtrOffsetInfo &PtrOffsetInfo::operator=(const PtrOffsetInfo &other) {
  setPtr(other.getPtr());
  setOffset(other.getOffset());
  setOffsets(other.getOffsets());
  setStructured(other.getStructured());
  setScalarLike(other.isScalarLike());
  setByteAddressed(other.isByteAddressed());
  return *this;
}

Value PtrOffsetInfo::getPtr() const { return this->ptr; }
Value PtrOffsetInfo::getOffset() const { return this->offset; }
SmallVector<Value> PtrOffsetInfo::getOffsets() const {
  return this->tptOffsets;
}
SmallVector<Value> &PtrOffsetInfo::getOffsetsRef() { return this->tptOffsets; }

bool PtrOffsetInfo::isScalarLike() const { return this->scalarLike; }
bool PtrOffsetInfo::isByteAddressed() const { return this->byteAddressed; }

SmallVector<PtrOffsetInfo::AxisInfo> &PtrOffsetInfo::getStructuredRef() {
  return this->structured;
}
const SmallVector<PtrOffsetInfo::AxisInfo> &
PtrOffsetInfo::getStructured() const {
  return this->structured;
}

int PtrOffsetInfo::getRank() const { return structured.size(); }

void PtrOffsetInfo::setPtr(const Value &ptr) { this->ptr = ptr; }
void PtrOffsetInfo::setOffset(const Value &offset) { this->offset = offset; }

void PtrOffsetInfo::setOffsets(ValueRange offsets) {
  tptOffsets.clear();
  for (auto offset : offsets)
    tptOffsets.push_back(offset);
}

void PtrOffsetInfo::setStructured() {
  assert(ptr && "ptr Should be to infer rank");
  this->structured.clear();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), AxisInfo::structured);
}

void PtrOffsetInfo::setStructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, AxisInfo::structured);
}

void PtrOffsetInfo::setStructured(int rank, AxisInfo info) {
  this->structured.clear();
  this->structured.resize(rank, info);
}

void PtrOffsetInfo::setUnstructured() {
  assert(ptr && "ptr Should be to infer rank");
  this->structured.clear();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), AxisInfo::unstructured);
}

void PtrOffsetInfo::setUnstructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, AxisInfo::unstructured);
}

void PtrOffsetInfo::setStructured(ArrayRef<AxisInfo> structured) {
  this->structured.resize(structured.size());
  for (size_t i = 0; i < structured.size(); i++)
    this->structured[i] = structured[i];
}

void PtrOffsetInfo::setStructured(const PtrOffsetInfo &other) {
  this->setStructured(other.getStructured());
}

void PtrOffsetInfo::setScalarLike(bool scalarLike) {
  this->scalarLike = scalarLike;
}

void PtrOffsetInfo::setByteAddressed(bool byteAddressed) {
  this->byteAddressed = byteAddressed;
}

bool PtrOffsetInfo::isStructured(int dim) const {
  return this->scalarLike || structured[dim] == AxisInfo::structured ||
         structured[dim] == AxisInfo::scalar;
}

bool PtrOffsetInfo::isStructured() const {
  return this->scalarLike || llvm::all_of(structured, [](auto dim) {
           return dim == AxisInfo::structured || dim == AxisInfo::scalar;
         });
}

bool PtrOffsetInfo::isUnstructured() const {
  return llvm::all_of(structured,
                      [](auto dim) { return dim == AxisInfo::unstructured; });
}

bool PtrOffsetInfo::isUnstructuredOrScalarlike() const {
  return llvm::all_of(structured, [](auto dim) {
    return dim == AxisInfo::unstructured || dim == AxisInfo::scalarlike ||
           dim == AxisInfo::scalar;
  });
}

void PtrOffsetInfo::setZeroOffset() {
  if (!ptr)
    return;
  Value offset;
  OpBuilder builder(ptr.getContext());
  builder.setInsertionPointToStart(ptr.getParentBlock());
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
    offset = builder.create<arith::ConstantOp>(
        ptr.getLoc(), DenseElementsAttr::get(
                          RankedTensorType::get(tensorType.getShape(),
                                                builder.getIntegerType(64)),
                          builder.getZeroAttr(builder.getIntegerType(64))));
  } else {
    offset = builder.create<arith::ConstantOp>(ptr.getLoc(),
                                               builder.getI64IntegerAttr(0));
  }
  setOffset(offset);
}

PtrOffsetInfo combineInfo(const PtrOffsetInfo &lhs, const PtrOffsetInfo &rhs) {
  PtrOffsetInfo info;
  assert(lhs.getRank() == rhs.getRank() && "Rank must be same to be combined");

  info.setScalarLike(lhs.isScalarLike() && rhs.isScalarLike());
  auto &structuredRef = info.getStructuredRef();
  auto lhsStructured = lhs.getStructured();
  auto rhsStructured = rhs.getStructured();
  structuredRef.resize(lhs.getRank());
  for (size_t i = 0; i < structuredRef.size(); i++)
    structuredRef[i] = std::min(lhsStructured[i], rhsStructured[i]);
  return info;
}

void parse(Value operand, const Location &loc, RewriterBase &rewriter,
           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  (void)parseChecked(operand, loc, rewriter, offsetMap);
}
LogicalResult parseChecked(Value operand, const Location &loc,
                           RewriterBase &rewriter,
                           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (offsetMap.contains(operand)) {
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "found\n" << operand << '\n';
    });
    return success();
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "parse\n" << operand << '\n';
  });

  LogicalResult result = success();
  if (auto *defOp = operand.getDefiningOp()) {
    if (isa<arith::ArithDialect>(defOp->getDialect())) {
      result = parseArithOp(defOp, loc, rewriter, offsetMap);
    } else if (isa<triton::TritonDialect>(defOp->getDialect())) {
      result = parseTritonOp(defOp, loc, rewriter, offsetMap);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      result = parseIf(ifOp, loc, rewriter, offsetMap, operand);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(defOp)) {
      result = parseYield(yieldOp, loc, rewriter, offsetMap);
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(defOp)) {
      result = parseLoopOp(loopOp, loc, rewriter, offsetMap, operand);
    } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(defOp)) {
      result = parseExtract(extractOp, loc, rewriter, offsetMap);
    } else if (auto insertOp = dyn_cast<tensor::InsertOp>(defOp)) {
      result = parseInsert(insertOp, loc, rewriter, offsetMap);
    } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      result = parseExtractSlice(extractSliceOp, loc, rewriter, offsetMap);
    } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(defOp)) {
      result = parseInsertSlice(insertSliceOp, loc, rewriter, offsetMap);
    } else if (isDistributedTypeCustomOp(defOp)) {
      auto opResult = dyn_cast<OpResult>(operand);
      if (!opResult) {
        defOp->emitError("expected distributed custom-op result");
        return failure();
      }
      result = parseStructuredCustomOp(defOp, loc, rewriter, offsetMap,
                                       opResult.getResultNumber());
    }
  } else if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
    auto parentOp = blockArgument.getOwner()->getParentOp();
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Handling block argument\n" << *blockArgument.getOwner() << '\n';
    });
    if (isa<FunctionOpInterface>(parentOp)) {
      if (isa<triton::PointerType>(operand.getType()))
        offsetMap[operand] =
            PtrOffsetInfo(operand, PtrOffsetInfo::AxisInfo::scalar);
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      result = parseLoopRegionIterArg(loopOp, loc, rewriter, offsetMap,
                                      blockArgument);
    }
  } else {
    llvm_unreachable("Unreachable");
  }

  if (failed(result))
    return failure();

  if (!offsetMap.contains(operand)) {
    offsetMap[operand] = PtrOffsetInfo();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType()))
      offsetMap[operand].setUnstructured(tensorType.getRank());
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "finish parse\n" << operand << '\n';
    auto data = offsetMap.at(operand);
    for (auto s : data.getStructuredRef())
      os << static_cast<int>(s);
    os << "\n";
  });

  // Missing scalar roots are diagnosed by the memory converter. Keeping
  // analysis non-fatal here turns unsupported tensor-pointer arguments into a
  // regular compilation error instead of terminating the compiler.
  return success();
}
LogicalResult
parseLoopRegionIterArg(LoopLikeOpInterface loopOp, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                       BlockArgument regionIterArg) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation());
      whileOp && whileOp.getAfterBody() == regionIterArg.getOwner()) {
    auto argNum = regionIterArg.getArgNumber();
    auto conditionArg = whileOp.getConditionOp().getArgs()[argNum];
    if (failed(parseChecked(conditionArg, loc, rewriter, offsetMap)))
      return failure();
    auto tmp = offsetMap[conditionArg];
    offsetMap[regionIterArg] = tmp;
    return success();
  }
  OpOperand *initArgOperand = loopOp.getTiedLoopInit(regionIterArg);
  if (!initArgOperand)
    return success();
  Value initArg = initArgOperand->get();
  if (failed(parseChecked(initArg, loc, rewriter, offsetMap)))
    return failure();
  auto tmp = offsetMap[initArg];
  offsetMap[regionIterArg] = tmp;
  return success();
}

LogicalResult parseArithOp(Operation *arithOp, const Location &loc,
                           RewriterBase &rewriter,
                           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  assert(isa<arith::ArithDialect>(arithOp->getDialect()));
  if (auto addIOp = dyn_cast<arith::AddIOp>(arithOp)) {
    return parseAddI(addIOp, loc, rewriter, offsetMap);
  } else if (auto subIOp = dyn_cast<arith::SubIOp>(arithOp)) {
    return parseSubI(subIOp, loc, rewriter, offsetMap);
  } else if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(arithOp)) {
    return parseIndexCast(indexCastOp, loc, rewriter, offsetMap);
  } else if (auto constantFloatOp = dyn_cast<arith::ConstantFloatOp>(arithOp)) {
    return parseConstantOp(constantFloatOp, loc, rewriter, offsetMap);
  } else if (auto constantIntOp = dyn_cast<arith::ConstantIntOp>(arithOp)) {
    return parseConstantOp(constantIntOp, loc, rewriter, offsetMap);
  } else if (auto constantOp = dyn_cast<arith::ConstantOp>(arithOp)) {
    return parseConstantOp(constantOp, loc, rewriter, offsetMap);
  } else if (auto extSIOp = dyn_cast<arith::ExtSIOp>(arithOp)) {
    return parseExtSI(extSIOp, loc, rewriter, offsetMap);
  } else if (auto mulIOp = dyn_cast<arith::MulIOp>(arithOp)) {
    return parseMulI(mulIOp, loc, rewriter, offsetMap);
  } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(arithOp)) {
    return parseBinaryOp(remSIOp, loc, rewriter, offsetMap);
  } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(arithOp)) {
    return parseBinaryOp(divSIOp, loc, rewriter, offsetMap);
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(arithOp)) {
    return parseSelect(selectOp, loc, rewriter, offsetMap);
  } else if (auto fPToSIOp = dyn_cast<arith::FPToSIOp>(arithOp)) {
    return parseFPToSI(fPToSIOp, loc, rewriter, offsetMap);
  } else if (auto sIToFPOp = dyn_cast<arith::SIToFPOp>(arithOp)) {
    return parseSIToFP(sIToFPOp, loc, rewriter, offsetMap);
  } else if (auto mulFOp = dyn_cast<arith::MulFOp>(arithOp)) {
    return parseBinaryOp(mulFOp, loc, rewriter, offsetMap);
  } else if (auto divFOp = dyn_cast<arith::DivFOp>(arithOp)) {
    return parseBinaryOp(divFOp, loc, rewriter, offsetMap);
  } else if (auto addFOp = dyn_cast<arith::AddFOp>(arithOp)) {
    return parseBinaryOp(addFOp, loc, rewriter, offsetMap);
  } else if (auto subFOp = dyn_cast<arith::SubFOp>(arithOp)) {
    return parseBinaryOp(subFOp, loc, rewriter, offsetMap);
  } else if (auto minNumFOp = dyn_cast<arith::MinNumFOp>(arithOp)) {
    return parseBinaryOp(minNumFOp, loc, rewriter, offsetMap);
  } else if (auto maxNumFOp = dyn_cast<arith::MaxNumFOp>(arithOp)) {
    return parseBinaryOp(maxNumFOp, loc, rewriter, offsetMap);
  } else if (auto maxSIOp = dyn_cast<arith::MaxSIOp>(arithOp)) {
    return parseBinaryOp(maxSIOp, loc, rewriter, offsetMap);
  } else if (auto minSIOp = dyn_cast<arith::MinSIOp>(arithOp)) {
    return parseBinaryOp(minSIOp, loc, rewriter, offsetMap);
  } else if (auto cmpIOp = dyn_cast<arith::CmpIOp>(arithOp)) {
    return parseBinaryOp(cmpIOp, loc, rewriter, offsetMap);
  } else if (auto andIOp = dyn_cast<arith::AndIOp>(arithOp)) {
    return parseBinaryOp(andIOp, loc, rewriter, offsetMap);
  } else if (auto orIOp = dyn_cast<arith::OrIOp>(arithOp)) {
    return parseBinaryOp(orIOp, loc, rewriter, offsetMap);
  }
  return success();
}

LogicalResult parseTritonOp(Operation *tritonOp, const Location &loc,
                            RewriterBase &rewriter,
                            llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  assert(isa<triton::TritonDialect>(tritonOp->getDialect()));
  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(tritonOp)) {
    return parseAddPtr(addPtrOp, loc, rewriter, offsetMap);
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(tritonOp)) {
    return parseSplat(splatOp, loc, rewriter, offsetMap);
  } else if (auto getProgramIdOp = dyn_cast<triton::GetProgramIdOp>(tritonOp)) {
    return parseConstantOp(getProgramIdOp, loc, rewriter, offsetMap);
  } else if (auto getNumProgramsOp =
                 dyn_cast<triton::GetNumProgramsOp>(tritonOp)) {
    return parseConstantOp(getNumProgramsOp, loc, rewriter, offsetMap);
  } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(tritonOp)) {
    return parseMakeRange(makeRangeOp, loc, rewriter, offsetMap);
  } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(tritonOp)) {
    return parseBitcast(bitcastOp, loc, rewriter, offsetMap);
  } else if (auto loadOp = dyn_cast<triton::LoadOp>(tritonOp)) {
    return parseLoad(loadOp, loc, rewriter, offsetMap);
  } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(tritonOp)) {
    return parseBroadcast(broadcastOp, loc, rewriter, offsetMap);
  } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(tritonOp)) {
    return parseExpandDims(expandDimsOp, loc, rewriter, offsetMap);
  } else if (auto clampFOp = dyn_cast<triton::ClampFOp>(tritonOp)) {
    return parseClampF(clampFOp, loc, rewriter, offsetMap);
  }
  // FIXME:Z|wait triton version upgrade to 3.4
  // else if (auto makeTensorDescOp =
  //                dyn_cast<triton::MakeTensorDescOp>(tritonOp)) {
  //   parseMakeTensorDesc(makeTensorDescOp, loc, rewriter, offsetMap);
  // }
  else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(tritonOp)) {
    return parseMakeTensorPtr(makeTensorPtrOp, loc, rewriter, offsetMap);
  } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(tritonOp)) {
    return parseReduce(reduceOp, loc, rewriter, offsetMap);
  } else if (auto reduceReturnOp = dyn_cast<triton::ReduceReturnOp>(tritonOp)) {
    return parseReduceReturn(reduceReturnOp, loc, rewriter, offsetMap);
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(tritonOp)) {
    return parseAdvance(advanceOp, loc, rewriter, offsetMap);
  } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(tritonOp)) {
    return parseIntToPtr(intToPtrOp, loc, rewriter, offsetMap);
  }
  return success();
}

static triton::PointerType getScalarPointerType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    type = shapedType.getElementType();
  return dyn_cast<triton::PointerType>(type);
}

static unsigned getPointeeByteWidth(Type pointerType) {
  auto scalarPointerType = getScalarPointerType(pointerType);
  if (!scalarPointerType)
    return 0;
  Type pointeeType = scalarPointerType.getPointeeType();
  if (auto shapedType = dyn_cast<ShapedType>(pointeeType))
    pointeeType = shapedType.getElementType();
  if (!pointeeType.isIntOrFloat())
    return 0;
  unsigned bitWidth = pointeeType.getIntOrFloatBitWidth();
  if (bitWidth == 1)
    bitWidth = 8;
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return 0;
  return bitWidth / 8;
}

static Value scaleOffsetToBytes(Value offset, unsigned byteWidth, Location loc,
                                RewriterBase &rewriter) {
  if (byteWidth == 1)
    return offset;
  Type elementType = offset.getType();
  if (auto shapedType = dyn_cast<ShapedType>(elementType))
    elementType = shapedType.getElementType();
  auto integerType = cast<IntegerType>(elementType);
  TypedAttr factor = rewriter.getIntegerAttr(integerType, byteWidth);
  if (auto shapedType = dyn_cast<ShapedType>(offset.getType()))
    factor = DenseElementsAttr::get(shapedType, factor);
  Value factorValue = rewriter.create<arith::ConstantOp>(loc, factor);
  return rewriter.create<arith::MulIOp>(loc, offset, factorValue);
}

LogicalResult parseAddPtr(triton::AddPtrOp op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addPtr base_ptr
  Value ptr = op.getPtr();
  if (failed(parseChecked(ptr, op.getLoc(), rewriter, offsetMap)))
    return failure();
  // Get addPtr offset
  Value offsetValue = op.getOffset();
  if (failed(parseChecked(offsetValue, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo ptrOffsetInfo = offsetMap.at(ptr);
  PtrOffsetInfo offsetOffsetInfo = offsetMap.at(offsetValue);

  // A tensor pointer argument has no scalar root from which an exact address
  // can be materialized. Keep the unit marker and structural information so the
  // memory converter can reject it with a stable diagnostic instead of trying
  // to perform arithmetic on an absent offset.
  if (!ptrOffsetInfo.getPtr() || !ptrOffsetInfo.getOffset()) {
    auto dstOffsetInfo = combineInfo(ptrOffsetInfo, offsetOffsetInfo);
    dstOffsetInfo.setPtr(ptrOffsetInfo.getPtr());
    dstOffsetInfo.setByteAddressed(ptrOffsetInfo.isByteAddressed());
    offsetMap[op.getResult()] = dstOffsetInfo;
    return success();
  }

  // Modify IR
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto offsetType = dyn_cast<RankedTensorType>(offsetValue.getType())) {
    auto offsetElementType = cast<IntegerType>(offsetType.getElementType());
    if (offsetElementType.getWidth() != 64) {
      auto newOffsetType = RankedTensorType::get(offsetType.getShape(),
                                                 rewriter.getIntegerType(64));
      offsetValue = rewriter.create<arith::ExtSIOp>(op.getLoc(), newOffsetType,
                                                    offsetValue);
    }
  } else {
    auto offsetIntType = cast<IntegerType>(offsetValue.getType());
    if (offsetIntType.getWidth() != 64) {
      offsetValue = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getIntegerType(64), offsetValue);
    }
  }
  if (ptrOffsetInfo.isByteAddressed()) {
    unsigned byteWidth = getPointeeByteWidth(ptr.getType());
    if (!byteWidth) {
      op.emitError("byte-addressed AddPtr requires a byte-addressable scalar "
                   "integer or floating-point pointee type");
      return failure();
    }
    offsetValue =
        scaleOffsetToBytes(offsetValue, byteWidth, op.getLoc(), rewriter);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr] Adding offset\n";
    os << ptrOffsetInfo.getOffset() << '\n' << offsetValue << '\n';
  });
  Value offset = rewriter.create<arith::AddIOp>(
      op.getLoc(), ptrOffsetInfo.getOffset(), offsetValue);
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr] offset is\n" << offset << '\n';
  });
  // Set addPtr offset map
  auto dst = op.getResult();
  auto dstOffsetInfo = combineInfo(ptrOffsetInfo, offsetOffsetInfo);
  dstOffsetInfo.setPtr(ptrOffsetInfo.getPtr());
  dstOffsetInfo.setOffset(offset);
  dstOffsetInfo.setByteAddressed(ptrOffsetInfo.isByteAddressed());
  offsetMap[dst] = dstOffsetInfo;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    auto &ptrStructured = ptrOffsetInfo.getStructuredRef();
    auto &offsetStructured = offsetOffsetInfo.getStructuredRef();
    os << "[parseAddPtr] ptrStructured: ";
    for (size_t i = 0; i < ptrStructured.size(); i++)
      os << static_cast<int>(ptrStructured[i]);
    os << "\n";
    os << "[parseAddPtr] offsetStructured: ";
    for (size_t i = 0; i < offsetStructured.size(); i++)
      os << static_cast<int>(offsetStructured[i]);
    os << "\n";
  });
  return success();
}

LogicalResult parseSplat(triton::SplatOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get splat src
  auto src = op.getSrc();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto dst = op.getResult();
  auto dstType = cast<RankedTensorType>(dst.getType());
  PtrOffsetInfo dstOffsetInfo(srcOffsetInfo.getPtr());
  dstOffsetInfo.setByteAddressed(srcOffsetInfo.isByteAddressed());
  // Modify IR
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseSplat] dst is\n" << dst << '\n';
  });
  if (isa<triton::PointerType>(dstType.getElementType())) {
    RewriterBase::InsertionGuard guard(rewriter);
    auto dstShape = dstType.getShape();
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(dstShape, rewriter.getIntegerType(64)),
        valueOffset);
    dstOffsetInfo.setOffset(offset);
  }
  // Set addPtr offset map
  auto &dstStructured = dstOffsetInfo.getStructuredRef();
  for (auto dim : dstType.getShape())
    dstStructured.push_back(dim == 1 ? PtrOffsetInfo::AxisInfo::scalar
                                     : PtrOffsetInfo::AxisInfo::scalarlike);
  dstOffsetInfo.setScalarLike(true);
  offsetMap[dst] = dstOffsetInfo;
  return success();
}

template <typename BinOpTy>
LogicalResult parseBinaryOp(BinOpTy op, const Location &loc,
                            RewriterBase &rewriter,
                            llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto lhs = op.getLhs();
  if (failed(parseChecked(lhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  auto &lhsStructured = lhsOffsetInfo.getStructuredRef();
  auto rhs = op.getRhs();
  if (failed(parseChecked(rhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  auto &rhsStructured = rhsOffsetInfo.getStructuredRef();
  auto dst = op->getResult(0);
  PtrOffsetInfo dstOffsetInfo;
  dstOffsetInfo.setScalarLike(lhsOffsetInfo.isScalarLike() &&
                              rhsOffsetInfo.isScalarLike());
  if (dstOffsetInfo.isScalarLike())
    dstOffsetInfo.setStructured(lhsStructured.size(),
                                PtrOffsetInfo::AxisInfo::scalarlike);
  else
    dstOffsetInfo.setUnstructured(lhsStructured.size());
  offsetMap[dst] = dstOffsetInfo;
  return success();
}

LogicalResult parseAddI(arith::AddIOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addi lhs
  auto lhs = op.getLhs();
  if (failed(parseChecked(lhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  // Get addi rhs
  auto rhs = op.getRhs();
  if (failed(parseChecked(rhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  // Set addi offset map
  auto dst = op.getResult();
  offsetMap[dst] = combineInfo(lhsOffsetInfo, rhsOffsetInfo);
  return success();
}

LogicalResult parseSubI(arith::SubIOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addi lhs
  auto lhs = op.getLhs();
  if (failed(parseChecked(lhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  // Get addi rhs
  auto rhs = op.getRhs();
  if (failed(parseChecked(rhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  // Set addi offset map
  auto dst = op.getResult();
  offsetMap[dst] = combineInfo(lhsOffsetInfo, rhsOffsetInfo);
  if (!(lhsOffsetInfo.isStructured() && rhsOffsetInfo.isScalarLike())) {
    offsetMap[dst].setUnstructured(offsetMap[dst].getRank());
  }
  return success();
}

LogicalResult parseIndexCast(arith::IndexCastOp op, const Location &loc,
                             RewriterBase &rewriter,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get indexCast input
  auto src = op.getIn();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  // Set indexCast offset map
  auto dst = op.getOut();
  auto srcOffsetInfo = offsetMap.at(src);
  offsetMap[dst] = srcOffsetInfo;
  return success();
}

template <typename ConstOpTy>
LogicalResult parseConstantOp(ConstOpTy dst, const Location &loc,
                              RewriterBase &rewriter,
                              llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set constant offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(true);
  if (auto tensorType =
          dyn_cast<RankedTensorType>(dst->getResult(0).getType())) {
    auto &dstStructured = offsetMap[dst].getStructuredRef();
    for (auto dim : tensorType.getShape())
      dstStructured.push_back(dim == 1 ? PtrOffsetInfo::AxisInfo::scalar
                                       : PtrOffsetInfo::AxisInfo::scalarlike);
  }
  return success();
}

LogicalResult parseMakeRange(triton::MakeRangeOp op, const Location &loc,
                             RewriterBase &rewriter,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set makeRange offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setStructured(1);
  return success();
}

LogicalResult parseExtSI(arith::ExtSIOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get extSI input
  auto src = op.getIn();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  // Set extSI offset map
  auto dst = op.getOut();
  auto srcOffsetInfo = offsetMap.at(src);
  offsetMap[dst] = srcOffsetInfo;
  return success();
}

LogicalResult parseBitcast(triton::BitcastOp op, const Location &loc,
                           RewriterBase &rewriter,
                           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  Value src = op.getSrc();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  Value dst = op.getResult();

  auto srcPtrType = getScalarPointerType(src.getType());
  auto dstPtrType = getScalarPointerType(dst.getType());
  if (!srcPtrType || !dstPtrType) {
    // This analysis also sees ordinary value bitcasts. Keep their established
    // behavior and do not attach pointer-address unit state to numeric values.
    offsetMap[dst] = PtrOffsetInfo(srcOffsetInfo.getStructured());
    offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
    return success();
  }
  if (srcPtrType.getAddressSpace() != dstPtrType.getAddressSpace()) {
    op.emitError("cannot bitcast pointers between different address spaces");
    return failure();
  }
  MLIRContext *context = op.getContext();
  if (srcPtrType.getPointeeType() != IntegerType::get(context, 1) &&
      dstPtrType.getPointeeType() == IntegerType::get(context, 1)) {
    op.emitError(
        "pointer bitcast to i1 from a different pointee type is unsupported "
        "because i1 pointers use i8 storage");
    return failure();
  }
  if (srcPtrType == dstPtrType) {
    offsetMap[dst] = srcOffsetInfo;
    return success();
  }

  unsigned srcByteWidth = getPointeeByteWidth(src.getType());
  unsigned dstByteWidth = getPointeeByteWidth(dst.getType());
  if (!srcByteWidth || !dstByteWidth) {
    op.emitError(
        "different-width pointer bitcast requires byte-addressable scalar "
        "integer or floating-point pointee types");
    return failure();
  }

  offsetMap[dst] = srcOffsetInfo;
  if (srcByteWidth == dstByteWidth) {
    // The offset unit is unchanged. Keep the live source root in analysis and
    // materialize the target pointee type only when rewriting a real memory
    // operation. An analysis-map entry is not an SSA use, so creating a cast
    // here can leave a dangling Value after greedy canonicalization.
    return success();
  }

  // Record the different-width boundary even when analysis cannot recover a
  // scalar root (for example, a tensor pointer function argument). The memory
  // converter owns the structural diagnostic for that unsupported form.
  if (!srcOffsetInfo.getPtr() || !srcOffsetInfo.getOffset()) {
    offsetMap[dst].setByteAddressed();
    return success();
  }

  // A different-width bitcast is an address boundary, not an offset
  // conversion. Convert the accumulated source-element offset to bytes once,
  // without division or alignment assumptions. Later AddPtr operations add
  // their own offset multiplied by the pointee width visible at that operation.
  // This preserves the exact address even across multiple bitcasts.
  if (!srcOffsetInfo.isByteAddressed()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value byteOffset = scaleOffsetToBytes(srcOffsetInfo.getOffset(),
                                          srcByteWidth, op.getLoc(), rewriter);
    offsetMap[dst].setOffset(byteOffset);
  }
  offsetMap[dst].setByteAddressed();
  return success();
}
LogicalResult parseLoad(triton::LoadOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get load ptr
  auto ptr = op.getPtr();
  if (failed(parseChecked(ptr, op.getLoc(), rewriter, offsetMap)))
    return failure();
  // Set load offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(offsetMap[ptr].isScalarLike());
  auto tensorType = dyn_cast<RankedTensorType>(dst.getType());
  if (!tensorType)
    return success();
  offsetMap[dst].setUnstructured(tensorType.getRank());
  return success();
}

LogicalResult parseMulI(arith::MulIOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get muli lhs
  auto lhs = op.getLhs();
  if (failed(parseChecked(lhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  auto &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get muli rhs
  auto rhs = op.getRhs();
  if (failed(parseChecked(rhs, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  auto &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set muli offset map
  size_t maxSize = std::max(lhsStructured.size(), rhsStructured.size());
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(maxSize);
  for (size_t i = 0; i < maxSize; i++)
    if (lhsScalarLike)
      dstStructured[i] = rhsStructured[i];
    else if (rhsScalarLike)
      dstStructured[i] = lhsStructured[i];
    else
      dstStructured[i] = PtrOffsetInfo::AxisInfo::unstructured;
  return success();
}

LogicalResult parseBroadcast(triton::BroadcastOp op, const Location &loc,
                             RewriterBase &rewriter,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get broadcast src
  auto src = op.getSrcMutable().get();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Get broadcast dim
  auto dst = op.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "tt.broadcast's input should be a tensor");
  auto srcType = cast<RankedTensorType>(src.getType());
  auto dstType = cast<RankedTensorType>(dst.getType());
  assert(srcType.getRank() == dstType.getRank() &&
         "rank of source shoule be equal to destnation");
  auto broadcastDim = ConverterUtils::getBroadcastDims(srcType, dstType);
  // Set broadcast offset map
  offsetMap[dst] = PtrOffsetInfo(srcOffsetInfo.getPtr());
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setByteAddressed(srcOffsetInfo.isByteAddressed());

  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::BroadcastOp>(
        loc,
        RankedTensorType::get(dstType.getShape(), rewriter.getIntegerType(64)),
        valueOffset);

    offsetMap[dst].setOffset(offset);
  }

  auto &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(srcStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (llvm::find(broadcastDim, i) != broadcastDim.end() && dstShape[i] != 1) {
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalarlike;
    } else {
      dstStructured[i] = srcStructured[i];
    }
  return success();
}

LogicalResult parseExpandDims(triton::ExpandDimsOp op, const Location &loc,
                              RewriterBase &rewriter,
                              llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get expandDims src
  auto src = op.getSrc();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set expandDims offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(srcOffsetInfo.getPtr());
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setByteAddressed(srcOffsetInfo.isByteAddressed());
  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset =
        rewriter.create<triton::ExpandDimsOp>(loc, valueOffset, op.getAxis());

    offsetMap[dst].setOffset(offset);
  }
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size() + 1);
  size_t j = 0;
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (i == op.getAxis()) {
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalar;
    } else {
      dstStructured[i] = srcStructured[j];
      j++;
    }
  return success();
}

LogicalResult parseClampF(triton::ClampFOp op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get clampF src
  auto src = op.getX();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Get clampF min
  auto clampMin = op.getMin();
  if (failed(parseChecked(clampMin, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo minOffsetInfo = offsetMap.at(clampMin);
  // Get clampF max
  auto clampMax = op.getMax();
  if (failed(parseChecked(clampMax, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo maxOffsetInfo = offsetMap.at(clampMax);
  // Set clampF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike() &&
                               minOffsetInfo.isScalarLike() &&
                               maxOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return success();
  offsetMap[dst].setUnstructured(dstType.getRank());
  return success();
}

LogicalResult parseSelect(arith::SelectOp op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get select condition
  auto condition = op.getCondition();
  if (failed(parseChecked(condition, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo conditionOffsetInfo = offsetMap.at(condition);
  bool conditionScalarLike = conditionOffsetInfo.isScalarLike();
  // Get select trueValue
  auto trueValue = op.getTrueValue();
  if (failed(parseChecked(trueValue, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo trueValueOffsetInfo = offsetMap.at(trueValue);
  auto &trueValueStructured = trueValueOffsetInfo.getStructuredRef();
  bool trueValueScalarLike = trueValueOffsetInfo.isScalarLike();
  // Get select falseValue
  auto falseValue = op.getFalseValue();
  if (failed(parseChecked(falseValue, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo falseValueOffsetInfo = offsetMap.at(falseValue);
  auto &falseValueStructured = falseValueOffsetInfo.getStructuredRef();
  bool falseValueScalarLike = falseValueOffsetInfo.isScalarLike();
  // Set select offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return success();

  auto dstIsScalar =
      trueValueScalarLike && falseValueScalarLike && conditionScalarLike;
  offsetMap[dst].setScalarLike(dstIsScalar);

  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(trueValueStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = (dstIsScalar) ? PtrOffsetInfo::AxisInfo::scalarlike
                                     : PtrOffsetInfo::AxisInfo::unstructured;
  return success();
}

LogicalResult parseFPToSI(arith::FPToSIOp op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get FPToSI src
  auto src = op.getIn();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set FPToSI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return success();
  if (offsetMap[dst].isScalarLike())
    offsetMap[dst].setStructured(dstType.getRank(),
                                 PtrOffsetInfo::AxisInfo::scalarlike);
  else
    offsetMap[dst].setUnstructured(dstType.getRank());
  return success();
}

LogicalResult parseSIToFP(arith::SIToFPOp op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get SIToFP src
  auto src = op.getIn();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set SIToFP offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return success();
  if (offsetMap[dst].isScalarLike())
    offsetMap[dst].setStructured(dstType.getRank(),
                                 PtrOffsetInfo::AxisInfo::scalarlike);
  else
    offsetMap[dst].setUnstructured(dstType.getRank());
  return success();
}

// FIXME:Z|wait triton version upgrade to 3.4
// void parseMakeTensorDesc(triton::MakeTensorDescOp op, const Location &loc,
//                          RewriterBase &rewriter,
//                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
//   // Set MakeTensorDesc offset map
//   auto dst = op.getResult();
//   offsetMap[dst] = PtrOffsetInfo();
//   auto dstType = dyn_cast<ShapedType>(dst.getType());
//   if (!dstType)
//     return;
//   offsetMap[dst].setStructured(dstType.getRank());
// }

LogicalResult
parseMakeTensorPtr(triton::MakeTensorPtrOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set MakeTensorPtr offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(dst);
  auto dstType = dyn_cast<ShapedType>(
      cast<triton::PointerType>(dst.getType()).getPointeeType());
  if (!dstType)
    return success();
  offsetMap[dst].setStructured(dstType.getRank());
  offsetMap[dst].setOffsets(op.getOffsets());
  return success();
}

LogicalResult parseAdvance(triton::AdvanceOp op, const Location &loc,
                           RewriterBase &rewriter,
                           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set Advance offset map
  auto ptr = op.getPtr();
  if (failed(parseChecked(ptr, op.getLoc(), rewriter, offsetMap)))
    return failure();
  auto dst = op.getResult();
  auto ptrOffsetInfo = offsetMap.at(ptr);
  offsetMap[dst] = ptrOffsetInfo;
  auto dstType = dyn_cast<ShapedType>(
      cast<triton::PointerType>(dst.getType()).getPointeeType());
  if (!dstType)
    return success();
  offsetMap[dst].setStructured(dstType.getRank());
  auto &offsets = offsetMap[dst].getOffsetsRef();

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  for (auto [curOffset, opOffset] : llvm::zip(offsets, op.getOffsets())) {
    curOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), curOffset, opOffset);
  }
  return success();
}

LogicalResult parseReduce(triton::ReduceOp op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get reduce src
  Value src = op->getOperand(0);
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  if (!dstType)
    return success();
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalar;
    else
      dstStructured[i] = srcStructured[i];
  return success();
}

LogicalResult
parseReduceReturn(triton::ReduceReturnOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get reduce src
  Value src = op->getOperand(0);
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  if (!dstType)
    return success();
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalar;
    else
      dstStructured[i] = srcStructured[i];
  return success();
}

LogicalResult parseIf(scf::IfOp op, const Location &loc, RewriterBase &rewriter,
                      llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                      Value dst) {
  const unsigned int index = cast<OpResult>(dst).getResultNumber();
  Block &thenBlock = op.getThenRegion().front();
  Value thenYieldedValue = thenBlock.getTerminator()->getOperand(index);
  if (failed(parseChecked(thenYieldedValue, op.getLoc(), rewriter, offsetMap)))
    return failure();
  PtrOffsetInfo thenOffsetInfo = offsetMap.at(thenYieldedValue);
  auto &thenStructured = thenOffsetInfo.getStructuredRef();
  Value thenSrcPtr = thenOffsetInfo.getPtr();
  bool dstIsScalar = thenOffsetInfo.isScalarLike();

  if (op.elseBlock()) {
    Block &elseBlock = op.getElseRegion().front();
    Value elseYieldedValue = elseBlock.getTerminator()->getOperand(index);
    if (failed(
            parseChecked(elseYieldedValue, op.getLoc(), rewriter, offsetMap)))
      return failure();
    PtrOffsetInfo elseOffsetInfo = offsetMap.at(elseYieldedValue);
    if (thenOffsetInfo.isByteAddressed() || elseOffsetInfo.isByteAddressed()) {
      if (thenOffsetInfo.isByteAddressed() !=
          elseOffsetInfo.isByteAddressed()) {
        op.emitError(
            "cannot merge pointer offsets with different address units");
        return failure();
      }
      if (thenSrcPtr != elseOffsetInfo.getPtr()) {
        op.emitError("cannot merge pointers from different scalar roots");
        return failure();
      }
    } else if (thenSrcPtr != elseOffsetInfo.getPtr()) {
      // Preserve the legacy element-addressed behavior. Different roots were
      // already unsupported there, but byte-address validation must not turn
      // unrelated control flow into a new hard analysis failure.
      emitError(loc)
          << "Currently ptr type from different source not supported";
    }
    dstIsScalar = dstIsScalar && elseOffsetInfo.isScalarLike();
  }

  // Populate the result only after every branch has been validated. A failure
  // must not leave a partially usable pointer state in the analysis map.
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setPtr(thenSrcPtr);
  offsetMap[dst].setByteAddressed(thenOffsetInfo.isByteAddressed());
  offsetMap[dst].setScalarLike(dstIsScalar);
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(thenStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++) {
    if (op.elseBlock())
      dstStructured[i] = dstIsScalar ? PtrOffsetInfo::AxisInfo::scalarlike
                                     : PtrOffsetInfo::AxisInfo::unstructured;
    else
      dstStructured[i] = thenStructured[i];
  }
  SmallVector<Value> dstOffsets(thenOffsetInfo.getOffsetsRef().size());
  if (!dstOffsets.empty() && op->getNumResults() >= index + dstOffsets.size()) {
    // replacePtrArguments expands block-pointer offsets into extra results.
    // Keep the expanded results as the materializable offset state.
    for (size_t i = 0; i < dstOffsets.size(); i++)
      dstOffsets[i] = op->getResult(index + i);
    offsetMap[dst].setOffsets(dstOffsets);
  }
  return success();
}
LogicalResult parseYield(scf::YieldOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get yield src
  for (auto src : op->getOperands())
    if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
      return failure();
  return success();
}

LogicalResult parseLoopOp(LoopLikeOpInterface op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                          Value dst) {
  auto resNum = cast<OpResult>(dst).getResultNumber();
  Value yieldedValue;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation()))
    yieldedValue = whileOp.getConditionOp().getArgs()[resNum];
  else
    yieldedValue = op.getYieldedValues()[resNum];
  if (failed(parseChecked(yieldedValue, op.getLoc(), rewriter, offsetMap)))
    return failure();
  offsetMap[dst] = offsetMap.at(yieldedValue);
  return success();
}

LogicalResult
parseExtractSlice(tensor::ExtractSliceOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get extractSlice src
  auto src = op.getSource();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  // Set extractSlice offset map
  auto dst = op.getResult();
  auto srcPtrInfo = offsetMap.at(src);
  auto srcPtr = srcPtrInfo.getPtr();
  auto srcOffset = srcPtrInfo.getOffset();
  auto srcStructured = srcPtrInfo.getStructured();
  auto droppedDims = op.getDroppedDims();
  if (srcOffset) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto offsetType = getExtractSlicedType(op.getMixedSizes(), droppedDims,
                                           getElementTypeOrSelf(srcOffset));
    srcOffset = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), offsetType, srcOffset, op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
  }
  SmallVector<PtrOffsetInfo::AxisInfo> dstStructured;
  for (size_t i = 0; i < srcStructured.size(); i++) {
    if (!droppedDims[i])
      dstStructured.push_back(srcStructured[i]);
  }
  offsetMap[dst] = PtrOffsetInfo(srcPtr, srcOffset, dstStructured);
  offsetMap[dst].setByteAddressed(srcPtrInfo.isByteAddressed());
  return success();
}

static bool isPointerTensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

static LogicalResult validatePointerInsertState(Operation *op,
                                                llvm::StringRef operationName,
                                                const PtrOffsetInfo &srcInfo,
                                                const PtrOffsetInfo &dstInfo) {
  if (!srcInfo.getPtr() || !srcInfo.getOffset() || !dstInfo.getPtr() ||
      !dstInfo.getOffset()) {
    op->emitError() << operationName
                    << " requires complete source and destination pointer "
                       "states";
    return failure();
  }
  if (srcInfo.isByteAddressed() != dstInfo.isByteAddressed()) {
    op->emitError() << operationName
                    << " cannot merge pointer offsets with different address "
                       "units";
    return failure();
  }
  if (srcInfo.getPtr() != dstInfo.getPtr()) {
    op->emitError() << operationName
                    << " cannot merge pointers from different scalar roots";
    return failure();
  }
  return success();
}

LogicalResult
parseInsertSlice(tensor::InsertSliceOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto src = op.getSource();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  auto dst = op.getDest();
  if (failed(parseChecked(dst, op.getLoc(), rewriter, offsetMap)))
    return failure();

  auto res = op.getResult();
  auto srcPtrInfo = offsetMap.at(src);
  auto dstPtrInfo = offsetMap.at(dst);
  const bool pointerResult = isPointerTensor(res.getType());
  const bool byteAddressedPointerResult =
      pointerResult &&
      (srcPtrInfo.isByteAddressed() || dstPtrInfo.isByteAddressed());
  if (byteAddressedPointerResult &&
      failed(validatePointerInsertState(
          op.getOperation(), "tensor.insert_slice", srcPtrInfo, dstPtrInfo)))
    return failure();

  PtrOffsetInfo resPtrInfo;
  if (byteAddressedPointerResult) {
    // Validation must precede construction: tensor ops cannot be built with an
    // absent destination offset, and a rejected merge must leave no new SSA.
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value resOffset = rewriter.create<tensor::InsertSliceOp>(
        op.getLoc(), srcPtrInfo.getOffset(), dstPtrInfo.getOffset(),
        op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides());
    resPtrInfo.setPtr(srcPtrInfo.getPtr());
    resPtrInfo.setOffset(resOffset);
    resPtrInfo.setByteAddressed(srcPtrInfo.isByteAddressed());
  } else if (Value srcOffset = srcPtrInfo.getOffset()) {
    // Preserve the main-dev behavior for every non-byte-addressed value.
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value resOffset = rewriter.create<tensor::InsertSliceOp>(
        op.getLoc(), srcOffset, dstPtrInfo.getOffset(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    Value srcPtr = srcPtrInfo.getPtr();
    Value dstPtr = dstPtrInfo.getPtr();
    assert(srcPtr == dstPtr && "ptrInfo for insert slice should be consistent");
    resPtrInfo.setPtr(srcPtr);
    resPtrInfo.setOffset(resOffset);
  }

  auto droppedDims = op.getDroppedDims();
  auto srcStructuredIter = srcPtrInfo.getStructured().begin();
  SmallVector<PtrOffsetInfo::AxisInfo> resStructured;
  auto srcShape = op.getStaticSizes();
  auto dstShape = cast<RankedTensorType>(dst.getType()).getShape();
  for (size_t i = 0; i < dstShape.size(); i++) {
    if (!ShapedType::isDynamic(srcShape[i]) && srcShape[i] == dstShape[i])
      resStructured.push_back(*srcStructuredIter);
    else
      resStructured.push_back(PtrOffsetInfo::AxisInfo::unstructured);
    if (!droppedDims[i])
      ++srcStructuredIter;
  }
  resPtrInfo.setStructured(resStructured);
  offsetMap[res] = resPtrInfo;
  return success();
}
LogicalResult parseExtract(tensor::ExtractOp op, const Location &loc,
                           RewriterBase &rewriter,
                           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto parentValue = op.getTensor();
  if (failed(parseChecked(parentValue, op.getLoc(), rewriter, offsetMap)))
    return failure();
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  if (isa<triton::PointerType>(dst.getType())) {
    offsetMap[dst].setPtr(dst);
    offsetMap[dst].setZeroOffset();
  }
  offsetMap[dst].setScalarLike(true);
  return success();
}

LogicalResult parseInsert(tensor::InsertOp op, const Location &loc,
                          RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto src = op.getScalar();
  if (failed(parseChecked(src, op.getLoc(), rewriter, offsetMap)))
    return failure();
  auto dst = op.getDest();
  if (failed(parseChecked(dst, op.getLoc(), rewriter, offsetMap)))
    return failure();

  auto res = op.getResult();
  auto srcPtrInfo = offsetMap.at(src);
  auto dstPtrInfo = offsetMap.at(dst);
  const bool pointerResult = isPointerTensor(res.getType());
  const bool byteAddressedPointerResult =
      pointerResult &&
      (srcPtrInfo.isByteAddressed() || dstPtrInfo.isByteAddressed());
  if (byteAddressedPointerResult &&
      failed(validatePointerInsertState(op.getOperation(), "tensor.insert",
                                        srcPtrInfo, dstPtrInfo)))
    return failure();

  PtrOffsetInfo resPtrInfo;
  if (byteAddressedPointerResult) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value resOffset = rewriter.create<tensor::InsertOp>(
        op.getLoc(), srcPtrInfo.getOffset(), dstPtrInfo.getOffset(),
        op.getIndices());
    resPtrInfo.setPtr(srcPtrInfo.getPtr());
    resPtrInfo.setOffset(resOffset);
    resPtrInfo.setByteAddressed(srcPtrInfo.isByteAddressed());
  } else if (Value srcOffset = srcPtrInfo.getOffset()) {
    // Preserve the main-dev behavior for every non-byte-addressed value.
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value resOffset = rewriter.create<tensor::InsertOp>(
        op.getLoc(), srcOffset, dstPtrInfo.getOffset(), op.getIndices());
    resPtrInfo.setPtr(srcPtrInfo.getPtr());
    resPtrInfo.setOffset(resOffset);
  }
  resPtrInfo.setUnstructured(dstPtrInfo.getRank());
  offsetMap[res] = resPtrInfo;
  return success();
}
LogicalResult parseIntToPtr(triton::IntToPtrOp op, const Location &loc,
                            RewriterBase &rewriter,
                            llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(dst);
  offsetMap[dst].setScalarLike(true);
  return success();
}

namespace {
template <typename CustomOpT>
LogicalResult parseStructuredCustomOpImpl(
    CustomOpT op, const Location &loc, RewriterBase &rewriter,
    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, unsigned resultIdx) {
  for (Value operand : op.getInputs()) {
    if (failed(parseChecked(operand, op->getLoc(), rewriter, offsetMap)))
      return failure();
  }
  Value dst = op->getResult(resultIdx);
  offsetMap[dst] = PtrOffsetInfo();
  auto tensorType = dyn_cast<RankedTensorType>(dst.getType());
  if (!tensorType) {
    if (isa<triton::PointerType>(dst.getType())) {
      offsetMap[dst].setPtr(dst);
      offsetMap[dst].setZeroOffset();
    } else if (isa<IntegerType>(dst.getType())) {
      offsetMap[dst].setOffset(dst);
    } else {
      emitError(loc) << "unsupported return type for hivm custom op: "
                     << dst.getType();
      offsetMap.erase(dst);
      return failure();
    }
    return success();
  }
  if (isa<triton::PointerType>(tensorType.getElementType())) {
    if (checkStructureAnnotated(op, rewriter)) {
      auto srcValArrayAttr = op->template getAttrOfType<DenseI32ArrayAttr>(
          ConverterUtils::customSrcPtrIndexAttrName);
      assert(srcValArrayAttr &&
             "structure hivm custom op should present src tensor<tt.ptr>");
      auto srcValArray = srcValArrayAttr.asArrayRef();
      assert(srcValArray[resultIdx] != -1 &&
             "tensor<tt.ptr> result should map to src tensor<tt.ptr>");
      offsetMap[dst] = offsetMap[op->getOperand(srcValArray[resultIdx])];
      return success();
    }
    emitError(loc) << "unsupported unstructured RankedTensor of tt.ptr "
                      "return for hivm custom op: "
                   << dst;
    offsetMap.erase(dst);
    return failure();
  }
  offsetMap[dst].setUnstructured(tensorType.getRank());
  return success();
}
} // namespace

LogicalResult parseStructuredCustomOp(
    Operation *op, const Location &loc, RewriterBase &rewriter,
    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, unsigned resultIdx) {
  if (auto customOp = dyn_cast<hivm::CustomOp>(op))
    return parseStructuredCustomOpImpl(customOp, loc, rewriter, offsetMap,
                                       resultIdx);
  if (auto macroOp = dyn_cast<hivm::CustomMacroOp>(op))
    return parseStructuredCustomOpImpl(macroOp, loc, rewriter, offsetMap,
                                       resultIdx);
  llvm_unreachable("expected hivm custom op");
}
} // namespace triton
} // namespace mlir
