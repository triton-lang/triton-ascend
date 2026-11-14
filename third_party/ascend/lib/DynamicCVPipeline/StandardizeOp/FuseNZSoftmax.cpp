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

#include <memory>
#include <type_traits>
#include <utility>

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"

using namespace mlir;
using namespace CVPipeline;

namespace {
bool hasTensorType(Value value, ArrayRef<int64_t> shape, Type element) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == shape && type.getElementType() == element &&
         !type.getEncoding();
}

bool isStaticReshape(tensor::ReshapeOp op, ArrayRef<int64_t> shape, Type elem) {
  DenseIntElementsAttr extents;
  return op && hasTensorType(op.getResult(), shape, elem) &&
         matchPattern(op.getShape(), m_Constant(&extents)) &&
         llvm::equal(extents.getValues<APInt>(), shape,
                     [](const APInt &extent, int64_t dim) {
                       return extent == static_cast<uint64_t>(dim);
                     });
}

template <typename Op> bool hasFastMath(Op op) {
  auto flags = op.getFastmathAttr();
  return flags && flags.getValue() != arith::FastMathFlags::none;
}

template <typename Combiner, typename InitMatcher>
bool isRowReduction(linalg::ReduceOp op, Value input, InitMatcher init) {
  if (!op || op.getNumResults() != 1 || op.getInputs().size() != 1 ||
      op.getInputs().front() != input ||
      op.getDimensions() != ArrayRef<int64_t>{1})
    return false;
  auto fill = op.getInits().front().getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs().front(), init))
    return false;
  Block &body = op.getCombiner().front();
  auto combine = dyn_cast<Combiner>(body.front());
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  return combine && combine->getNextNode() == yield &&
         yield.getValues().front() == combine.getResult() &&
         ((combine.getLhs() == body.getArgument(0) &&
           combine.getRhs() == body.getArgument(1)) ||
          (combine.getLhs() == body.getArgument(1) &&
           combine.getRhs() == body.getArgument(0))) &&
         !hasFastMath(combine);
}

Value getBroadcastInput(Value value, int64_t dim) {
  auto broadcast = value.getDefiningOp<linalg::BroadcastOp>();
  return broadcast && broadcast.getDimensions() == ArrayRef<int64_t>{dim}
             ? broadcast.getInput()
             : Value{};
}

bool isColumnRange(Value value) {
  if (!value ||
      !hasTensorType(value, {512}, IntegerType::get(value.getContext(), 32)))
    return false;
  auto range = value.getDefiningOp<linalg::GenericOp>();
  if (!range || !range.getInputs().empty() ||
      !llvm::equal(
          range.getIndexingMapsArray(),
          ArrayRef{AffineMap::getMultiDimIdentityMap(1, value.getContext())}) ||
      !llvm::equal(range.getIteratorTypesArray(),
                   ArrayRef{utils::IteratorType::parallel}))
    return false;
  Block &body = range.getRegion().front();
  auto index = dyn_cast<linalg::IndexOp>(body.front());
  auto indexCast =
      index ? dyn_cast_or_null<arith::IndexCastOp>(index->getNextNode())
            : nullptr;
  return indexCast && index.getDim() == 0 && indexCast.getIn() == index &&
         indexCast->getNextNode() == body.getTerminator() &&
         cast<linalg::YieldOp>(body.getTerminator()).getValues().front() ==
             indexCast.getResult();
}

struct SoftmaxMask {
  Value score, limits, required;
};

FailureOr<SoftmaxMask> matchSoftmaxMask(Value score) {
  auto select = score.getDefiningOp<arith::SelectOp>();
  if (!select)
    return SoftmaxMask{score, {}, {}};
  auto isNegativeInfinity = [](Value value) {
    if (auto fill = value.getDefiningOp<linalg::FillOp>())
      value = fill.getInputs().front();
    return matchPattern(value, m_NegInfFloat());
  };
  bool selectsHidden = isNegativeInfinity(select.getTrueValue());
  if (!selectsHidden && !isNegativeInfinity(select.getFalseValue()))
    return failure();
  Value pred = select.getCondition();
  Value required;
  if (auto andOp = pred.getDefiningOp<arith::AndIOp>()) {
    if (!selectsHidden)
      return failure();
    pred = andOp.getRhs();
    auto fill = andOp.getLhs().getDefiningOp<linalg::FillOp>();
    if (!fill) {
      pred = andOp.getLhs();
      fill = andOp.getRhs().getDefiningOp<linalg::FillOp>();
    }
    if (!fill || !fill.getInputs().front().getType().isInteger(1))
      return failure();
    required = fill.getInputs().front();
  }
  auto cmp = pred.getDefiningOp<arith::CmpIOp>();
  auto predicate =
      selectsHidden ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sle;
  if (!cmp || cmp.getPredicate() != predicate ||
      !isColumnRange(getBroadcastInput(cmp.getLhs(), 0)))
    return failure();
  Value limits = getBroadcastInput(cmp.getRhs(), 1);
  if (!limits)
    return failure();
  return SoftmaxMask{selectsHidden ? select.getFalseValue()
                                   : select.getTrueValue(),
                     limits, required};
}

Value createPackLUT(IRRewriter &rewriter, Location loc) {
  auto lutTy = RankedTensorType::get({256}, rewriter.getI32Type());
  Value init = rewriter.create<tensor::EmptyOp>(loc, lutTy.getShape(),
                                                lutTy.getElementType());
  auto range = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{lutTy}, ValueRange{}, ValueRange{init},
      ArrayRef{rewriter.getMultiDimIdentityMap(1)},
      ArrayRef{utils::IteratorType::parallel},
      [&](OpBuilder &b, Location loc, ValueRange) {
        Value i = b.create<linalg::IndexOp>(loc, 0);
        Value index = b.create<arith::IndexCastOp>(loc, b.getI32Type(), i);
        b.create<linalg::YieldOp>(loc, index);
      });
  auto splat = [&](int n) -> Value {
    Value scalar = rewriter.create<arith::ConstantIntOp>(loc, n, 32);
    return rewriter
        .create<linalg::FillOp>(loc, ValueRange{scalar}, ValueRange{init})
        .getResult(0);
  };
  // Materialize in UB like the explicit wrapper; the VF cannot load a global
  // tensor constant.
  Value i = range->getResult(0), group = splat(32), half = splat(16);
  Value high = rewriter.create<arith::DivSIOp>(loc, i, group);
  high = rewriter.create<arith::MulIOp>(loc, high, group);
  Value lane = rewriter.create<arith::RemSIOp>(loc, i, half);
  lane = rewriter.create<arith::MulIOp>(loc, lane, splat(2));
  Value low = rewriter.create<arith::RemSIOp>(loc, i, group);
  low = rewriter.create<arith::DivSIOp>(loc, low, half);
  Value index = rewriter.create<arith::AddIOp>(loc, high, lane);
  index = rewriter.create<arith::AddIOp>(loc, index, low);
  return rewriter.create<arith::TruncIOp>(
      loc, RankedTensorType::get({256}, rewriter.getI8Type()), index);
}

hivm::CustomOp createNZSoftmax(IRRewriter &rewriter, Location loc, Value score,
                               Value oldMax, Value limits, Value required,
                               Type probabilityType) {
  auto stateType = cast<RankedTensorType>(oldMax.getType());
  auto sumType =
      RankedTensorType::get(stateType.getShape(), rewriter.getF32Type());
  auto makeEmpty = [&](Type type) -> Value {
    auto tensor = cast<RankedTensorType>(type);
    return rewriter.create<tensor::EmptyOp>(loc, tensor.getShape(),
                                            tensor.getElementType());
  };
  SmallVector<Value> inputs{score, oldMax};
  SmallVector<Value, 3> outputs;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&score.getDefiningOp()
                                           ->getParentOfType<func::FuncOp>()
                                           .getBody()
                                           .front());
    inputs.push_back(createPackLUT(rewriter, loc));
    outputs = {makeEmpty(probabilityType), makeEmpty(stateType),
               makeEmpty(sumType)};
  }
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr n1, m1, m0, n0, lut;
  bindDims(ctx, n1, m1, m0, n0, lut);
  auto map = [&](ArrayRef<AffineExpr> exprs) -> Attribute {
    return AffineMapAttr::get(AffineMap::get(5, 0, exprs, ctx));
  };
  Attribute state = map({m1 * 16 + m0});
  SmallVector<Attribute> maps{map({n1, m1, m0, n0}), state, map({lut})};
  if (limits) {
    inputs.push_back(limits);
    maps.push_back(state);
  }
  if (required) {
    inputs.push_back(required);
    maps.push_back(map({}));
  }
  maps.append({map({n1.floorDiv(2), m1, m0, n1 % 2 * 16 + n0}), state, state});
  auto symbol =
      rewriter.getStringAttr(hivm::CustomOp::kBuiltinOnlineSoftmaxNZName);
  auto custom = rewriter.create<hivm::CustomOp>(
      loc, ValueRange(outputs).getTypes(), symbol.getValue(), inputs, outputs,
      ValueRange{});
  custom->setAttr(hivm::CustomOp::kSymbolAttrName, symbol);
  SmallVector<Attribute> argAttrs(inputs.size(),
                                  rewriter.getDictionaryAttr({}));
  auto view =
      AffineMapAttr::get(AffineMap::get(4, 0, {n1 * 2, m1, m0, n0}, ctx));
  auto binding = rewriter.getArrayAttr({rewriter.getI32IntegerAttr(0), view});
  argAttrs[0] = rewriter.getDictionaryAttr(
      {{hivm::CustomOp::kInplaceOutsAttrName, binding}});
  custom->setAttr(hivm::CustomOp::kArgAttrsName,
                  rewriter.getArrayAttr(argAttrs));
  custom.setCoreType(hivm::TCoreType::VECTOR);
  custom.setPipe(hivm::PIPE::PIPE_V);
  custom.setVFMode(hivm::VFMode::SIMD);
  custom.setInlineMode(hivm::InlineMode::AlwaysInline);
  custom.setGMAddrArgsIndices({});
  custom->setAttr(hivm::CustomOp::kIndexingMapName,
                  rewriter.getArrayAttr(maps));
  auto opaque = hivm::IteratorTypeAttr::get(ctx, hivm::IteratorType::kOpaque);
  auto parallel =
      hivm::IteratorTypeAttr::get(ctx, hivm::IteratorType::kParallel);
  custom->setAttr(
      hivm::CustomOp::kIteratorTypesName,
      rewriter.getArrayAttr({opaque, parallel, opaque, opaque, opaque}));
  return custom;
}

bool isFp8Bound(Value value, const APFloat &bound) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    value = fill.getInputs().front();
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant)))
    return false;
  if (auto scalar = dyn_cast<FloatAttr>(constant))
    return scalar.getValue().bitwiseIsEqual(bound);
  auto dense = dyn_cast<DenseFPElementsAttr>(constant);
  return dense && dense.isSplat() &&
         dense.getSplatValue<APFloat>().bitwiseIsEqual(bound);
}

math::ExpOp matchProbabilityExp(arith::TruncFOp castOp) {
  Value input = castOp.getIn();
  if (auto exp = input.getDefiningOp<math::ExpOp>())
    return llvm::hasNItems(input.getUses(), 2) ? exp : math::ExpOp{};

  // NaNs pass through the ordered clamp. The exponent of a score minus its
  // running row maximum is at most one for all non-NaN inputs to the cast.
  auto lower = input.getDefiningOp<arith::SelectOp>();
  if (!lower || !input.hasOneUse())
    return {};
  auto upper = lower.getFalseValue().getDefiningOp<arith::SelectOp>();
  auto below = lower.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!upper || !below || below.getPredicate() != arith::CmpFPredicate::OLT ||
      below.getLhs() != upper.getResult() ||
      below.getRhs() != lower.getTrueValue() ||
      !below.getResult().hasOneUse() ||
      !llvm::hasNItems(upper.getResult().getUses(), 2))
    return {};

  auto above = upper.getCondition().getDefiningOp<arith::CmpFOp>();
  auto exp = upper.getFalseValue().getDefiningOp<math::ExpOp>();
  if (!above || !exp || above.getPredicate() != arith::CmpFPredicate::OGT ||
      above.getLhs() != exp.getResult() ||
      above.getRhs() != upper.getTrueValue() ||
      !above.getResult().hasOneUse() ||
      !llvm::hasNItems(exp.getResult().getUses(), 3) ||
      !isFp8Bound(upper.getTrueValue(), APFloat(448.0f)) ||
      !isFp8Bound(lower.getTrueValue(), APFloat(-448.0f)))
    return {};
  return exp;
}

LogicalResult fuseNZSoftmax(linalg::TransposeOp pack, Value nz,
                            Value unpackedScore, IRRewriter &rewriter) {
  if (pack->getNumResults() != 1 ||
      pack.getPermutation() != ArrayRef<int64_t>{2, 0, 1, 3})
    return failure();
  auto probability = dyn_cast<RankedTensorType>(pack->getResult(0).getType());
  if (!probability || probability.getRank() != 4)
    return failure();
  auto fp8 = Float8E4M3FNType::get(pack.getContext());
  auto f16 = rewriter.getF16Type();
  auto f32 = rewriter.getF32Type();
  int64_t blocks = probability.getDimSize(1);
  if ((blocks != 2 && blocks != 4 && blocks != 8) ||
      !hasTensorType(pack->getResult(0), {16, blocks, 16, 32}, fp8))
    return failure();
  int64_t rows = blocks * 16;
  Block *block = pack->getBlock();
  auto reshapeP = pack.getInput().getDefiningOp<tensor::ReshapeOp>();
  if (!isStaticReshape(reshapeP, {blocks, 16, 16, 32}, fp8) ||
      !reshapeP.getResult().hasOneUse() ||
      !hasTensorType(reshapeP.getSource(), {rows, 512}, fp8))
    return failure();
  auto fp8Cast = reshapeP.getSource().getDefiningOp<arith::TruncFOp>();
  auto rounding =
      fp8Cast ? fp8Cast->getAttrOfType<hfusion::RoundModeAttr>("round_mode")
              : hfusion::RoundModeAttr{};
  if (!rounding || rounding.getValue() != hfusion::RoundMode::RINT ||
      fp8Cast.getRoundingmodeAttr() || hasFastMath(fp8Cast) ||
      !fp8Cast.getResult().hasOneUse())
    return failure();
  auto exp = matchProbabilityExp(fp8Cast);
  auto sub = exp ? exp.getOperand().getDefiningOp<arith::SubFOp>() : nullptr;
  if (!sub || exp->getBlock() != block || !sub.getResult().hasOneUse() ||
      hasFastMath(exp) || hasFastMath(sub) ||
      !hasTensorType(exp.getResult(), {rows, 512}, f32))
    return failure();
  auto sumUser = llvm::find_if(exp.getResult().getUsers(), [](Operation *user) {
    return isa<linalg::ReduceOp>(user);
  });
  if (sumUser == exp.getResult().getUsers().end())
    return failure();
  auto sum = cast<linalg::ReduceOp>(*sumUser);
  if (!isRowReduction<arith::AddFOp>(sum, exp, m_PosZeroFloat()) ||
      sum->getBlock() != block ||
      !hasTensorType(sum->getResult(0), {rows}, f32))
    return failure();

  Value broadcastInput = getBroadcastInput(sub.getRhs(), 1);
  auto nextExt =
      broadcastInput ? broadcastInput.getDefiningOp<arith::ExtFOp>() : nullptr;
  auto next =
      nextExt ? nextExt.getIn().getDefiningOp<arith::TruncFOp>() : nullptr;
  auto maximum =
      next ? next.getIn().getDefiningOp<arith::MaximumFOp>() : nullptr;
  if (!maximum || maximum->getBlock() != block || next->getBlock() != block ||
      nz.getDefiningOp()->getBlock() != block ||
      !maximum.getResult().hasOneUse() || !hasTensorType(next, {rows}, f16) ||
      hasFastMath(maximum) || hasFastMath(next) || next.getRoundingmodeAttr() ||
      hasFastMath(nextExt))
    return failure();
  if (auto round = next->getAttrOfType<hfusion::RoundModeAttr>("round_mode");
      round && round.getValue() != hfusion::RoundMode::RINT)
    return failure();
  Value score = sub.getLhs();
  Value old = maximum.getLhs();
  auto rowMax = maximum.getRhs().getDefiningOp<linalg::ReduceOp>();
  if (!isRowReduction<arith::MaximumFOp>(rowMax, score, m_NegInfFloat())) {
    rowMax = old.getDefiningOp<linalg::ReduceOp>();
    if (!isRowReduction<arith::MaximumFOp>(rowMax, score, m_NegInfFloat()))
      return failure();
    old = maximum.getRhs();
  }
  auto oldExt = old.getDefiningOp<arith::ExtFOp>();
  if (rowMax->getBlock() != block || !rowMax->getResult(0).hasOneUse() ||
      !oldExt || hasFastMath(oldExt) ||
      !hasTensorType(oldExt.getIn(), {rows}, f16) ||
      !llvm::hasNItems(score.getUses(), 2))
    return failure();
  auto mask = matchSoftmaxMask(score);
  if (failed(mask) || mask->score != unpackedScore ||
      (mask->limits &&
       !hasTensorType(mask->limits, {rows}, rewriter.getI32Type())))
    return failure();
  if (!hasTensorType(unpackedScore, {rows, 512}, f32) ||
      !hasTensorType(nz, {32, blocks, 16, 16}, f16) ||
      (mask->score != score && !mask->score.hasOneUse()))
    return failure();

  DominanceInfo dominance(pack->getParentOfType<func::FuncOp>());
  if (!dominance.dominates(nz, maximum) ||
      (mask->limits && !dominance.dominates(mask->limits, maximum)) ||
      (mask->required && !dominance.dominates(mask->required, maximum)))
    return failure();

  rewriter.setInsertionPoint(maximum);
  auto custom = createNZSoftmax(rewriter, pack.getLoc(), nz, oldExt.getIn(),
                                mask->limits, mask->required, probability);
  llvm::SmallSetVector<Operation *, 16> dead;
  auto collectInputs = [&](Operation *op) {
    for (Value operand : op->getOperands())
      if (auto *def = operand.getDefiningOp())
        dead.insert(def);
  };
  auto replace = [&](Operation *op, Value result) {
    collectInputs(op);
    rewriter.replaceOp(op, ValueRange{result});
  };
  replace(pack, custom->getResult(0));
  replace(next, custom->getResult(1));
  replace(sum, custom->getResult(2));
  while (!dead.empty()) {
    Operation *op = dead.pop_back_val();
    if (isOpTriviallyDead(op)) {
      collectInputs(op);
      rewriter.eraseOp(op);
    }
  }
  return success();
}

template <typename Op> Op getSingleUser(Value value) {
  if (!value.hasOneUse())
    return {};
  return dyn_cast<Op>(*value.getUsers().begin());
}

LogicalResult matchProbabilityPack(arith::TruncFOp trunc, Value nz,
                                   Value unpackedScore, IRRewriter &rewriter) {
  auto reshape = getSingleUser<tensor::ReshapeOp>(trunc.getResult());
  auto pack = reshape ? getSingleUser<linalg::TransposeOp>(reshape.getResult())
                      : nullptr;
  return pack ? fuseNZSoftmax(pack, nz, unpackedScore, rewriter) : failure();
}

LogicalResult matchSoftmaxUsers(Value score, Value nz, Value unpackedScore,
                                IRRewriter &rewriter) {
  for (Operation *user : score.getUsers()) {
    auto sub = dyn_cast<arith::SubFOp>(user);
    if (!sub || sub.getLhs() != score)
      continue;
    auto exp = getSingleUser<math::ExpOp>(sub.getResult());
    if (!exp)
      continue;
    for (Operation *user : exp->getUsers()) {
      if (auto trunc = dyn_cast<arith::TruncFOp>(user)) {
        if (succeeded(matchProbabilityPack(trunc, nz, unpackedScore, rewriter)))
          return success();
        continue;
      }
      auto upper = dyn_cast<arith::SelectOp>(user);
      if (!upper || upper.getFalseValue() != exp.getResult())
        continue;
      for (Operation *consumer : upper->getUsers()) {
        auto lower = dyn_cast<arith::SelectOp>(consumer);
        if (!lower || lower.getFalseValue() != upper.getResult())
          continue;
        auto trunc = getSingleUser<arith::TruncFOp>(lower.getResult());
        if (trunc &&
            succeeded(matchProbabilityPack(trunc, nz, unpackedScore, rewriter)))
          return success();
      }
    }
  }
  return failure();
}

LogicalResult matchScore(hivm::ConvertLayoutOp layout, arith::TruncFOp half,
                         IRRewriter &rewriter) {
  auto fixpipe = matchFixpipeOutputCast(half);
  if (!fixpipe || !fixpipe->writesNZ ||
      !llvm::is_contained(fixpipe->foldedOps, layout.getOperation()) ||
      fixpipe->matmulOp->getBlock() != half->getBlock() ||
      !half->getParentOfType<func::FuncOp>())
    return failure();

  auto unpack = getSingleUser<linalg::TransposeOp>(half.getResult());
  if (!unpack || unpack->getNumResults() != 1 ||
      unpack.getPermutation() != ArrayRef<int64_t>{1, 2, 0, 3})
    return failure();
  auto reshape = getSingleUser<tensor::ReshapeOp>(unpack->getResult(0));
  auto ext =
      reshape ? getSingleUser<arith::ExtFOp>(reshape.getResult()) : nullptr;
  auto scoreType = reshape ? dyn_cast<RankedTensorType>(reshape.getType())
                           : RankedTensorType{};
  if (!ext || hasFastMath(ext) || !scoreType || scoreType.getRank() != 2 ||
      !isStaticReshape(reshape, {scoreType.getDimSize(0), 512},
                       rewriter.getF16Type()))
    return failure();

  Value score = ext.getResult();
  if (succeeded(matchSoftmaxUsers(score, half, score, rewriter)))
    return success();
  for (Operation *user : score.getUsers())
    if (auto mask = dyn_cast<arith::SelectOp>(user);
        mask &&
        succeeded(matchSoftmaxUsers(mask.getResult(), half, score, rewriter)))
      return success();
  return failure();
}

LogicalResult matchFractalOutput(hivm::ConvertLayoutOp layout,
                                 IRRewriter &rewriter) {
  for (Operation *user : layout.getResult().getUsers()) {
    if (auto half = dyn_cast<arith::TruncFOp>(user)) {
      if (succeeded(matchScore(layout, half, rewriter)))
        return success();
    } else if (auto scale = dyn_cast<arith::MulFOp>(user)) {
      for (Operation *consumer : scale->getUsers())
        if (auto half = dyn_cast<arith::TruncFOp>(consumer);
            half && succeeded(matchScore(layout, half, rewriter)))
          return success();
    }
  }
  return failure();
}

// Older NPUIR versions only accept integer inplace_outs bindings.
template <typename Op, typename = void>
constexpr bool hasStridedInplaceOutputs = false;

template <typename Op>
constexpr bool hasStridedInplaceOutputs<
    Op, std::void_t<decltype(std::declval<Op>().getInplaceInputIndexForOutput(
            0))>> = true;

class FuseNZSoftmaxPass
    : public PassWrapper<FuseNZSoftmaxPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseNZSoftmaxPass);

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, hivm::HIVMDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  [[nodiscard]] StringRef getArgument() const final {
    return "ssbuf-standardize-op-fuse-nz-softmax";
  }

  void runOnOperation() override;
};

void FuseNZSoftmaxPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (!hasStridedInplaceOutputs<hivm::CustomOp> ||
      !hacc::utils::isAscend950(module))
    return;

  struct LayoutWorklist : RewriterBase::Listener {
    llvm::SmallSetVector<Operation *, 16> pending;

    void notifyOperationErased(Operation *op) override { pending.remove(op); }
  } layouts;
  module.walk([&](hivm::ConvertLayoutOp layout) {
    if (layout.getDstLayout().getDataLayout() == hivm::DataLayout::Fractal)
      layouts.pending.insert(layout);
  });
  IRRewriter rewriter(module.getContext(), &layouts);
  while (!layouts.pending.empty()) {
    auto layout = cast<hivm::ConvertLayoutOp>(layouts.pending.pop_back_val());
    // The replacement consumes the score, so the layout remains live.
    while (succeeded(matchFractalOutput(layout, rewriter)))
      continue;
  }
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> CVPipeline::createFuseNZSoftmaxPass() {
  return std::make_unique<FuseNZSoftmaxPass>();
}
