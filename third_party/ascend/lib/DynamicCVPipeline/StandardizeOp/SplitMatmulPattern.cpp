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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

#include "DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "SplitMatmul";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

struct MatmulInputs {
    Value a;
    Value b;
    Value bias;
};

} // namespace

static inline MatmulInputs parseMatmulInputs(linalg::MatmulOp matmulOp)
{
    auto inits = matmulOp.getDpsInits();
    auto inputs = matmulOp.getDpsInputs();

    return {inputs[0], inputs[1], inits[0]};
}

// the user is responsible for checking biasDefOp is not null
static bool operationIsFillZero(Operation *op)
{
    auto fillOp = dyn_cast<linalg::FillOp>(op);
    if (!fillOp) {
        return false;
    }
    auto filledVal = fillOp.getInputs()[0];
    return matchPattern(filledVal, m_Zero()) || matchPattern(filledVal, m_AnyZeroFloat());
}

// this generally should always be true, but just for safety...
static bool isFloatOrInt(RankedTensorType tensorType)
{
    auto elmType = tensorType.getElementType();
    return isa<FloatType, IntegerType>(elmType);
}

// Search outward from args to obtain the following information:
// 1. Whether args is only used and updated by matmul: bool argsLimitedInMatmul
// 2. Whether matmul is guaranteed to execute: bool mayNotExec
// 3. Outermost initial value: Value outerInVal
// 4. Outermost for or if, i.e., where to insert if/else.
static Value searchInArgsChain(Value nextValueOfC, bool &argsLimitedInMatmul, bool &mayNotExec, Value &outerInVal)
{
    if (outerInVal.getDefiningOp()) {
        return nextValueOfC;
    }
    auto op = nextValueOfC.getDefiningOp();
    auto parentOp = op->getParentOp();
    Value nextSearchValue = nextValueOfC;

    // update mayNotExec
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        IntegerAttr ubAttr, lbAttr;
        if (matchPattern(forOp.getUpperBound(), m_Constant(&ubAttr)) &&
            matchPattern(forOp.getLowerBound(), m_Constant(&lbAttr)))
        {
            if (ubAttr.getValue().sle(lbAttr.getValue())) {
                mayNotExec = true;
            }
        } else {
            mayNotExec = true;
        }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        if (!matchPattern(ifOp.getCondition(), m_One())) {
            mayNotExec = true;
        }
    }

    // update argsLimitedInMatmul
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        auto blockArg = dyn_cast_if_present<BlockArgument>(outerInVal);
        if (!blockArg || blockArg.getOwner() != forOp.getBody()) {
            argsLimitedInMatmul = false;
            return nextValueOfC;
        }
        int argIdx = blockArg.getArgNumber() - 1;
        for (auto &use : blockArg.getUses()) {
            auto user = use.getOwner();
            // Allowed: the op itself (mmad or inner for/if that chains to mmad).
            auto userInBlock = CVPipeline::getAncestorInBlock(user, op->getBlock());
            if (userInBlock == op) {
                continue;
            }
            if (auto yieldOp = dyn_cast<scf::YieldOp>(userInBlock)) {
                argsLimitedInMatmul = (use.getOperandNumber() == argIdx);
            } else {
                argsLimitedInMatmul = false;
                break;
            }
        }
        outerInVal = forOp.getInitArgs()[argIdx];
        nextSearchValue = forOp->getResult(argIdx);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        if (!ifOp.elseBlock()) {
            argsLimitedInMatmul = false;
        }
        auto otherYieldOp = op->getBlock() == ifOp->getBlock() ? cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator())
                                                               : cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
        auto opYieldOp = op->getBlock() == ifOp->getBlock() ? cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator())
                                                            : cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
        int resultIdx = -1;
        for (unsigned i = 0; i < otherYieldOp->getNumOperands(); ++i) {
            if (otherYieldOp->getOperand(i) == outerInVal && opYieldOp->getOperand(i) == nextSearchValue) {
                resultIdx = i;
                break;
            }
        }
        if (resultIdx == -1) {
            argsLimitedInMatmul = false;
        } else {
            nextSearchValue = ifOp.getResult(resultIdx);
        }
    } else {
        argsLimitedInMatmul = false;
        LOG_DEBUG("WARN: no for/if out to matmul.");
    }

    if (!argsLimitedInMatmul) {
        return nextValueOfC; // early return
    }
    return searchInArgsChain(nextSearchValue, argsLimitedInMatmul, mayNotExec, outerInVal);
}

template <typename Container> static Container filterNonIgnoredOps(const Container &container)
{
    auto filteredRange = llvm::make_filter_range(container, [](Operation *op) { return !isa<tensor::DimOp>(op); });
    return Container(filteredRange.begin(), filteredRange.end());
}

static OpOperand *getOnlyUse(Operation *op, Value value)
{
    auto uses =
        make_pointer_range(make_filter_range(op->getOpOperands(), [=](OpOperand &use) { return use.get() == value; }));
    llvm::SmallVector<OpOperand *> usesVec(uses.begin(), uses.end());
    if (usesVec.size() != 1) {
        return nullptr;
    }

    return usesVec[0];
}

/**
 * @brief Traces a chain of single-user operations starting from the given value.
 * Returns true if an operation matching the predicate is found in the chain.
 *
 * This function follows the def-use chain through operations that have exactly
 * one user. It traverses through:
 * - View-like operations (follows their single result)
 * - For loops (tracks init args to iteration arguments, then continues from yield to result)
 * - Yield operations within for/if (maps yield operands to parent operation results)
 * - Skip-able operations specified by isSkipOp callback
 *
 * @param value The starting value to trace from
 * @param isMatchedOp Callback to check if an operation matches the criteria
 * @param isSkipOp Callback to check if an operation should be skipped (continue tracing its result)
 * @return True if a matching operation is found, false otherwise
 */
static bool traceSingleChainUser(Value value, const std::function<bool(Operation *, Value v)> &isMatchedOp,
                                 const std::function<bool(Operation *, Value v)> &isSkipOp)
{
    if (!value) {
        return false;
    }

    auto users = filterNonIgnoredOps(llvm::DenseSet<Operation *>(value.user_begin(), value.user_end()));
    if (users.size() != 1) {
        return false;
    }

    auto *user = *users.begin();
    if (llvm::isa<ViewLikeOpInterface>(user)) {
        return traceSingleChainUser(user->getResult(0), isMatchedOp, isSkipOp);
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        auto initArgs = forOp.getInitArgs();
        int initIndx = -1;
        int useCnt = 0;
        for (auto [i, arg] : llvm::enumerate(initArgs)) {
            if (arg == value) {
                initIndx = i;
                useCnt++;
            }
        }
        if (useCnt == 1) {
            return traceSingleChainUser(forOp.getRegionIterArgs()[initIndx], isMatchedOp, isSkipOp);
        }
        return false;
    }

    // used in yield, we need find the for/if result;
    auto parentOp = user->getParentOp();
    if (parentOp && isa<scf::YieldOp>(user) && llvm::isa_and_present<scf::ForOp, scf::IfOp>(parentOp)) {
        auto use = getOnlyUse(user, value);
        if (!use) {
            return false;
        }
        return traceSingleChainUser(parentOp->getResult(use->getOperandNumber()), isMatchedOp, isSkipOp);
    }

    if (isMatchedOp(user, value)) {
        return true;
    }

    if (isSkipOp(user, value)) {
        if (user && user->getNumResults() == 1) {
            return traceSingleChainUser(user->getResult(0), isMatchedOp, isSkipOp);
        }
        return false;
    }
    return false;
}

static bool verifyAndHandleLoopCarriedL0C(linalg::MatmulOp matmulOp, PatternRewriter &rewriter, Value bias)
{
    if (matmulOp->hasAttr(CVPipeline::kLoopCarriedL0C)) {
        return false;
    }

    bool argsLimitedInMatmul = true;
    bool mayNotExec = false;
    Value outerInVal = bias;
    auto outerValue = searchInArgsChain(matmulOp.getResult(0), argsLimitedInMatmul, mayNotExec, outerInVal);
    auto *outerDefOp = outerValue.getDefiningOp();
    if (!argsLimitedInMatmul) {
        LOG_DEBUG("Split because bias is not limited in args" << matmulOp);
        return true;
    }

    if (!operationIsFillZero(outerInVal.getDefiningOp())) {
        // From one matmul
        auto defMatmul =
            dyn_cast_if_present<linalg::MatmulOp>(hivm::traceDefOp<linalg::MatmulOp>(bias).value_or(nullptr));
        if (!defMatmul) {
            LOG_DEBUG("Split because bias may not be zero, and the init value is not from matmul: " << matmulOp);
            return true;
        }
        LOG_DEBUG("Split because avoiding NPUIR insert fixpipe errors" << matmulOp);
        return true;
    }
    auto matchFunc = [=](Operation *op, Value value) {
        if (auto nextMatmulOp = dyn_cast<linalg::MatmulOp>(op)) {
            auto inputs = parseMatmulInputs(nextMatmulOp);
            return inputs.a != value && inputs.b != value && inputs.bias == value;
        }
        return false;
    };

    if (traceSingleChainUser(outerValue, matchFunc, [](Operation *op, Value value) { return false; })) {
        // To one matmul
        LOG_DEBUG("Split because avoiding NPUIR insert fixpipe errors" << matmulOp);
        return true;
    }

    matmulOp->setAttr(CVPipeline::kLoopCarriedL0C, rewriter.getUnitAttr());

    if (mayNotExec) {
        LOG_DEBUG("Split because the for loop may not execute" << matmulOp);
        return true;
    }
    LOG_DEBUG("Not Split because bias can remain in L0C" << matmulOp);
    return false;
}

/**
 * @brief Evaluates whether a Matmul operation is a candidate for splitting.
 *
 * This pattern identifies tensor-based matmul operations where the accumulator (bias) is
 * non-zero or dynamic. Splitting isolates the pure GEMM computation (ideal for CUBE hardware)
 * from the accumulator addition (ideal for VECTOR hardware), preventing hardware execution stalls.
 *
 * Matching Logic and Rules:
 *
 * 1. Validation Checks:
 *    - The matmul must operate on tensors (tensor mode).
 *    - The element type of the output must be an integer or a float type.
 *
 * 2. Rule 1: Dynamic / Block Argument Accumulator
 *    - If the bias tensor is a block argument (i.e., it has no defining operation in the
 *      current block), its compile-time value is unknown. We conservatively assume it is
 *      non-zero and trigger a split.
 *
 * 3. Rule 2: Zero-Accumulator Bypass (Do Not Split)
 *    - If the bias tensor is statically known to be a constant zero (e.g., initialized via
 *      linalg.fill with 0), the split is bypassed. Standard lowerings already optimize
 *      this cleanly without introducing a redundant vector addition.
 *
 * 4. Default Split:
 *    - If the bias is defined, non-zero, and does not fall under Rule 3, split the operation.
 */
static bool shouldSplit(linalg::MatmulOp matmulOp, PatternRewriter &rewriter)
{
    auto matmulResult = matmulOp->getResult(0);
    for (auto res : matmulResult.getUsers()) {
        if (auto nextMatmul = dyn_cast<linalg::MatmulOp>(res)) {
            auto inputs = parseMatmulInputs(nextMatmul);
            if (inputs.a == matmulResult || inputs.b == matmulResult) {
                LOG_DEBUG("Split because the user is matmul's A or B." << matmulOp);
                return true;
            }
        }
    }

    auto bias = parseMatmulInputs(matmulOp).bias;
    auto *biasDefOp = bias.getDefiningOp();
    // Rule 1: bias is block arg -> split if result cannot remain in l0c
    if (!biasDefOp) {
        return verifyAndHandleLoopCarriedL0C(matmulOp, rewriter, bias);
    }

    // Rule 2: matmul a b 0 -> do not split
    if (operationIsFillZero(biasDefOp)) {
        LOG_DEBUG("Not split because bias is zero: " << matmulOp);
        return false;
    }

    // Otherwise split:
    LOG_DEBUG("Should split: " << matmulOp);

    return true;
}

/**
 * @brief Transforms a matmul with a non-zero accumulator into a zero-initialized matmul
 * followed by an elementwise addition.
 *
 * This transformation breaks the combined GEMM-and-accumulation execution down to isolate
 * the high-throughput matrix multiply from vector-based bias additions.
 *
 * =========================================================================================
 * --- [BEFORE] ---
 * %bias = ... : tensor<MxNxf32>
 * %result = linalg.matmul ins(%a, %b : tensor<MxKxf32>, tensor<KxNxf32>)
 *                       outs(%bias : tensor<MxNxf32>) -> tensor<MxNxf32>
 *
 * =========================================================================================
 * --- [AFTER] ---
 * // Step 1: Create a placeholder empty tensor matching the output shape
 * %empty = tensor.empty() : tensor<MxNxf32>
 *
 * // Step 2: Initialize constant zero corresponding to the element type
 * %cst_0 = arith.constant 0.000000e+00 : f32
 *
 * // Step 3: Fill the empty tensor to establish a zero-accumulator
 * %zero_acc = linalg.fill ins(%cst_0 : f32) outs(%empty : tensor<MxNxf32>) -> tensor<MxNxf32>
 *
 * // Step 4: Perform the pure GEMM operation on the zero-accumulator
 * %matmul_res = linalg.matmul ins(%a, %b : tensor<MxKxf32>, tensor<KxNxf32>)
 *                           outs(%zero_acc : tensor<MxNxf32>) -> tensor<MxNxf32>
 *
 * // Step 5: Complete the operation by adding the original bias using an elementwise add
 * %result = arith.addf %matmul_res, %bias : tensor<MxNxf32>
 * =========================================================================================
 *
 * Assumptions:
 * 1. The operation is tensor-based (not buffer-based).
 * 2. The accumulator (bias) is not statically known to be zero.
 * 3. The element type is a primitive float or integer type.
 */
static void splitMatmul(linalg::MatmulOp matmulOp, PatternRewriter &rewriter)
{
    auto inputs = matmulOp.getDpsInputs();
    auto a = inputs[0];
    auto b = inputs[1];

    // this is the accumulator/out operand, not result in tensor mode
    auto bias = matmulOp.getDpsInits()[0];

    auto outputType = dyn_cast<RankedTensorType>(bias.getType());
    if (!outputType) {
        LOG_DEBUG("Not tensor mode: " << matmulOp
                                      << "; the caller does not ensure the assumption. Cowardly doing nothing");
        return;
    }
    auto elmType = outputType.getElementType();

    Location loc = matmulOp.getLoc();

    // [Step 1] Create tensor.empty for the new accumulator tensor
    // Same shape and type as original matmul output
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (outputType.isDynamicDim(i)) {
            dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, bias, i));
        }
    }
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, outputType, dynamicSizes);

    // [Step 2] Create zero constant based on element type
    // Supports both floating-point (arith.constant float) and integer types
    Value zeroValue;
    if (auto floatType = dyn_cast<FloatType>(elmType)) {
        APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
        zeroValue = rewriter.create<arith::ConstantFloatOp>(loc, zeroAPFloat, floatType).getResult();
    } else if (auto intType = dyn_cast<IntegerType>(elmType)) {
        zeroValue = rewriter.create<arith::ConstantIntOp>(loc, 0, intType).getResult();
    } else {
        // User does not ensure assumption 3.
        return;
    }

    // [Step 3] Use linalg.fill to populate empty tensor with zero -> zero accumulator
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange {zeroValue}, ValueRange {emptyOp.getResult()});

    // [Step 5] Create new matmul using zero-filled tensor as accumulator
    // New matmul runs entirely on CUBE with no VECTOR dependency
    auto newMatmul = rewriter.create<linalg::MatmulOp>(loc, ValueRange {a, b}, ValueRange {fillOp.getResult(0)});
    NamedAttrList attrs(matmulOp->getAttrDictionary());
    constexpr StringLiteral kShouldRemoveAttrs[] = {"operandSegmentSizes", "res_attrs", "arg_attrs"};
    for (auto attr : kShouldRemoveAttrs) {
        attrs.erase(attr);
    }
    newMatmul->setAttrs(attrs);
    auto newMatmulRes = newMatmul.getResult(0);

    // [Step 6] Create add: add(new_matmul_result, outs_value)
    // This is the "c" in a*b+c, added after the matmul result
    Operation *addOp;
    if (isa<FloatType>(elmType)) {
        addOp = rewriter.create<arith::AddFOp>(loc, newMatmulRes, bias).getOperation();
    } else {
        addOp = rewriter.create<arith::AddIOp>(loc, newMatmulRes, bias).getOperation();
    }

    addOp->setAttr(CVPipeline::kAddFromMatmul, rewriter.getUnitAttr());
    rewriter.replaceOp(matmulOp, addOp);
}

static bool verifyMatmul(linalg::MatmulOp matmulOp)
{
    auto inits = matmulOp.getDpsInits();
    auto inputs = matmulOp.getDpsInputs();
    if (inits.empty() || inputs.size() < 2) {
        LOG_DEBUG("Not split because op is illegal: " << matmulOp);
        return false;
    }

    auto bias = matmulOp.getDpsInits()[0];
    auto outputType = dyn_cast<RankedTensorType>(bias.getType());
    if (!outputType) {
        LOG_DEBUG("Not split because not tensor mode matmul: " << matmulOp);
        return false;
    }
    if (!isFloatOrInt(outputType)) {
        LOG_DEBUG("Not split because not integer or float: " << matmulOp);
        return false;
    }

    return true;
}

LogicalResult SplitMatmulPattern::matchAndRewrite(linalg::MatmulOp matmulOp, PatternRewriter &rewriter) const
{
    if (!verifyMatmul(matmulOp)) {
        return failure();
    }

    if (!shouldSplit(matmulOp, rewriter)) {
        return failure();
    }

    splitMatmul(matmulOp, rewriter);
    return success();
}
