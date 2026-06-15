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

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fixpipe-opt"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg << "\n")

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

class FixpipeOptPass : public PassWrapper<FixpipeOptPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixpipeOptPass)

    FixpipeOptPass() = default;
    void runOnOperation() override;

    llvm::StringRef getArgument() const final { return "fixpipe-opt"; }
    void getDependentDialects(DialectRegistry &registry) const override;

    llvm::StringRef getDescription() const final
    {
        return "Optimize matmul-cast-store pattern for fixpipe by setting core_type to CUBE";
    }

  private:
    bool matchFixpipePattern(linalg::MatmulOp matmulOp, SetVector<Operation *> &matchedOps);
    bool isFixpipeCastPattern(Operation *truncOp, SetVector<Operation *> &matchedOps);
    bool isFixpipeMulPattern(Operation *mulOp, SetVector<Operation *> &matchedOps);
    bool isStoreToGM(Operation *materializeOp, SetVector<Operation *> &matchedOps);
    bool applyFixpipeOpt(SetVector<Operation *> &matchedOps, const CVPipeline::MemoryDependenceGraph &memGraph,
                         CVPipeline::ComputeBlockIdManager &bm);
    bool isSubviewFromGlobalMemory(ViewLikeOpInterface viewOp, SetVector<Operation *> &matchedOps);
    bool isValidTrunc(Operation *op);
    bool isValidMul(Operation *op, Value matmulValues, SetVector<Operation *> &matchedOps);
};

namespace {
struct DependencyCycleDetector {
    llvm::DenseSet<mlir::Operation *> &opsInNewBlock;
    llvm::DenseSet<mlir::Operation *> visited;
    const CVPipeline::MemoryDependenceGraph &memGraph;
    CVPipeline::ComputeBlockIdManager &bm;
    Block *block;
    void clear() { visited.clear(); }
    bool dfs(Operation *cur);
    DependencyCycleDetector(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
                            llvm::DenseSet<mlir::Operation *> &opsInNewBlock, CVPipeline::ComputeBlockIdManager &bm)
        : block(block), memGraph(memGraph), opsInNewBlock(opsInNewBlock), bm(bm)
    {
    }
};

} // namespace

bool DependencyCycleDetector::dfs(Operation *cur)
{
    if (opsInNewBlock.contains(cur)) {
        return true;
    }
    if (!visited.insert(cur).second) {
        return false;
    }

    SmallVector<Operation *> allusers;
    allusers.append(cur->getUsers().begin(), cur->getUsers().end());
    allusers.append(memGraph.getExecAfter(cur).begin(), memGraph.getExecAfter(cur).end());
    for (auto *user : allusers) {
        auto *userInBlock = CVPipeline::getAncestorInBlock(user, block);
        if (bm.getBlockIdByOp(userInBlock) == -1) {
            if (dfs(userInBlock)) {
                return true;
            }
        } else {
            for (auto *nx : bm.getOpsByBlockId(bm.getBlockIdByOp(userInBlock))) {
                if (dfs(nx)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/**
 * Check if adding willaddOps to targetBlockId will create cycle.
 * Walk from every op in targetBlockId and willaddOps.
 * if reach other blockid ops and dfs find any targetBlockId op, then there is cycle.
 */
static std::optional<bool> willCreateCycle(SetVector<Operation *> &willaddOps, Block *block,
                                           const CVPipeline::MemoryDependenceGraph &memGraph, int targetBlockId,
                                           CVPipeline::ComputeBlockIdManager &bm)
{
    // Step1: Init, Add willaddOps to targetBlockId.
    // opsInNewBlock is new block, includes two part: 1. original ops in targetBlockId. 2. willaddOps.
    llvm::DenseSet<mlir::Operation *> opsInNewBlock;
    for (auto op : bm.getOpsByBlockId(targetBlockId)) {
        opsInNewBlock.insert(op);
    }
    llvm::DenseMap<mlir::Operation *, int> originBlockId;
    for (auto op : willaddOps) {
        opsInNewBlock.insert(op);
        // For backtracing
        originBlockId[op] = bm.getBlockIdByOp(op);
        bm.updateBlockId(op, targetBlockId);
    }
    DependencyCycleDetector detector = {block, memGraph, opsInNewBlock, bm};

    // Step2: Walk from every op in opsInNewBlock
    auto ret = false;
    for (mlir::Operation *testOp : opsInNewBlock) {
        SmallVector<Operation *> allusers;
        allusers.append(testOp->getUsers().begin(), testOp->getUsers().end());
        allusers.append(memGraph.getExecAfter(testOp).begin(), memGraph.getExecAfter(testOp).end());
        for (auto *user : allusers) {
            auto *userInBlock = CVPipeline::getAncestorInBlock(user, block);
            if (opsInNewBlock.contains(userInBlock)) {
                continue;
            }
            if (bm.getBlockIdByOp(userInBlock) == -1) {
                detector.clear();
                if (detector.dfs(userInBlock)) {
                    ret = true;
                    break;
                }
                continue;
            }
            auto opsUsedBlockId = bm.getOpsByBlockId(bm.getBlockIdByOp(userInBlock));
            for (auto *userOp : opsUsedBlockId) {
                detector.clear();
                if (detector.dfs(userOp)) {
                    ret = true;
                    break;
                }
            }
        }
        if (ret) {
            // early stop if find cycle.
            break;
        }
    }

    // Step3: Backtrace blockId change.
    for (auto op : willaddOps) {
        bm.updateBlockId(op, originBlockId[op]);
    }
    return ret;
}

bool FixpipeOptPass::isValidTrunc(Operation *op)
{
    // Just filter: arith.truncf(f32->bf16, f32->f16, i32->i8)
    if (auto truncFOp = dyn_cast<arith::TruncFOp>(op)) {
        Type inType = truncFOp.getIn().getType();
        Type outType = truncFOp.getResult().getType();
        if (auto shapedType = dyn_cast<ShapedType>(inType))
            inType = shapedType.getElementType();
        if (auto shapedType = dyn_cast<ShapedType>(outType))
            outType = shapedType.getElementType();

        return isa<Float32Type>(inType) && (isa<BFloat16Type>(outType) || isa<Float16Type>(outType));
    }
    if (auto truncIOp = dyn_cast<arith::TruncIOp>(op)) {
        Type inType = truncIOp.getIn().getType();
        Type outType = truncIOp.getResult().getType();
        if (auto shapedType = dyn_cast<ShapedType>(inType))
            inType = shapedType.getElementType();
        if (auto shapedType = dyn_cast<ShapedType>(outType))
            outType = shapedType.getElementType();

        return inType.isInteger(32) && outType.isInteger(8);
    }
    return false;
}

/** Determines if a value is "scalar-like" based on the following criteria:
 1. True scalar types (integer, index, or float)
 2. Tensor types with empty shape (e.g., tensor<f32>)
 3. Constant tensors where all elements have the same value (splat constants)
 4. Tensors with shape where all dimensions equal 1 (single-element tensors)
 */
static bool isScalarLike(Value value)
{
    Type type = value.getType();
    auto shapedType = dyn_cast<ShapedType>(type);

    // 1. scalar
    if (!shapedType) {
        return type.isIntOrIndexOrFloat();
    }

    // 2. tensor<f32> with empty shape is also considered scalar-like
    ArrayRef<int64_t> shape = shapedType.getShape();
    if (shape.empty()) {
        return true;
    }

    // 3. tensor with constant value
    Attribute attr;
    if (matchPattern(value, m_Constant(&attr))) {
        auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(attr);
        return denseAttr && denseAttr.isSplat() && denseAttr.getElementType().isIntOrIndexOrFloat();
    }

    // 4. tensor with one element
    return llvm::all_of(shape, [](int64_t dim) { return dim == 1; });
}

void transSource(Value value, SetVector<Operation *> &matchedOps, Block *block)
{
    if (!isScalarLike(value)) {
        return;
    }

    if (auto defOp = value.getDefiningOp()) {
        // We need to find one point, value changes from tensor to scalar.
        // So use specific op condition to stop searching.
        if (isa<tensor::ExtractOp>(defOp)) {
            return;
        }
        if (llvm::find(matchedOps, defOp) != matchedOps.end()) {
            return;
        }
        if (defOp->getBlock() != block) {
            return;
        }
        matchedOps.insert(defOp);
        defOp->walk([&](Operation *nestedOp) {
            for (Value operand : nestedOp->getOperands()) {
                transSource(operand, matchedOps, block);
            }
        });
    }
}

bool FixpipeOptPass::isValidMul(Operation *op, Value matmulValue, SetVector<Operation *> &matchedOps)
{
    // Just filter: arith.mulf/muli(scalar)
    if (!isa<arith::MulFOp>(op) && !isa<arith::MulIOp>(op)) {
        return false;
    }
    auto quantScalarValue = op->getOperand(0) == matmulValue ? op->getOperand(1) : op->getOperand(0);
    if (isScalarLike(quantScalarValue)) {
        transSource(quantScalarValue, matchedOps, op->getBlock());
        return true;
    }

    // From one fill op with constant value or args....
    if (auto defOp = quantScalarValue.getDefiningOp()) {
        if (auto fillOp = dyn_cast<linalg::FillOp>(defOp)) {
            auto operands = fillOp->getOperands();
            if (!operands.empty()) {
                Value fillValue = operands[0];
                if (isScalarLike(fillValue)) {
                    if (llvm::find(matchedOps, fillOp) == matchedOps.end() && fillOp.getBlock() == op->getBlock()) {
                        matchedOps.insert(fillOp);
                    }
                    transSource(fillValue, matchedOps, op->getBlock());
                    return true;
                } else {
                    LOG_DEBUG("Fill operand is not scalar-like, NOT match. fill=" << *fillOp);
                    return false;
                }
            }
        }
    }
    return false;
}

bool FixpipeOptPass::isSubviewFromGlobalMemory(ViewLikeOpInterface viewOp, SetVector<Operation *> &matchedOps)
{
    // Subview ops may be nested many layers deep through reinterpretation or other subviews.
    // like, subview (subview (reinterpret_cast (subview (reinterpret_cast (arg0)))))
    // so we need Search and only keep same block view-like op.
    Value source = viewOp.getViewSource();
    auto block = viewOp->getBlock();
    while (true) {
        LOG_DEBUG("Check view source: " << source << "\n");
        if (auto blockArg = dyn_cast<BlockArgument>(source)) {
            Operation *parentOp = blockArg.getOwner()->getParentOp();
            if (isa<func::FuncOp>(parentOp)) {
                return true;
            } else {
                LOG_DEBUG("Subview source block argument is not from func entry block.");
                return false;
            }
        }
        // From other view-like op
        if (auto viewLike = dyn_cast<ViewLikeOpInterface>(source.getDefiningOp())) {
            if (viewLike->getBlock() == block) {
                matchedOps.insert(viewLike.getOperation());
            }
            source = viewLike.getViewSource();
            continue;
        }
        LOG_DEBUG("Subview source defining op is not ViewLikeOpInterface: " << source);
        return false;
    }
    return false;
}

bool FixpipeOptPass::isStoreToGM(Operation *storeOp, SetVector<Operation *> &matchedOps)
{
    ViewLikeOpInterface viewOp = nullptr;
    if (auto materializeOp = dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
        Value destMemref = materializeOp.getDest();
        viewOp = destMemref.getDefiningOp<ViewLikeOpInterface>();
    } else if (auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
        auto dest = hivmStore.getDst();
        viewOp = dest.getDefiningOp<ViewLikeOpInterface>();
    } else {
        LOG_DEBUG("Cannot find store op, NOT match");
        return false;
    }

    if (!viewOp) {
        LOG_DEBUG("store destination is not from ViewLikeOpInterface, NOT match");
        return false;
    }
    matchedOps.insert(storeOp);
    matchedOps.insert(viewOp);
    if (!isSubviewFromGlobalMemory(viewOp, matchedOps)) {
        LOG_DEBUG("Subview is not from global memory (GM), NOT match.");
        return false;
    }
    return true;
}

/** Fixpipe supports cast, the pattern should be like below:
    linalg.matmul
        ↓
    arith.truncf(f32->bf16, f32->f16, i32->i8)
        ↓
    tensor.extract_slice
        ↓
    bufferization.materialize_in_destination memref.subview(gm)
    After optimization, all these ops will be in same block with matmul and set core_type to CUBE.
 */
bool FixpipeOptPass::isFixpipeCastPattern(Operation *truncOp, SetVector<Operation *> &matchedOps)
{
    Value truncResult = truncOp->getResult(0);
    if (!truncResult.hasOneUse()) {
        LOG_DEBUG("Trunc not only one user, NOT match.");
        return false;
    }
    auto maybeExtract = *truncResult.getUsers().begin();
    tensor::ExtractSliceOp extractSliceOp = nullptr;
    if (auto extract = dyn_cast<tensor::ExtractSliceOp>(maybeExtract)) {
        extractSliceOp = extract;
    } else {
        LOG_DEBUG("Cannot find extract slice op, NOT match");
        return false;
    }

    Value extractResult = extractSliceOp.getResult();
    if (!extractResult.hasOneUse()) {
        LOG_DEBUG("Extract Slice not only one user, NOT match.");
        return false;
    }
    auto maybeMaterialize = *extractResult.getUsers().begin();
    bufferization::MaterializeInDestinationOp materializeOp = nullptr;

    if (auto materialize = dyn_cast<bufferization::MaterializeInDestinationOp>(maybeMaterialize)) {
        materializeOp = materialize;
    } else {
        LOG_DEBUG("Cannot find materialize op, NOT match");
        return false;
    }

    matchedOps.insert(truncOp);
    matchedOps.insert(extractSliceOp);
    if (!isStoreToGM(materializeOp, matchedOps)) {
        LOG_DEBUG("Not store to GM pattern, NOT match.");
        return false;
    }
    return true;
}

/** Fixpipe supports scaling, the pattern should be like below:
    linalg.matmul
        ↓
    arith.mulf/muli (mul one scalar-like value for quantization)
        ↓
    tensor.extract_slice (optional)
        ↓
    bufferization.materialize_in_destination memref.subview(gm)
    After optimization, all these ops will be in same block with matmul and set core_type to CUBE.
 */
bool FixpipeOptPass::isFixpipeMulPattern(Operation *mulOp, SetVector<Operation *> &matchedOps)
{
    Value mulResult = mulOp->getResult(0);
    if (!mulResult.hasOneUse()) {
        LOG_DEBUG("Mul not only one user, NOT match.");
        return false;
    }
    auto maybeExtract = *mulResult.getUsers().begin();
    tensor::ExtractSliceOp extractSliceOp = nullptr;
    Value extractResult = mulResult;
    if (auto extract = dyn_cast<tensor::ExtractSliceOp>(maybeExtract)) {
        extractSliceOp = extract;
        extractResult = extractSliceOp.getResult();
        matchedOps.insert(extractSliceOp);
    }

    if (!extractResult.hasOneUse()) {
        LOG_DEBUG("Extract Slice not only one user, NOT match.");
        return false;
    }

    matchedOps.insert(mulOp);
    if (!isStoreToGM(*extractResult.getUsers().begin(), matchedOps)) {
        LOG_DEBUG("Not store to GM pattern, NOT match.");
        return false;
    }
    return true;
}

/** Match fixpipe optimization patterns starting from a matmul operation.
 Pattern 1 (Cast Pattern):
   linalg.matmul -> arith.truncf/i -> tensor.extract_slice ->
   bufferization.materialize_in_destination(memref.subview(gm))

 Pattern 2 (Quantization Pattern):
   linalg.matmul -> arith.mulf -> tensor.extract_slice ->
   bufferization.materialize_in_destination(memref.subview(gm))
 */
bool FixpipeOptPass::matchFixpipePattern(linalg::MatmulOp matmulOp, SetVector<Operation *> &matchedOps)
{
    LOG_DEBUG("Check matmul op: " << *matmulOp);
    Value matmulResult = matmulOp.getResult(0);
    if (!matmulResult.hasOneUse()) {
        LOG_DEBUG("Matmul not only one user, NOT match.");
        return false;
    }
    matchedOps.insert(matmulOp);

    auto matmulUser = *matmulResult.getUsers().begin();

    if (isValidTrunc(matmulUser)) {
        if (isFixpipeCastPattern(matmulUser, matchedOps)) {
            return true;
        }
    } else if (isValidMul(matmulUser, matmulResult, matchedOps)) {
        if (isFixpipeMulPattern(matmulUser, matchedOps)) {
            return true;
        }
    } else {
        LOG_DEBUG("Cannot find valid consumer op (trunc or mulf), NOT match.");
        return false;
    }

    return false;
}

bool FixpipeOptPass::applyFixpipeOpt(SetVector<Operation *> &matchedOps,
                                     const CVPipeline::MemoryDependenceGraph &memGraph,
                                     CVPipeline::ComputeBlockIdManager &bm)
{
    // If there are no cycle in Compute Block level, we apply:
    // 1. Change block_id to the matmul's block id
    // 2. Change core_type to CUBE.
    Operation *matmulOp = matchedOps[0];
    for (auto op : matchedOps) {
        if (isa<linalg::MatmulOp>(op)) {
            matmulOp = op;
            break;
        }
    }
    int targetBlockId = bm.getBlockIdByOp(matmulOp);
    auto block = matmulOp->getBlock();

    if (willCreateCycle(matchedOps, block, memGraph, targetBlockId, bm).value_or(true)) {
        return false;
    }
    for (Operation *op : matchedOps) {
        if (isa<scf::SCFDialect>(op->getDialect())) {
            op->walk([&](Operation *nestedOp) {
                bm.updateBlockId(nestedOp, targetBlockId);
                nestedOp->setAttr(CVPipeline::kCoreType, StringAttr::get(op->getContext(), "CUBE"));
            });
        } else {
            bm.updateBlockId(op, targetBlockId);
            op->setAttr(CVPipeline::kCoreType, StringAttr::get(op->getContext(), "CUBE"));
        }
    }
    return true;
}

void FixpipeOptPass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<hivm::HIVMDialect>();
}

void FixpipeOptPass::runOnOperation()
{
    ModuleOp module = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);
    LOG_DEBUG("== FixpipeOpt Pass Start ==\n");
    LOG_DEBUG(module);

    SmallVector<SetVector<Operation *>> allMatchedPatterns;

    module.walk([&](linalg::MatmulOp matmulOp) {
        SetVector<Operation *> matchedOps;
        if (matchFixpipePattern(matmulOp, matchedOps)) {
            allMatchedPatterns.push_back(matchedOps);
        }
    });
    LOG_DEBUG("== Found " << allMatchedPatterns.size() << " fixpipe patterns ==\n");

    /** Inorder to avoid cycle, clone scalar-like ops.
        A-> B -> C
        ↘      ↗
            D
        Now we want to fuse A/B/C, so clone A' for D to avoid cycle.
    */
    auto bmOriginal = CVPipeline::ComputeBlockIdManager(module);
    for (auto &matchedOps : allMatchedPatterns) {
        auto sorted = mlir::multiRootTopologicalSort(matchedOps);
        for (Operation *op : llvm::reverse(sorted)) {
            if (op->getNumResults() == 1 && isScalarLike(op->getResult(0))) {
                // replace op not in matchedOps with cloned op, and keep original op for other pattern.
                SmallVector<OpOperand *> otherUses;
                for (auto &use : op->getResult(0).getUses()) {
                    Operation *userOp = use.getOwner();
                    auto userInBlock = CVPipeline::getAncestorInBlock(userOp, op->getBlock());
                    if (llvm::find(matchedOps, userInBlock) == matchedOps.end() &&
                        bmOriginal.getBlockIdByOp(userInBlock) != bmOriginal.getBlockIdByOp(matchedOps[0]))
                    {
                        otherUses.push_back(&use);
                    }
                }
                if (otherUses.size() > 0) {
                    LOG_DEBUG("now cloned: " << *op << "\n");
                    OpBuilder builder(op);
                    auto clonedOp = builder.clone(*op);
                    for (auto use : otherUses) {
                        (*use).set(clonedOp->getResult(0));
                    }
                }
            }
        }
    }

    auto bm = CVPipeline::ComputeBlockIdManager(module);
    for (auto &matchedOps : allMatchedPatterns) {
        if (!applyFixpipeOpt(matchedOps, memDepGraph, bm)) {
            for (Operation *op : matchedOps) {
                LOG_DEBUG("Cannot set block id for op: " << *op);
            }
            LOG_DEBUG("Cannot set one Block Id, may be because cycle");
        }
    }

    LOG_DEBUG("== FixpipeOpt Pass Complete ==\n");
}

std::unique_ptr<OperationPass<ModuleOp>> createFixpipeOptPass()
{
    return std::make_unique<FixpipeOptPass>();
}

} // namespace triton
} // namespace mlir