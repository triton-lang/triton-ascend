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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Matchers.h"
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
    bool matchFixpipePattern(linalg::MatmulOp matmulOp, SmallVector<Operation *> &matchedOps);
    bool matchAfterForFixpipePattern(scf::ForOp forOp, SmallVector<Operation *> &matchedOps, CVPipeline::ComputeBlockIdManager &bm);
    bool isFixpipeCastPattern(Operation *op, SmallVector<Operation *> &matchedOps);
    bool isFixpipeMulfPattern(Operation *op, SmallVector<Operation *> &matchedOps);
    bool isStoreToGM(Operation *materializeOp, SmallVector<Operation *> &matchedOps);
    bool applyFixpipeOpt(SmallVector<Operation *> &matchedOps, const CVPipeline::MemoryDependenceGraph &memGraph,
                         CVPipeline::ComputeBlockIdManager &bm);
    bool isSubviewFromGlobalMemory(memref::SubViewOp subviewOp, SmallVector<Operation *> &matchedOps);
    bool isValidTrunc(Operation *op);
    bool isValidMul(Operation *op, Value matmulValue, SmallVector<Operation *> &matchedOps);
    bool hasCubeCoreType(scf::ForOp forOp);
    bool isCubeResult(scf::ForOp forOp, Value result);
};

namespace {
static std::vector<std::string> splitStringByComma(llvm::StringRef str) {
    std::vector<std::string> result;
    llvm::SmallVector<llvm::StringRef, 8> parts;
    str.split(parts, ',');
    for (auto part : parts) {
        result.push_back(part.trim().str());
    }
    return result;
}

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
static std::optional<bool> willCreateCycle(llvm::SmallVectorImpl<Operation *> &willaddOps, Block *block,
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
    if (matchPattern(value, m_Constant(&attr)))
    {
        auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(attr);
        return denseAttr && denseAttr.isSplat() && denseAttr.getElementType().isIntOrIndexOrFloat();

    }
    
    // 4. tensor with one element
    return llvm::all_of(shape, [](int64_t dim) { return dim == 1; });
}

bool FixpipeOptPass::isValidMul(Operation *op, Value matmulValue, SmallVector<Operation *> &matchedOps)
{
    // Just filter: arith.mulf(scalar)
    if (!isa<arith::MulFOp>(op) && !isa<arith::MulIOp>(op)) {
        return false;
    }
    auto quantScalarValue = op->getOperand(0) == matmulValue ? op->getOperand(1) : op->getOperand(0);
    if (isScalarLike(quantScalarValue)) {
        return true;
    }

    // From one fill op with constant value or args....
    if(auto defOp = quantScalarValue.getDefiningOp()) {
        if (auto fillOp = dyn_cast<linalg::FillOp>(defOp)) {
            auto operands = fillOp->getOperands();
            if (!operands.empty()) {
                Value fillValue = operands[0];
                if (isScalarLike(fillValue)) {
                    return true;
                }
                else {
                    LOG_DEBUG("Fill operand is not scalar-like, NOT match. fill=" << *fillOp);

                    return false;
                }
            }
        }
    }   
    return false;
}

bool FixpipeOptPass::isSubviewFromGlobalMemory(memref::SubViewOp subviewOp, SmallVector<Operation *> &matchedOps)
{
    // Subview ops may be nested many layers deep through reinterpretation or other subviews.
    // like, subview (subview (reinterpret_cast (subview (reinterpret_cast (arg0)))))
    // so we need Search and only keep same block view-like op.
    Value source = subviewOp.getSource();
    auto block = subviewOp->getBlock();
    while (true) {
        LOG_DEBUG("Check subview source: " << source << "\n");
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
                matchedOps.push_back(viewLike.getOperation());
            }
            source = viewLike.getViewSource();
            continue;
        }
        LOG_DEBUG("Subview source defining op is not ViewLikeOpInterface: " << source);
        return false;
    }
    return false;
}

bool FixpipeOptPass::isStoreToGM(Operation* storeOp,
                                 SmallVector<Operation *> &matchedOps)
{
    memref::SubViewOp subviewOp = nullptr;
    if (auto materializeOp = dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
        Value destMemref = materializeOp.getDest();
        subviewOp = destMemref.getDefiningOp<memref::SubViewOp>();
    } else if(auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
        auto dest = hivmStore.getDst();
        subviewOp = dest.getDefiningOp<memref::SubViewOp>();
    } else {
        LOG_DEBUG("Cannot find store op, NOT match");
        return false;
    }


    if (!subviewOp) {
        LOG_DEBUG("store destination is not from memref.subview, NOT match");
        return false;
    }
    matchedOps.push_back(storeOp);
    matchedOps.push_back(subviewOp);
    if (!isSubviewFromGlobalMemory(subviewOp, matchedOps)) {
        LOG_DEBUG("Subview is not from global memory (GM), NOT match.");
        return false;
    }
    return true;
}

/** To use fixpipe optimization, the pattern should be like below:
    linalg.matmul
        ↓
    arith.truncf(f32->bf16, f32->f16, i32->i8)
        ↓
    tensor.extract_slice
        ↓
    bufferization.materialize_in_destination memref.subview(gm)
    After optimization, all these ops will be in same block with matmul and set core_type to CUBE.
 */
bool FixpipeOptPass::isFixpipeCastPattern(Operation *truncOp, SmallVector<Operation *> &matchedOps)
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

    matchedOps.push_back(truncOp);
    matchedOps.push_back(extractSliceOp);
    if (!isStoreToGM(materializeOp, matchedOps)) {
        LOG_DEBUG("Not store to GM pattern, NOT match.");
        return false;
    }
    return true;
}

/** To use fixpipe optimization, the pattern should be like below:
    linalg.matmul
        ↓
    arith.mulf (mul one scalar-like value for quantization)
        ↓
    tensor.extract_slice
        ↓
    bufferization.materialize_in_destination memref.subview(gm)
    After optimization, all these ops will be in same block with matmul and set core_type to CUBE.
 */
bool FixpipeOptPass::isFixpipeMulfPattern(Operation *mulfOp, SmallVector<Operation *> &matchedOps)
{
    Value mulfResult = mulfOp->getResult(0);
    if (!mulfResult.hasOneUse()) {
        LOG_DEBUG("Mulf not only one user, NOT match.");
        return false;
    }
    auto maybeExtract = *mulfResult.getUsers().begin();
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
    
    matchedOps.push_back(mulfOp);
    matchedOps.push_back(extractSliceOp);
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
bool FixpipeOptPass::matchFixpipePattern(linalg::MatmulOp matmulOp, SmallVector<Operation *> &matchedOps)
{
    LOG_DEBUG("Check matmul op: " << *matmulOp);
    Value matmulResult = matmulOp.getResult(0);
    if (!matmulResult.hasOneUse()) {
        LOG_DEBUG("Matmul not only one user, NOT match.");
        return false;
    }
    matchedOps.push_back(matmulOp);

    auto matmulUser = *matmulResult.getUsers().begin();

    if (isValidTrunc(matmulUser)) {
        if (isFixpipeCastPattern(matmulUser, matchedOps)) {
            return true;
        }
    }
    else if (isValidMul(matmulUser, matmulResult, matchedOps))
    {
        if (isFixpipeMulfPattern(matmulUser, matchedOps)) {
            return true;
        }
    } else {
        LOG_DEBUG("Cannot find valid consumer op (trunc or mulf), NOT match.");
        return false;
    }

    return false;
}

bool FixpipeOptPass::matchAfterForFixpipePattern(scf::ForOp forOp, SmallVector<Operation *> &matchedOps, CVPipeline::ComputeBlockIdManager &bm)
{
    LOG_DEBUG("Check for op: " << *forOp);
    matchedOps.push_back(forOp);
    auto ret = false;
    for (auto result : forOp.getResults()) {
        if (!isCubeResult(forOp, result)) {
            continue;
        }
        if (!result.hasOneUse()) {
            continue;
        }
        auto matmulUser = *result.getUsers().begin();
        if (isValidTrunc(matmulUser)) {
            if (isFixpipeCastPattern(matmulUser, matchedOps)) {
                ret = true;
                // add subview's scalar into matchedOps for later use in pattern applying.
                std::queue<Value> q;
                DenseSet<Value> visited;
                for (auto op: matchedOps) {
                    if( auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(op)) {
                        for (auto operand : viewLikeOp->getOperands()) {
                            if (isScalarLike(operand)) {
                                q.push(operand);
                                visited.insert(operand);
                            }
                        }
                    }
                }
                while(!q.empty()) {
                    auto val = q.front();
                    q.pop();
                    auto defOp = val.getDefiningOp();
                    if (defOp) {

                        if(defOp->getBlock() == forOp->getBlock() && bm.getBlockIdByOp(defOp) == bm.getBlockIdByOp(matmulUser)) {
                            matchedOps.push_back(defOp);
                        }
                        for (auto operand : defOp->getOperands()) {
                            if (isScalarLike(operand) && !visited.contains(operand)) {
                                q.push(operand);
                                visited.insert(operand);
                            }
                        }
                    }
                }
                return true;
            }
        } else {
            LOG_DEBUG("Cannot find valid forop's consumer op (trunc), NOT match.");
            continue;
        }
    }
    return ret;

}

bool FixpipeOptPass::applyFixpipeOpt(SmallVector<Operation *> &matchedOps,
                                     const CVPipeline::MemoryDependenceGraph &memGraph,
                                     CVPipeline::ComputeBlockIdManager &bm)
{
    // If there are no cycle in Compute Block level, we apply:
    // 1. Change block_id to the matmul's block id
    // 2. Change core_type to CUBE.
    Operation *matmulOp = matchedOps[0];
    int targetBlockId = bm.getBlockIdByOp(matmulOp);
    auto block = matmulOp->getBlock();

    if (willCreateCycle(matchedOps, block, memGraph, targetBlockId, bm).value_or(true)) {
        return false;
    }
    for (Operation *op : matchedOps) {
        if (isa<scf::ForOp>(matmulOp) && op->isBeforeInBlock(matmulOp)) {
            continue;
        }
        bm.updateBlockId(op, targetBlockId);
    }
    for (Operation *op : matchedOps) {
        op->setAttr(CVPipeline::kCoreType, StringAttr::get(op->getContext(), "CUBE"));
    }
    return true;
}

bool FixpipeOptPass::hasCubeCoreType(scf::ForOp forOp)
{
    auto coreTypeAttr = forOp->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
    if (!coreTypeAttr) {
        return false;
    }
    auto coreTypes = splitStringByComma(coreTypeAttr.getValue());
    for (const auto &coreType : coreTypes) {
        if (coreType == "CUBE") {
            return true;
        }
    }
    return false;
}

bool FixpipeOptPass::isCubeResult(scf::ForOp forOp, Value result)
{
    auto coreTypeAttr = forOp->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
    if (!coreTypeAttr) {
        return false;
    }
    auto coreTypes = splitStringByComma(coreTypeAttr.getValue());
    auto results = forOp.getResults();
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] == result) {
            if (i < coreTypes.size() && coreTypes[i] == "CUBE") {
                return true;
            }
            return false;
        }
    }
    return false;
}

void FixpipeOptPass::getDependentDialects(DialectRegistry &registry) const
{
    registry
        .insert<hivm::HIVMDialect>();
}

void FixpipeOptPass::runOnOperation()
{
    ModuleOp module = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);
    LOG_DEBUG("== FixpipeOpt Pass Start ==\n");
    LOG_DEBUG(module);

    SmallVector<SmallVector<Operation *>> allMatchedPatterns;
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    module.walk([&](linalg::MatmulOp matmulOp) {
        SmallVector<Operation *> matchedOps;
        if (matchFixpipePattern(matmulOp, matchedOps)) {
            allMatchedPatterns.push_back(matchedOps);
        }
    });

    module.walk([&](scf::ForOp forOp) {
        if (hasCubeCoreType(forOp)) {
            if(bm.markOpBlockId(forOp).failed()) {
                LOG_DEBUG("ForOp has been marked with block id, skip. forOp=" << *forOp);
                return;
            }
            SmallVector<Operation *> matchedOps;
            if (matchAfterForFixpipePattern(forOp, matchedOps, bm)) {
                allMatchedPatterns.push_back(matchedOps);
            }
        }
    });

    

    LOG_DEBUG("== Found " << allMatchedPatterns.size() << " fixpipe patterns ==\n");

    for (auto &matchedOps : allMatchedPatterns) {
        if (!applyFixpipeOpt(matchedOps, memDepGraph, bm)) {
            for (Operation *op : matchedOps) {
                LOG_DEBUG("Cannot set block id for op: " << *op);
            }
            LOG_DEBUG("Cannot set one Block Id, may be because cycle");
        }
    }

    // aaaa
    module.walk([&](scf::ForOp forOp) {
        if (hasCubeCoreType(forOp)) {
            // delete forOp's block_id attr
            // forOp->removeAttr(CVPipeline::kBlockId);
            // change forOp's core_type attr to it's yieldOp's core_type attr
            auto yieldOp = forOp.getBody()->getTerminator();
            auto yieldCoreTypeAttr = yieldOp->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
            if (yieldCoreTypeAttr) {
                forOp->setAttr(CVPipeline::kCoreType, yieldCoreTypeAttr);
            }
        }
    });

    LOG_DEBUG("== FixpipeOpt Pass Complete ==\n");
}

std::unique_ptr<OperationPass<ModuleOp>> createFixpipeOptPass()
{
    return std::make_unique<FixpipeOptPass>();
}

} // namespace triton
} // namespace mlir