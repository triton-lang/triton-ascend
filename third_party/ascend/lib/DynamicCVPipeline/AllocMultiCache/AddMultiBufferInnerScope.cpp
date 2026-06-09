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

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include <climits>
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferInnerScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace hivm;
using namespace annotation;
using namespace triton;
using namespace CVPipeline;

using BufferPair = std::pair<Value, Value>;
using BufferMap = DenseMap<Value, SmallVector<BufferPair>>;

// Buffer count constants
constexpr int kBufferCountOne = 1;

namespace mlir {
namespace triton {

// Check if forOp has main_loop attribute
static bool hasMainLoopAttr(scf::ForOp forOp)
{
    if (forOp->hasAttr(kMainLoop)) {
        return true;
    }
    if (auto *term = forOp.getBody()->getTerminator())
        return term->hasAttr(kMainLoop);
    return false;
}

// Collect main_loop forOps in a single block
static int collectMainLoopsInBlock(Block &block, SmallVector<scf::ForOp> &mainLoopForOps)
{
    int count = 0;
    for (Operation &op : block) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            if (hasMainLoopAttr(forOp)) {
                mainLoopForOps.push_back(forOp);
                count++;
            }
        }
    }
    return count;
}

// Recursively collect main_loop forOps, returns count of collected items
static int collectMainLoopsRecursively(Region &region, SmallVector<scf::ForOp> &mainLoopForOps)
{
    int totalCount = 0;
    for (Block &block : region) {
        totalCount += collectMainLoopsInBlock(block, mainLoopForOps);
        for (Operation &op : block) {
            for (auto &nestedRegion : op.getRegions())
                totalCount += collectMainLoopsRecursively(nestedRegion, mainLoopForOps);
        }
    }
    return totalCount;
}

struct InnerBlockInfo {
    Value blockId;
    SmallVector<Operation *> ops;
};

// Get ssbuffer block_id attribute from op, returns INT_MIN if not found
static int getSsbufferId(Operation *op)
{
    if (auto idAttr = op->getAttrOfType<IntegerAttr>(kBlockId))
        return idAttr.getInt();
    return INT_MIN;
}

void collectNestedOps(Block *block, SmallVector<Operation *> &ops)
{
    for (auto &op : *block) {
        ops.push_back(&op);
        for (auto &region : op.getRegions())
            for (auto &innerBlock : region)
                collectNestedOps(&innerBlock, ops);
    }
}

// Get priority of forOp (lower value means higher priority)
// Priority order: main_loop (1) > block_id (2) > iter_args (3) > none (0)
// This is used to select the most relevant main loop when multiple candidates exist
static int getForOpPriority(scf::ForOp f)
{
    constexpr int priorityMainLoop = 1;
    constexpr int priorityBlockId = 2;
    constexpr int priorityIterArgs = 3;

    // Check if forOp itself has main_loop attribute
    bool hasMainloop = f->hasAttr(kMainLoop);
    bool bodyHasMainloop = false;
    bool bodyHasBlockId = false;

    // Check terminator for main_loop and block_id attributes
    if (auto *term = f.getBody()->getTerminator()) {
        bodyHasMainloop = term->hasAttr(kMainLoop);
        bodyHasBlockId = term->getAttrOfType<IntegerAttr>(kBlockId) != nullptr;
    }

    bool opHasBlockId = f->getAttrOfType<IntegerAttr>(kBlockId) != nullptr;
    bool hasIterArgs = f.getNumResults() > 0 || !f.getInitArgs().empty();

    if (hasMainloop || bodyHasMainloop) {
        return priorityMainLoop;
    }
    if (opHasBlockId || bodyHasBlockId) {
        return priorityIterArgs;
    }
    if (hasIterArgs) {
        return priorityIterArgs;
    }
    return 0;
}

scf::ForOp findMainloopInScope(scope::ScopeOp scope)
{
    SmallVector<Operation *> allOps;
    collectNestedOps(&scope.getBodyRegion().front(), allOps);

    scf::ForOp mainLoopForOp;
    int bestPriority = INT_MAX;

    for (Operation *op : allOps) {
        auto f = dyn_cast<scf::ForOp>(op);
        if (!f)
            continue;

        int priority = getForOpPriority(f);
        if (priority > 0 && priority < bestPriority) {
            mainLoopForOp = f;
            bestPriority = priority;
        }
    }
    return mainLoopForOp;
}

// Collect a single dependency value to depValueMap
static void collectDepValue(Value operand, Block *body, int currentBlockId, DenseMap<Value, int> &outputToBlockId,
                            DenseMap<Value, SmallVector<Value>> &depValueMap, Value groupKey)
{
    if (auto barg = dyn_cast<BlockArgument>(operand)) {
        if (barg.getOwner() == body && !llvm::is_contained(depValueMap[groupKey], barg))
            depValueMap[groupKey].push_back(barg);
        return;
    }

    if (outputToBlockId.count(operand) && outputToBlockId[operand] != currentBlockId &&
        !llvm::is_contained(depValueMap[groupKey], operand))
        depValueMap[groupKey].push_back(operand);
}

// Recursively find nested main_loop
static scf::ForOp findNestedMainloopInForOp(scf::ForOp forOp)
{
    SmallVector<Operation *> allOps;
    collectNestedOps(forOp.getBody(), allOps);

    for (Operation *op : allOps) {
        auto nestedFor = dyn_cast<scf::ForOp>(op);
        if (!nestedFor)
            continue;
        if (nestedFor->hasAttr(kMainLoop))
            return nestedFor;
    }
    return {};
}

bool isInsideMainLoopForOp(Operation *op)
{
    Operation *parent = op->getParentOp();
    if (!parent) {
        return false;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        return forOp->hasAttr(kMainLoop);
    }
    return false;
}

bool isInsideMainLoopForOpTraverse(Operation *op)
{
    Operation *parent = op->getParentOp();
    while (parent) {
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            if (forOp->hasAttr(kMainLoop)) {
                return true;
            }
        }
        parent = parent->getParentOp();
    }
    return false;
}

// Collect all ops with ssbuffer.id from allOps, grouped by id
// Returns 0=success, -1=invalid negative block ID from upstream pass
static int groupOpsBySsbufferId(SmallVector<Operation *> &allOps,
                                llvm::MapVector<int, SmallVector<Operation *>> &opsById)
{
    llvm::MapVector<Value, Operation *> opsByValue;
    for (Operation *op : allOps) {
        int id = getSsbufferId(op);
        if (id == INT_MIN)
            continue; // Attribute doesn't exist, skip
        for (auto res : op->getResults())
            opsByValue[res] = op;
    }
    for (auto &p : opsByValue) {
        int id = getSsbufferId(p.second);
        if (id == INT_MIN)
            continue;
        opsById[id].push_back(p.second);
    }
    return 0;
}

// Returns 0=success (including normal skip when blocks empty), -1=invalid negative block ID
static int collectInnerBlockInfo(scf::ForOp forOp, DenseMap<Value, InnerBlockInfo> &blocks,
                                 DenseMap<Value, SmallVector<Value>> &depValueMap,
                                 SmallVector<Operation *> &allOps)
{
    depValueMap.clear();
    Block *body = forOp.getBody();
    if (!body)
        return 0;

    collectNestedOps(body, allOps);

    llvm::MapVector<int, SmallVector<Operation *>> opsById;
    if (groupOpsBySsbufferId(allOps, opsById) != 0)
        return -1;
    if (opsById.empty())
        return 0;

    // Build mapping from output to block id
    DenseMap<Value, int> outputToBlockId;
    for (auto &p : opsById)
        for (Operation *op : p.second)
            for (auto res : op->getResults())
                outputToBlockId[res] = p.first;

    // Collect dependency values for each block
    for (auto &p : opsById) {
        Value groupKey = p.second.front()->getResult(0);
        InnerBlockInfo bi;
        bi.blockId = groupKey;
        bi.ops = p.second;
        blocks[groupKey] = bi;

        for (Operation *op : bi.ops)
            for (Value operand : op->getOperands())
                collectDepValue(operand, body, p.first, outputToBlockId, depValueMap, groupKey);
    }

    return 0;
}

// Check if a yieldOp is already processed in blocks
static bool isYieldAlreadyProcessed(scf::YieldOp yieldOp, DenseMap<Value, InnerBlockInfo> &blocks)
{
    for (auto &p : blocks) {
        if (llvm::is_contained(p.second.ops, yieldOp.getOperation())) {
            return true;
        }
    }
    return false;
}

// Process yield op that is not in blocks: add parent multi-region op as consumer
// Generic version: supports any op with >= 2 regions (scf.if, scf.while, etc.)
static void processYieldNotInBlocks(scf::YieldOp yieldOp, DenseMap<Value, InnerBlockInfo> &blocks,
                                    DenseMap<Value, SmallVector<Operation *>> &depUserMap)
{
    if (isYieldAlreadyProcessed(yieldOp, blocks))
        return;

    Operation *parentOp = yieldOp->getParentOp();
    // Generic: check if parent op has >= 2 regions
    if (!parentOp || parentOp->getNumRegions() < 2)
        return;

    for (Value operand : yieldOp->getOperands()) {
        // Add parent multi-region op as consumer for yield operands
        // This handles cases where depVal is only used in yield (not as direct operand)
        depUserMap[operand].push_back(parentOp);
    }
}

DenseMap<Value, SmallVector<Operation *>> buildDepUserMap(DenseMap<Value, InnerBlockInfo> &blocks,
                                                          SmallVector<Operation *> &allOps,
                                                          DenseMap<Value, SmallVector<Value>> &depValueMap)
{
    DenseMap<Value, SmallVector<Operation *>> depUserMap;

    // First pass: process operations in blocks
    for (auto &p : blocks)
        for (Operation *op : p.second.ops)
            for (Value operand : op->getOperands())
                depUserMap[operand].push_back(op);

    // Second pass: process yield operations that are not in blocks (e.g., INT_MIN block_id)
    // Generic version: supports any multi-region op's yield
    for (Operation *op : allOps) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
            processYieldNotInBlocks(yieldOp, blocks, depUserMap);
        }
    }

    return depUserMap;
}

SmallVector<Value> collectBufferValues(DenseMap<Value, SmallVector<Value>> &depValueMap)
{
    SmallVector<Value> valueList;
    SmallVector<Operation *> seenOps;

    for (auto &p : depValueMap) {
        for (Value depVal : p.second) {
            Operation *op = depVal.getDefiningOp();
            if (!op || llvm::is_contained(seenOps, op))
                continue;
            seenOps.push_back(op);

            auto shapedType = dyn_cast<ShapedType>(depVal.getType());
            if (!shapedType)
                continue;

            // Skip tensor::EmptyOp - it should only get dep_mark, not buffer allocation
            if (isa<tensor::EmptyOp>(op))
                continue;

            valueList.push_back(depVal);
        }
    }

    return valueList;
}

SmallVector<Value> collectScalarDeps(DenseMap<Value, SmallVector<Value>> &depValueMap,
                                     DenseMap<Value, SmallVector<Operation *>> &depUserMap)
{
    SmallVector<Value> scalarValueList;

    for (auto &p : depValueMap) {
        for (Value depVal : p.second) {
            if (isa<BlockArgument>(depVal))
                continue;

            Operation *depDefinedOp = depVal.getDefiningOp();
            if (!depDefinedOp)
                continue;

            if (isa<ShapedType>(depVal.getType())) {
                // tensor::EmptyOp should be treated like scalar, add dep_mark
                if (!isa<tensor::EmptyOp>(depDefinedOp))
                continue;
                // Check if definingOp's parentOp is a main_loop forOp
                auto *parentOp = depDefinedOp->getParentOp();
                if (!parentOp || !parentOp->hasAttr(kMainLoop))
                    continue;
            }

            auto userIt = depUserMap.find(depVal);
            if (userIt == depUserMap.end())
                continue;

            int producerId = getSsbufferId(depDefinedOp);
            if (producerId < 0)
                continue;

            SmallVector<Operation *> depUsers = userIt->second;
            bool hasCrossBlockUser = false;
            for (Operation *depUser : depUsers) {
                int userId = getSsbufferId(depUser);
                if (userId < 0 || userId != producerId) {
                    hasCrossBlockUser = true;
                    break;
                }
            }

            if (hasCrossBlockUser)
                scalarValueList.push_back(depVal);
        }
    }

    return scalarValueList;
}

// Compute iteration index: (iv - lb) / step, used for buffer selection in double buffering
static Value getIterCount(OpBuilder &builder, mlir::scf::ForOp forOp, Location loc, SmallVector<Operation *> *newOps,
                          int blockId = -1)
{
    auto i32Type = builder.getI32Type();
    Value iv = forOp.getInductionVar();
    Value lb = forOp.getLowerBound();
    Value step = forOp.getStep();
    Type ivType = iv.getType();

    // Check if lower bound is a constant zero
    bool lbIsZero = false;
    if (auto constOp = lb.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
            lbIsZero = (intAttr.getInt() == 0);
    }

    Value iterIdx;
    if (lbIsZero) {
        // Optimization: if lb is 0, use iv directly (or iv/step if step != 1)
        bool stepIsOne = false;
        if (auto constOp = step.getDefiningOp<mlir::arith::ConstantOp>())
            if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                stepIsOne = intAttr.getInt() == 1;
        if (stepIsOne) {
            iterIdx = iv;
        } else {
            iterIdx = builder.create<mlir::arith::DivUIOp>(loc, iv, step);
            if (newOps)
                newOps->push_back(iterIdx.getDefiningOp());
            if (blockId >= 0) {
                iterIdx.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
            }
        }
    } else {
        // General case: (iv - lb) / step
        Value diff = builder.create<mlir::arith::SubIOp>(loc, iv, lb);
        iterIdx = builder.create<mlir::arith::DivUIOp>(loc, diff, step);
        if (newOps) {
            newOps->push_back(diff.getDefiningOp());
            newOps->push_back(iterIdx.getDefiningOp());
        }
        if (blockId >= 0) {
            diff.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
            iterIdx.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
        }
    }

    // Cast to i32 if necessary
    if (ivType == i32Type)
        return iterIdx;

    Value result;
    constexpr int bits32 = 32;
    if (ivType.isIndex()) {
        result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
    } else if (auto intType = dyn_cast<mlir::IntegerType>(ivType)) {
        // Extend or truncate integer types to i32
        if (intType.getWidth() < bits32)
            result = builder.create<mlir::arith::ExtSIOp>(loc, i32Type, iterIdx);
        else if (intType.getWidth() > bits32)
            result = builder.create<mlir::arith::TruncIOp>(loc, i32Type, iterIdx);
        else
            return iterIdx;
    } else {
        result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
    }
    if (newOps)
        newOps->push_back(result.getDefiningOp());
    if (blockId >= 0) {
        result.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    }
    return result;
}

// yieldFn is used for normal op case, getNestedResults is used for nested if case
static void createConditionalYield(OpBuilder &builder, Location loc, bool hasResults,
                                 function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
                                 Operation *op, std::function<SmallVector<Value>()> getNestedResults = nullptr)
{
    if (hasResults && getNestedResults) {
        builder.create<mlir::scf::YieldOp>(loc, getNestedResults());
    } else if (hasResults && yieldFn && op) {
        builder.create<mlir::scf::YieldOp>(loc, yieldFn(builder, loc, op));
    } else {
        builder.create<mlir::scf::YieldOp>(loc);
    }
}

// Build if-else chain for N==2 (simple nested structure)
static int buildIfChainTwoBuffers(OpBuilder &builder, Location loc, Value indexVal, SmallVector<BufferPair> &buffers,
                                  SmallVector<Operation *> &newOps, SmallVector<Operation *> &outIfOps,
                                  function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
                                  function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
                                  mlir::TypeRange types, bool hasResults, int blockId)
{
    // Create condition: index == 0
    Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    Value firstCond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, indexVal, zero);
    auto firstIf = builder.create<mlir::scf::IfOp>(loc, types, firstCond, true, true);

    newOps.push_back(zero.getDefiningOp());
    newOps.push_back(firstCond.getDefiningOp());
    newOps.push_back(firstIf);
    outIfOps.push_back(firstIf);

    // Tag counter operations with block_id
    if (blockId >= 0) {
        zero.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
        firstCond.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    }

    // Then branch: use buffer[0]
    builder.setInsertionPointToStart(&firstIf.getThenRegion().front());
    Operation *op0 = createOpFn(builder, loc, buffers[0].second);
    if (!op0) return -1;
    newOps.push_back(op0);
    createConditionalYield(builder, loc, hasResults, yieldFn, op0, nullptr);

    // Else branch: use buffer[1]
    builder.setInsertionPointToStart(&firstIf.getElseRegion().front());
    Operation *op1 = createOpFn(builder, loc, buffers[1].second);
    if (!op1) return -1;
    newOps.push_back(op1);
    createConditionalYield(builder, loc, hasResults, yieldFn, op1, nullptr);

    builder.setInsertionPointAfter(firstIf);
    return 0;
}

// Build if-else chain for N>2 (if-else-if-else chain with proper yield passthrough)
static int buildIfChainMultiBuffers(OpBuilder &builder, Location loc, Value indexVal, SmallVector<BufferPair> &buffers,
                                    SmallVector<Operation *> &newOps, SmallVector<Operation *> &outIfOps,
                                    function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
                                    function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
                                    mlir::TypeRange types, bool hasResults, int blockId)
{
    int N = buffers.size();

    // Create rootIf (idx == 0)
    Value zeroVal = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    Value firstCond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, indexVal, zeroVal);
    if (blockId >= 0) {
        zeroVal.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
        firstCond.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    }
    newOps.push_back(zeroVal.getDefiningOp());
    newOps.push_back(firstCond.getDefiningOp());

    auto rootIf = builder.create<mlir::scf::IfOp>(loc, types, firstCond, true, true);
    if (!rootIf) return -1;
    newOps.push_back(rootIf);
    outIfOps.push_back(rootIf);

    // Then branch of rootIf: use buffer[0]
    builder.setInsertionPointToStart(&rootIf.getThenRegion().front());
    Operation *op0 = createOpFn(builder, loc, buffers[0].second);
    if (!op0) return -1;
    newOps.push_back(op0);
    createConditionalYield(builder, loc, hasResults, yieldFn, op0, nullptr);

    // Build the nested if chain in rootIf's else region
    Block *currentElseBlock = &rootIf.getElseRegion().front();

    for (int i = 1; i < N - 1; ++i) {
        // Set insertion to current else block
        builder.setInsertionPointToStart(currentElseBlock);

        // Create condition for idx == i
        Value iVal = builder.create<mlir::arith::ConstantIntOp>(loc, i, 32);
        Value cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, indexVal, iVal);
        if (blockId >= 0) {
            iVal.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
            cond.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
        }
        newOps.push_back(iVal.getDefiningOp());
        newOps.push_back(cond.getDefiningOp());

        // Create nested if for this level
        auto nestedIf = builder.create<mlir::scf::IfOp>(loc, types, cond, true, true);
        if (!nestedIf) return -1;
        newOps.push_back(nestedIf);

        // Then branch: use buffer[i]
        builder.setInsertionPointToStart(&nestedIf.getThenRegion().front());
        Operation *op = createOpFn(builder, loc, buffers[i].second);
        if (!op) return -1;
        newOps.push_back(op);
        createConditionalYield(builder, loc, hasResults, yieldFn, op, nullptr);

        // Set insertion to end of currentElseBlock and add yield
        builder.setInsertionPointToEnd(currentElseBlock);
        createConditionalYield(builder, loc, hasResults, yieldFn, nullptr, [&nestedIf]() { return nestedIf.getResults(); });

        // Move to nestedIf's else block for next iteration
        currentElseBlock = &nestedIf.getElseRegion().front();
    }

    // Final else: use buffer[N-1]
    builder.setInsertionPointToStart(currentElseBlock);
    Operation *opLast = createOpFn(builder, loc, buffers[N - 1].second);
    if (!opLast) return -1;
    newOps.push_back(opLast);
    createConditionalYield(builder, loc, hasResults, yieldFn, opLast, nullptr);

    builder.setInsertionPointAfter(rootIf);
    return 0;
}

// Build if-else chain for buffer selection: if (idx==0) -> buf[0] else ... else -> buf[N-1]
static int buildIfChain(OpBuilder &builder, Location loc, Value indexVal, SmallVector<BufferPair> &buffers,
                        SmallVector<Operation *> &newOps, SmallVector<Operation *> &outIfOps,
                        function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
                        function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
                        std::optional<mlir::TypeRange> resultTypes = std::nullopt, int blockId = -1)
{
    int N = buffers.size();
    auto types = resultTypes.value_or(mlir::TypeRange {});
    bool hasResults = !types.empty();

    if (N == 2) {
        return buildIfChainTwoBuffers(builder, loc, indexVal, buffers, newOps, outIfOps,
                                       createOpFn, yieldFn, types, hasResults, blockId);
    }
    return buildIfChainMultiBuffers(builder, loc, indexVal, buffers, newOps, outIfOps,
                                    createOpFn, yieldFn, types, hasResults, blockId);
}

// Compute buffer index: iterCount % N
static Value computeBufferIndex(OpBuilder &builder, mlir::scf::ForOp forOp, Location loc, int N,
                                SmallVector<Operation *> *newOps, int blockId = -1)
{
    Value iterCount = getIterCount(builder, forOp, loc, newOps, blockId);
    Value Nval = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
    Value bufIdx = builder.create<mlir::arith::RemSIOp>(loc, iterCount, Nval);
    if (newOps) {
        newOps->push_back(Nval.getDefiningOp());
        newOps->push_back(bufIdx.getDefiningOp());
    }
    // Tag counter operations with block_id
    if (blockId >= 0) {
        MLIRContext *ctx = builder.getContext();
        Nval.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
        bufIdx.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    }
    return bufIdx;
}

static SmallVector<Operation *> insertProducerLogic(OpBuilder &builder, Value depVal, SmallVector<BufferPair> &buffers,
                                                    mlir::scf::ForOp forOp)
{
    SmallVector<Operation *> newOps;
    int N = buffers.size();
    Location loc = depVal.getLoc();
    // Single buffer producer logic
    if (N == kBufferCountOne) {
        Operation *producerOp = builder.create<hivm::CopyOp>(
            loc, mlir::TypeRange{}, depVal, buffers[0].second);
        if (!producerOp)
            return newOps;
        newOps.push_back(producerOp);
        return newOps;
    }

    Value bufIdx = computeBufferIndex(builder, forOp, loc, N, &newOps);
    SmallVector<Operation *> dummyOutIfOps;
    if (buildIfChain(
        builder, loc, bufIdx, buffers, newOps, dummyOutIfOps,
        [&](OpBuilder &b, Location l, Value buffer) -> Operation* {
            return b.create<hivm::CopyOp>(
                l, mlir::TypeRange{}, depVal, buffer);
        },
        nullptr) != 0) {
    return {};
    }
    return newOps;
}

// Handle consumer when N==1 (directly return buffer)
static Operation *handleSingleBufferConsumer(OpBuilder &builder, Location loc, SmallVector<BufferPair> &buffers)
{
    auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
    auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
    return builder.create<mlir::bufferization::ToTensorOp>(loc, tensorType, buffers[0].second,
                                                           mlir::UnitAttr::get(builder.getContext()),
                                                           mlir::UnitAttr::get(builder.getContext()));
}

// Helper function to create ToTensorOp
static mlir::bufferization::ToTensorOp createToTensorOp(OpBuilder &builder, Location loc, mlir::Type tensorType,
                                                        Value buffer)
{
    return builder.create<mlir::bufferization::ToTensorOp>(
        loc, tensorType, buffer, mlir::UnitAttr::get(builder.getContext()), mlir::UnitAttr::get(builder.getContext()));
}

static int insertConsumerLogic(OpBuilder &builder, Value depVal, SmallVector<BufferPair> &buffers,
                               mlir::scf::ForOp forOp, SmallVector<Operation *> &outIfOps, int groupId = -1,
                               int blockId = -1)
{
    SmallVector<Operation *> newOps;
    int N = buffers.size();
    Location loc = builder.getInsertionPoint()->getLoc();

    if (N == kBufferCountOne) {
        Operation *consumerOp = handleSingleBufferConsumer(builder, loc, buffers);
        outIfOps.push_back(consumerOp);
        if (groupId >= 0) {
            consumerOp->setAttr("ssbuffer.intraDeps", builder.getI32ArrayAttr({groupId, 0}));
        }
        return 0;
    }

    Value readIdx = computeBufferIndex(builder, forOp, loc, N, &newOps, blockId);
    auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
    auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
    mlir::TypeRange resultTypes(tensorType);
    int ret = buildIfChain(
        builder, loc, readIdx, buffers, newOps, outIfOps,
        [&](OpBuilder &b, Location l, Value buffer) -> Operation* {
            return createToTensorOp(b, l, tensorType, buffer);
        },
        [&](OpBuilder &b, Location l, Operation *op) -> Value {
            return cast<mlir::bufferization::ToTensorOp>(op).getResult();
        },
        resultTypes, blockId);
    if (ret != 0) {
        return ret;
    }
    if (groupId >= 0 && !outIfOps.empty()) {
        outIfOps.front()->setAttr("ssbuffer.intraDeps", builder.getI32ArrayAttr({groupId, 0}));
    }
    return 0;
}

static void addBlockAttrForOps(SmallVector<Operation *> &newOps, int blockId, OpBuilder &builder)
{
    auto attr = builder.getI32IntegerAttr(blockId);
    for (auto *op : newOps)
        op->setAttr(kBlockId, attr);
}

// Add dep_mark attribute to operation
static void addDepMarkAttr(Operation *op, int depMark, OpBuilder &builder)
{
    if (auto existingAttr = op->getAttrOfType<mlir::ArrayAttr>("ssbuffer.dep_mark")) {
        SmallVector<int> marks;
        for (auto attr : existingAttr)
            marks.push_back(cast<mlir::IntegerAttr>(attr).getInt());
        marks.push_back(depMark);
        op->setAttr("ssbuffer.dep_mark", builder.getI32ArrayAttr(marks));
    } else {
        op->setAttr("ssbuffer.dep_mark", builder.getI32ArrayAttr({depMark}));
    }
}

static void addIntraBufferAttr(SmallVector<Operation *> &ops, OpBuilder &builder)
{
    for (auto *op : ops) {
        if (isa<scf::IfOp>(op) || isa<hivm::CopyOp>(op) || isa<bufferization::ToTensorOp>(op)) {
            op->setAttr(kIntraBuffer, builder.getUnitAttr());
        }
    }
}

// Collect cross-block user operations
static SmallVector<Operation *> collectCrossBlockUsers(Value depVal, int producerId,
                                                       DenseMap<Value, SmallVector<Operation *>> &depUserMap)
{
    SmallVector<Operation *> crossBlockUsers;

    auto userIt = depUserMap.find(depVal);
    if (userIt == depUserMap.end())
        return crossBlockUsers;

    for (Operation *depUser : userIt->second) {
        int userId = getSsbufferId(depUser);
        if ((userId < 0 || userId != producerId) && isInsideMainLoopForOpTraverse(depUser))
            crossBlockUsers.push_back(depUser);
    }
    return crossBlockUsers;
}

// Insert buffer selection logic (scf.if + to_tensor) at the start of the given region.
// Returns the scf::IfOp that performs the buffer selection, or nullptr on failure.
static Operation *insertBufferSelectionInRegion(OpBuilder &builder, Region &region, Location loc,
                                              Value depVal, SmallVector<BufferPair> &buffers,
                                              mlir::scf::ForOp forOp, int blockId)
{
    auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
    auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());

    // Insert at the start of the region
    builder.setInsertionPointToStart(&region.front());

    // Compute buffer index
    Value readIdx = computeBufferIndex(builder, forOp, loc, buffers.size(), nullptr, blockId);

    // Build buffer selection if-else chain
    SmallVector<Operation *> newIfOps;
    SmallVector<Operation *> outIfOps;
    int ret = buildIfChain(
        builder, loc, readIdx, buffers, newIfOps, outIfOps,
        [&](OpBuilder &b, Location l, Value buffer) -> Operation* {
            return createToTensorOp(b, l, tensorType, buffer);
        },
        [&](OpBuilder &b, Location l, Operation *op) -> Value {
            return cast<mlir::bufferization::ToTensorOp>(op).getResult();
        },
        tensorType, blockId);
    if (ret != 0)
        return nullptr;

    if (newIfOps.empty())
        return nullptr;

    // Tag all operations in newIfOps with block_id
    for (auto *op : newIfOps) {
        op->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
        op->setAttr(kIntraBuffer, builder.getUnitAttr());
    }

    // The main scf.if is the first one in outIfOps
    return outIfOps.front();
}

static void markScalarDeps(SmallVector<Value> &scalarValueList, DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                           OpBuilder &builder, int startDepMark)
{
    int nextDepMark = startDepMark;

    for (Value depVal : scalarValueList) {
        Operation *depDefinedOp = depVal.getDefiningOp();
        if (!depDefinedOp)
            continue;

        if (!isInsideMainLoopForOp(depDefinedOp))
            continue;

        int producerId = getSsbufferId(depDefinedOp);
        if (producerId < 0)
            continue;
        auto crossBlockUsers = collectCrossBlockUsers(depVal, producerId, depUserMap);
        if (crossBlockUsers.empty())
            continue;

        int depMark = nextDepMark++;
        // Add depmark to producer
        addDepMarkAttr(depDefinedOp, depMark, builder);
        // Add depmark to consumer
        for (Operation *depUser : crossBlockUsers) {
            addDepMarkAttr(depUser, depMark, builder);
        }
    }
}

// Check if depUser is a multi-region op and depVal is not a direct operand
// Returns true when op was added as consumer due to yield using depVal
// Generic version: supports any op with >= 2 regions (scf.if, scf.while, etc.)
static bool isMultiRegionConsumerFromYield(Operation *depUser, Value depVal)
{
    // Check if op has >= 2 regions (generic for any multi-region op)
    if (depUser->getNumRegions() < 2)
        return false;

    for (OpOperand &operand : depUser->getOpOperands()) {
        if (operand.get() == depVal)
            return false;  // depVal is a direct operand
    }
    return true;  // depVal comes from yield
}

// Process normal consumer for a block: generate one buffer selection and share among all ops in the block
static int processNormalConsumerBlock(OpBuilder &consumedBuilder, Value depVal, SmallVector<BufferPair> &buffers,
                                     mlir::scf::ForOp mainLoopForOp, SmallVector<Operation *> &opsInBlock,
                                     int userBlockId, int groupId, OpBuilder &globalBuilder)
{
    SmallVector<Operation *> resultIfOps;
    int ret = insertConsumerLogic(consumedBuilder, depVal, buffers, mainLoopForOp, resultIfOps, groupId, userBlockId);
    if (ret != 0)
        return -1;

    if (resultIfOps.empty())
        return 0;

    addBlockAttrForOps(resultIfOps, userBlockId, globalBuilder);
    if (buffers.size() > kBufferCountOne) {
        for (auto *op : resultIfOps) {
            if (isa<scf::IfOp>(op)) {
                op->setAttr(kIntraBuffer, globalBuilder.getUnitAttr());
            }
        }
    } else {
        addIntraBufferAttr(resultIfOps, globalBuilder);
    }

    Operation *resultIf = resultIfOps.back();
    Value selectedBuffer = resultIf->getResult(0);

    // Replace operands of all ops in this block
    for (Operation *opInBlock : opsInBlock) {
        for (OpOperand &use : opInBlock->getOpOperands()) {
            if (use.get() == depVal)
                use.set(selectedBuffer);
        }
    }
    return 0;
}

// Handle all regions (except region 0) when depVal is used in that region's yield
// Generic version: supports any op with >= 2 regions (scf.if, scf.while, etc.)
// For scf.if: handles else region (region index 1)
// For scf.while: handles after region (region index 1)
// For any op with 3+ regions: handles all regions from index 1 onwards
static int processMultiRegionAllYields(OpBuilder &consumedBuilder, Value depVal, SmallVector<BufferPair> &buffers,
                                       mlir::scf::ForOp mainLoopForOp, Operation *depUser, int userBlockId,
                                       int groupId)
{
    // Generic: check if op has >= 2 regions
    if (depUser->getNumRegions() < 2)
        return 0;

    // Iterate through all regions (from region 1 onwards, excluding region 0)
    for (size_t i = 1; i < depUser->getNumRegions(); ++i) {
        Region &region = depUser->getRegion(i);
        if (region.empty())
            continue;

        auto yieldOp = dyn_cast<scf::YieldOp>(region.back().getTerminator());
        if (!yieldOp)
            continue;

        for (OpOperand &operand : yieldOp->getOpOperands()) {
            if (operand.get() != depVal)
                continue;

            Operation *selectIf = insertBufferSelectionInRegion(
                consumedBuilder, region, yieldOp.getLoc(),
                depVal, buffers, mainLoopForOp, userBlockId);
            if (!selectIf)
                return -1;

            if (groupId >= 0) {
                selectIf->setAttr("ssbuffer.intraDeps", consumedBuilder.getI32ArrayAttr({groupId, 0}));
            }

            operand.set(selectIf->getResult(0));
            return 0;  // Only handle one
        }
    }
    return 0;
}

// Process producer and consumer for a single dependency value
static int processDepVal(Value depVal, mlir::scf::ForOp mainLoopForOp, BufferMap &bufferMap,
                         DenseMap<Value, SmallVector<Operation *>> &depUserMap, OpBuilder &globalBuilder,
                         int producerId, int groupId)
{
    Operation *depDefinedOp = depVal.getDefiningOp();
    if (!depDefinedOp)
        return 0;

    SmallVector<BufferPair> &buffers = bufferMap[depVal];

    auto userIt = depUserMap.find(depVal);
    if (userIt == depUserMap.end())
        return 0;
    SmallVector<Operation *> depUsers = userIt->second;

    // Create producer
    OpBuilder producedBuffers(mainLoopForOp.getContext());
    producedBuffers.setInsertionPointAfter(depDefinedOp);
    SmallVector<Operation *> producerNewOps = insertProducerLogic(producedBuffers, depVal, buffers, mainLoopForOp);
    addBlockAttrForOps(producerNewOps, producerId, globalBuilder);
    if (buffers.size() > kBufferCountOne) {
        for (auto *op : producerNewOps) {
            if (isa<scf::IfOp>(op)) {
                op->setAttr(kIntraBuffer, globalBuilder.getUnitAttr());
            }
        }
    } else {
        addIntraBufferAttr(producerNewOps, globalBuilder);
    }

    // Single pass: process both normal ops and multi-region ops
    // For normal ops: generate one buffer selection per block_id and share among all ops in that block
    // For multi-region ops: process independently
    DenseMap<int, SmallVector<Operation *>> opsByBlockId;
    DenseMap<int, Value> processedBlockSelections;

    for (Operation *depUser : depUsers) {
        int userBlockId = getSsbufferId(depUser);
        if (userBlockId < 0 || userBlockId == producerId)
            continue;

        if (isMultiRegionConsumerFromYield(depUser, depVal)) {
            // Multi-region op: process independently
            OpBuilder consumedBuilder(mainLoopForOp.getContext());
            consumedBuilder.setInsertionPoint(depUser);

            if (int ret = processMultiRegionAllYields(consumedBuilder, depVal, buffers, mainLoopForOp,
                                                     depUser, userBlockId, groupId))
                return ret;
        } else {
            // Normal op: collect by block_id for batch processing
            opsByBlockId[userBlockId].push_back(depUser);
        }
    }

    // For each block_id, generate only one set of buffer selection and share among all ops
    for (auto &blockPair : opsByBlockId) {
        int userBlockId = blockPair.first;
        SmallVector<Operation *> &opsInBlock = blockPair.second;
        if (opsInBlock.empty())
            continue;

        // Insert buffer selection at the position of the first op
        Operation *firstOp = opsInBlock.front();
        OpBuilder consumedBuilder(mainLoopForOp.getContext());
        consumedBuilder.setInsertionPoint(firstOp);

        if (int ret = processNormalConsumerBlock(consumedBuilder, depVal, buffers, mainLoopForOp,
                                               opsInBlock, userBlockId, groupId, globalBuilder))
            return ret;
    }

    return 0;
}

// Process cross-block tensor dependencies for double buffering
static int processTensorDependencies(mlir::scf::ForOp mainLoopForOp, DenseMap<Value, InnerBlockInfo> &blocks,
                                     DenseMap<Value, SmallVector<Value>> &depValueMap,
                                     DenseMap<Value, SmallVector<Operation *>> &depUserMap, BufferMap &bufferMap,
                                     OpBuilder &globalBuilder, int &groupId)
{
    SmallVector<Operation *> seenOps;

    for (auto &blockPair : blocks) {
        Value blockKey = blockPair.first;
        auto depIt = depValueMap.find(blockKey);
        if (depIt == depValueMap.end())
            continue;

        SmallVector<Value> &depValues = depIt->second;

        for (Value depVal : depValues) {
            // Skip if already processed
            if (llvm::is_contained(seenOps, depVal.getDefiningOp()))
                continue;
            seenOps.push_back(depVal.getDefiningOp());

            // Validate dependency value (skip BlockArgument, null definingOp, non-ShapedType)
            if (isa<BlockArgument>(depVal) || !depVal.getDefiningOp() || !isa<ShapedType>(depVal.getType()))
                continue;

            // Skip tensor::EmptyOp - it should only get dep_mark, not buffer allocation
            if (isa<tensor::EmptyOp>(depVal.getDefiningOp()))
                continue;

            // Check if definingOp's parentOp is the main_loop forOp
            auto *parentOp = depVal.getDefiningOp()->getParentOp();
            if (parentOp != mainLoopForOp.getOperation())
                continue;

            auto userIt = depUserMap.find(depVal);
            if (userIt == depUserMap.end())
                continue;
            SmallVector<Operation *> depUsers = userIt->second;

            int producerId = getSsbufferId(depVal.getDefiningOp());
            if (producerId < 0)
                continue; // Skip if producer has no ssbuffer attribute

            // Check if all users are in the same block
            bool allUsersSameBlock = true;
            for (Operation *depUser : depUsers) {
                int userId = getSsbufferId(depUser);
                if (userId < 0 || userId != producerId) {
                    allUsersSameBlock = false;
                    break;
                }
            }
            if (allUsersSameBlock)
                continue;

            // Process cross-block dependency with double buffering
            if (processDepVal(depVal, mainLoopForOp, bufferMap, depUserMap, globalBuilder, producerId, groupId) != 0)
                return -1;
            groupId++;
        }
    }
    return 0;
}

static BufferMap insertBuffersBeforeFor(mlir::scf::ForOp forOp, SmallVector<Value> &valueList, OpBuilder &builder,
                                        int groupId)
{
    BufferMap bufferMap;
    Block *parentBlock = forOp->getBlock();
    OpBuilder insertedBuffers(builder.getContext());
    insertedBuffers.setInsertionPoint(parentBlock, forOp->getIterator());

    using BufferCountManager = mlir::triton::BufferCountManager;
    int bufNum = BufferCountManager::getInstance().getBufferCountByType(BufferCountManager::DepType::IntraCore);

    for (Value depVal : valueList) {
        ShapedType shapedType = cast<ShapedType>(depVal.getType());
        Type elemType = shapedType.getElementType();
        AddressSpace addrSpace = AddressSpace::UB;

        SmallVector<BufferPair> buffers;
        for (int i = 0; i < bufNum; ++i) {
            MemRefType memrefType = MemRefType::get(shapedType.getShape(), elemType, MemRefLayoutAttrInterface {},
                                                    AddressSpaceAttr::get(insertedBuffers.getContext(), addrSpace));

            auto allocOp = insertedBuffers.create<memref::AllocOp>(forOp.getLoc(), memrefType);

            auto genericType = MemRefType::get(shapedType.getShape(), elemType, MemRefLayoutAttrInterface {}, 0u);

            auto casted =
                insertedBuffers.create<memref::MemorySpaceCastOp>(forOp.getLoc(), genericType, allocOp.getResult());

            casted->setAttr("ssbuffer.intraDeps", insertedBuffers.getI32ArrayAttr({groupId, 1}));

            buffers.push_back({casted.getResult(), casted.getResult()});
        }

        bufferMap[depVal] = buffers;
        groupId++;
    }

    return bufferMap;
}

static bool hasMemrefDepValue(DenseMap<Value, SmallVector<Value>> &depValueMap)
{
    for (auto &p : depValueMap) {
        for (Value depVal : p.second) {
            if (isa<MemRefType>(depVal.getType()))
                return true;
        }
    }
    return false;
}

static int addInnerMultiBuffer(mlir::scf::ForOp mainLoopForOp, OpBuilder &builder, scope::ScopeOp vectorScope,
                               int &groupId)
{
    DenseMap<Value, InnerBlockInfo> blocks;
    DenseMap<Value, SmallVector<Value>> depValueMap;
    SmallVector<Operation *> allOps;
    if (collectInnerBlockInfo(mainLoopForOp, blocks, depValueMap, allOps) != 0)
        return -1;

    if (blocks.empty())
        return -1;

    // If memref-type dependency values exist, skip double buffer processing
    if (hasMemrefDepValue(depValueMap)) {
        vectorScope->setAttr("ssbuffer.skip", builder.getUnitAttr());
        return 0;
    }

    auto depUserMap = buildDepUserMap(blocks, allOps, depValueMap);

    auto valueList = collectBufferValues(depValueMap);
    auto bufferMap = insertBuffersBeforeFor(mainLoopForOp, valueList, builder, groupId);

    auto scalarValueList = collectScalarDeps(depValueMap, depUserMap);

    OpBuilder globalBuilder(mainLoopForOp.getContext());
    markScalarDeps(scalarValueList, depUserMap, globalBuilder, 1);

    if (processTensorDependencies(mainLoopForOp, blocks, depValueMap, depUserMap, bufferMap, globalBuilder, groupId) !=
        0) {
        return -1;
    }

    return 0;
}

void AddMultiBufferInnerScopePass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<mlir::annotation::AnnotationDialect, mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                    mlir::hivm::HIVMDialect, mlir::scope::ScopeDialect>();
}

void AddMultiBufferInnerScopePass::runOnOperation()
{
    auto module = getOperation();
    OpBuilder builder(module.getContext());

    LDBG("Enter pass.");

    module.walk([&](scope::ScopeOp scope) -> WalkResult {
        // Step 1: Check if scope has coreType attribute
        auto coreTypeAttr = scope->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
        if (!coreTypeAttr)
            return WalkResult::advance();

        // Step 2: Check if core type is VECTOR
        hivm::TCoreType coreType = coreTypeAttr.getTcoretype();
        if (coreType != hivm::TCoreType::VECTOR) {
            LDBG("Not vector scope");
            return WalkResult::advance();
        }

        // Step 3: Collect all forOps with main_loop attribute
        SmallVector<scf::ForOp> mainLoopForOps;
        int foundCount = collectMainLoopsRecursively(scope.getBodyRegion(), mainLoopForOps);
        if (foundCount < 0) {
            LDBG("collectMainLoopsRecursively failed");
            signalPassFailure();
            return WalkResult::interrupt();
        }
        if (foundCount == 0)
            return WalkResult::advance();

        // Step 4: Process each main_loop forOp
        int groupId = 0;
        for (scf::ForOp mainLoopForOp : mainLoopForOps) {
            scf::ForOp nestedMainloop = findNestedMainloopInForOp(mainLoopForOp);
            if (nestedMainloop) {
                LDBG("Nested main_loop found, this is not allowed");
                signalPassFailure();
                return WalkResult::interrupt();
            }
            if (addInnerMultiBuffer(mainLoopForOp, builder, scope, groupId) != 0) {
                LDBG("addInnerMultiBuffer failed");
                signalPassFailure();
                return WalkResult::interrupt();
            }
        }

        return WalkResult::advance();
    });

    LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferInnerScopePass()
{
    return std::make_unique<AddMultiBufferInnerScopePass>();
}

void registerAddMultiBufferInnerScopePasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createAddMultiBufferInnerScopePass(); });
}

} // namespace triton
} // namespace mlir
