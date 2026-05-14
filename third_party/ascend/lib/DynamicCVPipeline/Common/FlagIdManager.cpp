/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "llvm/Support/Debug.h"

#include "mlir/IR/Operation.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "FlagIdManager";
#define LDBG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;
using namespace hivm;

static constexpr int kMaxOps = 1024;

FlagIdManager::FlagIdManager(ModuleOp module)
{
    this->module = module;
    scanExistingFlags(module);
    LDBG("FlagIdManager: Initialized with max_id = " << currentMaxId << "\n");
}

void FlagIdManager::scanExistingFlags(ModuleOp module)
{
    module.walk([&](Operation *op) {
        if (isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) {
            int flag = -1;
            if (auto intAttr = op->getAttrOfType<IntegerAttr>("static_flag_id")) {
                flag = (int)intAttr.getInt();
            } else if (auto intAttr = op->getAttrOfType<IntegerAttr>("flag")) {
                flag = (int)intAttr.getInt();
            }
            if (flag >= 0) {
                currentMaxId = std::max(currentMaxId, (int64_t)flag);
            }
        }
    });
}

int FlagIdManager::tryReuseFlag(Operation* insertionPoint)
{
    SmallVector<Operation *, kMaxOps> ops;
    module.walk([&](Operation *op) { ops.push_back(op); });

    SmallVector<int, MAX_FLAG_ID + 1> lastUse(MAX_FLAG_ID + 1, -1);

    int insertPos = -1;
    for (int i = 0; i < (int)ops.size(); ++i) {
        Operation *op = ops[i];
        if (op == insertionPoint) insertPos = i;
        if (isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) {
            int fid = -1;
            if (auto attr = op->getAttrOfType<IntegerAttr>("static_flag_id")) {
                fid = (int)attr.getInt();
            } else if (auto attr = op->getAttrOfType<IntegerAttr>("flag")) {
                fid = (int)attr.getInt();
            }
            if (fid >= 1 && fid <= MAX_FLAG_ID) lastUse[fid] = i;
        }
    }

    if (insertPos == -1) insertPos = (int)ops.size();

    for (int fid = 1; fid <= MAX_FLAG_ID; ++fid) {
        if (lastUse[fid] < insertPos) return fid;
    }
    return INVALID_FLAG_ID;
}

int FlagIdManager::acquireId(Operation* insertionPoint)
{
    // if (insertionPoint) {
    //     int reused = tryReuseFlag(insertionPoint);
    //     if (reused != INVALID_FLAG_ID) return reused;
    // }

    if (currentMaxId < MAX_FLAG_ID) {
        return ++currentMaxId;
    }

    return INVALID_FLAG_ID;
}
