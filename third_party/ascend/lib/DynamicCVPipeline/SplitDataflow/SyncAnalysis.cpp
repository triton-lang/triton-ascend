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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/SyncAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "sync-analysis";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace mlir::triton;

void mlir::triton::analyzeDepLoopInfo(DataDependencyInfo &info,
                                      DependencyInfo &dep) {
  auto blockInfoIt = info.getBlockInfoMap().find(dep.producerBlockId);
  if (blockInfoIt == info.getBlockInfoMap().end() ||
      blockInfoIt->second.Operations.empty()) {
    LOG_DEBUG("Producer block info is missing for dependency block "
              << dep.producerBlockId << "\n");
    return;
  }

  auto producerBlockOp = blockInfoIt->second.Operations.front();
  for (Operation *parent = producerBlockOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(parent)) {
      dep.loopRegion = parent;
      dep.loopDepth = getOpLoopDepth(parent);
      break;
    }
  }
}

namespace {

// Assign flag IDs by reusing those from the dominant loop region:
// dependencies are grouped by loop depth and loop region, then regions other
// than the one with the most dependencies reuse its unique IDs round-robin.
// Dependencies in the dominant region retain their IDs and do not need
// start/end synchronization; reused dependencies do.
void assignDepFlagIds(llvm::SmallVector<DependencyInfo> &deps) {
  llvm::DenseMap<int, llvm::SmallVector<DependencyInfo *>> loopDepthGroups;
  for (auto &dep : deps) {
    if (!dep.loopRegion) {
      continue;
    }
    loopDepthGroups[dep.loopDepth].push_back(&dep);
  }

  for (auto &[loopDepth, depList] : loopDepthGroups) {
    if (depList.empty()) {
      continue;
    }

    llvm::DenseMap<int, llvm::SmallVector<DependencyInfo *>> regionGroups;
    for (auto *dep : depList) {
      int regionID = CVPipeline::getOpBlockId(dep->loopRegion).value_or(-1);
      regionGroups[regionID].push_back(dep);
    }

    auto it = regionGroups.begin();
    int maxRegion = it->first;
    size_t maxSize = it->second.size();
    ++it;

    for (; it != regionGroups.end(); ++it) {
      size_t curSize = it->second.size();
      if (curSize > maxSize) {
        maxRegion = it->first;
        maxSize = curSize;
      }
    }

    llvm::SmallVector<int> masterFlags;
    llvm::DenseSet<int> seen;
    for (auto *dep : regionGroups[maxRegion]) {
      if (seen.insert(dep->flagId).second) {
        masterFlags.push_back(dep->flagId);
      }
    }

    if (masterFlags.empty()) {
      continue;
    }

    for (auto &[region, group] : regionGroups) {
      if (region == maxRegion) {
        for (auto *dep : group) {
          dep->reuseStartEnd = false;
        }
        continue;
      }

      size_t idx = 0;
      for (auto *dep : group) {
        dep->flagId = masterFlags[idx % masterFlags.size()];
        dep->reuseStartEnd = true;
        ++idx;
      }
    }
  }
}

// Remap all distinct IDs to a contiguous range starting at zero, preserving
// the relationship between dependencies that share an ID.
void remapFlagIdsToContinuous(
    llvm::SmallVector<llvm::SmallVector<DependencyInfo> *> &depLists) {
  llvm::DenseMap<int, int> idMap;
  int nextId = 0;

  for (auto *list : depLists) {
    for (auto &dep : *list) {
      auto it = idMap.find(dep.flagId);
      if (it == idMap.end()) {
        idMap[dep.flagId] = nextId;
        dep.flagId = nextId;
        ++nextId;
      } else {
        dep.flagId = it->second;
      }
    }
  }
}

} // namespace

void mlir::triton::analyzeSyncReuse(DataDependencyInfo &info) {
  int flag_id = 0;
  for (auto &dep : info.getV2CDependencies()) {
    dep.flagId = flag_id++;
  }
  for (auto &dep : info.getC2VDependencies()) {
    dep.flagId = flag_id++;
  }
  for (auto &dep : info.getMemoryDependencies()) {
    dep.flagId = flag_id++;
  }

  assignDepFlagIds(info.getV2CDependencies());
  assignDepFlagIds(info.getC2VDependencies());
  assignDepFlagIds(info.getMemoryDependencies());

  llvm::SmallVector<llvm::SmallVector<DependencyInfo> *> depLists;
  depLists.push_back(&info.getV2CDependencies());
  depLists.push_back(&info.getC2VDependencies());
  depLists.push_back(&info.getMemoryDependencies());
  remapFlagIdsToContinuous(depLists);
}
