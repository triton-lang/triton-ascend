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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "sync-analysis";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace mlir::triton;

void mlir::triton::analyzeDepLoopInfo(ModuleOp module, DependencyInfo &dep) {
  mlir::Operation *producerBlockOp = nullptr;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (CVPipeline::getOpBlockId(op) == dep.producerBlockId) {
      producerBlockOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!producerBlockOp) {
    LOG_DEBUG("Producer block info is missing for dependency block "
              << dep.producerBlockId << "\n");
    return;
  }

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
    if (!dep.loopRegion || dep.loopDepth == 0 || dep.flagId < 0) {
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
      if (dep.flagId < 0) {
        continue;
      }
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

void mlir::triton::analyzeSyncReuse(DataDependencyInfo &info, ModuleOp module) {
  int flag_id = 0;
  auto initFlagIds = [&](llvm::SmallVector<DependencyInfo> &deps) {
    for (auto &dep : deps) {
      if (dep.flagId >= 0) {
        dep.flagId = flag_id++;
      }
    }
  };
  initFlagIds(info.getV2CDependencies());
  initFlagIds(info.getC2VDependencies());
  initFlagIds(info.getMemoryDependencies());

  if (isOutlineReuseBlacklistKernel(module)) {
    return;
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

/// Redundant-edge analysis.
///
/// Stores the directed graph in a DenseMap and SmallVector, with a shared
/// addEdge interface to support adding other edge types later.
class RedundantEdgeAnalyzer {
public:
  struct Edge {
    int target;
    DependencyInfo *dependency;
  };
  using Graph = llvm::DenseMap<int, llvm::SmallVector<Edge>>;

  /// Adds a directed edge from -> to.
  void addEdge(int from, int to, DependencyInfo *dependency = nullptr) {
    graph_[from].push_back({to, dependency});
  }

  void analyze(DataDependencyInfo &info, ModuleOp module) {
    buildIntraCoreExecutionOrderEdges(info, module);
    buildInterCoreDependencyEdges(info);
    // Print the graph.
    for (auto &[from, edges] : graph_) {
      for (auto &edge : edges) {
        LOG_DEBUG("Edge: [" << from << "] -> [" << edge.target << "] \n");
      }
    }
    analyzeRedundantEdges();
  }

private:
  Graph graph_;

  /// Builds edges representing execution order between blocks on the same core.
  /// Adjacent participating blocks of the same core type follow IR order.
  void buildIntraCoreExecutionOrderEdges(DataDependencyInfo &info,
                                         ModuleOp module) {
    llvm::DenseSet<int> cubeBlockSet;
    llvm::DenseSet<int> vectorBlockSet;
    auto addCrossCoreBlocks = [&](const DependencyInfo &dep) {
      if (dep.type == DependencyType::VectorToCube) {
        vectorBlockSet.insert(dep.producerBlockId);
        cubeBlockSet.insert(dep.consumerBlockId);
      } else if (dep.type == DependencyType::CubeToVector) {
        cubeBlockSet.insert(dep.producerBlockId);
        vectorBlockSet.insert(dep.consumerBlockId);
      }
    };

    for (const auto &dep : info.getV2CDependencies()) {
      addCrossCoreBlocks(dep);
    }
    for (const auto &dep : info.getC2VDependencies()) {
      addCrossCoreBlocks(dep);
    }

    llvm::SmallVector<int> cubeBlockIds;
    llvm::SmallVector<int> vectorBlockIds;
    llvm::DenseSet<int> seenBlockIds;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto blockId = CVPipeline::getOpBlockId(op);
      if (!blockId || !seenBlockIds.insert(*blockId).second) {
        return;
      }
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        return;
      }
      if (cubeBlockSet.contains(*blockId)) {
        cubeBlockIds.push_back(*blockId);
      }
      if (vectorBlockSet.contains(*blockId)) {
        vectorBlockIds.push_back(*blockId);
      }
    });

    auto connectAdjacentBlocks = [&](const llvm::SmallVector<int> &blockIds) {
      for (size_t i = 1; i < blockIds.size(); ++i) {
        addEdge(blockIds[i - 1], blockIds[i]);
      }
    };
    connectAdjacentBlocks(cubeBlockIds);
    connectAdjacentBlocks(vectorBlockIds);
  }

  /// Builds edges representing dependencies between different cores and keeps
  /// the corresponding dependencies so their attributes can be updated.
  void buildInterCoreDependencyEdges(DataDependencyInfo &info) {
    auto addDependencies = [&](llvm::SmallVector<DependencyInfo> &deps) {
      for (auto &dep : deps) {
        dep.flagId = 0;
        addEdge(dep.producerBlockId, dep.consumerBlockId, &dep);
      }
    };
    addDependencies(info.getV2CDependencies());
    addDependencies(info.getC2VDependencies());
  }

  /// Computes a topological ordering of the graph. Returns false if it has a
  /// cycle.
  /// only used to obtain the reachability graph. Does not guarantee the
  /// determinism of the result.
  bool getTopologicalOrder(llvm::SmallVectorImpl<int> &topologicalOrder) const {
    llvm::DenseSet<int> nodes;
    for (const auto &entry : graph_) {
      nodes.insert(entry.first);
      for (const Edge &edge : entry.second) {
        nodes.insert(edge.target);
      }
    }

    llvm::DenseMap<int, unsigned> inDegree;
    for (int node : nodes) {
      inDegree[node] = 0;
    }
    for (const auto &entry : graph_) {
      for (const Edge &edge : entry.second) {
        ++inDegree[edge.target];
      }
    }

    llvm::SmallVector<int> ready;
    for (int node : nodes) {
      if (inDegree.lookup(node) == 0) {
        ready.push_back(node);
      }
    }

    while (!ready.empty()) {
      int node = ready.pop_back_val();
      topologicalOrder.push_back(node);
      auto edges = graph_.find(node);
      if (edges == graph_.end()) {
        continue;
      }
      for (const Edge &edge : edges->second) {
        unsigned remainingInDegree = --inDegree[edge.target];
        if (remainingInDegree == 0) {
          ready.push_back(edge.target);
        }
      }
    }

    return topologicalOrder.size() == nodes.size();
  }

  /// A Patch to fix the issue caused by same target v->c edge redundant:
  /// Checks whether another outgoing edge from \p source can reach \p target.
  /// While searching, ignores V->C dependency edges that directly enter
  /// \p target, so that edge alone cannot make the candidate appear redundant.
  bool isRealRedundantVectorToCubeEdge(int source, int target) const {
    auto sourceEdges = graph_.find(source);
    if (sourceEdges == graph_.end()) {
      return false;
    }

    llvm::SmallVector<int> worklist;
    for (const Edge &alternative : sourceEdges->second) {
      if (alternative.target != target) {
        worklist.push_back(alternative.target);
      }
    }

    llvm::DenseSet<int> visited;
    while (!worklist.empty()) {
      int node = worklist.pop_back_val();
      if (node == target) {
        return true;
      }
      if (!visited.insert(node).second) {
        continue;
      }

      auto edges = graph_.find(node);
      if (edges == graph_.end()) {
        continue;
      }
      for (const Edge &edge : edges->second) {
        // Preserve the V->C edge to this consumer; only an independent route
        // to the target counts as an alternative.
        if (edge.target == target && edge.dependency &&
            edge.dependency->type == DependencyType::VectorToCube) {
          continue;
        }
        worklist.push_back(edge.target);
      }
    }
    return false;
  }

  /// Computes reachability using topological sorting and marks explicit
  /// dependencies with alternative paths as redundant.
  void analyzeRedundantEdges() {
    llvm::SmallVector<int> topologicalOrder;
    if (!getTopologicalOrder(topologicalOrder)) {
      LOG_DEBUG("Dependency graph contains a cycle; skip redundant-edge "
                "marking.\n");
      return;
    }

    llvm::DenseMap<int, llvm::DenseSet<int>> reachable;
    for (auto nodeIt = topologicalOrder.rbegin();
         nodeIt != topologicalOrder.rend(); ++nodeIt) {
      int node = *nodeIt;
      auto edges = graph_.find(node);
      if (edges == graph_.end()) {
        continue;
      }
      for (const Edge &edge : edges->second) {
        reachable[node].insert(edge.target);
        for (int descendant : reachable[edge.target]) {
          reachable[node].insert(descendant);
        }
      }
    }

    for (const auto &entry : graph_) {
      for (const Edge &edge : entry.second) {
        if (!edge.dependency || edge.dependency->loopDepth > 0) {
          continue;
        }
        bool isRedundant = false;
        for (const Edge &alternative : entry.second) {
          if (alternative.target != edge.target &&
              reachable[alternative.target].contains(edge.target)) {
            isRedundant = true;
            break;
          }
        }
        if (isRedundant) {
          if (edge.dependency->type == DependencyType::VectorToCube &&
              !isRealRedundantVectorToCubeEdge(entry.first, edge.target)) {
            continue;
          }
          edge.dependency->flagId = -1;
          // Print the redundant edge.
          LOG_DEBUG("Redundant edge found: ["
                    << edge.dependency->producerBlockId << "] -> ["
                    << edge.dependency->consumerBlockId << "] "
                    << edge.dependency->value << "\n");
        }
      }
    }
  }
};

void mlir::triton::analyzeRedundantEdges(DataDependencyInfo &info,
                                         ModuleOp module) {
  RedundantEdgeAnalyzer analyzer;
  analyzer.analyze(info, module);
}
