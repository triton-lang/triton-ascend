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

#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "third_party/ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

// Role in dependency attribute: ssbuffer.crossDeps/intraDeps = [groupId,
// roleId] role: 1=producer, 0=consumer
static const int producerId = 1;
static const int consumerId = 0;
static constexpr const char *DEBUG_TYPE = "InitDependentMap";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace hivm;
using namespace triton;

// Function: Check if a consumer op is inside a given mainLoop
//            but not inside any nested mainloop, and push to consumers if true
// Input: Consumer operation, target mainLoop forOp, reference to consumers
// vector Output: consumers - push consumer to this vector if it's inside
// mainLoop (not in nested mainloop) Return: 0 for success (consumer pushed), -1
// for failure (not in mainLoop or in nested mainLoop)
static int isConsumerInMainLoop(Operation *consumer, scf::ForOp mainLoop,
                                SmallVector<Operation *> &consumers) {
  Operation *current = consumer->getParentOp();

  // Traverse up the parent chain until we reach the top (nullptr)
  while (current != nullptr) {
    if (auto forOp = dyn_cast<scf::ForOp>(current)) {
      if (forOp->hasAttr(CVPipeline::kMainLoop) && forOp != mainLoop) {
        // comsumer Op not in the current mainloop
        return 0;
      }
    }
    // If we reach the target mainLoop, consumer is inside it
    if (current == mainLoop) {
      consumers.push_back(consumer);
      return 0;
    }
    current = current->getParentOp();
  }

  LDBG("Can not find the consumer's mainloop!");
  return -1;
}

// Function: Collect ops with dependency attributes, grouped by group ID
// Input: Root operation to traverse (module or forOp), attribute name
// Output: depsByGroup - Ops grouped by group ID, format: group -> [(op, role),
// ...]
//         Attribute format: [group, role], role: 1=producer, 0=consumer
// Return: 0 for success, -1 for failure
static int
collectDepsByGroup(Operation *rootOp, const char *attrName,
                   llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
                       &depsByGroup) {
  // Attribute format: {ssbuffer.crossDeps/intraDeps = [group, role]}
  int ret = 0;
  int depSize = 2;

  rootOp->walk([&](Operation *op) {
    auto depsAttr = op->getAttrOfType<ArrayAttr>(attrName);
    if (!depsAttr)
      return;

    if (depsAttr.size() < depSize) {
      LDBG("format of dependency attribute error!");
      ret = -1;
      return;
    }

    if (!isa<IntegerAttr>(depsAttr[0]) || !isa<IntegerAttr>(depsAttr[1])) {
      LDBG("type of dependency attritbute is not Int! error op:" << *op);
      ret = -1;
      return;
    }

    int group = cast<IntegerAttr>(depsAttr[0]).getInt();
    int role = cast<IntegerAttr>(depsAttr[1]).getInt();
    depsByGroup[group].push_back({op, role});
  });

  return ret;
}

// Function: Build mapping from consumer Operation to producer Operation
// Input: Ops grouped by group ID, format: group -> [(op, role), ...]
//        role: 1=producer, 0=consumer
//        mainLoop: if not nullptr, only include consumers inside this mainLoop
// Output: result - Mapping from consumer Operation* to list of producer
// Operation* Return: 0 for success, -1 for failure
static int buildProducerConsumerMapping(
    llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> &depsByGroup,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &result,
    scf::ForOp mainLoop = nullptr) {
  for (auto &groupEntry : depsByGroup) {
    auto &ops = groupEntry.second;

    // Collect all producers and consumers in this group
    SmallVector<Operation *> producers;
    SmallVector<Operation *> consumers;

    for (auto &opRole : ops) {
      Operation *op = opRole.first;
      int role = opRole.second;
      if (role == producerId) {
        producers.push_back(op);
      } else if (role == consumerId) {
        // For intra-core mapping, only include consumers inside mainLoop
        if (mainLoop != nullptr) {
          if (isConsumerInMainLoop(op, mainLoop, consumers) != 0) {
            LDBG("isConsumerInMainLoop failed");
            return -1;
          }
        } else {
          consumers.push_back(op);
        }
      } else {
        LDBG("Get error role id in dependency attribute: OP: "
             << *op << ", role: " << role);
        return -1;
      }
    }

    // Skip if no consumers (for intra-core mapping with mainLoop filter)
    if (mainLoop != nullptr && consumers.empty())
      continue;

    // For each consumer, build mapping to all producers
    for (Operation *consumer : consumers) {
      result[consumer] = producers;
    }
  }

  return 0;
}

static int collectMainLoopById(ModuleOp module,
                               llvm::DenseMap<int, scf::ForOp> &mainLoopById) {
  int ret = 0;
  module.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr(CVPipeline::kMainLoop))
      return;
    auto mainLoopIdAttr =
        forOp->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
    if (mainLoopIdAttr) {
      mainLoopById[mainLoopIdAttr.getInt()] = forOp;
    }
  });
  return ret;
}

static int
findMainLoopIdContainingOp(Operation *op,
                           llvm::DenseMap<int, scf::ForOp> &mainLoopById) {
  for (auto &idLoopEntry : mainLoopById) {
    if (idLoopEntry.second->isAncestor(op)) {
      return idLoopEntry.first;
    }
  }
  return -1;
}

static int filterMemCrossCoreDepsByMainLoop(
    ModuleOp module,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &initialDepsMap,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &filteredDepsMap) {
  LDBG("memCrossCore dependencies before filter: " << initialDepsMap.size());

  // Step 1: Collect all main_loop forOps and their ids
  llvm::DenseMap<int, scf::ForOp> mainLoopById;
  if (collectMainLoopById(module, mainLoopById) != 0) {
    LDBG("collectMainLoopById Failed!");
    return -1;
  }

  // Step 2: Filter mapping - only keep producer/consumer pairs in the same
  // main_loop
  for (auto &entry : initialDepsMap) {
    Operation *consumer = entry.first;
    SmallVector<Operation *> &producers = entry.second;
    if (producers.empty()) {
      LDBG("Producers list is empty!");
      return -1;
    }

    // Find the main_loop id containing the consumer
    int consumerMainLoopId = findMainLoopIdContainingOp(consumer, mainLoopById);
    if (consumerMainLoopId == -1) {
      LDBG("Consumer op is not in any main_loop, skip: " << *consumer);
      continue;
    }

    // Find the main_loop id containing the producer
    int producerMainLoopId =
        findMainLoopIdContainingOp(producers[0], mainLoopById);
    if (producerMainLoopId == -1) {
      LDBG("producer op is not in any main_loop: " << *producers[0]);
      continue;
    }

    // Check all producers in the same mainloop
    for (size_t i = 1; i < producers.size(); i++) {
      int otherProducerMainLoopId =
          findMainLoopIdContainingOp(producers[i], mainLoopById);
      if (otherProducerMainLoopId != producerMainLoopId) {
        LDBG("Producers are not in the same main_loop. "
             << "First producer main_loop id: " << producerMainLoopId
             << ", Producer[" << i
             << "] main_loop id: " << otherProducerMainLoopId);
        return -1;
      }
    }

    // Check if consumer and producers are in the same main_loop
    if (consumerMainLoopId != producerMainLoopId) {
      LDBG("Consumer and producers are in different main_loop, skip. "
           << "Consumer main_loop id: " << consumerMainLoopId
           << ", Producer main_loop id: " << producerMainLoopId);
      continue;
    }

    filteredDepsMap[consumer] = producers;
  }

  LDBG("memCrossCore dependencies after filter: " << filteredDepsMap.size());

  return 0;
}

// Initialize crossCoreDependentMap (cross-core data dependency)
// Find ops with ssbuffer.crossDeps attribute
// Attribute value is a list: [group, role], role: 1=producer, 0=consumer
// Map key is consumer, value is list of all producers in the same group
// Return: 0 for success, -1 for failure
int initCrossCoreDependentMap(ModuleOp module, ControlFlowConditionInfo *info) {
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
      crossDepsByGroup;
  if (collectDepsByGroup(module, CVPipeline::kCrossCoreDeps.data(),
                         crossDepsByGroup) != 0) {
    LDBG("collectDepsByGroup on crossDeps Failed!");
    return -1;
  }

  llvm::DenseMap<Operation *, SmallVector<Operation *>> crossDepsMap;
  if (buildProducerConsumerMapping(crossDepsByGroup, crossDepsMap) != 0) {
    LDBG("buildProducerConsumerMapping on crossDeps Failed!");
    return -1;
  }
  info->crossCoreDependentMap = crossDepsMap;
  return 0;
}

// Initialize memCrossCoreDependentMap (memory cross-core data dependency)
// Find ops with ssbuffer.memCrossDeps attribute
// Attribute value is a list: [group, role], role: 1=producer, 0=consumer
// Map key is consumer Operation*, value is list of all producer Operation* in
// the same group Constraint: consumer and producer must be in the same
// main_loop (with same id) Return: 0 for success, -1 for failure
int initMemCrossCoreDependentMap(ModuleOp module,
                                 ControlFlowConditionInfo *info) {
  // Step 1: Collect all memcrossDeps by group
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
      memcrossDepsByGroup;
  if (collectDepsByGroup(module, CVPipeline::kMemCrossDeps.data(),
                         memcrossDepsByGroup) != 0) {
    LDBG("collectDepsByGroup on memcrossDeps Failed!");
    return -1;
  }

  // Step 2: Build initial mapping (all producers for each consumer)
  llvm::DenseMap<Operation *, SmallVector<Operation *>> initialMemcrossDepsMap;
  if (buildProducerConsumerMapping(memcrossDepsByGroup,
                                   initialMemcrossDepsMap) != 0) {
    LDBG("buildProducerConsumerMapping on memcrossDeps Failed!");
    return -1;
  }

  // Step 3: Filter by main_loop constraint
  llvm::DenseMap<Operation *, SmallVector<Operation *>> filteredMemcrossDepsMap;
  if (filterMemCrossCoreDepsByMainLoop(module, initialMemcrossDepsMap,
                                       filteredMemcrossDepsMap) != 0) {
    LDBG("filterMemCrossCoreDepsByMainLoop Failed!");
    return -1;
  }

  info->memCrossCoreDependentMap = filteredMemcrossDepsMap;
  return 0;
}

// Initialize intraCoreDependentMap (intra-core data dependency)
// Find forOp with ssbuffer.main_loop attribute
// Collect all intra-core deps from module (producers may be outside the loop)
// For each mainLoop, filter consumers that are inside it (not in nested
// mainloops) Return: 0 for success, -1 for failure
int initIntraCoreDependentMap(ModuleOp module, ControlFlowConditionInfo *info) {
  // Collect all intra-core deps from the entire module
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
      allIntraDepsByGroup;
  if (collectDepsByGroup(module, CVPipeline::kIntraDeps.data(),
                         allIntraDepsByGroup) != 0) {
    LDBG("collectDepsByGroup on intraDeps Failed!");
    return -1;
  }

  // For each mainLoop, build mapping with consumers inside it
  int ret = 0;
  module.walk([&](Operation *op) {
    if (!op->hasAttr(CVPipeline::kMainLoop))
      return;
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("Do not support other mainloop except forOp!");
      ret = -1;
      return;
    }

    llvm::DenseMap<Operation *, SmallVector<Operation *>> depMap;
    if (buildProducerConsumerMapping(allIntraDepsByGroup, depMap, forOp) != 0) {
      LDBG("buildProducerConsumerMapping on intraDeps Failed!");
      ret = -1;
      return;
    }

    // Only insert if there are dependencies for this mainLoop
    if (!depMap.empty()) {
      info->intraCoreDependentMap[forOp] = depMap;
    }
  });
  return ret;
}

// Print all dependent maps for verification
static void printDependentMaps(ControlFlowConditionInfo *info) {
  // Print crossCoreDependentMap
  LDBG("crossCoreDependentMap size: " << info->crossCoreDependentMap.size());
  LDBG("crossCoreDependentMap contents:");
  for (auto &entry : info->crossCoreDependentMap) {
    Operation *consumer = entry.first;
    SmallVector<Operation *> &producers = entry.second;
    LDBG("    Consumer: " << *consumer
                          << " (producers count: " << producers.size() << ")");
    for (Operation *producer : producers) {
      LDBG("      Producer: " << *producer);
    }
  }

  // Print memCrossCoreDependentMap
  LDBG("memCrossCoreDependentMap size: "
       << info->memCrossCoreDependentMap.size());
  LDBG("memCrossCoreDependentMap contents:");
  for (auto &entry : info->memCrossCoreDependentMap) {
    Operation *consumer = entry.first;
    SmallVector<Operation *> &producers = entry.second;
    LDBG("    Consumer: " << *consumer
                          << " (producers count: " << producers.size() << ")");
    for (Operation *producer : producers) {
      LDBG("      Producer: " << *producer);
    }
  }

  // Print intraCoreDependentMap
  LDBG("intraCoreDependentMap size: " << info->intraCoreDependentMap.size());
  LDBG("intraCoreDependentMap contents:");
  for (auto &forEntry : info->intraCoreDependentMap) {
    scf::ForOp forOp = forEntry.first;
    auto &depMap = forEntry.second;
    LDBG("  ForOp (depMap size: " << depMap.size() << "):");
    LDBG("    ");
    LLVM_DEBUG(llvm::dbgs() << '[' << DEBUG_TYPE << "] ";
               forOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
               llvm::dbgs() << "\n";);

    for (auto &entry : depMap) {
      Operation *consumer = entry.first;
      SmallVector<Operation *> &producers = entry.second;
      LDBG("    Consumer: " << *consumer << " (producers count: "
                            << producers.size() << ")");
      for (Operation *producer : producers) {
        LDBG("      Producer: " << *producer);
      }
    }
  }
}

// Find the IfOp that contains a given operation
static scf::IfOp findIfOpContainingOp(Operation *op) {
  if (!op) {
    return nullptr;
  }

  constexpr int maxDepth = 100;
  int depth = 0;

  Operation *current = op;
  while (current && depth < maxDepth) {
    if (auto ifOp = dyn_cast<scf::IfOp>(current)) {
      if (ifOp->hasAttr(CVPipeline::kIf)) {
        LDBG("Found ssbuffer.if at depth " << depth);
        return ifOp;
      }
    }
    current = current->getParentOp();
    depth++;
  }

  if (depth >= maxDepth) {
    LDBG("Warning: Max depth " << maxDepth
                               << " exceeded in findIfOpContainingOp");
  }

  return nullptr;
}

// Compute producer buffer count from dependency maps
// Rule: traverse cross-core and intra-core maps, assign max size
// If intra-core map is empty, use BufferCountManager's IntraCore value
static void computeProducerBufferCount(ControlFlowConditionInfo *info,
                                       ModuleOp module) {
  // Get cross-core buffer count (max size in the map)
  info->crossCoreBufferCount = 0;
  for (auto &entry : info->crossCoreDependentMap) {
    info->crossCoreBufferCount =
        std::max(info->crossCoreBufferCount, (int)entry.second.size());
  }
  LDBG("Cross-core buffer count (max): " << info->crossCoreBufferCount);

  // Get intra-core buffer count (max size across all forOps)
  info->intraCoreBufferCount = 0;
  for (auto &forOpEntry : info->intraCoreDependentMap) {
    auto &intraDepMap = forOpEntry.second;
    for (auto &entry : intraDepMap) {
      info->intraCoreBufferCount =
          std::max(info->intraCoreBufferCount, (int)entry.second.size());
    }
  }
  LDBG("Intra-core buffer count (max across all forOps): "
       << info->intraCoreBufferCount);

  // If intra-core map is empty, use BufferCountManager's IntraCore value
  if (info->intraCoreBufferCount == 0) {
    BufferCountManager bufferCountMgr(module);
    info->intraCoreBufferCount = bufferCountMgr.getBufferCountByType(
        BufferCountManager::DepType::IntraCore);
    LDBG("Intra-core map is empty, using BufferCountManager IntraCore value: "
         << info->intraCoreBufferCount);
  }
}

// Get the buffers that have the same tightly_coupled_buffer id
static int buildTightlyCoupledBufferGroups(
    ModuleOp module,
    DenseMap<int, SmallVector<Operation *>> &tightlyCoupledBufferGroups) {
  int ret = 0;
  WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (isa<annotation::MarkOp>(op)) {
      if (auto tcbAttr = op->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
              "hivm.tightly_coupled_buffer")) {
        auto id = tcbAttr.getId();
        if (id.has_value()) {
          int tcb = id.value();
          Operation *markedOp = op->getOperand(0).getDefiningOp();
          if (markedOp) {
            tightlyCoupledBufferGroups[tcb].push_back(markedOp);
          }
        } else {
          ret = -1;
          LDBG("hivm.tightly_coupled_buffer Attribute has no id!");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });

  if (ret == -1) {
    LDBG("Failed to build tightlyCoupledBufferGroups!");
    return -1;
  }

  return 0;
}

// Find producer IfOp for a given producer operation
// The producer op is linked to another equivalent op via
// tightly_coupled_buffer We need to find the equivalent op and its userOp
// (fixpipe or copy) to get the producer IfOp
static scf::IfOp findProducerIfOp(
    Operation *producerOp,
    DenseMap<int, SmallVector<Operation *>> &tightlyCoupledBufferGroups) {
  if (!producerOp) {
    LDBG("Producer op is null!");
    return nullptr;
  }

  // Find the tightly_coupled_buffer group id for this producer
  int tcbGroupId = -1;
  for (auto &tcbEntry : tightlyCoupledBufferGroups) {
    if (llvm::is_contained(tcbEntry.second, producerOp)) {
      tcbGroupId = tcbEntry.first;
      break;
    }
  }

  if (tcbGroupId == -1) {
    LDBG("Producer op not found in any tightly_coupled_buffer group: "
         << *producerOp);
    return nullptr;
  }

  // Find the equivalent op (another op in the same tcb group)
  Operation *equivalentOp = nullptr;
  for (Operation *op : tightlyCoupledBufferGroups[tcbGroupId]) {
    if (op != producerOp) {
      equivalentOp = op;
      break;
    }
  }

  if (!equivalentOp) {
    LDBG("No equivalent op found for producer: " << *producerOp);
    return nullptr;
  }

  // Find the user operation (fixpipe or copy) that uses results of this
  // equivalent op
  for (Value result : equivalentOp->getResults()) {
    for (Operation *userOp : result.getUsers()) {
      if (isa<hivm::FixpipeOp>(userOp) || isa<hivm::CopyOp>(userOp)) {
        scf::IfOp producerIf = findIfOpContainingOp(userOp);
        if (!producerIf) {
          LDBG("Equivalent op's user op not in any ssbuffer.if block: "
               << *userOp);
          return nullptr;
        }
        return producerIf;
      }
    }
  }

  LDBG("Equivalent op results not used by any fixpipe/copy: " << *equivalentOp);
  return nullptr;
}

// Build if block DAG from crossCoreDependentMap
// For consumer: its definingOp is inside an if block
// For producer: its definingOp is NOT inside an if block, need to find userOp
// (fixpipe/copy) that uses this producer
static int buildIfBlockCrossCoreDAG(ModuleOp module,
                                    ControlFlowConditionInfo *info) {
  // Step 0: Build tightlyCoupledBufferGroups map
  DenseMap<int, SmallVector<Operation *>> tightlyCoupledBufferGroups;
  if (buildTightlyCoupledBufferGroups(module, tightlyCoupledBufferGroups) !=
      0) {
    return -1;
  }

  // Traverse crossCoreDependentMap to build DAG
  for (auto &entry : info->crossCoreDependentMap) {
    Operation *consumerOp = entry.first;

    // Step 1: Find consumer IfOp
    // Consumer op is inside an if block
    scf::IfOp consumerIf = findIfOpContainingOp(consumerOp);
    if (!consumerIf) {
      LDBG("Consumer op not in any ssbuffer.if block: " << *consumerOp);
      return -1;
    }

    // Step 2: Find producer IfOps
    // Producer op is NOT inside an if block
    // Need to find the userOp (fixpipe or copy) that uses results of this
    // producer
    for (Operation *producerOp : entry.second) {
      scf::IfOp producerIf =
          findProducerIfOp(producerOp, tightlyCoupledBufferGroups);
      if (!producerIf) {
        return -1;
      }

      if (producerIf == consumerIf) {
        LDBG("Producer and consumer are in the same if block, this is invalid: "
             << *producerIf);
        return -1;
      }

      info->ifBlockCrossCoreDAG[producerIf].push_back(consumerIf);
    }
  }

  // Deduplicate edges
  for (auto &entry : info->ifBlockCrossCoreDAG) {
    llvm::SmallVector<scf::IfOp> uniqueConsumers;
    for (scf::IfOp consumer : entry.second) {
      if (!llvm::is_contained(uniqueConsumers, consumer)) {
        uniqueConsumers.push_back(consumer);
      }
    }
    entry.second = uniqueConsumers;
  }
  return 0;
}

// DFS helper function to find nodes at target distance from start node
static void dfsFindNodesAtDistance(
    scf::IfOp currentNode, int currentDistance, int targetDistance,
    llvm::DenseSet<scf::IfOp> &visited,
    llvm::SmallVector<scf::IfOp> &resultNodes,
    llvm::DenseMap<scf::IfOp, llvm::SmallVector<scf::IfOp>> &dag) {
  // Mark current node as visited
  visited.insert(currentNode);

  // If we've reached target distance, add to result and stop recursion
  if (currentDistance == targetDistance) {
    resultNodes.push_back(currentNode);
    return;
  }

  // Get consumers of current node
  auto it = dag.find(currentNode);
  if (it == dag.end() || it->second.empty()) {
    return;
  }
  auto &consumers = it->second;

  // Recursively visit all consumers
  for (scf::IfOp consumer : consumers) {
    if (!visited.contains(consumer)) {
      dfsFindNodesAtDistance(consumer, currentDistance + 1, targetDistance,
                             visited, resultNodes, dag);
    }
  }
}

// Collect flowOpt if block pairs from DAG using DFS
// Find all start nodes (in-degree = 0), then use DFS to find nodes at distance
// 2
static int collectFlowOptIfOpPairs(ModuleOp module,
                                   ControlFlowConditionInfo *info) {
  // Step 1: Calculate in-degree for each node
  llvm::DenseMap<scf::IfOp, int> inDegree;
  for (auto &entry : info->ifBlockCrossCoreDAG) {
    for (scf::IfOp consumer : entry.second) {
      inDegree[consumer]++;
    }
  }

  // Step 2: Find all start nodes (in-degree = 0)
  llvm::SmallVector<scf::IfOp> startNodes;
  for (auto &entry : info->ifBlockCrossCoreDAG) {
    if (inDegree.lookup(entry.first) == 0) {
      startNodes.push_back(entry.first);
      LDBG("Found start node (in-degree = 0)");
    }
  }

  LDBG("Number of start nodes: " << startNodes.size());

  // Step 3: For each start node, use DFS to find nodes at distance 2
  constexpr int targetDistance = 2;

  for (scf::IfOp start : startNodes) {
    // DFS data structures
    llvm::DenseSet<scf::IfOp> visited;
    llvm::SmallVector<scf::IfOp> thirdNodes;

    // Start DFS from start node at distance 0
    dfsFindNodesAtDistance(start, 0, targetDistance, visited, thirdNodes,
                           info->ifBlockCrossCoreDAG);

    // Record all third nodes found
    for (scf::IfOp thirdNode : thirdNodes) {
      info->flowOptIfOpPairs[thirdNode] = start;
    }
  }

  LDBG("flowOptIfOpPairs size: " << info->flowOptIfOpPairs.size());

  return 0;
}

// Print DAG and flowOpt pairs for verification
static void printDAGInfo(ControlFlowConditionInfo *info) {
  LDBG("ifBlockCrossCoreDAG contents:");
  for (auto &entry : info->ifBlockCrossCoreDAG) {
    scf::IfOp producer = entry.first;
    LDBG("  Producer IfOp has " << entry.second.size() << " consumers");
    for (scf::IfOp consumer : entry.second) {
      LDBG("    -> Consumer IfOp");
    }
  }

  LDBG("flowOptIfOpPairs contents:");
  for (auto &entry : info->flowOptIfOpPairs) {
    scf::IfOp target = entry.first;
    scf::IfOp source = entry.second;
    LDBG("  Target IfOp (third node) -> Source IfOp (start node)");
  }
}

void InitDependentMapPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter InitDependentMap pass.");

  // Step 1: Initialize crossCoreDependentMap
  if (initCrossCoreDependentMap(module, info) != 0) {
    LDBG("initCrossCoreDependentMap failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 2: Initialize memCrossCoreDependentMap
  if (initMemCrossCoreDependentMap(module, info) != 0) {
    LDBG("initMemCrossCoreDependentMap failed!");
    signalPassFailure();
    return;
  }

  // Step 3: Initialize intraCoreDependentMap
  if (initIntraCoreDependentMap(module, info) != 0) {
    LDBG("initIntraCoreDependentMap failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Print all dependent maps for verification
  LLVM_DEBUG(printDependentMaps(info));

  // Step 3: Compute producer buffer count for flowOpt condition
  computeProducerBufferCount(info, module);

  // Step 4: Build if block DAG from crossCoreDependentMap
  if (info->crossCoreBufferCount > CROSS_CORE_BUFFER_COUNT_THRESHOLD &&
      info->intraCoreBufferCount > INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    LDBG("Buffer counts meet requirements, building DAG and collecting flowOpt "
         "pairs.");

    if (buildIfBlockCrossCoreDAG(module, info) != 0) {
      LDBG("buildIfBlockCrossCoreDAG failed!");
      signalPassFailure();
      return;
    }

    // Step 5: Collect flowOpt if block pairs from DAG
    if (collectFlowOptIfOpPairs(module, info) != 0) {
      LDBG("collectFlowOptIfOpPairs failed!");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(printDAGInfo(info));
  }

  LDBG("Exit InitDependentMap pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createInitDependentMapPass() {
  return std::make_unique<InitDependentMapPass>();
}
} // namespace triton
} // namespace mlir
