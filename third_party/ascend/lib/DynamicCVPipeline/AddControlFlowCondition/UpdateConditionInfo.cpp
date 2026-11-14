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

#include <functional>
#include <optional>
#include <string>
#include <type_traits>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"

static constexpr const char *DEBUG_TYPE = "UpdateConditionInfoPass";
static constexpr int VECTOR_SSBUF_OFFSET = 1024;
static constexpr int VALUE_SSBUF_OFFSET = 4;
static constexpr int UPDATE_CONDITION_INFO_SUCCESS = 0;
static constexpr int UPDATE_CONDITION_INFO_FAILED = -1;
static constexpr size_t kVectorCount = 2;
// Block level counters start right after the area that used to hold the
// per-dependency-edge counters (3 edges * 8 bytes). Each block owns one slot,
// a vector block keeps a second copy at slot + VECTOR_SSBUF_OFFSET.
static constexpr int SSBUF_BLOCK_COUNTER_BASE = 24;
static constexpr int SSBUF_BLOCK_COUNTER_STRIDE = 8;
// Flag id of the PIPE_S handshake that brackets the counter initialization.
// Flag 15 is the flag that is reserved for the pipe synchronization: the cross
// core transfer flags are allocated in the range 0..14
// (AddMultiBufferOuterScope reserves 15 for PIPE_S), so every op of the
// handshake shares that one flag.
static constexpr int kPipeSFlagId = 15;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
  })
using namespace mlir;
using namespace triton;
using namespace hivm;
using namespace CVPipeline;

static void logConditionGroupIndices(llvm::StringRef label,
                                     llvm::ArrayRef<int> groupIndices) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << label;
  for (int groupIndex : groupIndices) {
    os << groupIndex << " ";
  }
  os << "\n";
  LDBG(os.str());
}

static void logOutputGroupValues(llvm::StringRef label,
                                 llvm::ArrayRef<Value> values) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << label;
  for (Value value : values) {
    os << value << " ";
  }
  os << "\n";
  LDBG(os.str());
}

// Read block id from ssbuffer.if on ifOp. Missing attr is unexpected.
static int getIfBlockId(scf::IfOp ifOp, int &outBlockId) {
  auto attr = ifOp->getAttrOfType<IntegerAttr>(kIf);
  if (!attr) {
    LDBG("ssbuffer.if missing block id on ifOp: " << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  outBlockId = static_cast<int>(attr.getInt());
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// for: body region iter args; while-do: after-region args.
static int getLoopRegionIterArgs(Operation *loopOp,
                                 MutableArrayRef<BlockArgument> &outArgs) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    outArgs = forOp.getRegionIterArgs();
    return UPDATE_CONDITION_INFO_SUCCESS;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    outArgs = whileOp.getAfterArguments();
    return UPDATE_CONDITION_INFO_SUCCESS;
  }
  LDBG("getLoopRegionIterArgs expects scf.for or scf.while, got " << *loopOp
                                                                  << "\n");
  return UPDATE_CONDITION_INFO_FAILED;
}

static int getLoopRegionIterArg(Operation *loopOp, int argIdx, Value &outArg) {
  MutableArrayRef<BlockArgument> args;
  if (getLoopRegionIterArgs(loopOp, args) == UPDATE_CONDITION_INFO_FAILED)
    return UPDATE_CONDITION_INFO_FAILED;
  if (argIdx < 0 || argIdx >= static_cast<int>(args.size())) {
    LDBG("Invalid loop region iter arg index: "
         << argIdx << ", iter args " << args.size() << ", loopOp=" << *loopOp
         << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  outArg = args[argIdx];
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Clone only the SSA def-chain that produces `condition` when it lives in
// beforeRegion.
static int cloneConditionDefChain(Value value, Region &beforeRegion,
                                  IRMapping &mapping, OpBuilder &builder,
                                  Value &outValue) {
  if (Value mapped = mapping.lookupOrNull(value)) {
    outValue = mapped;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getParentRegion() == &beforeRegion) {
      LDBG("Before-region arg is not in whileBlockArgMap mapping: " << value
                                                                    << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    outValue = value;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    outValue = value;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }
  // Defined outside before-region: reuse as-is (constants, outer values).
  if (defOp->getParentRegion() != &beforeRegion) {
    outValue = value;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  for (Value operand : defOp->getOperands()) {
    Value remappedOperand;
    if (cloneConditionDefChain(operand, beforeRegion, mapping, builder,
                               remappedOperand) == UPDATE_CONDITION_INFO_FAILED)
      return UPDATE_CONDITION_INFO_FAILED;
    if (!mapping.lookupOrNull(operand))
      mapping.map(operand, remappedOperand);
  }

  Operation *cloned = builder.clone(*defOp, mapping);
  outValue = cloned->getResult(cast<OpResult>(value).getResultNumber());
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Remap scf.condition(x): clone only ops that produce x into the after region,
// remapping before args via whileBlockArgMap[while][blockId]:
// {new_arg_idx(after) -> old_arg_idx(before)}.
static int buildWhileCounterCondition(
    scf::WhileOp whileOp, scf::IfOp ifOp, ControlFlowConditionInfo *info,
    OpBuilder &builder, const DenseMap<Value, Value> &controlVarToLatestValue,
    Value &outCond) {
  int blockId;
  if (getIfBlockId(ifOp, blockId) == UPDATE_CONDITION_INFO_FAILED)
    return UPDATE_CONDITION_INFO_FAILED;

  auto whileIt = info->whileBlockArgMap.find(whileOp);
  if (whileIt == info->whileBlockArgMap.end()) {
    LDBG("whileBlockArgMap has no entry for whileOp=" << whileOp << ", ifOp="
                                                      << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  auto blockIt = whileIt->second.find(blockId);
  if (blockIt == whileIt->second.end()) {
    LDBG("whileBlockArgMap has no entry for blockId "
         << blockId << ", whileOp=" << whileOp << ", ifOp=" << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  const DenseMap<int, int> &argIdxMap = blockIt->second;
  auto beforeArgs = whileOp.getBeforeArguments();
  auto afterArgs = whileOp.getAfterArguments();

  IRMapping mapping;
  for (auto [newArgIdx, oldArgIdx] : argIdxMap) {
    if (oldArgIdx < 0 || oldArgIdx >= static_cast<int>(beforeArgs.size()) ||
        newArgIdx < 0 || newArgIdx >= static_cast<int>(afterArgs.size())) {
      LDBG("Invalid whileBlockArgMap entry (new="
           << newArgIdx << ", old=" << oldArgIdx << "), whileOp=" << whileOp
           << ", ifOp=" << ifOp << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    Value afterArg = afterArgs[newArgIdx];
    auto latestIt = controlVarToLatestValue.find(afterArg);
    if (latestIt != controlVarToLatestValue.end())
      afterArg = latestIt->second;
    mapping.map(beforeArgs[oldArgIdx], afterArg);
  }

  // Only take condition value x from scf.condition(x);
  scf::ConditionOp condOp = whileOp.getConditionOp();
  Value beforeCond = condOp.getCondition();
  if (cloneConditionDefChain(beforeCond, whileOp.getBefore(), mapping, builder,
                             outCond) == UPDATE_CONDITION_INFO_FAILED) {
    LDBG("Failed to remap while before-region condition expression, whileOp="
         << whileOp << ", ifOp=" << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Find the scope that owns `op`. Returns nullptr when there is none.
static scope::ScopeOp findOwningScope(Operation *op) {
  if (!op) {
    return nullptr;
  }
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(parent)) {
      return scopeOp;
    }
    parent = parent->getParentOp();
  }
  return nullptr;
}

// Read the core type of a scope. Returns false when the attribute is missing
// or holds an unexpected value.
static bool getScopeCoreType(scope::ScopeOp scopeOp, bool &isAIC, bool &isAIV) {
  isAIC = false;
  isAIV = false;
  if (!scopeOp || !scopeOp->hasAttr("hivm.tcore_type")) {
    return false;
  }

  auto attr = scopeOp->getAttr("hivm.tcore_type");
  auto aiCAttr =
      hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::CUBE);
  auto aiVAttr =
      hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::VECTOR);
  if (attr == aiCAttr) {
    isAIC = true;
    return true;
  }
  if (attr == aiVAttr) {
    isAIV = true;
    return true;
  }
  return false;
}

// Build the block level cross-core edges out of the op level dependency map.
// One dependency group becomes one edge per producer computation block: the
// edge depth is the number of producer ops of the group living in that block.
// A group whose producers spread over several blocks therefore produces one
// edge per block, every block owning its own counter.
int UpdateConditionInfoPass::buildCrossCoreBlockEdges(ModuleOp module) {
  info->crossCoreBlockEdges.clear();

  // The dependency map is a DenseMap, so collect the edges first and sort
  // them afterwards to get a stable result.
  SmallVector<CrossCoreBlockEdge> edges;
  DenseMap<std::pair<scf::IfOp, scf::IfOp>, int> edgeIndex;

  for (auto &entry : info->crossCoreDependentMap) {
    Operation *consumerOp = entry.first;
    scf::IfOp consumerIf = findIfOpContainingOp(consumerOp);
    if (!consumerIf) {
      LDBG("Cross-core consumer op is not inside any ssbuffer.if block: "
           << *consumerOp << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    for (SmallVector<Operation *> &producers : entry.second) {
      if (producers.empty()) {
        LDBG("Cross-core producer list is empty, consumer=" << *consumerOp
                                                            << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }

      // Group the producers of this dependency group by their block. Every
      // producer block owns its own counter and produces its own buffers, so
      // a group whose producers spread over several blocks produces one edge
      // per block.
      SmallVector<std::pair<scf::IfOp, int>> producersByBlock;
      for (Operation *producerOp : producers) {
        scf::IfOp producerBlock = findIfOpContainingOp(producerOp);
        if (!producerBlock) {
          LDBG("Cross-core producer op is not inside any ssbuffer.if block: "
               << *producerOp << "\n");
          return UPDATE_CONDITION_INFO_FAILED;
        }
        auto blockIt = producersByBlock.end();
        for (auto it = producersByBlock.begin(); it != producersByBlock.end();
             ++it) {
          if (it->first == producerBlock) {
            blockIt = it;
            break;
          }
        }
        if (blockIt == producersByBlock.end()) {
          producersByBlock.push_back({producerBlock, 1});
        } else {
          blockIt->second++;
        }
      }

      for (const auto &producerEntry : producersByBlock) {
        scf::IfOp producerIf = producerEntry.first;
        int bufferNum = producerEntry.second;
        if (producerIf == consumerIf) {
          // Producer and consumer share the block: the difference of their
          // counters is always zero, the edge carries no information.
          LDBG("Skip the cross-core edge inside a single block: " << producerIf
                                                                  << "\n");
          continue;
        }

        auto key = std::make_pair(producerIf, consumerIf);
        auto it = edgeIndex.find(key);
        if (it == edgeIndex.end()) {
          edgeIndex[key] = static_cast<int>(edges.size());
          edges.push_back({producerIf, consumerIf, bufferNum});
        } else if (bufferNum < edges[it->second].bufferNum) {
          // Several groups share the same block pair: keep the strictest depth.
          edges[it->second].bufferNum = bufferNum;
        }
      }
    }
  }

  // Sort the edges by block id to get a deterministic slot assignment.
  auto blockIdOf = [](scf::IfOp ifOp) {
    int blockId = -1;
    getIfBlockId(ifOp, blockId);
    return blockId;
  };
  llvm::sort(edges,
             [&](const CrossCoreBlockEdge &a, const CrossCoreBlockEdge &b) {
               int aProducer = blockIdOf(a.producer);
               int bProducer = blockIdOf(b.producer);
               if (aProducer != bProducer) {
                 return aProducer < bProducer;
               }
               return blockIdOf(a.consumer) < blockIdOf(b.consumer);
             });

  for (const CrossCoreBlockEdge &edge : edges) {
    info->crossCoreBlockEdges[edge.producer].push_back(edge);
    info->crossCoreBlockEdges[edge.consumer].push_back(edge);
  }

  LDBG("Built " << edges.size() << " block level cross-core edges." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Give every computation block taking part in a cross-core dependency a
// counter slot. The vector blocks come first (sorted by block id), then the
// cube blocks (sorted by block id).
int UpdateConditionInfoPass::assignBlockCounterSlots(ModuleOp module) {
  info->blockCounterSlots.clear();
  blockCounterOwners.clear();

  struct BlockEntry {
    int blockId;
    bool isVector;
    scf::IfOp ifOp;
  };
  SmallVector<BlockEntry> vectorBlocks;
  SmallVector<BlockEntry> cubeBlocks;
  bool failed = false;

  // Collect the blocks in module walk order to have a stable base order.
  module.walk([&](Operation *op) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp || !ifOp->hasAttr(CVPipeline::kIf)) {
      return;
    }
    if (!info->crossCoreBlockEdges.count(ifOp)) {
      return;
    }

    int blockId;
    if (getIfBlockId(ifOp, blockId) == UPDATE_CONDITION_INFO_FAILED) {
      failed = true;
      return;
    }
    scope::ScopeOp scopeOp = findOwningScope(ifOp);
    bool isAIC = false;
    bool isAIV = false;
    if (!getScopeCoreType(scopeOp, isAIC, isAIV)) {
      LDBG("Cross-core block is not in a cube or vector scope: " << ifOp
                                                                 << "\n");
      failed = true;
      return;
    }
    if (isAIV) {
      vectorBlocks.push_back({blockId, true, ifOp});
    } else {
      cubeBlocks.push_back({blockId, false, ifOp});
    }
  });

  if (failed) {
    LDBG("assignBlockCounterSlots failed!");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  auto byBlockId = [](const BlockEntry &a, const BlockEntry &b) {
    return a.blockId < b.blockId;
  };
  llvm::stable_sort(vectorBlocks, byBlockId);
  llvm::stable_sort(cubeBlocks, byBlockId);

  int slot = 0;
  SmallVector<BlockEntry> allBlocks(vectorBlocks);
  llvm::append_range(allBlocks, cubeBlocks);
  for (const BlockEntry &entry : allBlocks) {
    BlockCounterSlot counterSlot;
    counterSlot.slot = slot++;
    counterSlot.byteOffset = SSBUF_BLOCK_COUNTER_BASE +
                             counterSlot.slot * SSBUF_BLOCK_COUNTER_STRIDE;
    counterSlot.isVector = entry.isVector;
    counterSlot.blockId = entry.blockId;
    info->blockCounterSlots[entry.ifOp] = counterSlot;
    blockCounterOwners.push_back(entry.ifOp);
  }

  LDBG("Assigned " << blockCounterOwners.size() << " block level counter slots."
                   << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// The depth of every block of the dependency graph, and the blocks every
// block depends on.
//
// The blocks that have no producer open a dependency chain, they are on depth
// one. The depth of a block is the depth of its deepest producer, plus one
// when the edge that links them crosses the cores, plus zero when the edge
// stays inside a core. The blocks of the same depth are on the same level.
//
// The ancestors of a block are the blocks it depends on, directly or through
// other blocks. Two blocks that belong to the same dependency chain always
// have a common ancestor, the blocks of two different chains never have one.
// The preload conditions only compare a block with its descendants: the
// counters of two unrelated chains do not advance together, comparing them
// would compare the rounds of two different pipelines.
static void computeBlockDepths(
    const llvm::DenseMap<
        scf::IfOp, llvm::SmallVector<std::pair<scf::IfOp, IfBlockDepKind>>>
        &ifBlockDAG,
    llvm::DenseMap<scf::IfOp, int> &depths,
    llvm::DenseMap<scf::IfOp, llvm::DenseSet<scf::IfOp>> &ancestors) {
  // Every block of the graph, and the number of its producers.
  llvm::SmallVector<scf::IfOp> nodes;
  llvm::DenseMap<scf::IfOp, int> inDegree;
  auto addNode = [&](scf::IfOp node) {
    if (!inDegree.count(node)) {
      nodes.push_back(node);
    }
  };
  for (auto &entry : ifBlockDAG) {
    addNode(entry.first);
    for (auto &edge : entry.second) {
      addNode(edge.first);
      inDegree[edge.first]++;
    }
  }

  // The blocks that open a chain are on depth one. The others get their depth
  // from their producers, the graph is walked in topological order.
  llvm::SmallVector<scf::IfOp> ready;
  for (scf::IfOp node : nodes) {
    if (inDegree.lookup(node) == 0) {
      depths[node] = 1;
      ready.push_back(node);
    }
  }

  while (!ready.empty()) {
    scf::IfOp node = ready.pop_back_val();
    auto it = ifBlockDAG.find(node);
    if (it == ifBlockDAG.end()) {
      continue;
    }
    for (auto &edge : it->second) {
      int step = (edge.second == IfBlockDepKind::CrossCore) ? 1 : 0;
      int candidate = depths.lookup(node) + step;
      int &depth = depths[edge.first];
      if (candidate > depth) {
        depth = candidate;
      }
      // The consumer depends on the producer, and on everything the producer
      // depends on.
      llvm::DenseSet<scf::IfOp> &consumerAncestors = ancestors[edge.first];
      consumerAncestors.insert(node);
      for (scf::IfOp ancestor : ancestors.lookup(node)) {
        consumerAncestors.insert(ancestor);
      }
      if (--inDegree[edge.first] == 0) {
        ready.push_back(edge.first);
      }
    }
  }
}

// Collect the computation blocks of the two cores, ordered by level.
//
// The preload conditions compare a block with the blocks of the higher levels
// of the same core, so the level of every block has to be known before the
// conditions are built. The level comes from the dependency graph: the blocks
// that open a dependency chain are on the first level of their core, a block
// is one level below its producer when the edge that links them crosses the
// cores. The blocks of the same depth are on the same level, they are not
// compared with each other.
//
// Only the blocks that take part in a cross-core dependency are collected:
// they are the ones the preload conditions compare, they are the ones that own
// a counter. A block that takes part in no cross-core dependency is not
// numbered at all, it does not take a place in the order.
int UpdateConditionInfoPass::collectPreloadBlocks(ModuleOp module) {
  info->preloadVectorBlocks.clear();
  info->preloadCubeBlocks.clear();
  preloadAncestors.clear();

  // The depth of every block of the dependency graph, and the blocks every
  // block depends on.
  llvm::DenseMap<scf::IfOp, int> depths;
  llvm::DenseMap<scf::IfOp, llvm::DenseSet<scf::IfOp>> ancestors;
  computeBlockDepths(info->ifBlockDAG, depths, ancestors);

  // The ancestors are kept by block id: the id of a block does not change when
  // its if op is replaced, the if op itself does.
  for (auto &entry : ancestors) {
    int blockId;
    if (getIfBlockId(entry.first, blockId) == UPDATE_CONDITION_INFO_FAILED) {
      continue;
    }
    llvm::DenseSet<int> &blockAncestors = preloadAncestors[blockId];
    for (scf::IfOp ancestor : entry.second) {
      int ancestorId;
      if (getIfBlockId(ancestor, ancestorId) == UPDATE_CONDITION_INFO_FAILED) {
        continue;
      }
      blockAncestors.insert(ancestorId);
    }
  }

  struct BlockEntry {
    int blockId;
    int depth;
    scf::IfOp ifOp;
  };
  SmallVector<BlockEntry> vectorBlocks;
  SmallVector<BlockEntry> cubeBlocks;
  bool failed = false;

  module.walk([&](Operation *op) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp || !ifOp->hasAttr(CVPipeline::kIf)) {
      return;
    }
    // Only the blocks that own a counter are numbered, they are the ones
    // the preload conditions compare.
    if (!info->blockCounterSlots.count(ifOp)) {
      return;
    }

    int blockId;
    if (getIfBlockId(ifOp, blockId) == UPDATE_CONDITION_INFO_FAILED) {
      failed = true;
      return;
    }
    scope::ScopeOp scopeOp = findOwningScope(ifOp);
    bool isAIC = false;
    bool isAIV = false;
    if (!getScopeCoreType(scopeOp, isAIC, isAIV)) {
      LDBG("Computation block is not in a cube or vector scope: " << ifOp
                                                                  << "\n");
      failed = true;
      return;
    }
    // A block that is not part of the dependency graph opens a chain of its
    // own, it is on depth one.
    int depth = depths.lookup(ifOp);
    if (depth <= 0) {
      depth = 1;
    }
    if (isAIV) {
      vectorBlocks.push_back({blockId, depth, ifOp});
    } else {
      cubeBlocks.push_back({blockId, depth, ifOp});
    }
  });

  if (failed) {
    LDBG("collectPreloadBlocks failed!");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // The blocks of a core are ordered by their depth, the ones that open a
  // chain come first.
  auto byDepth = [](const BlockEntry &a, const BlockEntry &b) {
    if (a.depth != b.depth) {
      return a.depth < b.depth;
    }
    return a.blockId < b.blockId;
  };
  llvm::stable_sort(vectorBlocks, byDepth);
  llvm::stable_sort(cubeBlocks, byDepth);

  // The level of a block is the rank of its depth among the depths of the
  // blocks of its core. The blocks of the same depth share their level.
  auto assignLevels = [](const SmallVector<BlockEntry> &blocks,
                         SmallVector<std::pair<scf::IfOp, int>> &levels) {
    int level = 0;
    int lastDepth = 0;
    for (const BlockEntry &entry : blocks) {
      if (entry.depth != lastDepth) {
        level++;
        lastDepth = entry.depth;
      }
      levels.push_back({entry.ifOp, level});
    }
  };
  assignLevels(vectorBlocks, info->preloadVectorBlocks);
  assignLevels(cubeBlocks, info->preloadCubeBlocks);

  auto printLevels = [](const char *core,
                        const SmallVector<std::pair<scf::IfOp, int>> &levels) {
    for (auto &block : levels) {
      LDBG("Preload " << core << " block on level " << block.second << ".\n");
    }
  };
  printLevels("vector", info->preloadVectorBlocks);
  printLevels("cube", info->preloadCubeBlocks);

  LDBG("Collected " << info->preloadVectorBlocks.size() << " vector and "
                    << info->preloadCubeBlocks.size()
                    << " cube computation blocks." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// The level of `ifOp` in the block list of its core, 0 when the block is not
// in the list.
static int
getPreloadLevel(const llvm::SmallVector<std::pair<scf::IfOp, int>> &blocks,
                scf::IfOp ifOp) {
  for (auto &block : blocks) {
    if (block.first == ifOp) {
      return block.second;
    }
  }
  return 0;
}

// The blocks of the core of `ifOp` that `ifOp` is compared with.
//
// They are the blocks of the higher levels of the same core that depend on
// `ifOp`, directly or through other blocks. The blocks of another dependency
// chain are left out: they run in another round of the pipeline, their
// counters do not advance together with the ones of `ifOp`, comparing them
// would compare the rounds of two unrelated pipelines.
//
// Every entry holds the target and the level it is on.
SmallVector<std::pair<scf::IfOp, int>>
UpdateConditionInfoPass::getPreloadTargets(scf::IfOp ifOp, bool isAIC) {
  SmallVector<std::pair<scf::IfOp, int>> targets;
  const SmallVector<std::pair<scf::IfOp, int>> &blocks =
      isAIC ? info->preloadCubeBlocks : info->preloadVectorBlocks;
  int ownLevel = getPreloadLevel(blocks, ifOp);
  if (ownLevel == 0) {
    return targets;
  }
  int ownBlockId;
  if (getIfBlockId(ifOp, ownBlockId) == UPDATE_CONDITION_INFO_FAILED) {
    return targets;
  }
  for (auto &block : blocks) {
    // Only the blocks of the higher levels are compared, the blocks of the
    // same level are independent of each other.
    if (block.second <= ownLevel) {
      continue;
    }
    // The target has to depend on `ifOp`: the blocks of another dependency
    // chain are not part of the same pipeline.
    int targetBlockId;
    if (getIfBlockId(block.first, targetBlockId) ==
        UPDATE_CONDITION_INFO_FAILED) {
      continue;
    }
    auto ancestorsIt = preloadAncestors.find(targetBlockId);
    if (ancestorsIt == preloadAncestors.end() ||
        !ancestorsIt->second.count(ownBlockId)) {
      continue;
    }
    targets.push_back(block);
  }
  return targets;
}

// The number of the cross-core buffers, the base of the preload bounds.
//
// The count is the number of buffers the cross-core data passes through, it is
// the depth of the pipeline between the two cores. It is the same number the
// producer side of the cross-core conditions uses.
//
// The attribute does not always live on the module: the buffer count manager
// puts it there, but the function that carries the pipeline may hold it as
// well. The function is looked at first, it is the more specific one. When
// neither of them holds the attribute, the default of the buffer count
// manager is used.
int UpdateConditionInfoPass::getPreloadBufferCount(ModuleOp module) {
  int count = 0;
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (auto attr = funcOp->getAttrOfType<IntegerAttr>(
            CVPipeline::kInterCoreBufCount)) {
      count = static_cast<int>(attr.getInt());
      break;
    }
  }
  if (count <= 0) {
    if (auto attr = module->getAttrOfType<IntegerAttr>(
            CVPipeline::kInterCoreBufCount)) {
      count = static_cast<int>(attr.getInt());
    }
  }
  if (count <= 0) {
    count = 2;
  }
  LDBG("Preload buffer count: " << count);
  return count;
}

// Preload conditions of `ifOp`.
//
// The computation blocks of a core that take part in a cross-core dependency
// are put on levels: B1 is the level of the blocks that open a dependency
// chain, B2 the level of the blocks that depend on them, and so on. Every
// block Bi is compared with the blocks of the higher levels:
//
//   i == 1 : c_Bi - c_Bj <= j - i + (buffer_num - 1)
//   i >  1 : c_Bi - c_Bj <= j - i
//
// i and j are the levels of the two blocks. The condition holds the block
// back: it may not run more than that many rounds ahead of the block it is
// compared with. The first level of the core is the one that fills the
// buffers, it is allowed to run buffer_num - 1 rounds further ahead than the
// other levels.
//
// Both blocks belong to the same core, so the comparison is always done in
// the scope of that core: the cube reads the counters of the cube blocks at
// their fixed address, the vector reads the counters of the vector blocks
// through the dynamic address of its own core.
//
// A block is only compared with the blocks that depend on it: the blocks of
// another dependency chain run in another round of the pipeline, their
// counters do not advance together with the ones of the block, the
// comparison would be meaningless.
Value UpdateConditionInfoPass::addPreloadConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp, bool isAIC,
    DenseMap<int, Value> &vectorSSBufferMemrefs,
    SmallVector<SmallVector<Value>> ssbufferPtrs,
    DenseMap<Value, Value> &loadedCounters) {
  const SmallVector<std::pair<scf::IfOp, int>> &blocks =
      isAIC ? info->preloadCubeBlocks : info->preloadVectorBlocks;
  // The level of the block in its core, 1 based.
  int ownLevel = getPreloadLevel(blocks, ifOp);
  if (ownLevel == 0) {
    return nullptr;
  }

  SmallVector<Value> ownMemrefs =
      getBlockCounterMemrefs(isAIC, ifOp, vectorSSBufferMemrefs, ssbufferPtrs);
  if (ownMemrefs.empty()) {
    // The counter of the block cannot be addressed, it cannot be compared.
    LDBG("No counter for the block " << ifOp
                                     << ", it has no preload condition.\n");
    return nullptr;
  }
  Value ownValue = loadCounter(builder, loc, ownMemrefs[0], loadedCounters);

  Value conditions = nullptr;
  auto combineCondition = [&](Value newCond) {
    if (!newCond) {
      return;
    }
    if (conditions) {
      conditions = builder.create<arith::AndIOp>(loc, conditions, newCond);
    } else {
      conditions = newCond;
    }
  };

  for (auto &block : getPreloadTargets(ifOp, isAIC)) {
    scf::IfOp target = block.first;
    // The level of the target, 1 based. Only the blocks of the higher levels
    // are compared, the blocks of the same level are independent of each
    // other.
    int targetLevel = block.second;

    SmallVector<Value> targetMemrefs = getBlockCounterMemrefs(
        isAIC, target, vectorSSBufferMemrefs, ssbufferPtrs);
    if (targetMemrefs.empty()) {
      // The counter of the target cannot be addressed, it cannot be
      // compared.
      continue;
    }

    // The first level of the core fills the buffers, it may run buffer_num - 1
    // rounds further ahead than the other levels. The other levels are only
    // held back by the distance that separates them.
    int limit = (ownLevel == 1)
                    ? (targetLevel - ownLevel) + (preloadBufferCount - 1)
                    : (targetLevel - ownLevel);

    Value targetValue =
        loadCounter(builder, loc, targetMemrefs[0], loadedCounters);
    Value diff = builder.create<arith::SubIOp>(loc, ownValue, targetValue);
    Value limitConst =
        builder.create<arith::ConstantIntOp>(loc, limit, CONST_INT_TYPE);
    combineCondition(builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, diff, limitConst));
  }

  return conditions;
}

// Allocate the SSBuffer pointers of the block level counters.
//   vector block: one counter per vector core, at byteOffset and
//                 byteOffset + VECTOR_SSBUF_OFFSET
//   cube block:   a single counter at byteOffset
// Every counter is reset to 0 before the first scope.
SmallVector<SmallVector<Value>>
UpdateConditionInfoPass::allocSSBuffer(ModuleOp module) {
  OpBuilder builder(module.getContext());
  auto i32Type = builder.getIntegerType(CONST_INT_TYPE);

  SmallVector<SmallVector<Value>> ssbufferMemrefs;
  if (blockCounterOwners.empty()) {
    LDBG("No block level counter to allocate!" << "\n");
    return ssbufferMemrefs;
  }

  int numSlots = static_cast<int>(blockCounterOwners.size());
  SmallVector<Value> core0Memrefs(numSlots);
  SmallVector<Value> core1Memrefs(numSlots);

  // Boundaries of the initialization block, they are needed to place the
  // PIPE_S synchronization around it.
  Operation *firstInitOp = nullptr;
  Operation *lastInitOp = nullptr;

  module->walk([&](Operation *op) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      builder.setInsertionPoint(scopeOp);
      auto loc = scopeOp.getLoc();
      auto zeroConst = builder.create<arith::ConstantOp>(
          scopeOp->getLoc(), i32Type, builder.getIntegerAttr(i32Type, 0));
      if (!firstInitOp) {
        firstInitOp = zeroConst.getOperation();
      }

      for (int slot = 0; slot < numSlots; slot++) {
        auto slotIt = info->blockCounterSlots.find(blockCounterOwners[slot]);
        if (slotIt == info->blockCounterSlots.end()) {
          continue;
        }
        const BlockCounterSlot &counterSlot = slotIt->second;

        auto memref0 =
            createPointerCastOp(builder, loc, counterSlot.byteOffset);
        core0Memrefs[slot] = memref0.getResult();
        auto store0 = builder.create<memref::StoreOp>(scopeOp->getLoc(),
                                                      zeroConst, memref0);
        lastInitOp = store0.getOperation();

        if (counterSlot.isVector) {
          // The second vector core keeps its own copy of the counter.
          auto memref1 = createPointerCastOp(
              builder, loc, counterSlot.byteOffset + VECTOR_SSBUF_OFFSET);
          core1Memrefs[slot] = memref1.getResult();
          auto store1 = builder.create<memref::StoreOp>(scopeOp->getLoc(),
                                                        zeroConst, memref1);
          lastInitOp = store1.getOperation();
        } else {
          core1Memrefs[slot] = memref0.getResult();
        }
      }
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  counterInitFirstOp = firstInitOp;
  counterInitLastOp = lastInitOp;

  ssbufferMemrefs.push_back(core0Memrefs);
  ssbufferMemrefs.push_back(core1Memrefs);
  return ssbufferMemrefs;
}

// Insert the PIPE_S synchronization around the counter initialization.
//
// The counters are initialized in front of the first scope of the module. The
// initialization is a part of both the cube and the vector kernel, so a core
// can read a counter that the other core has not initialized yet. The
// synchronization is a handshake between the two cores that brackets the
// initialization:
//
//   // before the initialization
//   sync_block_set  (vector)   the vector core announces that it reached the
//   sync_block_wait (vector)   initialization region and waits for the cube
//   sync_block_wait (cube)     core, the cube core waits for the vector core
//   sync_block_set  (cube)     and announces that it reached it as well
//   ... counter initialization ...
//   sync_block_set  (vector)   the vector core announces that the counters
//   sync_block_wait (cube)     are initialized and waits for the cube core,
//   sync_block_set  (cube)     the cube core announces the same and waits
//   sync_block_wait (vector)   for the vector core
//
// All the ops share the same flag: 15 is the flag that is reserved for the
// pipe synchronization, the cross core transfer flags live in the range
// 0..14 and must not be reused here.
//
// The ops carry an explicit hivm.tcore_type attribute, so the mixed kernel
// splitter puts the vector ops into the vector kernel and the cube ops into
// the cube kernel. Both kernels wait for each other at the end of the
// initialization, so no core enters its main loop before the counters of the
// other core are initialized.
void UpdateConditionInfoPass::insertCounterInitPipeS(ModuleOp module) {
  // Without cross core dependencies there is no counter to synchronize.
  if (!info || info->crossCoreDependentMap.empty()) {
    return;
  }

  if (!counterInitFirstOp || !counterInitLastOp) {
    return;
  }

  // The handshake needs both cores, a single core kernel has no partner to
  // wait for.
  auto cubeCoreType =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto vectorCoreType =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  bool hasCubeScope = false;
  bool hasVectorScope = false;
  module.walk([&](scope::ScopeOp scopeOp) {
    auto scopeTypeAttr =
        scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!scopeTypeAttr) {
      return;
    }
    if (scopeTypeAttr == cubeCoreType) {
      hasCubeScope = true;
    } else if (scopeTypeAttr == vectorCoreType) {
      hasVectorScope = true;
    }
  });
  if (!hasCubeScope || !hasVectorScope) {
    return;
  }

  auto pipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  Location loc = counterInitFirstOp->getLoc();

  OpBuilder builder(module.getContext());
  auto pipeSFlag = builder.getIntegerAttr(builder.getI64Type(), kPipeSFlagId);

  // Before the initialization: both cores announce that they reached the
  // initialization region and wait for the other one.
  builder.setInsertionPoint(counterInitFirstOp);
  builder.create<SyncBlockSetOp>(loc, vectorCoreType, pipeType, pipeType,
                                 pipeSFlag);
  builder.create<SyncBlockWaitOp>(loc, vectorCoreType, pipeType, pipeType,
                                  pipeSFlag);
  builder.create<SyncBlockWaitOp>(loc, cubeCoreType, pipeType, pipeType,
                                  pipeSFlag);
  builder.create<SyncBlockSetOp>(loc, cubeCoreType, pipeType, pipeType,
                                 pipeSFlag);

  // After the initialization: both cores announce that the counters are
  // initialized and wait for the other one, so that no core reads a counter
  // that the other core has not initialized yet.
  builder.setInsertionPointAfter(counterInitLastOp);
  builder.create<SyncBlockSetOp>(loc, vectorCoreType, pipeType, pipeType,
                                 pipeSFlag);
  builder.create<SyncBlockWaitOp>(loc, cubeCoreType, pipeType, pipeType,
                                  pipeSFlag);
  builder.create<SyncBlockSetOp>(loc, cubeCoreType, pipeType, pipeType,
                                 pipeSFlag);
  builder.create<SyncBlockWaitOp>(loc, vectorCoreType, pipeType, pipeType,
                                  pipeSFlag);
}

// Collect dependency buffer
void UpdateConditionInfoPass::collectDependencyBuffers(
    ModuleOp module, SmallVector<Operation *> &mainLoopOps,
    DenseMap<Operation *,
             DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
        &intraCoreBuffersMap) {
  // Collect intraCoreBuffers for all main_loop for/while ops
  for (Operation *loopOp : mainLoopOps) {
    auto depsIt = info->intraCoreDependentMap.find(loopOp);
    if (depsIt == info->intraCoreDependentMap.end())
      continue;
    auto &loopDeps = depsIt->second;
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        intraCoreBuffers;
    int intraCoreIdx = 0;
    loopOp->walk([&](Operation *op) {
      auto entryIt = loopDeps.find(op);
      if (entryIt == loopDeps.end())
        return;
      intraCoreBuffers[intraCoreIdx][entryIt->first] = entryIt->second;
      intraCoreIdx++;
    });
    intraCoreBuffersMap[loopOp] = std::move(intraCoreBuffers);
  }
}

// Helper: Find the tcb group id that contains op
// Returns the group id if found, -1 otherwise
static int findTcbGroupId(
    Operation *op,
    DenseMap<int, SmallVector<Operation *>> &tightlyCoupledBufferGroups) {
  for (auto &tcbEntry : tightlyCoupledBufferGroups) {
    if (llvm::is_contained(tcbEntry.second, op)) {
      return tcbEntry.first;
    }
  }
  return UPDATE_CONDITION_INFO_FAILED;
}

// Helper: Add all equivalent ops from tcbOps to ops (excluding op itself)
int addEquivalentOps(Operation *op, SmallVector<Operation *> &tcbOps,
                     SmallVector<Operation *> &ops) {
  int ret = -1;
  for (Operation *equivOp : tcbOps) {
    if (equivOp != op && !llvm::is_contained(ops, equivOp)) {
      ret = 0;
      ops.push_back(equivOp);
    }
  }
  return ret;
}

int UpdateConditionInfoPass::buildIdxToVarMap(
    Operation *loopOp,
    const DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar) {
  int varIdx = 0;
  MutableArrayRef<BlockArgument> regionIterArgs;
  if (getLoopRegionIterArgs(loopOp, regionIterArgs) ==
      UPDATE_CONDITION_INFO_FAILED)
    return UPDATE_CONDITION_INFO_FAILED;
  int iterArgNum = static_cast<int>(regionIterArgs.size());

  const auto &innerDepIndices = info->innerDepConds[loopOp];
  if (innerDepIndices.size() < intraCoreBuffers.size()) {
    LDBG("Not enough inner dependency condition indices: assigned "
         << innerDepIndices.size() << ", expected " << intraCoreBuffers.size()
         << ", loopOp=" << *loopOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // intraCoreBuffers stays a lookup table. Group order is this vector.
  SmallVector<int> groupIndices;
  groupIndices.reserve(intraCoreBuffers.size());
  for (const auto &entry : intraCoreBuffers)
    groupIndices.push_back(entry.first);
  llvm::sort(groupIndices);

  for (int idx : groupIndices) {
    int argIdx = innerDepIndices[varIdx];
    if (argIdx < 0 || argIdx >= iterArgNum) {
      LDBG("Invalid inner dependency arg index: "
           << argIdx << ", iter args " << iterArgNum << ", loopOp=" << *loopOp
           << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    idxToVar[idx] = regionIterArgs[argIdx];
    LDBG("Assign intraCore buffer group " << idx << " to iter arg index "
                                          << argIdx << "\n");
    varIdx++;
  }

  LDBG("Assigned " << idxToVar.size() << " intraCore condition variables."
                   << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Helper function to build buffer dependency mappings (fully Operation* based)
// Outputs two reverse lookup tables for O(1) lookup during IR walk:
//   - consumerToGroups: consumer Op -> [groupIdx] (one op may consume several
//     groups)
//   - producerToGroups: producer Op -> [groupIdx]
static int buildBufferDependencyMappings(
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &buffers,
    DenseMap<Operation *, SmallVector<int>> &consumerToGroups,
    DenseMap<Operation *, SmallVector<int>> &outputToGroups) {
  for (auto &[groupIdx, deps] : buffers) {
    for (auto &[consumer, producers] : deps) {
      consumerToGroups[consumer].push_back(groupIdx);

      for (Operation *producer : producers) {
        outputToGroups[producer].push_back(groupIdx);
      }
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// getInputOutputValues - Analyze the input/output buffer groups used in a
// single ifOp This function traverses all operations within ifOp, identifies
// FixpipeOp, CopyOp and MaterializeInDestinationOp, and extracts intra-core
// buffer group indices from their operands and yield values.

// Data structure description
//   - intraCoreBuffers: {groupIdx -> (consumer -> [producers])}
//   - intraCoreInputValues: list of intra-core group indices read by this ifOp
//   - intraCoreOutputValues: list of intra-core group indices written by this
//   ifOp

// Processing flow
//   1. Build reverse mapping table from Value to groupIdx
//   2. Traverse Fixpipe/Copy/MaterializeInDestination ops in ifOp, collect
//   input/output groups
//   3. Traverse ifOp.thenYield() operands, collect yield output groups
//   4. Deduplicate and output the two groups of index values
int UpdateConditionInfoPass::getInputOutputValues(
    scf::IfOp ifOp,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        intraCoreBuffers,
    SmallVector<int> &intraCoreInputValues,
    SmallVector<int> &intraCoreOutputValues) {
  DenseSet<int> intraCoreInputSet;
  DenseSet<int> intraCoreOutputSet;

  // Build output mappings for intra-core
  // Same producer/output can be used by multiple consumers/inputs, so we need
  // to track all related groups
  DenseMap<Operation *, SmallVector<int>> intraCoreOutputToGroups;

  // Add consumer mappings for input dependency identification
  DenseMap<Operation *, SmallVector<int>> intraCoreConsumerToGroups;

  // Build intra-core mappings
  if (buildBufferDependencyMappings(intraCoreBuffers, intraCoreConsumerToGroups,
                                    intraCoreOutputToGroups) ==
      UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // collect input/output groups
  ifOp.walk([&](Operation *op) {
    if (op == ifOp)
      return WalkResult::advance();

    // Check if this op is a consumer (for input dependency)
    if (intraCoreConsumerToGroups.count(op)) {
      for (int idx : intraCoreConsumerToGroups[op])
        intraCoreInputSet.insert(idx);
    }

    // Check if this op is a producer (for output dependency)
    if (intraCoreOutputToGroups.count(op)) {
      for (int idx : intraCoreOutputToGroups[op]) {
        intraCoreOutputSet.insert(idx);
      }
    }
    return WalkResult::advance();
  });

  intraCoreInputValues.assign(intraCoreInputSet.begin(),
                              intraCoreInputSet.end());
  intraCoreOutputValues.assign(intraCoreOutputSet.begin(),
                               intraCoreOutputSet.end());
  llvm::sort(intraCoreInputValues);
  llvm::sort(intraCoreOutputValues);

  LDBG("==== Intra Core Values ====" << "\n");
  LLVM_DEBUG(
      logConditionGroupIndices("intraCoreInputValues: ", intraCoreInputValues));
  LLVM_DEBUG(logConditionGroupIndices("intraCoreOutputValues: ",
                                      intraCoreOutputValues));

  return UPDATE_CONDITION_INFO_SUCCESS;
}

Value UpdateConditionInfoPass::getVarValue(Operation *loopOp, int varIndex) {
  if (!info->innerDepConds.count(loopOp))
    return Value();
  SmallVector<int> &innerDepIndices = info->innerDepConds[loopOp];
  if (varIndex < (int)innerDepIndices.size()) {
    int argIdx = innerDepIndices[varIndex];
    Value arg;
    if (getLoopRegionIterArg(loopOp, argIdx, arg) ==
        UPDATE_CONDITION_INFO_FAILED)
      return Value();
    return arg;
  }
  return Value();
}

// Build the information of the producer group.
int UpdateConditionInfoPass::buildOutputGroups(
    SmallVector<int> &intraCoreOutputValues,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar,
    SmallVector<OutputGroupInfo> &outputGroups) {
  outputGroups.clear();

  for (int idx : intraCoreOutputValues) {
    auto bufferIt = intraCoreBuffers.find(idx);
    if (bufferIt == intraCoreBuffers.end()) {
      LDBG("Failed to build output groups: no buffer entry for intraCore "
           "output group "
           << idx << "." << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) {
      LDBG("Failed to build output groups: no control variable for intraCore "
           "output group "
           << idx << "." << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    Value var = varIt->second;

    for (auto &entry : bufferIt->second) {
      SmallVector<Operation *> outputOps;
      for (Operation *producer : entry.second) {
        outputOps.push_back(producer);
      }
      if (outputOps.empty())
        continue;

      bool flag = true;
      for (auto &outputGroup : outputGroups) {
        if (outputGroup.outputs == outputOps) {
          outputGroup.inputVars.push_back(var);
          flag = false;
          break;
        }
      }
      if (flag) {
        OutputGroupInfo groupInfo;
        groupInfo.outputs = outputOps;
        groupInfo.inputVars.push_back(var);
        outputGroups.push_back(groupInfo);
      }
    }
  }

  LDBG("Built " << outputGroups.size() << " intraCore output groups." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// ---------------------------------------------------------------------------
// Block level cross-core conditions
// ---------------------------------------------------------------------------

// Memrefs holding the counter of `targetBlock` as seen by the reading core.
//   cube reader  : vector target -> {core0 copy, core1 copy};
//                  cube target   -> {the single copy}.
//   vector reader: vector target -> {the copy of the running core};
//                  cube target   -> {the single copy}.
SmallVector<Value> UpdateConditionInfoPass::getBlockCounterMemrefs(
    bool isAIC, scf::IfOp targetBlock,
    DenseMap<int, Value> &vectorSSBufferMemrefs,
    ArrayRef<SmallVector<Value>> ssbufferPtrs) {
  SmallVector<Value> memrefs;

  auto slotIt = info->blockCounterSlots.find(targetBlock);
  if (slotIt == info->blockCounterSlots.end()) {
    LDBG("No counter slot for block " << targetBlock << "\n");
    return memrefs;
  }
  const BlockCounterSlot &counterSlot = slotIt->second;
  int slot = counterSlot.slot;

  if (!counterSlot.isVector) {
    // A cube block owns a single counter.
    memrefs.push_back(ssbufferPtrs[0][slot]);
    return memrefs;
  }

  if (isAIC) {
    // The cube reads both copies of a vector block at their fixed addresses.
    memrefs.push_back(ssbufferPtrs[0][slot]);
    memrefs.push_back(ssbufferPtrs[1][slot]);
    return memrefs;
  }

  // The vector reads the copy of its own core through a dynamic address.
  auto vectorIt = vectorSSBufferMemrefs.find(slot);
  if (vectorIt == vectorSSBufferMemrefs.end()) {
    LDBG("No dynamic address for the vector counter of slot " << slot << "\n");
    return memrefs;
  }
  memrefs.push_back(vectorIt->second);
  return memrefs;
}

// Compute the dynamic addresses of the vector side counters. For every slot
// the result holds the memref of the counter copy owned by the vector core
// running the code (slotBase + sub_block_idx * 1024). The copy of the other
// vector core is only read by the cube core, which addresses it through a
// fixed address.
std::optional<DenseMap<int, Value>>
UpdateConditionInfoPass::computeVectorSSBufferMemrefs(OpBuilder &builder,
                                                      Location loc,
                                                      Operation *scopeOp,
                                                      ArrayRef<int> slots) {
  if (!scopeOp) {
    LDBG("Scope Op is null pointer!");
    return std::nullopt;
  }

  DenseMap<int, Value> &cached = vectorSSBufferMemrefCache[scopeOp];

  SmallVector<int> missingSlots;
  for (int slot : slots) {
    if (!cached.count(slot)) {
      missingSlots.push_back(slot);
    }
  }
  if (missingSlots.empty()) {
    return cached;
  }

  Value ssbAddrOffset;
  auto offsetIt = vectorSubBlockOffsetCache.find(scopeOp);
  if (offsetIt == vectorSubBlockOffsetCache.end()) {
    builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
    Value vec1OffsetValue = builder.create<arith::ConstantIntOp>(
        loc, VECTOR_SSBUF_OFFSET, ADDR_INT_TYPE);
    auto subIdOp = builder.create<GetSubBlockIdxOp>(
        loc, builder.getIntegerType(ADDR_INT_TYPE));
    ssbAddrOffset =
        builder.create<arith::MulIOp>(loc, subIdOp, vec1OffsetValue);
    vectorSubBlockOffsetCache[scopeOp] = ssbAddrOffset;
  } else {
    ssbAddrOffset = offsetIt->second;
    // The offset is already in the block: the new addresses have to be built
    // after it, otherwise they would use a value that is defined later.
    builder.setInsertionPointAfter(ssbAddrOffset.getDefiningOp());
  }

  for (int slot : missingSlots) {
    if (slot < 0 || slot >= static_cast<int>(blockCounterOwners.size())) {
      continue;
    }
    auto slotIt = info->blockCounterSlots.find(blockCounterOwners[slot]);
    if (slotIt == info->blockCounterSlots.end()) {
      continue;
    }
    int byteOffset = slotIt->second.byteOffset;

    Value baseAddr =
        builder.create<arith::ConstantIntOp>(loc, byteOffset, ADDR_INT_TYPE);
    // The copy of the running core: base + sub_block_idx * 1024
    Value ownAddr = builder.create<arith::AddIOp>(loc, baseAddr, ssbAddrOffset);

    cached[slot] = builder.create<PointerCastOp>(
        loc, getSsbufMemrefType(builder), ownAddr);
  }

  return cached;
}

// Build the cross-core conditions of `ifOp`.
//
// Every cross-core dependency edge P -> C is described by the difference of
// the two block counters:
//   consumer side: c_P - c_C > 0    the producer produced more rounds than
//                                    the consumer consumed
//   producer side: c_P - c_C < N_e  the producer is less than N_e rounds
//                                    ahead of the consumer
// A cube block consuming a vector block additionally waits until both vector
// cores produced the same number of rounds (c_V0 == c_V1).
//
// A vector block only ever looks at the counter copy of its own core: the
// number of rounds the other core produced does not constrain it.
Value UpdateConditionInfoPass::addCrossCoreBlockConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp, bool isAIC,
    Value zeroConst, DenseMap<int, Value> &vectorSSBufferMemrefs,
    SmallVector<SmallVector<Value>> ssbufferPtrs,
    DenseMap<Value, Value> &loadedCounters) {
  Value conditions = nullptr;

  auto combineCondition = [&](Value newCond) {
    if (!newCond) {
      return;
    }
    if (conditions) {
      conditions = builder.create<arith::AndIOp>(loc, conditions, newCond);
    } else {
      conditions = newCond;
    }
  };

  auto edgesIt = info->crossCoreBlockEdges.find(ifOp);
  if (edgesIt == info->crossCoreBlockEdges.end()) {
    return nullptr;
  }

  // Every counter is read once, the loaded value is shared by all conditions.
  for (const CrossCoreBlockEdge &edge : edgesIt->second) {
    bool currentIsProducer = (edge.producer == ifOp);
    scf::IfOp producer = edge.producer;
    scf::IfOp consumer = edge.consumer;

    SmallVector<Value> producerMemrefs = getBlockCounterMemrefs(
        isAIC, producer, vectorSSBufferMemrefs, ssbufferPtrs);
    SmallVector<Value> consumerMemrefs = getBlockCounterMemrefs(
        isAIC, consumer, vectorSSBufferMemrefs, ssbufferPtrs);
    if (producerMemrefs.empty() || consumerMemrefs.empty()) {
      LDBG("Missing counter address for a cross-core edge of " << ifOp << "\n");
      return nullptr;
    }

    bool producerIsVector = producerMemrefs.size() > 1;
    bool consumerIsVector = consumerMemrefs.size() > 1;

    // The pairs of counters to compare.
    SmallVector<std::pair<Value, Value>> pairs;
    if (isAIC) {
      if (currentIsProducer) {
        // The cube produces: it must not run more than N_e rounds ahead of
        // any of the vector cores, so both copies of the consumer counter are
        // compared against the single cube counter.
        pairs.push_back({producerMemrefs[0], consumerMemrefs[0]});
        if (consumerIsVector) {
          pairs.push_back({producerMemrefs[0], consumerMemrefs[1]});
        }
      } else {
        // The cube consumes: it waits for every vector core, so both copies
        // of the producer counter are compared against the single cube
        // counter.
        pairs.push_back({producerMemrefs[0], consumerMemrefs[0]});
        if (producerIsVector) {
          pairs.push_back({producerMemrefs[1], consumerMemrefs[0]});
        }
      }
    } else {
      // The vector only compares the counters of the running core.
      pairs.push_back({producerMemrefs[0], consumerMemrefs[0]});
    }

    for (const auto &pair : pairs) {
      Value producerValue =
          loadCounter(builder, loc, pair.first, loadedCounters);
      Value consumerValue =
          loadCounter(builder, loc, pair.second, loadedCounters);
      Value diff =
          builder.create<arith::SubIOp>(loc, producerValue, consumerValue);
      if (currentIsProducer) {
        Value bufferNum = builder.create<arith::ConstantIntOp>(
            loc, edge.bufferNum, CONST_INT_TYPE);
        combineCondition(builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, diff, bufferNum));
      } else {
        combineCondition(builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sgt, diff, zeroConst));
      }
    }

    // A cube block reads the whole slot filled by the two vector cores: it
    // has to wait until both cores produced the same number of rounds.
    if (!currentIsProducer && isAIC && producerIsVector) {
      Value core0Value =
          loadCounter(builder, loc, producerMemrefs[0], loadedCounters);
      Value core1Value =
          loadCounter(builder, loc, producerMemrefs[1], loadedCounters);
      combineCondition(builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, core0Value, core1Value));
    }
  }

  return conditions;
}

// Body of the main loop: the counter loads are placed at its beginning.
// The body of an scf.for becomes the do region of the scf.while the main
// loop is turned into, the order of its operations is kept.
Block *UpdateConditionInfoPass::getMainLoopBody(Operation *loopOp) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    return forOp.getBody();
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    return whileOp.getAfterBody();
  }
  return nullptr;
}

// Read the counter held by `memref`.
//
// The load is not placed where the condition using it is built, but at the
// beginning of the body of the main loop: every counter read of a loop
// iteration is gathered there, in front of the compute blocks that use them.
// The loads stay inside the loop, so every iteration reads the counters
// again.
//
// `loadedCounters` maps the counter address to the value that was loaded
// from it, a counter is only read once for the conditions of a block.
Value UpdateConditionInfoPass::loadCounter(
    OpBuilder &builder, Location loc, Value memref,
    DenseMap<Value, Value> &loadedCounters) {
  auto it = loadedCounters.find(memref);
  if (it != loadedCounters.end()) {
    return it->second;
  }

  OpBuilder loadBuilder(builder.getContext());
  if (counterLoadTail) {
    loadBuilder.setInsertionPointAfter(counterLoadTail);
  } else if (counterLoadBlock) {
    loadBuilder.setInsertionPointToStart(counterLoadBlock);
  } else {
    // No main loop body known, keep the load where it was asked for.
    loadBuilder.setInsertionPoint(builder.getInsertionBlock(),
                                  builder.getInsertionPoint());
  }

  auto loadOp = loadBuilder.create<memref::LoadOp>(loc, memref, ValueRange{});
  auto markOp = loadBuilder.create<annotation::MarkOp>(loc, loadOp);
  markOp->setAttr(kMemrefExtVolatile, loadBuilder.getUnitAttr());
  counterLoadTail = markOp;
  loadedCounters[memref] = loadOp.getResult();
  return loadOp.getResult();
}

// Increment the counter owned by `ifOp` at the end of its then block.
// Every counter has a single writer: the block owning it. A vector block owns
// one counter per vector core and only bumps the copy of the core it runs on.
void UpdateConditionInfoPass::updateBlockCounterVars(
    OpBuilder &builder, Location loc, scf::IfOp ifOp, bool isAIC,
    Value oneConst, DenseMap<int, Value> &vectorSSBufferMemrefs,
    SmallVector<SmallVector<Value>> ssbufferPtrs,
    DenseMap<Value, Value> &loadedCounters) {
  auto slotIt = info->blockCounterSlots.find(ifOp);
  if (slotIt == info->blockCounterSlots.end()) {
    return;
  }
  const BlockCounterSlot &counterSlot = slotIt->second;

  Value counterMemref;
  if (counterSlot.isVector) {
    // The running vector core bumps the copy of its own core.
    auto vectorIt = vectorSSBufferMemrefs.find(counterSlot.slot);
    if (vectorIt == vectorSSBufferMemrefs.end()) {
      LDBG("No dynamic address for the vector counter of slot "
           << counterSlot.slot << "\n");
      return;
    }
    counterMemref = vectorIt->second;
  } else {
    counterMemref = ssbufferPtrs[0][counterSlot.slot];
  }

  Block *thenBlock = &ifOp.getThenRegion().front();
  auto yieldOp = cast<scf::YieldOp>(thenBlock->getTerminator());
  builder.setInsertionPoint(yieldOp);

  // The counter has a single writer: the block that owns it. Nobody writes it
  // between the load of the conditions and this update, so the value the
  // conditions loaded is still the current one and the counter does not have
  // to be loaded again.
  Value currentValue = nullptr;
  auto loadedIt = loadedCounters.find(counterMemref);
  if (loadedIt != loadedCounters.end()) {
    currentValue = loadedIt->second;
  } else {
    // The conditions of this block did not read its counter, load it.
    currentValue = loadCounter(builder, loc, counterMemref, loadedCounters);
  }
  Value newValue = builder.create<arith::AddIOp>(loc, currentValue, oneConst);
  builder.create<memref::StoreOp>(loc, newValue, counterMemref, ValueRange{});
}

// Set the crossCore condition of `ifOp` and bump its own counter.
int UpdateConditionInfoPass::setCrossCoreCondition(
    scf::IfOp ifOp, Operation *loopOp,
    SmallVector<SmallVector<Value>> ssbufferPtrs, Value &crossCoreCond,
    Value &preloadCond) {
  OpBuilder builder(ifOp);
  Location loc = ifOp.getLoc();
  crossCoreCond = nullptr;
  preloadCond = nullptr;

  // The counter loads are gathered at the beginning of the body of the main
  // loop, in front of the compute blocks that read them.
  if (!counterLoadBlock) {
    counterLoadBlock = getMainLoopBody(loopOp);
  }

  // Only the blocks taking part in a cross-core dependency own a counter.
  if (!info->blockCounterSlots.count(ifOp)) {
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  // ========== Part 1: Preparation ==========
  // Determine whether the current ifblock is on cube or vector core
  scope::ScopeOp scopeOp = findOwningScope(ifOp);
  bool isAIC = false;
  bool isAIV = false;
  if (!getScopeCoreType(scopeOp, isAIC, isAIV)) {
    LDBG("ifblock not in a correct scope block" << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // The vector side addresses the counters of the vector blocks through the
  // dynamic address of its own core.
  DenseMap<int, Value> vectorSSBufferMemrefs;
  if (!isAIC) {
    SmallVector<int> neededSlots;
    auto edgesIt = info->crossCoreBlockEdges.find(ifOp);
    if (edgesIt != info->crossCoreBlockEdges.end()) {
      for (const CrossCoreBlockEdge &edge : edgesIt->second) {
        for (scf::IfOp block : {edge.producer, edge.consumer}) {
          auto slotIt = info->blockCounterSlots.find(block);
          if (slotIt == info->blockCounterSlots.end()) {
            continue;
          }
          if (!slotIt->second.isVector) {
            continue;
          }
          if (!llvm::is_contained(neededSlots, slotIt->second.slot)) {
            neededSlots.push_back(slotIt->second.slot);
          }
        }
      }
    }
    // The preload conditions compare the counters of the blocks of the same
    // core, the addresses of those counters are needed as well.
    for (auto &target : getPreloadTargets(ifOp, isAIC)) {
      auto slotIt = info->blockCounterSlots.find(target.first);
      if (slotIt == info->blockCounterSlots.end()) {
        continue;
      }
      if (!slotIt->second.isVector) {
        continue;
      }
      if (!llvm::is_contained(neededSlots, slotIt->second.slot)) {
        neededSlots.push_back(slotIt->second.slot);
      }
    }
    auto result = computeVectorSSBufferMemrefs(
        builder, loc, scopeOp.getOperation(), neededSlots);
    if (!result) {
      LDBG("computeVectorSSBufferMemrefs failed!");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    vectorSSBufferMemrefs = std::move(*result);
  }

  builder.setInsertionPoint(ifOp);

  Value zeroConst =
      builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
  Value oneConst = builder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);

  // ========== Part 2: Add cross-core conditions ==========
  // The counters that are read by the conditions are kept, the update of the
  // counter of this block reuses the value that was loaded here. The cache is
  // shared by all the blocks of the loop: a counter is read once per
  // iteration, every block that needs it reuses that value.
  crossCoreCond = addCrossCoreBlockConditions(builder, loc, ifOp, isAIC,
                                              zeroConst, vectorSSBufferMemrefs,
                                              ssbufferPtrs, loadedCounters);

  // ========== Part 2.5: Add the preload conditions ==========
  // The blocks of a core are held back by the blocks that come after them: a
  // block may not run more than a few rounds ahead of them.
  preloadCond =
      addPreloadConditions(builder, loc, ifOp, isAIC, vectorSSBufferMemrefs,
                           ssbufferPtrs, loadedCounters);

  // ========== Part 3: Update the own counter ==========
  updateBlockCounterVars(builder, loc, ifOp, isAIC, oneConst,
                         vectorSSBufferMemrefs, ssbufferPtrs, loadedCounters);

  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Collect the conditions for intra-core consumer values.
void UpdateConditionInfoPass::collectIntraCoreInputConditions(
    OpBuilder &builder, Location loc, SmallVector<int> &intraCoreInputValues,
    DenseMap<int, Value> &idxToVar, SmallVector<Value> &conditions,
    SetVector<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (intraCoreInputValues.empty()) {
    LDBG("No intraCore input conditions to collect." << "\n");
    return;
  }

  size_t beforeConditionNum = conditions.size();
  Value zeroConst =
      builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
  for (int idx : intraCoreInputValues) {
    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) {
      LDBG("Skip intraCore input group " << idx << ": no control variable."
                                         << "\n");
      continue;
    }

    Value var = varIt->second;
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) {
      varToUse = latestIt->second;
    }

    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               varToUse, zeroConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
    LDBG("Add intraCore input condition for group " << idx << "." << "\n");
  }
  LDBG("Collected " << (conditions.size() - beforeConditionNum)
                    << " intraCore input conditions." << "\n");
}

// Collect the conditions for intra-core producer values.
int UpdateConditionInfoPass::collectIntraCoreOutputConditions(
    OpBuilder &builder, Location loc,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
    SmallVector<Value> &conditions, SetVector<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (intraCoreOutputValues.empty()) {
    LDBG("No intraCore output conditions to collect." << "\n");
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  size_t beforeConditionNum = conditions.size();
  SmallVector<OutputGroupInfo> outputGroups;
  if (buildOutputGroups(intraCoreOutputValues, intraCoreBuffers, idxToVar,
                        outputGroups) == UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  for (auto &group : outputGroups) {
    int size = group.outputs.size();
    Value limitVal =
        builder.create<arith::ConstantIntOp>(loc, size, CONST_INT_TYPE);
    for (Value var : group.inputVars) {
      Value varToUse = var;
      auto latestIt = controlVarToLatestValue.find(var);
      if (latestIt != controlVarToLatestValue.end()) {
        varToUse = latestIt->second;
      }

      Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 varToUse, limitVal);
      conditions.push_back(cond);
      usedVarsSet.insert(var);
      varUpdateTypes[var] = VarUpdateType::INC;
      LDBG("Add intraCore output condition with producer limit " << size << "."
                                                                 << "\n");
    }
  }
  LDBG("Collected " << (conditions.size() - beforeConditionNum)
                    << " intraCore output conditions." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Build the ifOp variable mapping for the tensor iter_args
int UpdateConditionInfoPass::buildTensorIterArgIfOpVarMap(Operation *loopOp) {
  // Clear any previous data
  tensorIterArgIfOpVars.clear();

  if (!info->tensorIterArgDepsMap.count(loopOp) ||
      !info->tensorIterArgIndicesMap.count(loopOp)) {
    LDBG("Skip buildTensorIterArgIfOpVarMap: no tensor iter_args info for "
         << *loopOp << "\n");
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  auto &depsVec = info->tensorIterArgDepsMap[loopOp];
  auto &indicesMap = info->tensorIterArgIndicesMap[loopOp];

  llvm::DenseMap<scf::IfOp, SetVector<Value>> producerVars;
  llvm::DenseMap<scf::IfOp, SetVector<Value>> consumerVars;

  for (auto &depEntry : depsVec) {
    Value origIterArg = depEntry.iterArg;
    TensorIterArgIfOpRelation &relation = depEntry;

    if (!indicesMap.count(origIterArg)) {
      LDBG("[Error]: origIterArg not found in indicesMap, loopOp=" << *loopOp
                                                                   << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    SmallVector<int> &argIndices = indicesMap[origIterArg];

    if (relation.consumers.size() != argIndices.size()) {
      LDBG("[Error]: consumers size mismatch: "
           << relation.consumers.size() << " vs " << argIndices.size()
           << ", loopOp=" << *loopOp << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    // Establish a mapping (one-to-one) from consumers to variables
    // for: region iter args; while-do: after-region args.
    SmallVector<std::pair<scf::IfOp, Value>> consumerToVar;
    for (size_t i = 0; i < relation.consumers.size(); ++i) {
      scf::IfOp consumer = relation.consumers[i];
      Value var;
      if (getLoopRegionIterArg(loopOp, argIndices[i], var) ==
          UPDATE_CONDITION_INFO_FAILED) {
        LDBG("[Error]: invalid tensor iter_arg index "
             << argIndices[i] << ", loopOp=" << *loopOp
             << ", consumer=" << consumer << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }
      consumerToVar.push_back({consumer, var});
    }

    if (relation.producer) {
      for (auto &[consumer, var] : consumerToVar)
        producerVars[relation.producer].insert(var);
    }

    for (auto &[consumer, var] : consumerToVar)
      consumerVars[consumer].insert(var);
  }

  // Convert the temporary data structure to tensorIfOpVarMap
  for (auto &[producer, vars] : producerVars) {
    auto &ifOpVars = tensorIterArgIfOpVars[producer];
    for (Value var : vars) {
      ifOpVars.producerVars.push_back(var);
    }
  }

  for (auto &[consumer, vars] : consumerVars) {
    auto &ifOpVars = tensorIterArgIfOpVars[consumer];
    for (Value var : vars) {
      ifOpVars.consumerVars.push_back(var);
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Collect the conditions for tensor iter arg consumer values.
void UpdateConditionInfoPass::collectTensorIterArgInputConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp,
    SmallVector<Value> &conditions, SetVector<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (!tensorIterArgIfOpVars.count(ifOp)) {
    return;
  }

  auto &ifOpVars = tensorIterArgIfOpVars[ifOp];
  for (Value var : ifOpVars.consumerVars) {
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) {
      varToUse = latestIt->second;
    }

    Value oneConst =
        builder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               varToUse, oneConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
    LDBG("Add tensor iter arg consumer condition for var.\n");
  }
}

// Collect the conditions for tensor iter arg producer values.
void UpdateConditionInfoPass::collectTensorIterArgOutputConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp,
    SmallVector<Value> &conditions, SetVector<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (!tensorIterArgIfOpVars.count(ifOp)) {
    return;
  }

  auto &ifOpVars = tensorIterArgIfOpVars[ifOp];
  for (Value var : ifOpVars.producerVars) {
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) {
      varToUse = latestIt->second;
    }

    Value zeroConst =
        builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               varToUse, zeroConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::INC;
    LDBG("Add tensor iter arg producer condition (var == 0) and +1 update for "
         "var.\n");
  }
}

// Set the intraCore condition.
int UpdateConditionInfoPass::setIntraCoreCondition(
    ModuleOp module, scf::IfOp ifOp,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    SmallVector<int> &intraCoreInputValues,
    SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, Value &intraCoreCond) {
  LDBG("Enter set intraCore condition." << "\n");
  intraCoreCond = Value();
  OpBuilder builder(ifOp.getContext());
  builder.setInsertionPoint(ifOp);
  Location loc = ifOp.getLoc();

  SmallVector<Value> conditions;
  SetVector<Value> usedVarsSet;
  LDBG("Collect intraCore conditions: inputs "
       << intraCoreInputValues.size() << ", outputs "
       << intraCoreOutputValues.size() << "\n");
  // Collect the conditions for intra-core consumer values.
  collectIntraCoreInputConditions(builder, loc, intraCoreInputValues, idxToVar,
                                  conditions, usedVarsSet, varUpdateTypes);
  // Collect the conditions for intra-core producer values.
  if (collectIntraCoreOutputConditions(
          builder, loc, intraCoreBuffers, intraCoreOutputValues, idxToVar,
          conditions, usedVarsSet,
          varUpdateTypes) == UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  // Collect tensor iter_args conditions
  collectTensorIterArgInputConditions(builder, loc, ifOp, conditions,
                                      usedVarsSet, varUpdateTypes);
  collectTensorIterArgOutputConditions(builder, loc, ifOp, conditions,
                                       usedVarsSet, varUpdateTypes);

  if (!conditions.empty()) {
    intraCoreCond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i) {
      intraCoreCond =
          builder.create<arith::AndIOp>(loc, intraCoreCond, conditions[i]);
    }
  }

  currentUsedVars.clear();
  for (Value var : usedVarsSet) {
    currentUsedVars.push_back(var);
  }
  LDBG("Built " << conditions.size() << " intraCore conditions using "
                << currentUsedVars.size() << " control variables." << "\n");

  LDBG("Exit set intraCore condition." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Set the FlowOpt extra condition for the third if block in the DAG
int UpdateConditionInfoPass::setFlowOptCondition(scf::IfOp currentIfOp,
                                                 Operation *loopOp,
                                                 Value &flowOptCond) {
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp) {
    flowOptCond = nullptr;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  // Check if current ifOp is a target node (third node) in flowOptIfOpPairs
  if (!info->flowOptIfOpPairs.count(currentIfOp)) {
    LDBG("Current ifOp is not a flowOpt target node, skip.");
    flowOptCond = nullptr;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  // Check if flowOpt condition is needed based on buffer counts
  // - Cross-core buffer count > CROSS_CORE_BUFFER_COUNT_THRESHOLD
  // - Intra-core buffer count > INTRA_CORE_BUFFER_COUNT_THRESHOLD
  if (info->crossCoreBufferCount <= CROSS_CORE_BUFFER_COUNT_THRESHOLD ||
      info->intraCoreBufferCount <= INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    flowOptCond = nullptr;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  // Get the start node
  scf::IfOp sourceIfOp = info->flowOptIfOpPairs[currentIfOp];
  if (!info->cntArgs.count(sourceIfOp)) {
    LDBG("[Error] Start node has no counter in cntArgs, cannot build flowOpt "
         "condition. currentIfOp="
         << currentIfOp << ", sourceIfOp=" << sourceIfOp << ", forOp=" << forOp
         << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // Create builder and location
  OpBuilder builder(currentIfOp);
  Location loc = currentIfOp.getLoc();

  Value counter = info->cntArgs[sourceIfOp];
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Build condition: counter >= upperBound
  Value cond1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              counter, upperBound);

  // Build condition: counter >= lowerBound + step * opt_num
  // opt_num = min(intraCoreBufferCount - 1, crossCoreBufferCount)
  int optInt =
      std::min(info->intraCoreBufferCount - 1, info->crossCoreBufferCount);

  auto stepIntType = dyn_cast<IntegerType>(step.getType());
  if (!stepIntType) {
    LDBG("[Error] forOp step is expected to be an integer type, got "
         << step.getType() << ", forOp=" << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  Value optNum =
      builder.create<arith::ConstantIntOp>(loc, optInt, stepIntType.getWidth());
  Value optOffset = builder.create<arith::MulIOp>(loc, step, optNum);
  Value lowerPlusOffset =
      builder.create<arith::AddIOp>(loc, lowerBound, optOffset);
  Value cond2 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              counter, lowerPlusOffset);

  // Combine: cond1 OR cond2
  flowOptCond = builder.create<arith::OrIOp>(loc, cond1, cond2);

  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Update DAG nodes after ifOp replacement
void UpdateConditionInfoPass::updateDAGAfterIfOpReplacement(scf::IfOp oldIfOp,
                                                            scf::IfOp newIfOp) {
  // 1. Update ifBlockDAG
  if (info->ifBlockDAG.count(oldIfOp)) {
    auto consumers = info->ifBlockDAG[oldIfOp];
    info->ifBlockDAG.erase(oldIfOp);
    info->ifBlockDAG[newIfOp] = consumers;
  }

  for (auto &entry : info->ifBlockDAG) {
    for (auto &edge : entry.second) {
      if (edge.first == oldIfOp) {
        edge.first = newIfOp;
      }
    }
  }

  // 2. Update flowOptIfOpPairs
  if (info->flowOptIfOpPairs.count(oldIfOp)) {
    auto source = info->flowOptIfOpPairs[oldIfOp];
    info->flowOptIfOpPairs.erase(oldIfOp);
    info->flowOptIfOpPairs[newIfOp] = source;
  }

  for (auto &entry : info->flowOptIfOpPairs) {
    if (entry.second == oldIfOp) {
      entry.second = newIfOp;
    }
  }

  // 3. Update the block level cross-core edges
  if (info->crossCoreBlockEdges.count(oldIfOp)) {
    auto edges = info->crossCoreBlockEdges[oldIfOp];
    info->crossCoreBlockEdges.erase(oldIfOp);
    info->crossCoreBlockEdges[newIfOp] = edges;
  }

  for (auto &entry : info->crossCoreBlockEdges) {
    for (auto &edge : entry.second) {
      if (edge.producer == oldIfOp) {
        edge.producer = newIfOp;
      }
      if (edge.consumer == oldIfOp) {
        edge.consumer = newIfOp;
      }
    }
  }

  // 4. Update the block counter slots
  if (info->blockCounterSlots.count(oldIfOp)) {
    auto slot = info->blockCounterSlots[oldIfOp];
    info->blockCounterSlots.erase(oldIfOp);
    info->blockCounterSlots[newIfOp] = slot;
  }

  // 5. Update the counter slot owners
  for (auto &owner : blockCounterOwners) {
    if (owner == oldIfOp) {
      owner = newIfOp;
    }
  }

  // 6. Update the preload block lists, the level of the block stays the same
  for (auto &block : info->preloadVectorBlocks) {
    if (block.first == oldIfOp) {
      block.first = newIfOp;
    }
  }
  for (auto &block : info->preloadCubeBlocks) {
    if (block.first == oldIfOp) {
      block.first = newIfOp;
    }
  }
}

// Update the mapping of control variables to their latest values
void UpdateConditionInfoPass::updateControlVarToLatestValue(scf::IfOp newIfOp,
                                                            scf::IfOp oldIfOp,
                                                            bool hasCounter,
                                                            Value counter) {
  if (currentUsedVars.empty() && !hasCounter) {
    LDBG("No control variable latest values to update." << "\n");
    return;
  }

  size_t origResultCount = oldIfOp.getNumResults();

  for (size_t i = 0; i < currentUsedVars.size(); ++i) {
    Value var = currentUsedVars[i];
    Value newValue = newIfOp.getResult(origResultCount + i);
    controlVarToLatestValue[var] = newValue;
    LDBG("Record latest intraCore control value at result index "
         << (origResultCount + i) << "." << "\n");
  }

  if (hasCounter) {
    size_t counterResultIdx = origResultCount + currentUsedVars.size();
    Value newCounterValue = newIfOp.getResult(counterResultIdx);
    controlVarToLatestValue[counter] = newCounterValue;
    LDBG("Record latest counter value at result index " << counterResultIdx
                                                        << "." << "\n");
  }
  LDBG("[DEBUG] controlVarToLatestValue size: "
       << controlVarToLatestValue.size() << "\n");

  for (auto &entry : controlVarToLatestValue) {
    LDBG("[DEBUG]   key = " << entry.first
                            << "  -->  new value = " << entry.second << "\n");
  }
}

// Update the yield in the forOp
int UpdateConditionInfoPass::updateForOpYield(scf::ForOp forOp) {
  if (controlVarToLatestValue.empty()) {
    LDBG("Failed to update forOp yield: no latest control variable values, "
         << "forOp=" << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  Location loc = forOp.getLoc();
  Block *forBody = forOp.getBody();
  auto yieldOp = dyn_cast<scf::YieldOp>(forBody->getTerminator());
  if (!yieldOp) {
    LDBG("Failed to update forOp yield: terminator is not scf.yield, forOp="
         << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
  if (newYieldOperands.size() != forOp.getNumRegionIterArgs()) {
    LDBG("Failed to update forOp yield: yield operands "
         << newYieldOperands.size() << ", iter args "
         << forOp.getNumRegionIterArgs() << ", forOp=" << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  DenseMap<Value, unsigned> iterArgToIndex;
  for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j) {
    iterArgToIndex[forOp.getRegionIterArgs()[j]] = j;
  }

  for (auto &entry : controlVarToLatestValue) {
    Value origVar = entry.first;
    Value latestValue = entry.second;
    auto it = iterArgToIndex.find(origVar);
    if (it == iterArgToIndex.end()) {
      LDBG("Failed to update forOp yield: control variable is not a region "
           "iter arg, forOp="
           << forOp << ", var=" << origVar << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    newYieldOperands[it->second] = latestValue;
    LDBG("Update forOp yield operand index " << it->second << "\n");
  }

  OpBuilder yieldBuilder(yieldOp);
  yieldBuilder.create<scf::YieldOp>(loc, newYieldOperands);
  yieldOp.erase();
  LDBG("Updated forOp yield with " << controlVarToLatestValue.size()
                                   << " latest control values." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::updateWhileOpYield(scf::WhileOp whileOp) {
  LDBG("Enter update whileOp yield " << "\n");
  if (controlVarToLatestValue.empty()) {
    LDBG("Skip update whileOp yield: no latest control variable values, "
         << "whileOp=" << whileOp << "\n");
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  Location loc = whileOp.getLoc();
  Block *afterBody = whileOp.getAfterBody();
  auto yieldOp = dyn_cast<scf::YieldOp>(afterBody->getTerminator());
  if (!yieldOp) {
    LDBG("Failed to update whileOp yield: terminator is not scf.yield, "
         << "whileOp=" << whileOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
  auto afterArgs = whileOp.getAfterArguments();
  if (newYieldOperands.size() != afterArgs.size()) {
    LDBG("Failed to update whileOp yield: yield operands "
         << newYieldOperands.size() << ", after args " << afterArgs.size()
         << ", whileOp=" << whileOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  DenseMap<Value, unsigned> iterArgToIndex;
  for (auto [idx, arg] : llvm::enumerate(afterArgs))
    iterArgToIndex[arg] = idx;

  for (auto &entry : controlVarToLatestValue) {
    Value origVar = entry.first;
    Value latestValue = entry.second;
    auto it = iterArgToIndex.find(origVar);
    if (it == iterArgToIndex.end()) {
      LDBG("Failed to update whileOp yield: control variable is not an after "
           "region iter arg, whileOp="
           << whileOp << ", var=" << origVar << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    newYieldOperands[it->second] = latestValue;
    LDBG("Update whileOp yield operand index " << it->second << "\n");
  }

  OpBuilder yieldBuilder(yieldOp);
  yieldBuilder.create<scf::YieldOp>(loc, newYieldOperands);
  yieldOp.erase();
  LDBG("Updated whileOp yield with " << controlVarToLatestValue.size()
                                     << " latest control values." << "\n");
  LDBG("Exit update whileOp yield " << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::updateLoopYield(Operation *loopOp) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
    return updateForOpYield(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp))
    return updateWhileOpYield(whileOp);
  LDBG("updateLoopYield expects scf.for or scf.while, got " << *loopOp << "\n");
  return UPDATE_CONDITION_INFO_FAILED;
}

int UpdateConditionInfoPass::collectSSBufferIfOps(
    Operation *loopOp, SmallVector<scf::IfOp> &ifOps) {
  auto collectIfOps = [&](Operation *op) -> WalkResult {
    if (!op->hasAttr(kIf)) {
      return WalkResult::advance();
    }

    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp) {
      LDBG("Found unsupported ssbuffer if op: "
           << op->getName() << " under loopOp=" << *loopOp << "\n");
      return WalkResult::interrupt();
    }

    ifOps.push_back(ifOp);
    return WalkResult::advance();
  };

  WalkResult ifWalkResult;
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    ifWalkResult = forOp.walk(collectIfOps);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    // while-do: only after-region.
    ifWalkResult = whileOp.getAfterBody()->walk(collectIfOps);
  } else {
    LDBG("collectSSBufferIfOps expects scf.for or scf.while, got " << *loopOp
                                                                   << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  if (ifWalkResult.wasInterrupted()) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::validateBlockCounters(Operation *loopOp,
                                                   size_t ifOpCount) {
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp) {
    // blockCounters / cntArgs are for-only.
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  auto counterIt = info->blockCounters.find(forOp);
  if (counterIt == info->blockCounters.end()) {
    LDBG("Failed to assign counters for ssbuffer if ops: no counters for "
         << "forOp=" << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  size_t counterNum = counterIt->second.size();
  if (ifOpCount > counterNum) {
    LDBG("Failed to assign counters for all ssbuffer if ops: if ops "
         << ifOpCount << ", counters " << counterNum << ", forOp=" << forOp
         << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

SmallVector<Type>
UpdateConditionInfoPass::buildNewIfResultTypes(scf::IfOp oldIfOp,
                                               bool hasCounter, Value counter) {
  SmallVector<Type> resultTypes;
  for (Value result : oldIfOp.getResults()) {
    resultTypes.push_back(result.getType());
  }
  for (Value var : currentUsedVars) {
    resultTypes.push_back(var.getType());
  }
  if (hasCounter) {
    resultTypes.push_back(counter.getType());
  }
  LDBG("Build new if result types: old results "
       << oldIfOp.getNumResults() << ", control vars " << currentUsedVars.size()
       << ", has counter " << hasCounter << "." << "\n");
  return resultTypes;
}

void UpdateConditionInfoPass::collectYieldOperands(
    Block &block, Operation *&yieldOp, SmallVector<Value> &yieldOperands) {
  yieldOp = nullptr;
  yieldOperands.clear();
  if (block.empty()) {
    LDBG("Collect yield operands: block is empty." << "\n");
    return;
  }

  Operation *lastOp = &block.back();
  if (!isa<scf::YieldOp>(lastOp)) {
    LDBG("Collect yield operands: block terminator is not scf.yield." << "\n");
    return;
  }

  yieldOp = lastOp;
  auto scfYieldOp = cast<scf::YieldOp>(lastOp);
  yieldOperands.assign(scfYieldOp.getOperands().begin(),
                       scfYieldOp.getOperands().end());
  LDBG("Collected " << yieldOperands.size() << " yield operands." << "\n");
}

void UpdateConditionInfoPass::populateNewThenBlock(
    scf::IfOp newIfOp, Block &oldThenBlock, Operation *oldThenYieldOp,
    ArrayRef<Value> oldYieldOperands,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  Location loc = newIfOp.getLoc();
  Block &newThenBlock = newIfOp.getThenRegion().front();
  for (Operation &op : llvm::make_early_inc_range(oldThenBlock)) {
    if (&op != oldThenYieldOp) {
      op.moveBefore(&newThenBlock, newThenBlock.end());
    }
  }
  LDBG("Populate new then block with " << oldYieldOperands.size()
                                       << " original yield operands." << "\n");

  OpBuilder thenBuilder(&newThenBlock, newThenBlock.end());
  SmallVector<Value> thenYieldOperands(oldYieldOperands.begin(),
                                       oldYieldOperands.end());
  if (!currentUsedVars.empty()) {
    Value one =
        thenBuilder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
    for (Value var : currentUsedVars) {
      Value varToUse = var;
      auto latestIt = controlVarToLatestValue.find(var);
      if (latestIt != controlVarToLatestValue.end()) {
        varToUse = latestIt->second;
      }

      Value yieldVal = varToUse;
      auto it = varUpdateTypes.find(var);
      if (it != varUpdateTypes.end()) {
        if (it->second == VarUpdateType::DEC) {
          yieldVal = thenBuilder.create<arith::SubIOp>(loc, varToUse, one);
        } else if (it->second == VarUpdateType::INC) {
          yieldVal = thenBuilder.create<arith::AddIOp>(loc, varToUse, one);
        }
      }
      thenYieldOperands.push_back(yieldVal);
    }
  }

  if (hasCounter) {
    Value newCounter = thenBuilder.create<arith::AddIOp>(loc, counter, step);
    thenYieldOperands.push_back(newCounter);
    LDBG("Append updated counter to then yield." << "\n");
  }

  LDBG("Create then yield with " << thenYieldOperands.size() << " operands."
                                 << "\n");
  thenBuilder.create<scf::YieldOp>(loc, thenYieldOperands);
}

void UpdateConditionInfoPass::populateNewElseBlock(scf::IfOp newIfOp,
                                                   scf::IfOp oldIfOp,
                                                   bool oldHasElse,
                                                   bool hasCounter,
                                                   Value counter) {
  Location loc = newIfOp.getLoc();
  Block &newElseBlock = newIfOp.getElseRegion().front();
  SmallVector<Value> oldElseYieldOperands;
  Operation *oldElseYieldOp = nullptr;

  if (oldHasElse) {
    Block &oldElseBlock = oldIfOp.getElseRegion().front();
    collectYieldOperands(oldElseBlock, oldElseYieldOp, oldElseYieldOperands);
    for (Operation &op : llvm::make_early_inc_range(oldElseBlock)) {
      if (&op != oldElseYieldOp) {
        op.moveBefore(&newElseBlock, newElseBlock.end());
      }
    }
    LDBG("Moved old else block ops and collected "
         << oldElseYieldOperands.size() << " old else yield operands." << "\n");
  }

  OpBuilder elseBuilder(&newElseBlock, newElseBlock.end());
  SmallVector<Value> elseYieldOperands;
  for (Value operand : oldElseYieldOperands) {
    Value newOperand = operand;
    auto it = controlVarToLatestValue.find(operand);
    if (it != controlVarToLatestValue.end()) {
      newOperand = it->second;
    }
    elseYieldOperands.push_back(newOperand);
  }

  for (Value var : currentUsedVars) {
    Value varToUse = var;
    auto it = controlVarToLatestValue.find(var);
    if (it != controlVarToLatestValue.end()) {
      varToUse = it->second;
    }
    elseYieldOperands.push_back(varToUse);
  }

  if (hasCounter) {
    Value counterToUse = counter;
    auto it = controlVarToLatestValue.find(counter);
    if (it != controlVarToLatestValue.end()) {
      counterToUse = it->second;
    }
    elseYieldOperands.push_back(counterToUse);
  }

  LDBG("Create else yield with " << elseYieldOperands.size() << " operands."
                                 << "\n");
  elseBuilder.create<scf::YieldOp>(loc, elseYieldOperands);
  if (oldElseYieldOp) {
    oldElseYieldOp->erase();
  }
}

// Create new IfOp with new then and else blocks.
scf::IfOp UpdateConditionInfoPass::createNewIfOpWithBlocks(
    scf::IfOp oldIfOp, Value combinedCond,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  Location loc = oldIfOp.getLoc();
  OpBuilder builder(oldIfOp);

  bool needsYield = !currentUsedVars.empty() || hasCounter;
  bool oldHasElse = oldIfOp.getElseRegion().hasOneBlock();
  // Only create else when we will populate it;
  bool withElse = needsYield || oldHasElse;
  LDBG("Create replacement if op: needs yield "
       << needsYield << ", old has else " << oldHasElse << ", with else "
       << withElse << ", current used vars " << currentUsedVars.size() << "."
       << "\n");

  Block &oldThenBlock = oldIfOp.getThenRegion().front();
  Operation *oldThenYieldOp = nullptr;
  SmallVector<Value> oldYieldOperands;
  collectYieldOperands(oldThenBlock, oldThenYieldOp, oldYieldOperands);
  SmallVector<Type> resultTypes =
      buildNewIfResultTypes(oldIfOp, hasCounter, counter);
  scf::IfOp newIfOp =
      builder.create<scf::IfOp>(loc, resultTypes, combinedCond, withElse);
  LDBG("Created replacement if op with " << resultTypes.size() << " results."
                                         << "\n");

  for (auto &attr : oldIfOp->getAttrs()) {
    newIfOp->setAttr(attr.getName(), attr.getValue());
  }

  populateNewThenBlock(newIfOp, oldThenBlock, oldThenYieldOp, oldYieldOperands,
                       varUpdateTypes, hasCounter, counter, step);
  if (withElse) {
    populateNewElseBlock(newIfOp, oldIfOp, oldHasElse, hasCounter, counter);
  }

  for (size_t i = 0; i < oldIfOp.getNumResults(); ++i) {
    oldIfOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));
  }
  LDBG("Replaced " << oldIfOp.getNumResults() << " old if results." << "\n");

  return newIfOp;
}

// Combine the conditions: crossCore condition + intraCore condition + counter
// condition + flowOpt condition + the flag of the loop
int UpdateConditionInfoPass::combineConditions(
    ModuleOp module, Value crossCoreCond, Value preloadCond,
    Value intraCoreCond, Value flowOptCond, scf::IfOp ifOp, Operation *loopOp,
    size_t &usedCounterNum, DenseMap<Value, VarUpdateType> &varUpdateTypes,
    Value &blockFlag, bool isLastBlock, bool isFirstBlock, bool isPureScalar) {
  Location loc = ifOp.getLoc();
  SmallVector<Value> validConditions;
  Value counter;
  // Only for updates counter args inside if then  (+step).
  bool updateCounterArg = false;

  if (crossCoreCond) {
    validConditions.push_back(crossCoreCond);
  }
  if (preloadCond) {
    validConditions.push_back(preloadCond);
  }
  if (intraCoreCond) {
    validConditions.push_back(intraCoreCond);
  }
  if (flowOptCond) {
    validConditions.push_back(flowOptCond);
  }

  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  auto whileOp = dyn_cast<scf::WhileOp>(loopOp);
  OpBuilder condBuilder(ifOp);

  if (forOp) {
    if (!info->blockCounters.count(forOp)) {
      LDBG("Missing block counters for forOp=" << forOp << ", ifOp=" << ifOp
                                               << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    SmallVector<int> &counterIndices = info->blockCounters[forOp];

    if (info->cntArgs.count(ifOp)) {
      counter = info->cntArgs[ifOp];
      updateCounterArg = true;
    } else {
      if (usedCounterNum >= counterIndices.size()) {
        LDBG("Not enough counters for ssbuffer if ops: used "
             << usedCounterNum << ", counters " << counterIndices.size()
             << ", forOp=" << forOp << ", ifOp=" << ifOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }

      int argIdx = counterIndices[usedCounterNum];
      int iterArgNum = static_cast<int>(forOp.getNumRegionIterArgs());
      if (argIdx < 0 || argIdx >= iterArgNum) {
        LDBG("Invalid counter arg index: " << argIdx << ", iter args "
                                           << iterArgNum << ", forOp=" << forOp
                                           << ", ifOp=" << ifOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }

      counter = forOp.getRegionIterArgs()[argIdx];
      updateCounterArg = true;
      info->cntArgs[ifOp] = counter;
      usedCounterNum++;
      LDBG("Assign counter iter arg index " << argIdx << " to ssbuffer if op."
                                            << "\n");
    }

    LDBG("this ifop used counter is: " << counter << "\n");
    Value upperBound = forOp.getUpperBound();
    Value counterToUse = counter;
    auto latestIt = controlVarToLatestValue.find(counter);
    if (latestIt != controlVarToLatestValue.end()) {
      counterToUse = latestIt->second;
    }
    Value counterCond = condBuilder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, counterToUse, upperBound);
    validConditions.push_back(counterCond);
  } else if (whileOp) {
    // Only add remapped scf.condition(x) expression;
    Value counterCond;
    if (buildWhileCounterCondition(whileOp, ifOp, info, condBuilder,
                                   controlVarToLatestValue,
                                   counterCond) == UPDATE_CONDITION_INFO_FAILED)
      return UPDATE_CONDITION_INFO_FAILED;
    validConditions.push_back(counterCond);
  } else {
    LDBG("Unsupported loop op for counter condition, loopOp="
         << *loopOp << ", ifOp=" << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  if (validConditions.empty()) {
    LDBG("Failed to build any condition for ssbuffer if op, ifOp="
         << ifOp << ", loopOp=" << *loopOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  LDBG("Combine " << validConditions.size() << " conditions for ssbuffer if op."
                  << "\n");
  OpBuilder builder(ifOp);
  Value combinedCond = validConditions[0];
  for (size_t i = 1; i < validConditions.size(); ++i) {
    combinedCond =
        builder.create<arith::AndIOp>(loc, combinedCond, validConditions[i]);
  }

  // The flag of the loop is the last condition: the block only runs when no
  // other block of the iteration has run yet. The first block of the flag
  // chain reads the flag the loop starts with, a constant zero: its check is
  // always true and is left out. A pure scalar block has no interaction with
  // the other blocks, it is not part of the flag chain: it does not take the
  // flag as a condition.
  Value guardCond = combinedCond;
  if (blockFlag && !isFirstBlock && !isPureScalar) {
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
    Value flagCond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, blockFlag, zero);
    guardCond = builder.create<arith::AndIOp>(loc, combinedCond, flagCond);
    LDBG("Add the flag condition of the loop." << "\n");
  }

  Value step = updateCounterArg ? forOp.getStep() : Value();
  scf::IfOp newIfOp = createNewIfOpWithBlocks(ifOp, guardCond, varUpdateTypes,
                                              updateCounterArg, counter, step);

  if (blockFlag && !isLastBlock && !isPureScalar) {
    // The flag of the loop is not yielded by the block, it is computed here
    // with a select on the condition of the block: the block has run, the flag
    // of the loop becomes 1, the block did not run, the flag of the loop keeps
    // the value it had. The flag is cumulative: the select of the previous
    // block is the false value of this one, so the flag of the loop is 1 as
    // soon as one block has run. The select is placed after the block, so that
    // the flag of the next block is read after the body of this block has run.
    OpBuilder flagBuilder(ifOp);
    Value one =
        flagBuilder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
    blockFlag =
        flagBuilder.create<arith::SelectOp>(loc, guardCond, one, blockFlag);
    LDBG("Compute the flag of the loop with a select." << "\n");
  }

  // Update DAG nodes
  updateDAGAfterIfOpReplacement(ifOp, newIfOp);

  if (updateCounterArg) {
    info->cntArgs.erase(ifOp);
    info->cntArgs[newIfOp] = counter;
  }

  // Update mappings that refer to the old ifOp BEFORE erasing it
  // Update the tensorIterArgIfOpVars mapping
  if (tensorIterArgIfOpVars.count(ifOp)) {
    auto ifOpVars = tensorIterArgIfOpVars[ifOp];
    tensorIterArgIfOpVars.erase(ifOp);
    tensorIterArgIfOpVars[newIfOp] = ifOpVars;
  }

  // Update tensorIterArgDepsMap with new ifOp
  if (info->tensorIterArgDepsMap.count(loopOp)) {
    auto &depsVec = info->tensorIterArgDepsMap[loopOp];
    for (auto &relation : depsVec) {
      // Update producer ifOp - use pointer comparison
      if (relation.producer.getOperation() == ifOp.getOperation()) {
        relation.producer = newIfOp;
      }
      // Update consumer ifOps
      for (auto &consumerIfOp : relation.consumers) {
        if (consumerIfOp.getOperation() == ifOp.getOperation()) {
          consumerIfOp = newIfOp;
        }
      }
    }
  }

  updateControlVarToLatestValue(newIfOp, ifOp, updateCounterArg, counter);

  ifOp.erase();
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Update the conditions of ifOp.
int UpdateConditionInfoPass::updateIfConds(
    ModuleOp module, SmallVector<SmallVector<Value>> ssbufferPtrs) {
  // Walk main_loop for/while ops to update the conditions of ifOp
  SmallVector<Operation *> mainLoopOps;
  WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (CVPipeline::isMainLoopOp(op)) {
      mainLoopOps.push_back(op);
      return WalkResult::advance();
    }

    if (op->hasAttr(kMainLoop)) {
      LDBG("Found unsupported main loop op: " << *op << "\n");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // Step0: Collect dependency buffers once outside the loop
  DenseMap<Operation *,
           DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
      intraCoreBuffersMap;
  collectDependencyBuffers(module, mainLoopOps, intraCoreBuffersMap);

  for (Operation *loopOp : mainLoopOps) {
    controlVarToLatestValue.clear();

    // The counter loads of this loop are gathered at the beginning of its
    // body. The main loop is an scf.for here, it is turned into an scf.while
    // later on, keeping the order of the body.
    counterLoadBlock = getMainLoopBody(loopOp);
    counterLoadTail = nullptr;
    // The counters read by the blocks of this loop are read once, the blocks
    // share the values.
    loadedCounters.clear();

    // The flag of the loop. It is 0 at the beginning of an iteration and
    // becomes 1 as soon as a compute block has run. Every compute block takes
    // it as an extra condition (flag == 0), so that at most one block runs per
    // iteration. The flag is not yielded by the block, it is computed with a
    // select on the condition of the block: the block that ran makes it 1, a
    // block that did not run keeps the flag of the previous block. The first
    // block keeps the constant zero the loop starts with.
    Value blockFlag;
    if (counterLoadBlock) {
      OpBuilder flagBuilder(counterLoadBlock, counterLoadBlock->begin());
      auto flagOp = flagBuilder.create<arith::ConstantIntOp>(loopOp->getLoc(),
                                                             0, CONST_INT_TYPE);
      blockFlag = flagOp.getResult();
      // The counter loads are placed after the flag.
      counterLoadTail = flagOp.getOperation();
    }

    // Step1: Get intraCoreBuffers from pre-collected map
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        intraCoreBuffers;
    if (intraCoreBuffersMap.count(loopOp)) {
      intraCoreBuffers = intraCoreBuffersMap[loopOp];
    }

    // for/while requires at least one of cross/intra deps;
    if (info->crossCoreBlockEdges.empty() && intraCoreBuffers.empty()) {
      LDBG("crossCoreBlockEdges and intraCoreBuffers are all empty! loopOp="
           << *loopOp << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    // Step2: Assign control variables (intraCore inputs / tensor iter_args)
    DenseMap<int, Value> idxToVar;
    if (buildIdxToVarMap(loopOp, intraCoreBuffers, idxToVar) ==
        UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
    if (buildTensorIterArgIfOpVarMap(loopOp) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }

    size_t usedCounterNum = 0;
    SmallVector<scf::IfOp> ifOps;
    if (collectSSBufferIfOps(loopOp, ifOps) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
    if (validateBlockCounters(loopOp, ifOps.size()) ==
        UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }

    // The blocks that exchange data with the other blocks form the flag chain
    // of the loop: at most one of them runs per iteration. A block that has
    // neither a cross-core nor an intra-core interaction is a pure scalar
    // computation: it is not part of the chain, it neither checks the flag of
    // the loop nor updates it.
    //
    // The chain index of a block is its position in the chain, -1 for a pure
    // scalar block.
    SmallVector<int> chainIdx(ifOps.size(), -1);
    SmallVector<SmallVector<int>> intraCoreInputValuesList(ifOps.size());
    SmallVector<SmallVector<int>> intraCoreOutputValuesList(ifOps.size());
    int chainNum = 0;
    for (size_t ifIdx = 0; ifIdx < ifOps.size(); ++ifIdx) {
      scf::IfOp ifOp = ifOps[ifIdx];
      if (getInputOutputValues(ifOp, intraCoreBuffers,
                               intraCoreInputValuesList[ifIdx],
                               intraCoreOutputValuesList[ifIdx]) != 0) {
        LDBG("getInputOutputValues failed! ifOp=" << ifOp << ", loopOp="
                                                  << *loopOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // A cross-core interaction: the block takes part in a dependency that
      // crosses the cores, it owns a counter slot.
      bool hasCrossCoreInteraction = info->blockCounterSlots.count(ifOp) > 0;
      // An intra-core interaction: the block reads or writes a buffer of
      // another block of the same core, or a tensor of the loop.
      bool hasIntraCoreInteraction =
          !intraCoreInputValuesList[ifIdx].empty() ||
          !intraCoreOutputValuesList[ifIdx].empty() ||
          tensorIterArgIfOpVars.count(ifOp) > 0;
      if (hasCrossCoreInteraction || hasIntraCoreInteraction) {
        chainIdx[ifIdx] = chainNum++;
      } else {
        LDBG("Block " << ifIdx
                      << " is a pure scalar computation, it is not part of the "
                      << "flag chain of the loop." << "\n");
      }
    }

    // Update the conditions of ifOp in this loop op.
    for (size_t ifIdx = 0; ifIdx < ifOps.size(); ++ifIdx) {
      scf::IfOp ifOp = ifOps[ifIdx];
      // A pure scalar block is not part of the flag chain of the loop.
      bool isPureScalar = (chainIdx[ifIdx] < 0);
      // The last block of the chain yields a flag that no block reads: the
      // flag of the loop ends with it.
      bool isLastBlock = (!isPureScalar && chainIdx[ifIdx] == chainNum - 1);
      // The first block of the chain reads the flag the loop starts with, a
      // constant zero: its check is always true.
      bool isFirstBlock = (!isPureScalar && chainIdx[ifIdx] == 0);
      SmallVector<int> &intraCoreInputValues = intraCoreInputValuesList[ifIdx];
      SmallVector<int> &intraCoreOutputValues =
          intraCoreOutputValuesList[ifIdx];

      // Step3:Set the crossCore condition
      Value crossCoreCond;
      Value preloadCond;
      if (setCrossCoreCondition(ifOp, loopOp, ssbufferPtrs, crossCoreCond,
                                preloadCond) != 0) {
        LDBG("setCrossCoreCondition failed! ifOp=" << ifOp << ", loopOp="
                                                   << *loopOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step4:Set the intraCore condition
      DenseMap<Value, VarUpdateType> varUpdateTypes;
      Value intraCoreCond;
      if (setIntraCoreCondition(module, ifOp, intraCoreBuffers,
                                intraCoreInputValues, intraCoreOutputValues,
                                idxToVar, varUpdateTypes, intraCoreCond) ==
          UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step5:Set the flowOpt condition (for only; needs lb/ub/step)
      Value flowOptCond;
      if (setFlowOptCondition(ifOp, loopOp, flowOptCond) ==
          UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step6:Combine the conditions: crossCore + preload + intraCore +
      // counter + flowOpt + the flag of the loop
      if (combineConditions(module, crossCoreCond, preloadCond, intraCoreCond,
                            flowOptCond, ifOp, loopOp, usedCounterNum,
                            varUpdateTypes, blockFlag, isLastBlock,
                            isFirstBlock,
                            isPureScalar) == UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
    }
    // Step7: Update loop yield with latest control values
    if (updateLoopYield(loopOp) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

void UpdateConditionInfoPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter UpdateConditionInfo pass." << "\n");
  // Step1:Build the block level cross-core edges out of the op level
  // crossCoreDependentMap. Every producer of a dependency group must live in
  // the same ssbuffer.if block.
  if (buildCrossCoreBlockEdges(module) != UPDATE_CONDITION_INFO_SUCCESS) {
    LDBG("buildCrossCoreBlockEdges failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step2:Give every block taking part in a cross-core dependency a counter
  // slot.
  if (assignBlockCounterSlots(module) != UPDATE_CONDITION_INFO_SUCCESS) {
    LDBG("assignBlockCounterSlots failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step2.5:Collect the computation blocks of the two cores, ordered by block
  // id. The preload conditions compare the blocks of a core with each other.
  if (collectPreloadBlocks(module) != UPDATE_CONDITION_INFO_SUCCESS) {
    LDBG("collectPreloadBlocks failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
  preloadBufferCount = getPreloadBufferCount(module);
  LDBG("The preload conditions use " << preloadBufferCount
                                     << " cross core buffers as base.\n");

  // Step3:Init the ssbufferPtrs
  SmallVector<SmallVector<Value>> ssbufferPtrs = allocSSBuffer(module);
  // Step3.1:Bracket the counter initialization with the PIPE_S
  // synchronization, so that a core never reads a counter that is not
  // initialized yet.
  insertCounterInitPipeS(module);

  // Step4:Update the conditions of ifOp based on the intraCoreDependentMap and
  // the block level cross-core edges
  int updateResult = updateIfConds(module, ssbufferPtrs);

  if (updateResult != UPDATE_CONDITION_INFO_SUCCESS) {
    LDBG("updateIfConds failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  LDBG("Exit UpdateConditionInfo pass." << "\n");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createUpdateConditionInfoPass() {
  return std::make_unique<UpdateConditionInfoPass>();
}
} // namespace triton
} // namespace mlir
