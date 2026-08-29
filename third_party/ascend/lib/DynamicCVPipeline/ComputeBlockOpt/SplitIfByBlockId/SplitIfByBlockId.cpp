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
#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "Common.h"
#include "DynamicCVPipeline/Common/FallbackHelper.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

static constexpr const char *DEBUG_TYPE = "SplitIfByBlockId";
static constexpr llvm::StringLiteral kSkippedDeltaformerKernel =
    "parallel_deltaformer_fwd_kernel";
static constexpr llvm::StringLiteral kSkippedChunkwiseKernel =
    "chunkwise_fwd_kernel";
static constexpr llvm::StringLiteral kSkippedParallelNsaFwdKernel =
    "parallel_nsa_fwd_kernel";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;
using namespace SplitIf;

/// Register a cross-group SSA dependency: a value defined in one group
/// is consumed by an op in a different group.
static void addCrossGroupDependency(
    Value val, Operation *consumer, unsigned consumerGroup,
    llvm::SmallDenseMap<Operation *, unsigned> &opToGroup,
    llvm::SmallDenseMap<Value,
                        std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>>
        &crossValueMap) {
  auto *defOp = val.getDefiningOp();
  if (!defOp) {
    return;
  }
  auto it = opToGroup.find(defOp);
  if (it == opToGroup.end() || it->second == consumerGroup) {
    return;
  }

  auto &entry = crossValueMap[val];
  entry.first = it->second;
  entry.second.insert(consumer);
}

namespace {

/// A value-level cross-group SSA dependency.
/// Tracks which group produces a value and which groups consume it.
struct CrossGroupValue {
  Value value;                             // the SSA value crossing groups
  unsigned fromGroupIdx;                   // producer group index
  SmallVector<unsigned, 0> toGroupIndices; // consumer group indices
};

/// Per-group output description for materialization.
/// Describes what values a group yields from its then-block.
struct GroupOutputInfo {

  /// Values this group yields (cross-group values it produces;
  /// for non-last groups in Case B, also includes original yield values
  /// produced by this group).
  SmallVector<Value, 0> outputValues;

  /// Case B: original else-side yield value for each output slot.
  /// Set when the output value is an original yield slot (not a pure
  /// cross-group augmentation). Empty for Case A or cross-group-only values.
  SmallVector<Value, 0> origElseValues;

  SmallVector<Type, 0> outputTypes;

  bool isVoid() const { return outputValues.empty(); }
};

/// Each split-if gets its own result signature based on what
/// it actually produces. No uniform slot plan.
struct YieldAugmentation {
  /// Cross-group values (deduplicated, in producer-group order).
  SmallVector<CrossGroupValue, 0> crossValues;

  /// Per-group output info for non-last groups only.
  /// Last group is handled separately in materialization.
  SmallVector<GroupOutputInfo, 0> groupOutputs;

  // Case B only: info for building the last-if (carries original result types)
  unsigned numOriginalSlots = 0;
  SmallVector<Value, 0> origYieldValues;
  SmallVector<int, 0> origYieldProducerGroup;

  bool empty() const { return crossValues.empty(); }
};

struct CandidateIf {
  scf::IfOp ifOp;
  int64_t selfBlockId;
  bool hasYield;
  SmallVector<BlockGroup, 0> thenGroups;
  SmallVector<BlockGroup, 0> elseGroups;

  // value-level cross-group + yield plan
  YieldAugmentation yieldAug;

  bool needsSplit() const {
    return thenGroups.size() >= 2 || elseGroups.size() >= 2;
  }
};

} // namespace

static bool hasNestedIfs(const CandidateIf &c) {
  for (auto &g : c.thenGroups) {
    if (!g.nestedIfs.empty()) {
      return true;
    }
  }
  for (auto &g : c.elseGroups) {
    if (!g.nestedIfs.empty()) {
      return true;
    }
  }
  return false;
}

static inline void dumpCandidate(CandidateIf &candidate) {
  LDBG("Processing: " << candidate.ifOp);
  LDBG("  selfBlockId=" << candidate.selfBlockId);
  LDBG("  groups(then=" << candidate.thenGroups.size()
                        << " else=" << candidate.elseGroups.size()
                        << ")  yield=" << candidate.hasYield
                        << "  nested=" << hasNestedIfs(candidate));
  for (auto &g : candidate.thenGroups) {
    LDBG("  then block_id=" << g.blockId << "  ops=" << g.ops.size()
                            << "  nestedIfs=" << g.nestedIfs.size());
  }
  for (auto &g : candidate.elseGroups) {
    LDBG("  else block_id=" << g.blockId << "  ops=" << g.ops.size()
                            << "  nestedIfs=" << g.nestedIfs.size());
  }
}

/// Walk a single block (then or else body), grouping ops by block_id.
/// Nested ifs attach to the nearest preceding group.
/// Ops without a splittable id are kept in the current active group.
static SmallVector<BlockGroup> groupOpsInBlock(Block &block) {
  auto *parentOp = block.getParentOp();
  if (!parentOp) {
    return {};
  }
  Builder builder(parentOp->getContext());
  SmallVector<BlockGroup> groups;
  llvm::SmallDenseMap<int64_t, unsigned> idToIdx; // block_id -> index in groups

  auto getOrCreateGroup = [&](int bid) -> BlockGroup & {
    auto it = idToIdx.find(bid);
    if (it != idToIdx.end()) {
      return groups[it->second];
    }
    idToIdx[bid] = groups.size();
    groups.push_back({bid, {}, {}});
    return groups.back();
  };

  int currentId = -1; // which group nested / ambient ops attach to
  SmallVector<Operation *, 0>
      pendingAmbient; // ambient ops before first real group

  auto flushPending = [&](int targetBid) {
    if (pendingAmbient.empty()) {
      return;
    }
    auto &g = getOrCreateGroup(targetBid);
    for (auto *op : pendingAmbient) {
      g.ops.push_back(op);
    }
    pendingAmbient.clear();
  };

  for (auto &op : block) {
    if (isa<scf::YieldOp>(op)) {
      continue;
    }

    if (auto nestedIf = dyn_cast<scf::IfOp>(op)) {
      // Nested ifs form their own group based on their block_id so
      // that the parent if can be split when inner ifs of different
      // block_ids coexist in the same region after earlier iterations.
      auto bid = CVPipeline::getOpBlockId(nestedIf);
      if (bid.has_value() && *bid != -1) {
        flushPending(*bid);
        getOrCreateGroup(*bid).nestedIfs.push_back(nestedIf);
        currentId = *bid;
      } else if (currentId != -1) {
        getOrCreateGroup(currentId).nestedIfs.push_back(nestedIf);
      } else {
        pendingAmbient.push_back(&op);
      }
      continue;
    }

    auto bid = CVPipeline::getOpBlockId(&op);
    if (bid.has_value() && *bid != -1) {
      flushPending(*bid);
      getOrCreateGroup(*bid).ops.push_back(&op);
      currentId = *bid;
    } else if (currentId != -1) {
      // ambient op (id == -1 or 16): keep in the current group
      op.setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(currentId));
      getOrCreateGroup(currentId).ops.push_back(&op);
    } else {
      pendingAmbient.push_back(&op);
    }
  }

  // Flush remaining pending ambient ops into the first real group (if any),
  // otherwise they're the only content and don't need splitting.
  if (!pendingAmbient.empty() && !groups.empty()) {
    int firstBlockId = groups[0].blockId;
    for (auto *op : pendingAmbient) {
      op->setAttr(CVPipeline::kBlockId,
                  builder.getI32IntegerAttr(firstBlockId));
      groups[0].ops.push_back(op);
    }
  }

  return groups;
}

static CandidateIf getCandidate(scf::IfOp ifOp) {
  CandidateIf cand;
  cand.ifOp = ifOp;
  cand.hasYield = (ifOp->getNumResults() > 0);

  // self block_id
  auto selfBlockId = CVPipeline::getOpBlockId(ifOp);
  cand.selfBlockId = selfBlockId.value_or(-1);

  // Group then region
  cand.thenGroups = groupOpsInBlock(*ifOp.thenBlock());

  // Group else region
  Block *elseBlk = ifOp.elseBlock();
  if (elseBlk) {
    cand.elseGroups = groupOpsInBlock(*elseBlk);
  }

  return cand;
}

// Clone the outer scalar dependencies into the if ops
static void preprocessScalarDependencies(CandidateIf &cand) {
  auto processGroup = [](BlockGroup &group) {
    auto allOpsIter = llvm::concat<Operation *>(group.ops, group.nestedIfs);
    SmallVector<Operation *> allOps{allOpsIter.begin(), allOpsIter.end()};
    ScalarClosure closure{group, allOps};
    closure.collect();

    auto [newOps, mapper] = closure.capture(group.blockId);
    if (newOps.empty()) {
      return;
    }

    // insert them to the start of group, so that their inner operands can also
    // be rewritten
    allOps.append(newOps);
    group.nestedIfs.clear();
    group.ops.clear();

    llvm::sort(allOps, [&closure](Operation *a, Operation *b) {
      return closure.isBefore(a, b);
    });

    for (Operation *op : allOps) {
      op->walk([&, &mapper = mapper](Operation *innerOp) {
        for (OpOperand &operand : innerOp->getOpOperands()) {
          Value target = mapper.lookupOrNull(operand.get());
          if (target) {
            operand.set(target);
          }
        }
      });
      if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
        group.nestedIfs.push_back(ifOp);
      } else {
        group.ops.push_back(op);
      }
    }
  };

  llvm::for_each(llvm::concat<BlockGroup>(cand.thenGroups, cand.elseGroups),
                 processGroup);
}

// ============================================================================
// Part2: Dependency Analysis & Yield Planning
// ============================================================================
//
// For each candidate, in a single pass over operands:
//   - Group-level: detect cross-group SSA dependencies (for topo sort).
//   - Value-level: track which specific SSA values cross group boundaries.
//
// Then plan yield augmentation:
//   - Case A: promote all cross-group values to new yield slots.
//   - Case B: augment yield with extra slots for cross-group values
//     not in original yield (→ unified yield chain, no memref bridges).
//   - Compute slot actions for each split if.
//   - Topo sort groups so producers execute before consumers.
//
// Part4 consumes the resulting YieldAugmentation plan.
static void dumpYieldAugmentation(const CandidateIf &c) {
  auto &ya = c.yieldAug;
  bool splitThen = c.thenGroups.size() >= 2;
  auto &groups = splitThen ? c.thenGroups : c.elseGroups;
  const char *region = splitThen ? "then" : "else";

  // --- value-level cross-group deps ---
  if (ya.crossValues.empty()) {
    LDBG("[Part2] no cross-group values (region=" << region << ")");
  } else {
    LDBG("[Part2] value-level cross-group deps ("
         << region << ", " << ya.crossValues.size() << " values):");
    for (auto &cv : ya.crossValues) {
      std::string consumerStr;
      for (unsigned toG : cv.toGroupIndices) {
        if (!consumerStr.empty()) {
          consumerStr += ", ";
        }
        consumerStr += std::to_string(groups[toG].blockId);
      }
      LDBG("  val from block_id=" << groups[cv.fromGroupIdx].blockId
                                  << " -> consumed by block_id(s): ["
                                  << consumerStr << "]"
                                  << "  type=" << cv.value.getType());
    }
  }

  // --- per-group output info ---
  for (unsigned gi = 0; gi < ya.groupOutputs.size(); ++gi) {
    auto &output = ya.groupOutputs[gi];
    if (output.isVoid()) {
      LDBG("[Part2] group[" << gi << "] (block_id=" << groups[gi].blockId
                            << "): void if");
    } else {
      LDBG("[Part2] group[" << gi << "] (block_id=" << groups[gi].blockId
                            << "): " << output.outputValues.size()
                            << " output(s)");
      for (unsigned idx = 0; idx < output.outputValues.size(); ++idx) {
        LDBG("  output[" << idx << "] = " << output.outputValues[idx]
                         << "  type=" << output.outputTypes[idx]);
      }
    }
  }

  // --- last-if info (Case B) ---
  if (c.hasYield) {
    LDBG("[Part2] Case B: last-if original yield (" << ya.numOriginalSlots
                                                    << " slot(s))");
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      LDBG("  slot[" << slot << "] = " << ya.origYieldValues[slot]
                     << "  producer=group " << ya.origYieldProducerGroup[slot]);
    }
  } else {
    LDBG("  [Part2] Case A: no original yield");
  }
}

/// Single-pass scan: value-level crossValueMap only.
/// Group order is implicitly the natural discovery order (block_ids appear
/// in dependency order within a sequential basic block).
/// Used for both then and else regions.
static void scanRegion(
    SmallVector<BlockGroup, 0> &groups,
    llvm::SmallDenseMap<Operation *, unsigned> &opToGroup,
    llvm::SmallDenseMap<Value,
                        std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>>
        &crossValueMap) {
  unsigned n = groups.size();
  if (n < 2) {
    return;
  }

  for (unsigned gi = 0; gi < n; ++gi) {
    for (auto *op : groups[gi].ops) {
      for (auto &operand : op->getOpOperands()) {
        addCrossGroupDependency(operand.get(), op, gi, opToGroup,
                                crossValueMap);
      }
      for (auto &region : op->getRegions()) {
        region.walk([&](Operation *nestedOp) {
          for (auto &nestedOperand : nestedOp->getOpOperands()) {
            addCrossGroupDependency(nestedOperand.get(), op, gi, opToGroup,
                                    crossValueMap);
          }
        });
      }
    }
    for (auto nestedIf : groups[gi].nestedIfs) {
      for (auto &operand : nestedIf->getOpOperands()) {
        addCrossGroupDependency(operand.get(), nestedIf.getOperation(), gi,
                                opToGroup, crossValueMap);
      }
    }
    for (auto nestedIf : groups[gi].nestedIfs) {
      nestedIf->walk([&](Operation *innerOp) {
        for (auto &innerOperand : innerOp->getOpOperands()) {
          addCrossGroupDependency(innerOperand.get(), nestedIf.getOperation(),
                                  gi, opToGroup, crossValueMap);
        }
      });
    }
    // Scan nested if results for cross-group consumers.
    // A nested if in group[gi] may produce values consumed by ops
    // or nested ifs in a different group. Walk up from each user to
    // find the nearest ancestor tracked in opToGroup.
    for (auto nestedIf : groups[gi].nestedIfs) {
      for (auto result : nestedIf->getResults()) {
        for (auto *user : result.getUsers()) {
          Operation *trackedOp = user;
          while (trackedOp) {
            auto it = opToGroup.find(trackedOp);
            if (it != opToGroup.end()) {
              if (it->second != gi) {
                addCrossGroupDependency(result, trackedOp, it->second,
                                        opToGroup, crossValueMap);
              }
              break;
            }
            trackedOp = trackedOp->getParentOp();
          }
        }
      }
    }
  }
}

/// Compute per-group output info for Case B.
/// Non-last groups produce cross-group values + original yield values they own.
/// Last-group info (origYieldValues, origYieldProducerGroup) is stored
/// in YieldAugmentation for Part3 to build the last-if.
static void
planYieldCaseB(CandidateIf &c, bool splitThen, ArrayRef<BlockGroup> groups,
               llvm::SmallDenseMap<Operation *, unsigned> &opToGroup) {
  unsigned nGroups = groups.size();
  auto &ya = c.yieldAug;

  auto splitYieldOp =
      cast<scf::YieldOp>(splitThen ? c.ifOp.thenBlock()->getTerminator()
                                   : c.ifOp.elseBlock()->getTerminator());

  ya.numOriginalSlots = splitYieldOp->getNumOperands();
  ya.origYieldValues.clear();
  ya.origYieldProducerGroup.clear();
  for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
    Value val = splitYieldOp->getOperand(slot);
    ya.origYieldValues.push_back(val);
    auto *defOp = val.getDefiningOp();
    if (defOp) {
      auto it = opToGroup.find(defOp);
      ya.origYieldProducerGroup.push_back(
          it != opToGroup.end() ? static_cast<int>(it->second) : -1);
    } else {
      ya.origYieldProducerGroup.push_back(-1); // block argument
    }
  }

  auto otherYieldOp =
      cast<scf::YieldOp>(splitThen ? c.ifOp.elseBlock()->getTerminator()
                                   : c.ifOp.thenBlock()->getTerminator());
  SmallVector<Value> otherYieldValues;
  for (auto v : otherYieldOp->getOperands()) {
    otherYieldValues.push_back(v);
  }

  llvm::SmallDenseMap<Value, unsigned> valueToSlot;
  for (unsigned si = 0; si < ya.crossValues.size(); ++si) {
    valueToSlot[ya.crossValues[si].value] = si;
  }

  ya.groupOutputs.resize(nGroups > 0 ? nGroups - 1 : 0);
  for (unsigned gi = 0; gi < nGroups - 1; ++gi) {
    auto &output = ya.groupOutputs[gi];

    // Build map: original yield value produced by this group → slot number.
    llvm::SmallDenseMap<Value, unsigned> origValToSlot;
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      if (ya.origYieldProducerGroup[slot] == static_cast<int>(gi)) {
        origValToSlot[ya.origYieldValues[slot]] = slot;
      }
    }

    SmallPtrSet<Value, 4> addedValues;

    // Step 1: Cross-group values produced by this group.
    // For each cross-group value, also check whether it is an original
    // yield slot so we can track the corresponding else-side yield value.
    for (auto *op : groups[gi].ops) {
      for (auto result : op->getResults()) {
        auto it = valueToSlot.find(result);
        if (it != valueToSlot.end()) {
          output.outputValues.push_back(result);
          output.outputTypes.push_back(result.getType());
          addedValues.insert(result);

          auto origIt = origValToSlot.find(result);
          if (origIt != origValToSlot.end()) {
            output.origElseValues.push_back(otherYieldValues[origIt->second]);
          } else {
            output.origElseValues.push_back(Value());
          }
        }
      }
    }
    for (auto nestedIf : groups[gi].nestedIfs) {
      for (auto result : nestedIf->getResults()) {
        auto it = valueToSlot.find(result);
        if (it != valueToSlot.end()) {
          output.outputValues.push_back(result);
          output.outputTypes.push_back(result.getType());
          addedValues.insert(result);

          auto origIt = origValToSlot.find(result);
          if (origIt != origValToSlot.end()) {
            output.origElseValues.push_back(otherYieldValues[origIt->second]);
          } else {
            output.origElseValues.push_back(Value());
          }
        }
      }
    }

    // Step 2: Original yield values produced by this group that were NOT
    // already added as cross-group values
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      if (ya.origYieldProducerGroup[slot] == static_cast<int>(gi)) {
        Value origVal = ya.origYieldValues[slot];
        if (!addedValues.contains(origVal)) {
          output.outputValues.push_back(origVal);
          output.outputTypes.push_back(origVal.getType());
          output.origElseValues.push_back(otherYieldValues[slot]);
        }
      }
    }
  }
}

/// Compute per-group output info for Case A.
/// Each non-last group yields the cross-group values it produces.
/// The last group is always void (no original yield values).
static void planYieldCaseA(CandidateIf &c, ArrayRef<BlockGroup> groups) {
  unsigned nGroups = groups.size();
  auto &ya = c.yieldAug;

  llvm::SmallDenseMap<Value, unsigned> valueToSlot;
  for (unsigned si = 0; si < ya.crossValues.size(); ++si) {
    valueToSlot[ya.crossValues[si].value] = si;
  }

  // Per-group output for non-last groups only.
  ya.groupOutputs.resize(nGroups > 0 ? nGroups - 1 : 0);
  for (unsigned gi = 0; gi < nGroups - 1; ++gi) {
    auto &output = ya.groupOutputs[gi];
    for (auto *op : groups[gi].ops) {
      for (auto result : op->getResults()) {
        auto it = valueToSlot.find(result);
        if (it != valueToSlot.end()) {
          output.outputValues.push_back(result);
          output.outputTypes.push_back(result.getType());
        }
      }
    }
    for (auto nestedIf : groups[gi].nestedIfs) {
      for (auto result : nestedIf->getResults()) {
        auto it = valueToSlot.find(result);
        if (it != valueToSlot.end()) {
          output.outputValues.push_back(result);
          output.outputTypes.push_back(result.getType());
        }
      }
    }
  }
}

/// Plan per-group output from the collected crossValueMap.
/// Groups are already in natural dependency order (block_ids appear in
/// execution order within a sequential basic block), so group index gi
/// is used directly as split-if position.
static void planYield(
    CandidateIf &c, bool splitThen, ArrayRef<BlockGroup> groups,
    llvm::SmallDenseMap<Operation *, unsigned> &opToGroup,
    llvm::SmallDenseMap<Value,
                        std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>>
        &crossValueMap) {
  unsigned nGroups = groups.size();
  auto &ya = c.yieldAug;
  ya.crossValues.clear();
  ya.groupOutputs.clear();

  for (auto &[val, info] : crossValueMap) {
    unsigned fromG = info.first;
    SmallVector<unsigned, 2> toGroups;
    SmallVector<bool, 8> groupHasConsumer(nGroups, false);
    for (auto *consumerOp : info.second) {
      auto git = opToGroup.find(consumerOp);
      if (git != opToGroup.end() && !groupHasConsumer[git->second]) {
        groupHasConsumer[git->second] = true;
        toGroups.push_back(git->second);
      }
    }
    ya.crossValues.push_back({val, fromG, std::move(toGroups)});
  }

  // Stable sort for deterministic order: by producer group, then value ptr.
  llvm::sort(
      ya.crossValues, [](const CrossGroupValue &a, const CrossGroupValue &b) {
        if (a.fromGroupIdx != b.fromGroupIdx) {
          return a.fromGroupIdx < b.fromGroupIdx;
        }
        return a.value.getAsOpaquePointer() < b.value.getAsOpaquePointer();
      });

  if (c.hasYield) {
    planYieldCaseB(c, splitThen, groups, opToGroup);
  } else {
    planYieldCaseA(c, groups);
  }
}

static void analyzeDependencies(CandidateIf &candidate) {
  // Step 2.1: Build op → group index maps (including nested ifs)
  llvm::SmallDenseMap<Operation *, unsigned> opToThenGroup;
  for (unsigned gi = 0; gi < candidate.thenGroups.size(); ++gi) {
    for (auto *op : candidate.thenGroups[gi].ops) {
      opToThenGroup[op] = gi;
    }
    for (auto nestedIf : candidate.thenGroups[gi].nestedIfs) {
      opToThenGroup[nestedIf.getOperation()] = gi;
    }
  }

  llvm::SmallDenseMap<Operation *, unsigned> opToElseGroup;
  for (unsigned gi = 0; gi < candidate.elseGroups.size(); ++gi) {
    for (auto *op : candidate.elseGroups[gi].ops) {
      opToElseGroup[op] = gi;
    }
    for (auto nestedIf : candidate.elseGroups[gi].nestedIfs) {
      opToElseGroup[nestedIf.getOperation()] = gi;
    }
  }

  // Step 2.2: Single-pass scan — value-level crossValueMap only
  llvm::SmallDenseMap<Value,
                      std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>>
      thenValueMap, elseValueMap;

  scanRegion(candidate.thenGroups, opToThenGroup, thenValueMap);
  scanRegion(candidate.elseGroups, opToElseGroup, elseValueMap);

  // Step 2.3: Plan yield augmentation for the active region
  // Groups are in natural discovery order (block_ids appear in
  // dependency order within a sequential basic block).
  bool splitThen = candidate.thenGroups.size() >= 2;
  if (splitThen) {
    planYield(candidate, /*splitThen=*/true, ArrayRef(candidate.thenGroups),
              opToThenGroup, thenValueMap);
  } else if (candidate.elseGroups.size() >= 2) {
    planYield(candidate, /*splitThen=*/false, ArrayRef(candidate.elseGroups),
              opToElseGroup, elseValueMap);
  }

  // Step 2.4: Debug yield augmentation
  dumpYieldAugmentation(candidate);
}

// ============================================================================
// Part3: Materialization
// ============================================================================
//
// Unified yield-chain approach for both Case A and Case B.
// For each candidate, create a chain of split ifs.  Each if has the same
// result types; non-producing slots passthrough from the previous if.
// Case A: cross-group values become new yield slots; first if's else
//   uses zero placeholders (since original if has no else values).
// Case B: original yield slots preserved; first if's else passes through
//   original else values.

/// Walk through reinterpret_cast and subview ops to find the root memref.
/// Returns the ultimate source (typically a function argument = GM, or an
/// alloc/alloca) that downstream provenance analyses should see.
static Value getRootMemRef(Value v) {
  while (v) {
    if (auto *op = v.getDefiningOp()) {
      if (isa<memref::ReinterpretCastOp, memref::SubViewOp>(op)) {
        v = op->getOperand(0);
        continue;
      }
      // Trace through split-if chain results: the then branch carries
      // the real computation, whose root we want for the placeholder.
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        unsigned resultIdx = cast<OpResult>(v).getResultNumber();
        auto thenTerm = ifOp.thenBlock()->getTerminator();
        if (auto thenYield = dyn_cast<scf::YieldOp>(thenTerm)) {
          v = thenYield->getOperand(resultIdx);
          continue;
        }
      }
    }
    break;
  }
  return v;
}

/// Walk through scf.if (then branch) and scf.for (init args) to find the
/// ultimate source tensor. This allows tensor placeholders to preserve
/// provenance instead of creating a fresh tensor.empty().
static Value getRootTensor(Value v) {
  while (v) {
    if (auto *op = v.getDefiningOp()) {
      // Trace through split-if chain: then branch carries the real
      // computation whose source we want for the placeholder.
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        unsigned resultIdx = cast<OpResult>(v).getResultNumber();
        auto thenTerm = ifOp.thenBlock()->getTerminator();
        if (auto thenYield = dyn_cast<scf::YieldOp>(thenTerm)) {
          v = thenYield->getOperand(resultIdx);
          continue;
        }
      }
      // Trace through scf.for: init arg is the source before the loop.
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        unsigned resultIdx = cast<OpResult>(v).getResultNumber();
        v = forOp.getInitArgs()[resultIdx];
        continue;
      }
    }
    break;
  }
  return v;
}

static llvm::FailureOr<Value>
createMatmulPlaceHolderValue(OpBuilder &builder, scf::IfOp ifOp,
                             linalg::MatmulOp matmulOp,
                             RankedTensorType tensorType, Location loc) {
  auto bias = matmulOp.getDpsInitOperand(0)->get();
  auto block = ifOp->getBlock();
  return llvm::TypeSwitch<Value, llvm::FailureOr<Value>>(bias)
      .Case([&](BlockArgument barg) -> llvm::FailureOr<Value> {
        if (barg.getParentBlock()->findAncestorOpInBlock(*ifOp))
          return bias;
        else
          return llvm::failure();
      })
      .Case([&](OpResult res) -> llvm::FailureOr<Value> {
        auto defOp = res.getDefiningOp();
        auto fillOp = llvm::dyn_cast_if_present<linalg::FillOp>(defOp);
        if (!fillOp) {
          return llvm::failure();
        }

        auto sourceOp = fillOp.getDpsInitOperand(0)->get().getDefiningOp();
        auto sourceEmpty = llvm::dyn_cast_if_present<tensor::EmptyOp>(sourceOp);
        if (!sourceEmpty)
          return llvm::failure();

        auto emptyOp = builder.create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType());
        return emptyOp.getResult();
      })
      .Default([](auto) { return llvm::failure(); });
}

/// Create a zero/default SSA value of the given type.
/// Used for placeholder values in Case A else blocks when the original
/// scf.if has no yield results to passthrough.
/// When \p referenceValue traces to a function argument, the placeholder
/// preserves that provenance instead of creating a fresh alloca.
static llvm::FailureOr<Value>
createPlaceholderValue(int blockId, OpBuilder &builder, Location loc, Type type,
                       Value referenceValue, scf::IfOp ifOp) {
  Value result;
  bool usedTrace = false;
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    // When a reference value traces to a real tensor source (not a
    // tensor.empty placeholder we created), use it directly so downstream
    // provenance analyses see the same origin in both branches.
    // The traced root must dominate the placeholder insertion point:
    // its parent region must contain (be ancestor of, or equal to)
    // the builder's region. Values defined inside inner regions
    // (e.g. scf.if then-blocks) or sibling regions are rejected.
    if (referenceValue) {
      auto matmulOpt = hivm::traceDefOp<linalg::MatmulOp>(referenceValue);
      if (matmulOpt.has_value()) {
        auto matmulOp = llvm::dyn_cast<linalg::MatmulOp>(matmulOpt.value());
        if (matmulOp) {
          auto resultRes = createMatmulPlaceHolderValue(builder, ifOp, matmulOp,
                                                        tensorType, loc);
          if (llvm::failed(resultRes)) {
            return llvm::failure();
          }
          result = resultRes.value();
          usedTrace = true;
        }
      }
      if (!usedTrace) {
        Value root = getRootTensor(referenceValue);
        auto *rootRegion = root.getParentRegion();
        auto *builderRegion = builder.getBlock()->getParent();
        bool dominates = (rootRegion == builderRegion) ||
                         rootRegion->isAncestor(builderRegion);
        if (dominates) {
          if (auto *rootOp = root.getDefiningOp()) {
            auto existingId =
                rootOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
            if (!existingId || existingId.getInt() != -1) {
              result = root;
              usedTrace = true;
            }
          }
        }
      }
    }
    if (!usedTrace) {
      result = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                               tensorType.getElementType());
    }
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    result = builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(floatType, 0.0));
  } else if (auto intType = dyn_cast<IntegerType>(type)) {
    result = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(intType, 0));
  } else if (type.isIndex()) {
    result = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
  } else if (auto memrefType = dyn_cast<MemRefType>(type)) {
    // Create a base memref placeholder.
    // For memrefs with address_space (cbuf/ub), use alloca since these
    // special memory spaces require a valid allocation.
    // For plain memrefs, use a non-alloc-like placeholder (tensor.empty +
    // to_memref) so that downstream traceback (e.g. bisheng's
    // tracebackMemRefToAlloc) doesn't stop here prematurely and pick the
    // wrong alloc when both branches yield alloc-like values.
    Value baseMemref;
    bool usedReference = false;
    // When a reference value traces to a function argument (GM), use that
    // argument as the base memref so downstream provenance analysis sees
    // the same memory space in both branches of the split-if chain.
    // Only applicable for strided memrefs without explicit memory_space
    // (which would require a specific allocation).
    if (referenceValue && !memrefType.getMemorySpace() &&
        isa<StridedLayoutAttr>(memrefType.getLayout())) {
      Value root = getRootMemRef(referenceValue);
      if (auto blockArg = dyn_cast<BlockArgument>(root)) {
        if (blockArg.getOwner()->isEntryBlock()) {
          baseMemref = root;
          usedReference = true;
        }
      }
    }
    if (!usedReference) {
      if (memrefType.getMemorySpace()) {
        SmallVector<Value> allocaSizes;
        for (int64_t i = 0; i < memrefType.getRank(); ++i) {
          if (memrefType.isDynamicDim(i)) {
            allocaSizes.push_back(
                builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1))
                    .getResult());
          }
        }
        auto simpleType =
            MemRefType::get(memrefType.getShape(), memrefType.getElementType());
        baseMemref =
            builder.create<memref::AllocaOp>(loc, simpleType, allocaSizes)
                .getResult();
      } else {
        // For plain memrefs without address_space, use memref.alloc()
        // for static shapes and memref.alloca() for dynamic shapes.
        // The else branch is dead (never executed), and DCE will
        // eliminate it. block_id = -1 marks it as a placeholder.
        bool hasDynamic =
            llvm::any_of(memrefType.getShape(), ShapedType::isDynamic);
        if (hasDynamic) {
          SmallVector<Value> allocaSizes;
          for (int64_t i = 0; i < memrefType.getRank(); ++i) {
            if (memrefType.isDynamicDim(i)) {
              allocaSizes.push_back(
                  builder
                      .create<arith::ConstantOp>(loc, builder.getIndexAttr(1))
                      .getResult());
            }
          }
          auto simpleType = MemRefType::get(memrefType.getShape(),
                                            memrefType.getElementType());
          baseMemref =
              builder.create<memref::AllocaOp>(loc, simpleType, allocaSizes)
                  .getResult();
        } else {
          baseMemref =
              builder.create<memref::AllocOp>(loc, memrefType).getResult();
        }
      }
    }
    // Apply strided layout via reinterpret_cast if needed.
    auto layout = memrefType.getLayout();
    if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(layout)) {
      SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
      SmallVector<Value> dynOffsets, dynSizes, dynStrides;

      int64_t off = stridedLayout.getOffset();
      staticOffsets.push_back(off);
      if (ShapedType::isDynamic(off)) {
        dynOffsets.push_back(
            builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0))
                .getResult());
      }

      for (int64_t sz : memrefType.getShape()) {
        staticSizes.push_back(sz);
        if (ShapedType::isDynamic(sz)) {
          dynSizes.push_back(
              builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1))
                  .getResult());
        }
      }

      for (int64_t stride : stridedLayout.getStrides()) {
        staticStrides.push_back(stride);
        if (ShapedType::isDynamic(stride)) {
          dynStrides.push_back(
              builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1))
                  .getResult());
        }
      }

      SmallVector<OpFoldResult> mixedOffsets =
          getMixedValues(staticOffsets, dynOffsets, builder);
      SmallVector<OpFoldResult> mixedSizes =
          getMixedValues(staticSizes, dynSizes, builder);
      SmallVector<OpFoldResult> mixedStrides =
          getMixedValues(staticStrides, dynStrides, builder);
      auto castOp = builder.create<memref::ReinterpretCastOp>(
          loc, memrefType, baseMemref, mixedOffsets.front(), mixedSizes,
          mixedStrides);
      result = castOp.getResult();
    } else {
      result = baseMemref;
    }
  } else {
    LDBG(
        "[Error]: unsupported type for placeholder value in Case A else block");
    return llvm::failure();
  }

  if (!usedTrace) {
    result.getDefiningOp()->setAttr(CVPipeline::kBlockId,
                                    builder.getI32IntegerAttr(blockId));
  }
  return result;
}

/// Rewire cross-group SSA uses in a group's ops and nested ifs, then move
/// them into the target block.
static void
rewireAndMoveOps(BlockGroup &group,
                 llvm::SmallDenseMap<Value, Value> &crossValueReplacement,
                 Block &targetBlock) {
  auto rewireOperand = [&](OpOperand &operand) {
    auto it = crossValueReplacement.find(operand.get());
    if (it != crossValueReplacement.end()) {
      operand.set(it->second);
    }
  };

  for (auto *op : group.ops) {
    for (auto &operand : op->getOpOperands()) {
      rewireOperand(operand);
    }

    if (!isa<scf::IfOp>(op)) {
      for (auto &region : op->getRegions()) {
        // Use region walk (not op->walk) to avoid re-visiting the parent op
        // whose operands were already rewired above.
        region.walk([&](Operation *nestedOp) {
          for (auto &operand : nestedOp->getOpOperands()) {
            rewireOperand(operand);
          }
        });
      }
    }
  }
  for (auto nestedIf : group.nestedIfs) {
    for (auto &operand : nestedIf->getOpOperands()) {
      rewireOperand(operand);
    }

    for (auto &region : nestedIf->getRegions()) {
      region.walk([&](Operation *nestedOp) {
        for (auto &operand : nestedOp->getOpOperands()) {
          rewireOperand(operand);
        }
      });
    }
  }

  // Merge ops and nestedIfs, then sort by original position to preserve
  // interleaving order before moving into the split-if's then block.
  SmallVector<Operation *> allOps;
  allOps.reserve(group.ops.size() + group.nestedIfs.size());
  allOps.append(group.ops.begin(), group.ops.end());
  for (auto nestedIf : group.nestedIfs) {
    allOps.push_back(nestedIf.getOperation());
  }
  if (allOps.empty()) {
    return;
  }
  llvm::sort(allOps,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  OpBuilder builder{allOps.front()};
  for (auto *op : allOps) {
    op->moveBefore(&targetBlock, targetBlock.end());
    op->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(group.blockId));
  }
}

/// Ensure a value can be safely yielded from the else block.
/// Clones cheap alias ops (reinterpret_cast, subview, etc.) and allocs
/// locally in the else block so the else branch doesn't reference values
/// defined outside the if.
/// Constants (arith::ConstantOp) are NOT cloned — the pre-created placeholder
/// already dominates and cloning would perturb cross-group tracking.
/// Other values are returned unchanged.
static Value ensureLocalValue(int blockId, Value val, Block &elseBlock,
                              OpBuilder &builder) {
  // Block arguments always dominate
  if (isa<BlockArgument>(val)) {
    return val;
  }

  auto *defOp = val.getDefiningOp();
  if (!defOp) {
    return val;
  }

  if (defOp->getBlock() == &elseBlock) {
    return val;
  }

  if (isa<ViewLikeOpInterface>(defOp)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&elseBlock);
    auto *cloned = builder.clone(*defOp);
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
    return cloned->getResult(0);
  }

  return val;
}

/// Build then yield for a non-last group that produces output values.
/// Simply yields all output values in order.
static void buildThenYieldForGroup(const GroupOutputInfo &output,
                                   Block &thenBlock, Location loc,
                                   OpBuilder &builder) {
  builder.setInsertionPointToEnd(&thenBlock);
  SmallVector<Value> yieldVals;
  for (auto val : output.outputValues) {
    yieldVals.push_back(val);
  }
  builder.create<scf::YieldOp>(loc, yieldVals);
}

/// Check whether \p val (a defined SSA value) dominates \p block.
/// Walks up from \p block to see if the defining op's block is an ancestor.
static bool valueDominates(Value val, Block *block) {
  if (isa<BlockArgument>(val)) {
    return true;
  }
  auto *defOp = val.getDefiningOp();
  if (!defOp) {
    return true;
  }
  Block *defBlock = defOp->getBlock();
  for (Block *cur = block; cur;
       cur = cur->getParentOp() ? cur->getParentOp()->getBlock() : nullptr) {
    if (cur == defBlock) {
      return true;
    }
  }
  return false;
}

/// Build else yield for a non-last group that produces output values.
/// For Case B, uses original else-side yield values for slots that correspond
/// to original yield slots (only when the value dominates the else block).
/// For other slots (pure cross-group augmentation, Case A, or when the
/// original else value is defined inside the original else block and does
/// not dominate the new split-if), creates placeholder values.
static llvm::LogicalResult
buildElseYieldForGroup(const int blockId, const GroupOutputInfo &output,
                       Block &elseBlock, Location loc, OpBuilder &builder) {
  auto ifOp = llvm::dyn_cast<scf::IfOp>(elseBlock.getParentOp());
  if (!ifOp)
    return llvm::failure();
  builder.setInsertionPointToEnd(&elseBlock);
  SmallVector<Value> yieldVals;

  // Cache placeholders by type to avoid creating duplicate constants
  // (e.g., 5 identical "arith.constant 0 : index" for 5 index slots).
  llvm::SmallDenseMap<Type, FailureOr<Value>> placeholderCache;
  auto getOrCreatePlaceholder = [&](Type type, Value ref) -> FailureOr<Value> {
    auto it = placeholderCache.find(type);
    if (it != placeholderCache.end()) {
      return it->second;
    }
    auto ph = createPlaceholderValue(blockId, builder, loc, type, ref, ifOp);
    placeholderCache[type] = ph;
    return ph;
  };

  for (unsigned idx = 0; idx < output.outputValues.size(); ++idx) {
    // Check if this slot has a corresponding original else-side value
    // (Case B non-last group producing an original yield slot).
    if (idx < output.origElseValues.size() && output.origElseValues[idx]) {
      Value elseVal = output.origElseValues[idx];
      // The original else value may be defined inside the original else
      // block (e.g., Scene 3/4). Such values do NOT dominate the new
      // split-if's else block. In that case, fall back to a placeholder.
      if (valueDominates(elseVal, &elseBlock)) {
        yieldVals.push_back(
            ensureLocalValue(blockId, elseVal, elseBlock, builder));
      }
      continue;
    }
    auto phRes = getOrCreatePlaceholder(output.outputTypes[idx],
                                        output.outputValues[idx]);
    if (llvm::failed(phRes)) {
      return failure();
    }
    yieldVals.push_back(phRes.value());
  }
  builder.create<scf::YieldOp>(loc, yieldVals);
  return llvm::success();
}

struct OtherSideContext {
  bool hasOps;
  SmallVector<Operation *> ops;
  SmallVector<Value> yieldValues;
};

/// Collect ops (and yield values for Case B) from the side that is NOT being
/// split.  In Scene 3/4 the last split-if's else block absorbs them.
static OtherSideContext collectOtherSideInfo(const CandidateIf &c,
                                             bool splitThen) {
  OtherSideContext ctx;
  auto &otherGroups = splitThen ? c.elseGroups : c.thenGroups;
  ctx.hasOps = !otherGroups.empty();

  if (ctx.hasOps) {
    auto originalIf = c.ifOp;
    for (auto &g : otherGroups) {
      for (auto *op : g.ops) {
        ctx.ops.push_back(op);
      }
      for (auto nestedIf : g.nestedIfs) {
        ctx.ops.push_back(nestedIf.getOperation());
      }
    }
    llvm::sort(ctx.ops, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
  }

  // Always collect yield values for Case B, even when the other side has no
  // ops. The original else-branch yield values dominate all split-ifs and must
  // be preserved in the last split-if's else block to maintain correctness.
  if (c.hasYield) {
    auto originalIf = c.ifOp;
    Block *otherBlk =
        splitThen ? originalIf.elseBlock() : originalIf.thenBlock();
    auto otherYield = cast<scf::YieldOp>(otherBlk->getTerminator());
    for (auto v : otherYield->getOperands()) {
      ctx.yieldValues.push_back(v);
    }
  }
  return ctx;
}

// ============================================================================
// Part3: Materialization (orchestrator)
// ============================================================================

/// Build then yield for the last-if in Case B.
/// The last-if carries ALL original result types. For slots whose value is
/// produced by an earlier group, re-yield from that group's if result
/// (jump reference, no passthrough chain).
static void buildThenYieldForLastIf(
    unsigned lastGi, const YieldAugmentation &ya,
    llvm::SmallDenseMap<Value, Value> &crossValueReplacement, Block &thenBlock,
    Location loc, OpBuilder &builder) {
  builder.setInsertionPointToEnd(&thenBlock);
  SmallVector<Value> yieldVals;
  for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
    int prodGroup = ya.origYieldProducerGroup[slot];
    if (prodGroup == static_cast<int>(lastGi) || prodGroup < 0) {
      // Produced by last group, or is a block arg / external value:
      // yield directly (it dominates the split-if).
      yieldVals.push_back(ya.origYieldValues[slot]);
    } else {
      // Produced by an earlier group: re-yield via jump reference.
      Value oldVal = ya.origYieldValues[slot];
      auto it = crossValueReplacement.find(oldVal);
      if (it != crossValueReplacement.end()) {
        yieldVals.push_back(it->second);
      } else {
        yieldVals.push_back(oldVal);
      }
    }
  }
  builder.create<scf::YieldOp>(loc, yieldVals);
}

/// Build else yield for the last-if in Case B.
/// Absorbs other-side ops (Scene 3/4) or creates placeholders.
static llvm::LogicalResult
buildElseYieldForLastIf(int blockId, const OtherSideContext &otherCtx,
                        ArrayRef<Type> lastIfTypes, Block &elseBlock,
                        Location loc, OpBuilder &builder) {
  auto ifOp = llvm::dyn_cast<scf::IfOp>(elseBlock.getParentOp());
  if (elseBlock.mightHaveTerminator()) {
    elseBlock.getTerminator()->erase();
  }

  builder.setInsertionPointToEnd(&elseBlock);
  SmallVector<Value> yieldVals;

  if (otherCtx.hasOps) {
    // Scene 3/4: absorb other side ops into else block.
    for (auto *op : otherCtx.ops) {
      op->moveBefore(&elseBlock, elseBlock.end());
    }
  }

  // Use original else-branch yield values when available (Case B).
  // These values are defined outside the original if and dominate all
  // split-ifs, preserving the original semantics when condition is false.
  // For slots beyond the original yield count, use placeholders.
  for (unsigned slot = 0; slot < lastIfTypes.size(); ++slot) {
    if (slot < otherCtx.yieldValues.size()) {
      yieldVals.push_back(ensureLocalValue(blockId, otherCtx.yieldValues[slot],
                                           elseBlock, builder));
    } else {
      auto phRes = createPlaceholderValue(blockId, builder, loc,
                                          lastIfTypes[slot], Value(), ifOp);
      if (llvm::failed(phRes)) {
        return failure();
      }
      yieldVals.push_back(phRes.value());
    }
  }

  builder.create<scf::YieldOp>(loc, yieldVals);
  return llvm::success();
}

/// Register a non-last group's produced values so downstream groups can
/// rewire their cross-group SSA uses to the new results.
static void updateCrossValueReplacementGroup(
    const GroupOutputInfo &output, scf::IfOp splitIf, Block &thenBlock,
    llvm::SmallDenseMap<Value, Value> &crossValueReplacement) {
  SmallPtrSet<Block *, 4> thenBlocks;
  thenBlock.walk([&](Block *b) { thenBlocks.insert(b); });

  for (unsigned idx = 0; idx < output.outputValues.size(); ++idx) {
    Value oldVal = output.outputValues[idx];
    Value newVal = splitIf.getResult(idx);
    oldVal.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
      return !thenBlocks.contains(operand.getOwner()->getBlock());
    });
    crossValueReplacement[oldVal] = newVal;
  }
}

static scf::YieldOp safeGetTerminator(Block *block) {
  if (!block) {
    return nullptr;
  }
  return llvm::dyn_cast_if_present<scf::YieldOp>(block->getTerminator());
}

constexpr llvm::StringRef kSplittedIf = "ssbuffer.splitted_if";

static void postProcess(scf::IfOp ifOp, scf::IfOp sourceIfOp, int blockId) {
  OpBuilder builder{ifOp};

  // clone attrs
  ifOp->setAttrs(sourceIfOp->getAttrs());

  // remap core types
  auto thenYield = safeGetTerminator(ifOp.thenBlock());
  if (thenYield && thenYield.getNumOperands() > 0) {
    std::string joinedAttr =
        llvm::join(llvm::map_range(thenYield.getOperands(),
                                   [](Value value) {
                                     CVPipeline::CoreType ct =
                                         CVPipeline::getValueCoreType(value);
                                     return CVPipeline::coreTypeToString(ct);
                                   }),
                   ", ");
    ifOp->setAttr(CVPipeline::kCoreType, builder.getStringAttr(joinedAttr));
  } else {
    ifOp->removeAttr(CVPipeline::kCoreType);
  }

  // mark if and yields as the same block id
  auto setBlockId = [&](Operation *op) {
    op->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
  };
  setBlockId(ifOp);
  if (thenYield) {
    setBlockId(thenYield);
  }
  if (auto elseYield = safeGetTerminator(ifOp.elseBlock())) {
    setBlockId(elseYield);
  }

  ifOp->setAttr(kSplittedIf, builder.getUnitAttr());
}

/// Materialization for Case A and Case B using per-group signatures.
///
/// New design: each split-if gets its own result signature based on what
/// it actually produces. Consumers directly reference producer if results.
/// No passthrough chain. Pure side-effect groups become void ifs.
///
/// Flow for each group:
///   1. Non-last group producing values -> result-bearing if with its output
///   types
///   2. Non-last group producing nothing -> void if (no results, no else)
///   3. Last group (Case B) -> result-bearing if with ALL original result types
///   4. Last group (Case A) -> void if (no original yield)
static llvm::LogicalResult
materializeCandidate(CandidateIf &c, CVPipeline::ComputeBlockIdManager &bm) {
  OpBuilder builder(c.ifOp);
  auto originalIf = c.ifOp;
  auto loc = originalIf.getLoc();
  Value condition = originalIf.getCondition();
  auto &ya = c.yieldAug;

  LDBG("[Part3] enter materializeCandidate hasYield=" << c.hasYield);

  bool splitThen = c.thenGroups.size() >= 2;
  auto &groups = splitThen ? c.thenGroups : c.elseGroups;
  unsigned nGroups = groups.size();
  if (nGroups < 2) {
    return llvm::success();
  }

  auto otherCtx = collectOtherSideInfo(c, splitThen);

  LDBG("[Part3] splitThen=" << splitThen << " nGroups=" << nGroups
                            << " otherSideHasOps=" << otherCtx.hasOps);

  builder.setInsertionPoint(originalIf);

  // Step 3.2: Negate condition when splitting else region.
  if (!splitThen) {
    auto trueVal = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(builder.getI1Type(), 1));
    condition =
        builder.create<arith::XOrIOp>(loc, condition, trueVal).getResult();
  }

  llvm::SmallDenseMap<Value, Value> crossValueReplacement;
  Operation *lastCreatedIf = originalIf.getOperation();

  // Phase 1: Materialize each non-last group.
  // Each group gets its own if with types specific to what it produces.
  // The last group is handled in Phase 2.
  // IMPORTANT: Reset builder insertion point after each created if, because
  // filling then/else blocks leaves the builder INSIDE those blocks.
  for (unsigned gi = 0; gi < nGroups - 1; ++gi) {
    auto &output = ya.groupOutputs[gi];
    scf::IfOp splittedIf;

    if (output.isVoid()) {
      // Pure side-effect group: void if (no results, no else).
      LDBG("[Part3] group[" << gi << "] void if");

      builder.setInsertionPointAfter(lastCreatedIf);
      splittedIf = builder.create<scf::IfOp>(loc, condition, /*hasElse=*/false);
      splittedIf->setAttr(CVPipeline::kBlockId,
                          builder.getI32IntegerAttr(groups[gi].blockId));
      lastCreatedIf = splittedIf.getOperation();
      if (splittedIf.getThenRegion().empty()) {
        splittedIf.getThenRegion().emplaceBlock();
      }
      Block &thenBlock = splittedIf.getThenRegion().front();
      if (thenBlock.mightHaveTerminator()) {
        thenBlock.getTerminator()->erase();
      }

      rewireAndMoveOps(groups[gi], crossValueReplacement, thenBlock);

      builder.setInsertionPointToEnd(&thenBlock);
      builder.create<scf::YieldOp>(loc);

      // No crossValueReplacement update needed (void if produces no values).
    } else {
      // Result-bearing non-last group: if with per-group output types.
      LDBG("[Part3] group[" << gi << "] result-bearing if ("
                            << output.outputValues.size() << " outputs)");

      builder.setInsertionPointAfter(lastCreatedIf);
      splittedIf = builder.create<scf::IfOp>(loc, output.outputTypes, condition,
                                             /*hasElse=*/true);
      splittedIf->setAttr(CVPipeline::kBlockId,
                          builder.getI32IntegerAttr(groups[gi].blockId));
      lastCreatedIf = splittedIf.getOperation();
      if (splittedIf.getThenRegion().empty()) {
        splittedIf.getThenRegion().emplaceBlock();
      }
      if (splittedIf.getElseRegion().empty()) {
        splittedIf.getElseRegion().emplaceBlock();
      }

      // Fill then block.
      Block &thenBlock = splittedIf.getThenRegion().front();
      if (thenBlock.mightHaveTerminator()) {
        thenBlock.getTerminator()->erase();
      }
      rewireAndMoveOps(groups[gi], crossValueReplacement, thenBlock);
      buildThenYieldForGroup(output, thenBlock, loc, builder);

      // Fill else block with placeholders.
      Block &elseBlock = splittedIf.getElseRegion().front();
      if (elseBlock.mightHaveTerminator()) {
        elseBlock.getTerminator()->erase();
      }
      if (buildElseYieldForGroup(groups[gi].blockId, output, elseBlock, loc,
                                 builder)
              .failed()) {
        return llvm::failure();
      }

      // Register produced values for downstream groups.
      updateCrossValueReplacementGroup(output, splittedIf, thenBlock,
                                       crossValueReplacement);
    }
    postProcess(splittedIf, originalIf, groups[gi].blockId);
  }

  // Phase 2: Materialize the last group.
  scf::IfOp lastIf;
  unsigned lastGi = nGroups - 1;
  if (c.hasYield) {
    // Case B: last-if carries ALL original result types.
    SmallVector<Type> lastIfTypes;
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      lastIfTypes.push_back(originalIf.getResult(slot).getType());
    }

    LDBG("[Part3] last-if (Case B, " << ya.numOriginalSlots << " results)");

    builder.setInsertionPointAfter(lastCreatedIf);
    lastIf = builder.create<scf::IfOp>(loc, lastIfTypes, condition,
                                       /*hasElse=*/true);
    lastIf->setAttr(CVPipeline::kBlockId,
                    builder.getI32IntegerAttr(groups[lastGi].blockId));
    if (lastIf.getThenRegion().empty()) {
      lastIf.getThenRegion().emplaceBlock();
    }
    if (lastIf.getElseRegion().empty()) {
      lastIf.getElseRegion().emplaceBlock();
    }

    // Fill then block.
    Block &thenBlock = lastIf.getThenRegion().front();
    if (thenBlock.mightHaveTerminator()) {
      thenBlock.getTerminator()->erase();
    }
    rewireAndMoveOps(groups[lastGi], crossValueReplacement, thenBlock);
    buildThenYieldForLastIf(lastGi, ya, crossValueReplacement, thenBlock, loc,
                            builder);

    // Fill else block (with Scene 3/4 absorption if needed).
    Block &elseBlock = lastIf.getElseRegion().front();
    if (buildElseYieldForLastIf(groups[lastGi].blockId, otherCtx, lastIfTypes,
                                elseBlock, loc, builder)
            .failed()) {
      return llvm::failure();
    }

    // Replace original if's uses with last-if's results.
    for (unsigned ri = 0; ri < ya.numOriginalSlots; ++ri) {
      originalIf.getResult(ri).replaceAllUsesWith(lastIf.getResult(ri));
    }
  } else {
    // Case A: last group is void (no original yield, no cross-group values).
    LDBG("[Part3] last-if (Case A, void if)");

    builder.setInsertionPointAfter(lastCreatedIf);
    bool hasElse = otherCtx.hasOps;
    lastIf = builder.create<scf::IfOp>(loc, condition, hasElse);
    lastIf->setAttr(CVPipeline::kBlockId,
                    builder.getI32IntegerAttr(groups[lastGi].blockId));
    if (lastIf.getThenRegion().empty()) {
      lastIf.getThenRegion().emplaceBlock();
    }
    Block &thenBlock = lastIf.getThenRegion().front();
    if (thenBlock.mightHaveTerminator()) {
      thenBlock.getTerminator()->erase();
    }

    rewireAndMoveOps(groups[lastGi], crossValueReplacement, thenBlock);

    builder.setInsertionPointToEnd(&thenBlock);
    builder.create<scf::YieldOp>(loc);

    // Scene 3/4: else block absorbs other side's ops.
    if (hasElse) {
      if (lastIf.getElseRegion().empty()) {
        lastIf.getElseRegion().emplaceBlock();
      }
      Block &elseBlock = lastIf.getElseRegion().front();
      if (elseBlock.mightHaveTerminator()) {
        elseBlock.getTerminator()->erase();
      }

      for (auto *op : otherCtx.ops) {
        op->moveBefore(&elseBlock, elseBlock.end());
      }

      builder.setInsertionPointToEnd(&elseBlock);
      builder.create<scf::YieldOp>(loc);
    }
  }
  postProcess(lastIf, originalIf, groups[lastGi].blockId);

  // Erase the original if.
  originalIf->erase();

  return llvm::success();
}

static void rearrangeIfOp(scf::IfOp ifOp,
                          CVPipeline::MemoryDependenceGraph &memGraph) {
  Operation *lastDependency = nullptr;
  Block *block = ifOp->getBlock();
  ifOp->walk([&](Operation *nestedOp) {
    for (Value operand : nestedOp->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!defOp || ifOp->isAncestor(defOp)) {
        continue;
      }
      auto defOpInBlock = block->findAncestorOpInBlock(*defOp);
      if (!defOpInBlock) {
        continue;
      }
      if (!lastDependency || lastDependency->isBeforeInBlock(defOpInBlock)) {
        lastDependency = defOpInBlock;
      }
    }
  });
  for (auto *memDep : memGraph.getExecBefore(ifOp)) {
    auto depInBlock = block->findAncestorOpInBlock(*memDep);
    if (!depInBlock) {
      continue;
    }
    if (!lastDependency || lastDependency->isBeforeInBlock(depInBlock)) {
      lastDependency = depInBlock;
    }
  }
  if (lastDependency) {
    ifOp->moveAfter(lastDependency);
  }
}

namespace {

class SplitIfByBlockIdPass
    : public PassWrapper<SplitIfByBlockIdPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitIfByBlockIdPass)

  SplitIfByBlockIdPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "split-if-by-block-id"; }

  llvm::StringRef getDescription() const final {
    return "Split scf.if operations so that each if contains only ops of a "
           "single block_id";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }
};

} // namespace

void SplitIfByBlockIdPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module)) {
    return;
  }

  FallbackHelper fallback{module};

  LDBG("Before:\n" << module << "\n----------");

  CVPipeline::ComputeBlockIdManager bm(module);
  auto mainRes = walkMainLoop(module, [&](Operation *op) {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (funcOp && (funcOp.getSymName() == kSkippedDeltaformerKernel ||
                   funcOp.getSymName() == kSkippedChunkwiseKernel ||
                   funcOp.getSymName() == kSkippedParallelNsaFwdKernel)) {
      LDBG("Skip kernel: " << funcOp.getSymName());
      return llvm::success();
    }
    LDBG("Detected main loop: " << *op);
    auto walkRes = op->walk([&](scf::IfOp ifOp) {
      auto candidate = getCandidate(ifOp);
      preprocessScalarDependencies(candidate);
      LLVM_DEBUG(dumpCandidate(candidate));
      analyzeDependencies(candidate);
      if (materializeCandidate(candidate, bm).failed()) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
    if (walkRes.wasInterrupted()) {
      return llvm::failure();
    }
    return llvm::success();
  });

  if (llvm::failed(mainRes)) {
    LDBG("Split if failed, fallback to original");
    fallback.restore();
    return;
  }

  auto &aa = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph{module, aa};
  module->walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr(kSplittedIf)) {
      rearrangeIfOp(ifOp, memGraph);
      ifOp->removeAttr(kSplittedIf);
    }
  });

  // Dump the post-split IR in debug builds
  LDBG("After: \n" << module << "\n----------");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSplitIfByBlockIdPass() {
  return std::make_unique<SplitIfByBlockIdPass>();
}

} // namespace triton
} // namespace mlir
