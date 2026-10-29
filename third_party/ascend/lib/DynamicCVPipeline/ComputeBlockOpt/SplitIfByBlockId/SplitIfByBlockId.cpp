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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/Common/DependencyHelper.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"
#include "DynamicCVPipeline/Common/FallbackHelper.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

static constexpr const char *DEBUG_TYPE = "SplitIfByBlockId";
static constexpr llvm::StringLiteral kSkippedKernels[1] = {""};
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

struct CrossGroupValue {
  Value value;                             // the SSA value crossing groups
  unsigned fromGroupIdx;                   // producer group index
  SmallVector<unsigned, 0> toGroupIndices; // consumer group indices
};

struct GroupOutputInfo {

  /// Values this group yields (cross-group values it produces; in Case B also
  /// the original yield slots it produces or has to carry).
  SmallVector<Value, 0> outputValues;

  /// Case B: original else-side yield value for each output slot.
  /// Set for original yield slots; empty for Case A or cross-group-only values.
  SmallVector<Value, 0> origElseValues;

  SmallVector<Type, 0> outputTypes;

  bool isVoid() const { return outputValues.empty(); }
};

/// Case B: how an original result slot of the source if reaches the consumers
/// of the split-if chain.
///
/// Classification keys on the then-side provenance and on whether the else
/// operand is usable, not on whether the else operand is internal: the else
/// side is materialized by one uniform path, so both variants of a row behave
/// alike. then/else provenance combos and the row each one lands in:
///   then external, else external  => External
///   then external, else internal  => External
///   then internal, else external  => Delegated / NativeLast
///   then internal, else internal  => Relay / NativeLast
enum class SlotKind {
  /// Produced by a non-last group, whose else operand is already the original
  /// else value: consumers read that group's split-if result directly.
  Delegated,
  /// Produced by the last group: no redirect needed, it is native there.
  NativeLast,
  /// Produced by an earlier group, but the slot's else operand is not usable
  /// there: the last group forwards it through the producer's if result.
  Relay,
  /// Not produced inside the split region (block argument or defined before the
  /// source if): the last group carries it as `if(cond) then v else w`, where v
  /// and w are the slot's then/else operands (they may differ). w may also be
  /// an
  /// internal value: the un-split side's ops are moved into the last group's
  /// else
  /// block before its yield is built, so w dominates there.
  External,
};

static llvm::StringRef slotKindName(SlotKind kind) {
  switch (kind) {
  case SlotKind::Delegated:
    return "Delegated";
  case SlotKind::NativeLast:
    return "NativeLast";
  case SlotKind::Relay:
    return "Relay";
  case SlotKind::External:
    return "External";
  }
  return "Unknown";
}

/// Each split-if gets its own result signature based on what
/// it actually produces. No uniform slot plan.
struct YieldAugmentation {
  /// Cross-group values (deduplicated, in producer-group order).
  SmallVector<CrossGroupValue, 0> crossValues;

  /// Per-group output info, one entry per group (last group included).
  SmallVector<GroupOutputInfo, 0> groupOutputs;

  unsigned numGroups = 0;

  // Case B only, all indexed by original slot s in [0, numOriginalSlots):
  //   val(s)     := origYieldValues[s]         value yielded by the source if
  //   prod(s)    := origYieldProducerGroup[s]  group that produces val(s), or
  //                                            none when val(s) is defined
  //                                            outside the split region
  //   isLast(s)  := prod(s) is the last group
  //   domElse(s) := elseVal(s) exists and dominates the source if
  //   kind(s)    := origYieldSlotKind[s], the first line whose condition holds:
  //                   prod(s) is none                 => External
  //                   isLast(s)                       => NativeLast
  //                   domElse(s)                      => Delegated
  //                   otherwise                       => Relay
  //   carrier(s) := the group whose split-if exposes slot s:
  //                   kind(s) == Delegated ? *prod(s) : last group
  //
  //   kind(s)    | carrier(s) | how slot s reaches the carrier
  //   -----------+------------+-----------------------------------------
  //   Delegated  | *prod(s)   | producer's if result, read there
  //   NativeLast | last       | produced in the last group
  //   Relay      | last       | forwarded via *prod(s)'s if result
  //   External   | last       | carried as if(cond) then v else w
  //
  // Carriers are decided once in planYieldCaseB and only read afterwards.
  // These per-slot vectors replace std::vector per element, so they carry no
  // inline storage.
  unsigned numOriginalSlots = 0;
  SmallVector<Value, 0> origYieldValues;
  SmallVector<std::optional<unsigned>, 0> origYieldProducerGroup;
  SmallVector<SlotKind, 0> origYieldSlotKind;

  /// Carrier group of original slot `slot` (see the table above).
  unsigned carrierGroup(unsigned slot) const {
    return origYieldSlotKind[slot] == SlotKind::Delegated
               ? *origYieldProducerGroup[slot]
               : numGroups - 1;
  }

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

  // Pre-computed split decisions: bidirectional cross-core data deps
  // (mixed CUBE/VECTOR groups with data consumption in both directions)
  bool shouldSplitThen = false;
  bool shouldSplitElse = false;

  bool needsSplit() const { return shouldSplitThen || shouldSplitElse; }
};

} // namespace

static bool hasNestedIfs(const CandidateIf &cand) {
  for (const auto &group : cand.thenGroups) {
    if (!group.nestedIfs.empty()) {
      return true;
    }
  }
  for (const auto &group : cand.elseGroups) {
    if (!group.nestedIfs.empty()) {
      return true;
    }
  }
  return false;
}

static CoreType groupIsCube(const BlockGroup &group) {
  bool seenCube = false;
  bool seenVector = false;
  auto classifyOp = [&](Operation *op) {
    switch (CVPipeline::getCoreTypeOfSimpleOpOrCf(op)) {
    case CVPipeline::CoreType::CUBE_AND_VECTOR:
      return true;
    case CVPipeline::CoreType::CUBE_ONLY:
      seenCube = true;
      break;
    case CVPipeline::CoreType::VECTOR_ONLY:
      seenVector = true;
      break;
    default:
      break;
    }
    return false;
  };
  for (auto *op : group.ops) {
    if (classifyOp(op) || (seenCube && seenVector)) {
      return CoreType::CUBE_AND_VECTOR;
    }
  }
  for (auto nestedIf : group.nestedIfs) {
    if (classifyOp(nestedIf) || (seenCube && seenVector)) {
      return CoreType::CUBE_AND_VECTOR;
    }
  }
  if (seenCube) {
    return CoreType::CUBE_ONLY;
  }
  if (seenVector) {
    return CoreType::VECTOR_ONLY;
  }
  return CoreType::UNDETERMINED;
}

static bool groupContainsOp(const BlockGroup &group, Operation *op) {
  if (llvm::is_contained(group.ops, op)) {
    return true;
  }
  for (auto nestedIf : group.nestedIfs) {
    if (nestedIf == op || nestedIf->isAncestor(op)) {
      return true;
    }
  }
  return false;
}

static bool groupFeedsGroup(const BlockGroup &producer,
                            const BlockGroup &consumer,
                            const CVPipeline::MemoryDependenceGraph &memGraph) {
  DependencyHelper depHelper(memGraph);
  auto checkOp = [&](Operation *op) {
    auto result = depHelper.forEachUser(op, [&](Operation *user) {
      return groupContainsOp(consumer, user) ? WalkResult::interrupt()
                                             : WalkResult::advance();
    });
    return result.wasInterrupted();
  };
  for (auto *op : producer.ops) {
    if (checkOp(op)) {
      return true;
    }
  }
  for (auto nestedIf : producer.nestedIfs) {
    if (checkOp(nestedIf)) {
      return true;
    }
  }
  return false;
}

/// Returns true when groups mix CUBE and VECTOR core types AND have
/// bidirectional data flow (CUBE↔VECTOR in both directions).
static bool hasBidirectionalCrossCoreDeps(
    ArrayRef<BlockGroup> groups,
    const CVPipeline::MemoryDependenceGraph &memGraph) {
  SmallVector<CoreType, 8> isCube;
  isCube.reserve(groups.size());
  bool hasCubeGroup = false;
  bool hasVectorGroup = false;
  for (size_t i = 0; i < groups.size(); ++i) {
    auto cubeStatus = groupIsCube(groups[i]);
    isCube.push_back(cubeStatus);
    if (cubeStatus == CoreType::UNDETERMINED ||
        cubeStatus == CoreType::CUBE_AND_VECTOR) {
      continue;
    }
    hasCubeGroup = hasCubeGroup || (cubeStatus == CoreType::CUBE_ONLY);
    hasVectorGroup = hasVectorGroup || (cubeStatus == CoreType::VECTOR_ONLY);
  }
  if (!hasCubeGroup || !hasVectorGroup) {
    return false;
  }

  // Cross-core data dependency: CUBE data consumed by VECTOR and VECTOR
  // data consumed by CUBE. Ordering alone is not enough to split.
  bool cubeFeedsVector = false;
  bool vectorFeedsCube = false;
  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = 0; j < groups.size(); ++j) {
      if (i == j || isCube[i] == CoreType::UNDETERMINED ||
          isCube[i] == CoreType::CUBE_AND_VECTOR ||
          isCube[j] == CoreType::UNDETERMINED ||
          isCube[j] == CoreType::CUBE_AND_VECTOR || isCube[i] == isCube[j]) {
        continue;
      }
      if (!groupFeedsGroup(groups[i], groups[j], memGraph)) {
        continue;
      }
      if (isCube[i] == CoreType::CUBE_ONLY) {
        cubeFeedsVector = true;
      } else {
        vectorFeedsCube = true;
      }
    }
  }
  return cubeFeedsVector && vectorFeedsCube;
}

static inline void dumpCandidate(CandidateIf &candidate) {
  LDBG("Processing: " << candidate.ifOp);
  LDBG("  selfBlockId=" << candidate.selfBlockId);
  LDBG("  groups(then=" << candidate.thenGroups.size()
                        << " else=" << candidate.elseGroups.size()
                        << ")  yield=" << candidate.hasYield
                        << "  nested=" << hasNestedIfs(candidate));
  for (const auto &group : candidate.thenGroups) {
    LDBG("  then block_id=" << group.blockId << "  ops=" << group.ops.size()
                            << "  nestedIfs=" << group.nestedIfs.size());
  }
  for (const auto &group : candidate.elseGroups) {
    LDBG("  else block_id=" << group.blockId << "  ops=" << group.ops.size()
                            << "  nestedIfs=" << group.nestedIfs.size());
  }
}

/// Walk a single block, grouping ops by block_id.
/// Nested ifs attach to the nearest preceding group.
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
      // Nested ifs form their own group by block_id so the parent if can
      // be split when inner ifs of different block_ids coexist in the same
      // region.
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

static CandidateIf
getCandidate(scf::IfOp ifOp,
             const CVPipeline::MemoryDependenceGraph &memGraph) {
  CandidateIf cand;
  cand.ifOp = ifOp;
  cand.hasYield = (ifOp->getNumResults() > 0);

  // self block_id
  auto selfBlockId = CVPipeline::getOpBlockId(ifOp);
  cand.selfBlockId = selfBlockId.value_or(-1);

  // Split decision runs on raw per-blockId groups before merging,
  // so producer/consumer roles reflect original block structure.
  auto rawThenGroups = groupOpsInBlock(*ifOp.thenBlock());
  SmallVector<BlockGroup> rawElseGroups;
  Block *elseBlk = ifOp.elseBlock();
  if (elseBlk) {
    rawElseGroups = groupOpsInBlock(*elseBlk);
  }

  // Pre-compute split decisions on raw per-blockId groups.
  // Only split when both CUBE and VECTOR groups exchange data bidirectionally.
  cand.shouldSplitThen = hasBidirectionalCrossCoreDeps(rawThenGroups, memGraph);
  cand.shouldSplitElse = hasBidirectionalCrossCoreDeps(rawElseGroups, memGraph);
  if (!cand.needsSplit()) {
    return cand;
  }

  cand.thenGroups = std::move(rawThenGroups);
  cand.elseGroups = std::move(rawElseGroups);

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
// Scan cross-group SSA deps, then plan yield augmentation (Case A/B).
static void dumpYieldAugmentation(const CandidateIf &cand) {
  auto &ya = cand.yieldAug;
  bool splitThen = cand.shouldSplitThen;
  auto &groups = splitThen ? cand.thenGroups : cand.elseGroups;
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

  // --- original slot plan (Case B) ---
  if (cand.hasYield) {
    LDBG("[Part2] Case B: original yield (" << ya.numOriginalSlots
                                            << " slot(s))");
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      const auto producer = ya.origYieldProducerGroup[slot];
      LDBG("  slot[" << slot << "] = " << ya.origYieldValues[slot]
                     << "  producer="
                     << (producer
                             ? llvm::Twine("group ") + llvm::Twine(*producer)
                             : llvm::Twine("external"))
                     << "  kind=" << slotKindName(ya.origYieldSlotKind[slot])
                     << "  carrier=group " << ya.carrierGroup(slot));
    }
  } else {
    LDBG("  [Part2] Case A: no original yield");
  }
}

/// Single-pass scan building value-level crossValueMap.
/// Group order is implicit (block_ids in sequential basic block order).
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
    // Walk up from each user to find the nearest ancestor in opToGroup.
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

/// Compute per-group output info for Case B (original yield exists).
/// Tracks else-side yield values for non-last groups' original slots.
static void
planYieldCaseB(CandidateIf &c, bool splitThen, ArrayRef<BlockGroup> groups,
               llvm::SmallDenseMap<Operation *, unsigned> &opToGroup,
               DominanceInfo &dom) {
  unsigned nGroups = groups.size();
  auto &ya = c.yieldAug;

  auto splitYieldOp =
      cast<scf::YieldOp>(splitThen ? c.ifOp.thenBlock()->getTerminator()
                                   : c.ifOp.elseBlock()->getTerminator());

  ya.numGroups = nGroups;
  ya.numOriginalSlots = splitYieldOp->getNumOperands();
  ya.origYieldValues.clear();
  ya.origYieldProducerGroup.clear();
  for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
    Value val = splitYieldOp->getOperand(slot);
    ya.origYieldValues.push_back(val);
    auto *defOp = val.getDefiningOp();
    if (!defOp) {
      ya.origYieldProducerGroup.push_back(std::nullopt); // block argument
      continue;
    }
    auto it = opToGroup.find(defOp);
    ya.origYieldProducerGroup.push_back(
        it == opToGroup.end() ? std::nullopt
                              : std::optional<unsigned>(it->second));
  }

  auto otherYieldOp =
      cast<scf::YieldOp>(splitThen ? c.ifOp.elseBlock()->getTerminator()
                                   : c.ifOp.thenBlock()->getTerminator());
  SmallVector<Value> otherYieldValues;
  for (auto v : otherYieldOp->getOperands()) {
    otherYieldValues.push_back(v);
  }

  // domElse(s): the slot's else operand exists and is usable inside the
  // producer's else block, i.e. it dominates the source if.
  auto domElse = [&](unsigned slot) {
    return slot < otherYieldValues.size() && otherYieldValues[slot] &&
           dom.properlyDominates(otherYieldValues[slot], c.ifOp);
  };

  // kind(s), i.e. which row of the taxonomy table a slot falls into. opToGroup
  // only maps ops of this if, so prod(s) <= lastGi by construction.
  unsigned lastGi = nGroups - 1;
  ya.origYieldSlotKind.assign(ya.numOriginalSlots, SlotKind::External);
  for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
    const auto producer = ya.origYieldProducerGroup[slot];
    if (!producer) {
      continue; // External, the assigned default
    }
    if (*producer >= lastGi) {
      ya.origYieldSlotKind[slot] = SlotKind::NativeLast;
      continue;
    }
    ya.origYieldSlotKind[slot] =
        domElse(slot) ? SlotKind::Delegated : SlotKind::Relay;
  }

  llvm::SmallDenseMap<Value, unsigned> valueToSlot;
  for (unsigned si = 0; si < ya.crossValues.size(); ++si) {
    valueToSlot[ya.crossValues[si].value] = si;
  }

  ya.groupOutputs.resize(nGroups);
  for (unsigned gi = 0; gi < nGroups; ++gi) {
    auto &output = ya.groupOutputs[gi];

    // Build map: original yield value produced by this group → slot number.
    llvm::SmallDenseMap<Value, unsigned> origValToSlot;
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      if (ya.origYieldProducerGroup[slot] == gi) {
        origValToSlot[ya.origYieldValues[slot]] = slot;
      }
    }

    SmallPtrSet<Value, 4> addedValues;

    // Step 1: Cross-group values produced by this group.
    // Also check if each value is an original yield slot for else-side
    // tracking.
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
      if (ya.origYieldProducerGroup[slot] == gi) {
        Value origVal = ya.origYieldValues[slot];
        if (!addedValues.contains(origVal)) {
          output.outputValues.push_back(origVal);
          output.outputTypes.push_back(origVal.getType());
          output.origElseValues.push_back(otherYieldValues[slot]);
        }
      }
    }
  }

  // Append the slots this group carries but does not produce: the Relay and
  // External rows of the taxonomy table. Delegated slots live in their
  // producer's group, and NativeLast slots were pushed by Step 1/2 above (the
  // duplicate check below skips them).
  auto &lastOutput = ya.groupOutputs[lastGi];
  for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
    if (ya.carrierGroup(slot) != lastGi) {
      continue;
    }
    Value origVal = ya.origYieldValues[slot];
    if (llvm::is_contained(lastOutput.outputValues, origVal)) {
      continue;
    }
    lastOutput.outputValues.push_back(origVal);
    lastOutput.outputTypes.push_back(origVal.getType());
    lastOutput.origElseValues.push_back(
        slot < otherYieldValues.size() ? otherYieldValues[slot] : Value());
  }
}

/// Compute per-group output info for Case A (no original yield).
/// Only cross-group values are yielded; last group is void.
static void planYieldCaseA(CandidateIf &c, ArrayRef<BlockGroup> groups) {
  unsigned nGroups = groups.size();
  auto &ya = c.yieldAug;

  llvm::SmallDenseMap<Value, unsigned> valueToSlot;
  for (unsigned si = 0; si < ya.crossValues.size(); ++si) {
    valueToSlot[ya.crossValues[si].value] = si;
  }

  // Per-group output; the last group stays void (no cross-group consumer).
  ya.groupOutputs.resize(nGroups);
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

static void planYield(
    CandidateIf &c, bool splitThen, ArrayRef<BlockGroup> groups,
    llvm::SmallDenseMap<Operation *, unsigned> &opToGroup,
    llvm::SmallDenseMap<Value,
                        std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>>
        &crossValueMap,
    DominanceInfo &dom) {
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
    planYieldCaseB(c, splitThen, groups, opToGroup, dom);
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

  // Step 2.3: Plan yield augmentation for the active region.
  // Redirected slots need the original else value to dominate the source if;
  // a per-candidate DominanceInfo answers that without ad-hoc block walks.
  // Planning below is read-only, so this instance cannot go stale: the IR edits
  // of materializeCandidate happen after this function returns, and every
  // candidate builds its own instance.
  DominanceInfo dom(candidate.ifOp->getParentOp());

  // Groups are in natural dependency order from sequential basic block.
  bool splitThen = candidate.shouldSplitThen;
  if (splitThen) {
    planYield(candidate, /*splitThen=*/true, ArrayRef(candidate.thenGroups),
              opToThenGroup, thenValueMap, dom);
  } else if (candidate.shouldSplitElse) {
    planYield(candidate, /*splitThen=*/false, ArrayRef(candidate.elseGroups),
              opToElseGroup, elseValueMap, dom);
  }

  // Step 2.4: Debug yield augmentation
  dumpYieldAugmentation(candidate);
}

// ============================================================================
// Part3: Materialization
// ============================================================================
// Build split-if chain with per-group signatures; no passthrough chain.

/// Walk through reinterpret_cast and subview ops to find the root memref.
/// Returns the ultimate source for downstream provenance analyses.
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

/// Walk through scf.if (then branch) and scf.for (init args) to find
/// the ultimate source tensor for provenance-preserving placeholders.
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

static llvm::FailureOr<Value> createMatmulPlaceHolderValue(
    OpBuilder &builder, scf::IfOp ifOp, linalg::MatmulOp matmulOp,
    RankedTensorType tensorType, Location loc, int blockId) {
  auto bias = matmulOp.getDpsInitOperand(0)->get();
  auto block = ifOp->getBlock();
  return llvm::TypeSwitch<Value, llvm::FailureOr<Value>>(bias)
      .Case([&](BlockArgument barg) -> llvm::FailureOr<Value> {
        if (barg.getParentBlock()->findAncestorOpInBlock(*ifOp)) {
          return bias;
        } else {
          return llvm::failure();
        }
      })
      .Case([&](OpResult res) -> llvm::FailureOr<Value> {
        auto defOp = res.getDefiningOp();
        auto fillOp = llvm::dyn_cast_if_present<linalg::FillOp>(defOp);
        if (!fillOp) {
          return llvm::failure();
        }

        auto sourceOp = fillOp.getDpsInitOperand(0)->get().getDefiningOp();
        auto sourceEmpty = llvm::dyn_cast_if_present<tensor::EmptyOp>(sourceOp);
        if (!sourceEmpty) {
          return llvm::failure();
        }

        auto emptyOp = builder.create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType());
        emptyOp->setAttr(CVPipeline::kBlockId,
                         builder.getI32IntegerAttr(blockId));
        return emptyOp.getResult();
      })
      .Default([](auto) { return llvm::failure(); });
}

/// Create a zero/default SSA value of the given type for else-block
/// placeholders. Traces referenceValue provenance when possible to preserve
/// downstream analysis.
static llvm::FailureOr<Value>
createPlaceholderValue(int blockId, OpBuilder &builder, Location loc, Type type,
                       Value referenceValue, scf::IfOp ifOp) {
  Value result;
  bool usedTrace = false;
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    // When referenceValue traces to a real tensor source, use it directly
    // so downstream provenance analyses see the same origin in both branches.
    if (referenceValue) {
      auto matmulOpt = hivm::traceDefOp<linalg::MatmulOp>(referenceValue);
      if (matmulOpt.has_value()) {
        auto matmulOp = llvm::dyn_cast<linalg::MatmulOp>(matmulOpt.value());
        if (matmulOp) {
          auto resultRes = createMatmulPlaceHolderValue(
              builder, ifOp, matmulOp, tensorType, loc, blockId);
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
      auto emptyOp = builder.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), tensorType.getElementType());
      emptyOp->setAttr(CVPipeline::kBlockId,
                       builder.getI32IntegerAttr(blockId));
      Value fillVal = builder.create<arith::ConstantOp>(
          loc, tensorType.getElementType(),
          builder.getZeroAttr(tensorType.getElementType()));
      auto fillOp = builder.create<linalg::FillOp>(
          loc, ValueRange{fillVal}, ValueRange{emptyOp.getResult()});
      fillOp->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
      result = fillOp.getResult(0);
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
    // Create a base memref placeholder. For cbuf/ub, use alloca; for plain
    // memref, use alloc to avoid misleading downstream traceback.
    Value baseMemref;
    bool usedReference = false;
    // When referenceValue traces to a function argument (GM), use that
    // argument so both branches of the split-if chain share the same memory
    // space.
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
        // For plain memrefs without address_space, use alloc for static shapes
        // and alloca for dynamic shapes. block_id = -1 marks it as placeholder.
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
    if (!op->hasAttr(CVPipeline::kBlockId) && !isa<scf::YieldOp>(op)) {
      op->setAttr(CVPipeline::kBlockId,
                  builder.getI32IntegerAttr(group.blockId));
    }
  }
}

/// Ensure a value can be safely yielded from the else block.
/// Clones alias ops and allocs locally; constants pass through unchanged.
static Value ensureLocalValue(int blockId, Value val, Block &elseBlock,
                              OpBuilder &builder) {
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

static void buildThenYieldForGroup(
    const GroupOutputInfo &output,
    const llvm::SmallDenseMap<Value, Value> &crossValueReplacement,
    Block &thenBlock, Location loc, OpBuilder &builder) {
  builder.setInsertionPointToEnd(&thenBlock);
  SmallVector<Value> yieldVals;
  for (auto val : output.outputValues) {
    //   thenYield(v) = map[v]  if v in dom(map)
    //                = v      otherwise
    // map := crossValueReplacement. The first case is a Relay slot: v is
    // produced by an earlier group, whose defining ops now live inside that
    // group's if, so only that if's result is visible here.
    auto it = crossValueReplacement.find(val);
    yieldVals.push_back(it != crossValueReplacement.end() ? it->second : val);
  }
  builder.create<scf::YieldOp>(loc, yieldVals);
}

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

static llvm::LogicalResult
buildElseYieldForGroup(const int blockId, const GroupOutputInfo &output,
                       Block &elseBlock, Location loc, OpBuilder &builder) {
  auto ifOp = llvm::dyn_cast<scf::IfOp>(elseBlock.getParentOp());
  if (!ifOp) {
    return llvm::failure();
  }
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
    // Case B: yield the slot's original else value when it dominates this else
    // block. A slot without one (a cross-group value) gets a placeholder below,
    // so the yield keeps the planned arity.
    if (idx < output.origElseValues.size() && output.origElseValues[idx]) {
      Value elseVal = output.origElseValues[idx];
      // Non-dominating else operand: emit nothing for this slot on purpose, see
      // the latent-issue note in AGENTS/DynamicCVPipeline/SplitIfByBlockId.md.
      // The else yield is then short and the verifier rejects the if.
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

/// Collect ops and yield values from the side NOT being split.
/// In Scene 3/4 the last split-if's else block absorbs them.
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
  // ops. These dominate all split-ifs and must be preserved.
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

/// Effect: publish group g's outputs to the rest of the source-if body.
///   - replace users of each output value (outside g's own then block) with the
///     matching split-if result
///   - record that replacement in crossValueReplacement for the later groups
///
/// Precondition: g is not the last group, so every value of output(g) is
/// produced inside the split region by g itself (planYieldCaseB pushes an
/// original slot into output(g) only when prod(s) == g).
///
/// Why the last group must not call this, per taxonomy row it carries:
///   kind(s)    | re-pointing val(s) at splitIf_last#k breaks because
///   -----------+----------------------------------------------------------
///   External   | replaceUsesWithIf covers the whole enclosing function, so
///              | re-pointing an outer value rewrites unrelated blocks and
///              | earlier loop iterations
///   Relay      | the producer group already published val(s), and a middle
///              | group's body (emitted before splitIf_last) may still read it
///              | => use before def
///   NativeLast | safe, but no later group reads it, so nothing to publish
///
/// Provenance, not position: a redirect moves a slot's *result* into an earlier
/// if, while the original else operand stays an operand there (origElseValues,
/// never an entry here), and rearrangeIfOp only moves a split-if after its last
/// dependency. Neither turns a region-internal value into an External one.
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

struct SplittedIfTagGenerator {
  int counter = 0;

  int next() { return counter++; }
};

static void postProcess(scf::IfOp ifOp, scf::IfOp sourceIfOp, int blockId,
                        int splittedIfTag) {
  OpBuilder builder{ifOp};

  ifOp->setAttrs(sourceIfOp->getAttrs());

  auto thenYield = safeGetTerminator(ifOp.thenBlock());
  auto elseYield = safeGetTerminator(ifOp.elseBlock());
  if (thenYield && thenYield.getNumOperands() > 0) {
    std::string joinedAttr =
        llvm::join(llvm::map_range(thenYield.getOperands(),
                                   [](Value value) {
                                     CVPipeline::CoreType ct =
                                         CVPipeline::getValueCoreType(value);
                                     return CVPipeline::coreTypeToString(ct);
                                   }),
                   ", ");
    auto coreTypeAttr = builder.getStringAttr(joinedAttr);
    ifOp->setAttr(CVPipeline::kCoreType, coreTypeAttr);
    thenYield->setAttr(CVPipeline::kCoreType, coreTypeAttr);
    if (elseYield) {
      elseYield->setAttr(CVPipeline::kCoreType, coreTypeAttr);
    }
  } else {
    ifOp->removeAttr(CVPipeline::kCoreType);
  }

  auto setBlockId = [&](Operation *op) {
    op->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
  };
  setBlockId(ifOp);
  // Don't set block_id on scf.yield terminators

  ifOp->setAttr(kSplittedIf, builder.getI32IntegerAttr(splittedIfTag));
}

/// Materialize a split-if chain with per-group signatures.
/// Non-last groups get their own result types; last group carries original
/// results.
static llvm::LogicalResult
materializeCandidate(CandidateIf &c, CVPipeline::ComputeBlockIdManager &bm,
                     SplittedIfTagGenerator &tagGen) {
  OpBuilder builder(c.ifOp);
  auto originalIf = c.ifOp;
  auto loc = originalIf.getLoc();
  Value condition = originalIf.getCondition();
  auto &ya = c.yieldAug;

  int splittedIfTag = tagGen.next();

  LDBG("[Part3] enter materializeCandidate hasYield=" << c.hasYield);

  bool splitThen = c.shouldSplitThen;
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

  // One split-if per group, in dependency order. The last group additionally
  // carries the original slots it either produces itself or cannot delegate.
  SmallVector<Value> slotResults(ya.numOriginalSlots);
  for (unsigned gi = 0; gi < nGroups; ++gi) {
    auto &output = ya.groupOutputs[gi];
    bool isLastGroup = gi == nGroups - 1;
    // Absorbing the un-split side needs an else region even for a void group.
    // hasElse <= !output.isVoid() || (isLastGroup && otherCtx.hasOps).
    // A redirect never invalidates isVoid(): buildThenYieldForGroup yields
    // exactly output.outputValues, and a redirect only rewrites uses of those
    // values, so a non-void output list always means a non-empty then yield.
    bool hasElse = !output.isVoid() || (isLastGroup && otherCtx.hasOps);
    scf::IfOp splittedIf;

    LDBG("[Part3] group[" << gi << "] block_id=" << groups[gi].blockId
                          << " outputs=" << output.outputValues.size()
                          << " last=" << isLastGroup << " hasElse=" << hasElse);

    builder.setInsertionPointAfter(lastCreatedIf);
    splittedIf =
        builder.create<scf::IfOp>(loc, output.outputTypes, condition, hasElse);
    splittedIf->setAttr(CVPipeline::kBlockId,
                        builder.getI32IntegerAttr(groups[gi].blockId));
    lastCreatedIf = splittedIf.getOperation();
    if (splittedIf.getThenRegion().empty()) {
      splittedIf.getThenRegion().emplaceBlock();
    }
    if (hasElse && splittedIf.getElseRegion().empty()) {
      splittedIf.getElseRegion().emplaceBlock();
    }

    Block &thenBlock = splittedIf.getThenRegion().front();
    if (thenBlock.mightHaveTerminator()) {
      thenBlock.getTerminator()->erase();
    }
    rewireAndMoveOps(groups[gi], crossValueReplacement, thenBlock);
    buildThenYieldForGroup(output, crossValueReplacement, thenBlock, loc,
                           builder);

    if (hasElse) {
      Block &elseBlock = splittedIf.getElseRegion().front();
      if (elseBlock.mightHaveTerminator()) {
        elseBlock.getTerminator()->erase();
      }
      if (isLastGroup && otherCtx.hasOps) {
        // Scene 3/4: the un-split side runs when the condition is false, so its
        // ops move into this (last) else block instead of being dropped.
        for (auto *op : otherCtx.ops) {
          op->moveBefore(&elseBlock, elseBlock.end());
        }
      }
      if (buildElseYieldForGroup(groups[gi].blockId, output, elseBlock, loc,
                                 builder)
              .failed()) {
        return llvm::failure();
      }
    }

    // Skipped for the last group, whose output list also holds the Relay and
    // External slots: see the hazard table on updateCrossValueReplacementGroup.
    if (!isLastGroup) {
      updateCrossValueReplacementGroup(output, splittedIf, thenBlock,
                                       crossValueReplacement);
    }

    // Fill slotResults[s] with the result of carrier(s)'s split-if, for the
    // slots this group carries. Every slot gets exactly one entry, checked
    // after the loop over all groups.
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      if (ya.carrierGroup(slot) != gi) {
        continue;
      }
      auto it = llvm::find(output.outputValues, ya.origYieldValues[slot]);
      if (output.isVoid() || it == output.outputValues.end()) {
        LDBG("[Part3] original slot " << slot << " missing in group " << gi);
        return llvm::failure();
      }
      slotResults[slot] =
          splittedIf.getResult(it - output.outputValues.begin());
    }

    postProcess(splittedIf, originalIf, groups[gi].blockId, splittedIfTag);
  }

  // Replace the source if's results with their carriers, so no slot travels
  // through a downstream if.
  // Fail: a slot without a carrier means the plan and the materialization
  // disagree.
  for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
    if (!slotResults[slot]) {
      LDBG("[Part3] original slot " << slot << " has no carrier");
      return llvm::failure();
    }
    originalIf.getResult(slot).replaceAllUsesWith(slotResults[slot]);
  }

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
  SplittedIfTagGenerator tagGen;
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module)) {
    return;
  }

  FallbackHelper fallback{module};

  LDBG("Before:\n" << module << "\n----------");

  CVPipeline::ComputeBlockIdManager bm(module);
  auto mainRes = walkMainLoop(module, [&](Operation *op) {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (funcOp && llvm::is_contained(kSkippedKernels, funcOp.getSymName())) {
      LDBG("Skip kernel: " << funcOp.getSymName());
      return llvm::success();
    }
    LDBG("Detected main loop: " << *op);
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph{op, aa};
    auto walkRes = op->walk([&](scf::IfOp ifOp) {
      auto candidate = getCandidate(ifOp, memGraph);
      if (!candidate.needsSplit()) {
        return WalkResult::advance();
      }
      preprocessScalarDependencies(candidate);
      LLVM_DEBUG(dumpCandidate(candidate));
      analyzeDependencies(candidate);
      if (materializeCandidate(candidate, bm, tagGen).failed()) {
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
    if (ifOp->hasAttr(CVPipeline::kSplittedIf)) {
      rearrangeIfOp(ifOp, memGraph);
    }
  });

  LDBG("After: \n" << module << "\n----------");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSplitIfByBlockIdPass() {
  return std::make_unique<SplitIfByBlockIdPass>();
}

} // namespace triton
} // namespace mlir
