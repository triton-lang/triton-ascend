/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/OuterEventSchedule.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/OuterEventLoop.h"
#include "ascend/include/TritonToLinalg/MarkTensorKindPass.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <map>
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::CVPipeline;

namespace {

constexpr llvm::StringLiteral kPairedCubeWait = "ssbuffer.paired_cube_wait";

// The count prepass is cloned into the two scopes. Prove equivalence of its
// scalar/read-only expression graph, including region arguments and captures.
// Never equate unrelated function arguments or ignore operation attributes.
struct SameCount {
  IRMapping mapping;
  unsigned budget = 4096;
  bool same(Value a, Value b) {
    if (a == b)
      return true;
    if (a.getType() != b.getType() || budget == 0)
      return false;
    --budget;
    if (mapping.contains(a))
      return mapping.lookup(a) == b;
    auto lhs = dyn_cast<OpResult>(a), rhs = dyn_cast<OpResult>(b);
    if (!lhs || !rhs || lhs.getResultNumber() != rhs.getResultNumber())
      return false;
    Operation *def = lhs.getOwner();
    auto safe = def->walk([](Operation *op) {
      if (isa<scf::ForOp, scf::IfOp, scf::YieldOp, memref::LoadOp>(op) ||
          (op->getNumRegions() == 0 && isMemoryEffectFree(op)))
        return WalkResult::advance();
      return WalkResult::interrupt();
    });
    if (safe.wasInterrupted())
      return false;
    return OperationEquivalence::isEquivalentTo(
        def, rhs.getOwner(),
        [&](Value x, Value y) { return success(same(x, y)); },
        [&](Value x, Value y) { mapping.map(x, y); },
        OperationEquivalence::IgnoreLocations);
  }
};

bool constant(Value value, int64_t expected) {
  APInt integer;
  return matchPattern(value, m_ConstantInt(&integer)) &&
         integer.getSExtValue() == expected;
}

struct BlockSync {
  bool set;
  IntegerAttr flag;
  hivm::TCoreType core;
  hivm::PIPE from, to;

  bool usesScalarPipe() const {
    return from == hivm::PIPE::PIPE_S || to == hivm::PIPE::PIPE_S;
  }

  bool isSweep() const {
    return flag && from == hivm::PIPE::PIPE_S && to == hivm::PIPE::PIPE_S;
  }

  bool isNativeTransfer(Operation *op) const {
    return flag && !op->getNumOperands() && !usesScalarPipe() &&
           op->hasAttr(kAnalyzeFlagId);
  }
};

std::optional<BlockSync> readBlockSync(Operation *op) {
  auto read = [](auto sync, bool set) {
    return BlockSync{set, sync.getStaticFlagId().value_or(IntegerAttr{}),
                     sync.getTcoreType().getTcoretype(),
                     sync.getTpipe().getPipe(), sync.getPipe().getPipe()};
  };
  if (auto set = dyn_cast_or_null<hivm::SyncBlockSetOp>(op))
    return read(set, true);
  if (auto wait = dyn_cast_or_null<hivm::SyncBlockWaitOp>(op))
    return read(wait, false);
  return std::nullopt;
}

std::optional<BlockSync> readSweep(Operation *op) {
  auto sync = readBlockSync(op);
  return sync && sync->isSweep() ? sync : std::nullopt;
}

struct Side {
  scf::ForOp loop;
  hivm::TCoreType core;
  SmallVector<scf::IfOp> guards;
  bool first;
  int64_t flag;
};

FailureOr<Side> collectSide(Operation *op,
                            const ControlFlowConditionInfo &info) {
  auto loop = dyn_cast<scf::ForOp>(op);
  auto count = op->getAttrOfType<IntegerAttr>(kOuterEventControlCount);
  if (!loop || !count || count.getInt() <= 0 ||
      !op->hasAttr(kExplicitMainLoop) || !op->hasAttr(kMainLoop) ||
      !loop.getInductionVar().getType().isInteger(64) ||
      !constant(loop.getLowerBound(), 0) || !constant(loop.getStep(), 1))
    return failure();
  auto scope = op->getParentOfType<scope::ScopeOp>();
  auto core = scope
                  ? scope->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type")
                  : hivm::TCoreTypeAttr();
  if (!core || (core.getTcoretype() != hivm::TCoreType::CUBE &&
                core.getTcoretype() != hivm::TCoreType::VECTOR))
    return failure();
  Side side{loop, core.getTcoretype(), {}, false, 0};
  auto start = readSweep(&loop.getBody()->front());
  auto end = readSweep(loop.getBody()->getTerminator()->getPrevNode());
  if (!start || !end || start->set || !end->set ||
      start->flag.getInt() != end->flag.getInt() || start->core != side.core ||
      end->core != side.core)
    return failure();
  side.flag = start->flag.getInt();
  unsigned insideSyncs = 0;
  auto valid = loop.walk([&](Operation *nested) {
    if (auto sync = readBlockSync(nested); sync && sync->usesScalarPipe()) {
      if (!sync->isSweep() || nested->getBlock() != loop.getBody())
        return WalkResult::interrupt();
      ++insideSyncs;
    }
    if (nested->hasAttr(kMemCrossDeps))
      return WalkResult::interrupt();
    if (!nested->hasAttr(kIf))
      return WalkResult::advance();
    auto guard = dyn_cast<scf::IfOp>(nested);
    if (!guard || guard->getBlock() != loop.getBody())
      return WalkResult::interrupt();
    if (!matchPattern(guard.getCondition(), m_One()))
      return WalkResult::interrupt();
    side.guards.push_back(guard);
    return WalkResult::advance();
  });
  if (valid.wasInterrupted() || insideSyncs != 2 || side.guards.empty())
    return failure();
  auto counters = info.blockCounters.find(loop);
  if (counters == info.blockCounters.end() ||
      counters->second.size() != side.guards.size())
    return failure();
  for (int index : counters->second)
    if (index < 0 ||
        static_cast<unsigned>(index) >= loop.getInitArgs().size() ||
        !constant(loop.getInitArgs()[index], 0))
      return failure();

  // Only generated condition arithmetic and SSBuffer reads may run outside
  // a ready guard. In particular, an unguarded wait/store or another control
  // region is not represented by the stage graph.
  for (Operation &root : *loop.getBody()) {
    if (root.hasAttr(kIf) || readSweep(&root) || isa<scf::YieldOp>(root))
      continue;
    if (root.getNumRegions() != 0 ||
        (!isa<memref::LoadOp, annotation::MarkOp>(root) &&
         !isMemoryEffectFree(&root)))
      return failure();
  }
  if (auto conditions = info.innerDepConds.find(loop);
      conditions != info.innerDepConds.end())
    for (int index : conditions->second)
      if (index < 0 ||
          static_cast<unsigned>(index) >= loop.getInitArgs().size() ||
          !constant(loop.getInitArgs()[index], 0))
        return failure();

  unsigned initialSets = 0, finalWaits = 0;
  for (Operation &sibling : *loop->getBlock()) {
    auto sync = readBlockSync(&sibling);
    if (!sync || !sync->usesScalarPipe())
      continue;
    if (!sync->isSweep() || sync->flag.getInt() != side.flag ||
        sync->core != side.core)
      return failure();
    if (sync->set && sibling.isBeforeInBlock(loop))
      ++initialSets;
    else if (!sync->set && loop->isBeforeInBlock(&sibling))
      ++finalWaits;
    else
      return failure();
  }
  // The side without the initial SET waits for the other side's seed and
  // executes first. Its final WAIT consumes the last release from the peer.
  if (initialSets == 0 && finalWaits == 1)
    side.first = true;
  else if (initialSets != 1 || finalWaits != 0)
    return failure();
  return side;
}

struct Edge {
  unsigned from, to;
  int lag;
};

FailureOr<std::array<Side, 2>>
collectOrderedSides(ArrayRef<Operation *> loops,
                    const ControlFlowConditionInfo &info) {
  if (loops.size() != 2)
    return failure();
  auto a = collectSide(loops[0], info);
  auto b = collectSide(loops[1], info);
  if (failed(a) || failed(b) || a->core == b->core || a->first == b->first ||
      a->flag != b->flag ||
      loops[0]->getAttr(kMainLoop) != loops[1]->getAttr(kMainLoop) ||
      loops[0]->getAttr(kVectorFirst) != loops[1]->getAttr(kVectorFirst) ||
      !SameCount().same(a->loop.getUpperBound(), b->loop.getUpperBound()))
    return failure();
  if ((a->first ? a->core : b->core) != (loops[0]->hasAttr(kVectorFirst)
                                             ? hivm::TCoreType::VECTOR
                                             : hivm::TCoreType::CUBE))
    return failure();
  if (!a->first)
    std::swap(*a, *b);
  return std::array<Side, 2>{std::move(*a), std::move(*b)};
}

struct NativeCreditGroup {
  scf::IfOp producer, consumer;
  size_t slots;
};

// The transfer pass emits one data signal and one reverse credit per slot,
// with one seed and one drain for the reverse credit. Only group the scalar
// sweep grants when every cross-core edge also has that generated protocol.
bool hasNativeCreditProtocol(
    ArrayRef<Side> sides, const std::map<int64_t, NativeCreditGroup> &groups) {
  if (groups.empty())
    return false;
  enum Role { Publish, Acquire, Release, Reuse, Seed, Drain, NumRoles };
  std::map<std::pair<int64_t, int64_t>,
           std::array<std::optional<BlockSync>, NumRoles>>
      slots;
  for (Side side : sides) {
    auto record = [&](Operation *op, bool inside) -> WalkResult {
      if (llvm::any_of(op->getOperandTypes(), [](Type type) {
            auto memref = dyn_cast<BaseMemRefType>(type);
            auto space = memref ? dyn_cast_or_null<hivm::AddressSpaceAttr>(
                                      memref.getMemorySpace())
                                : hivm::AddressSpaceAttr();
            return space &&
                   space.getAddressSpace() == hivm::AddressSpace::SSBUF;
          }))
        return WalkResult::interrupt();
      auto sync = readBlockSync(op);
      if (!sync || sync->isSweep())
        return WalkResult::advance();
      auto tid = op->getAttrOfType<IntegerAttr>(kTransferId);
      auto group = tid ? groups.find(tid.getInt()) : groups.end();
      if (group == groups.end())
        return inside ? WalkResult::interrupt() : WalkResult::advance();
      if (!sync->isNativeTransfer(op) || sync->core != side.core ||
          sync->flag.getInt() == side.flag)
        return WalkResult::interrupt();
      scf::IfOp producer = group->second.producer;
      scf::IfOp consumer = group->second.consumer;
      Role role;
      if (inside) {
        Operation *guard = side.loop.getBody()->findAncestorOpInBlock(*op);
        if (guard == producer.getOperation())
          role = sync->set ? Publish : Reuse;
        else if (guard == consumer.getOperation())
          role = sync->set ? Release : Acquire;
        else
          return WalkResult::interrupt();
      } else if (sync->set && consumer->getParentOp() == side.loop &&
                 op->isBeforeInBlock(side.loop)) {
        role = Seed;
      } else if (!sync->set && producer->getParentOp() == side.loop &&
                 side.loop->isBeforeInBlock(op)) {
        role = Drain;
      } else {
        return WalkResult::interrupt();
      }
      auto &slot = slots[{tid.getInt(), sync->flag.getInt()}];
      if (slot[role])
        return WalkResult::interrupt();
      slot[role] = sync;
      return WalkResult::advance();
    };
    if (side.loop->walk([&](Operation *op) { return record(op, true); })
            .wasInterrupted())
      return false;
    for (Operation &op : *side.loop->getBlock())
      if (&op != side.loop.getOperation() &&
          record(&op, false).wasInterrupted())
        return false;
  }

  DenseMap<int64_t, size_t> slotCounts;
  DenseSet<int64_t> usedFlags;
  for (const auto &entry : slots) {
    const auto &key = entry.first;
    const auto &slot = entry.second;
    if (!usedFlags.insert(key.second).second ||
        !llvm::all_of(slot, [](const auto &sync) { return sync.has_value(); }))
      return false;
    auto samePipes = [&](Role a, Role b) {
      return slot[a]->from == slot[b]->from && slot[a]->to == slot[b]->to;
    };
    if (!samePipes(Publish, Acquire) || !samePipes(Release, Reuse) ||
        !samePipes(Release, Seed) || !samePipes(Release, Drain) ||
        slot[Publish]->from != slot[Release]->to ||
        slot[Publish]->to != slot[Release]->from)
      return false;
    ++slotCounts[key.first];
  }
  return llvm::all_of(groups, [&](const auto &entry) {
    return slotCounts.lookup(entry.first) == entry.second.slots;
  });
}

// A firing at ordinal k occurs at II*k+offset. Earlier actors in the
// PIPE_S order publish in this sweep, later actors in the next sweep.
FailureOr<SmallVector<int>> solveOffsets(unsigned count, ArrayRef<Edge> edges) {
  SmallVector<int> offset(count, 0);
  for (unsigned round = 0; round < count; ++round) {
    bool changed = false;
    for (const Edge &edge : edges) {
      int delay = edge.from < edge.to ? 0 : 1;
      int required = offset[edge.from] + delay - edge.lag;
      if (offset[edge.to] < required) {
        offset[edge.to] = required;
        changed = true;
      }
    }
    if (!changed)
      return offset;
  }
  return failure();
}

LogicalResult addCarryEdges(ArrayRef<Side> sides,
                            const ControlFlowConditionInfo &info,
                            const DenseMap<Operation *, unsigned> &number,
                            SmallVectorImpl<Edge> &edges) {
  for (const Side &side : sides) {
    auto loop = side.loop;
    auto relations = info.tensorIterArgDepsMap.find(loop);
    if (relations == info.tensorIterArgDepsMap.end())
      continue;
    auto indices = info.tensorIterArgIndicesMap.find(loop);
    if (!relations->second.empty() &&
        indices == info.tensorIterArgIndicesMap.end())
      return failure();
    for (const auto &relation : relations->second) {
      if (!relation.producer || !number.contains(relation.producer) ||
          relation.producer->getParentOp() != loop ||
          relation.consumers.empty())
        return failure();
      auto credits = indices->second.find(relation.iterArg);
      if (credits == indices->second.end() ||
          credits->second.size() != relation.consumers.size())
        return failure();
      for (int index : credits->second)
        if (index < 0 ||
            static_cast<unsigned>(index) >= loop.getInitArgs().size() ||
            !constant(loop.getInitArgs()[index], 1))
          return failure();
      unsigned producer = number.lookup(relation.producer);
      for (scf::IfOp reader : relation.consumers) {
        if (!number.contains(reader) || reader == relation.producer ||
            reader->getParentOp() != loop)
          return failure();
        unsigned consumer = number.lookup(reader);
        // Each carry reader owns an initially-full 1-slot credit. The writer
        // waits for every reader of the old value, not just the first reader.
        edges.push_back({producer, consumer, 1});
        edges.push_back({consumer, producer, 0});
      }
    }
  }
  return success();
}

} // namespace

FailureOr<OuterEventStaticSchedule> mlir::triton::inferOuterEventStaticSchedule(
    ArrayRef<Operation *> loops, const ControlFlowConditionInfo &info) {
  auto sides = collectOrderedSides(loops, info);
  if (failed(sides))
    return failure();
  SmallVector<scf::IfOp> guards;
  DenseMap<Operation *, unsigned> number;
  for (Side &side : *sides)
    for (scf::IfOp guard : side.guards) {
      number[guard] = guards.size();
      guards.push_back(guard);
    }
  if (guards.size() > 256)
    return failure();

  auto owner = [&](Operation *op) -> std::optional<unsigned> {
    for (Side &side : *sides)
      if (Operation *root = side.loop.getBody()->findAncestorOpInBlock(*op))
        if (auto entry = number.find(root); entry != number.end() && root != op)
          return entry->second;
    return std::nullopt;
  };
  SmallVector<Edge> edges;
  bool hasCrossCredit = false, hasTransferIds = true;
  std::map<int64_t, NativeCreditGroup> nativeGroups;
  auto addGroup = [&](Operation *consumer, ArrayRef<Operation *> producers,
                      bool cross) -> LogicalResult {
    auto dst = owner(consumer);
    if (producers.empty() || producers.size() > 256)
      return failure();
    auto src = owner(producers.front());
    if (!llvm::all_of(producers,
                      [&](Operation *op) { return owner(op) == src; }))
      return failure();
    if (!src && !dst) {
      for (Operation *loop : loops) {
        if (loop->isAncestor(consumer) ||
            llvm::any_of(producers, [&](Operation *producer) {
              return loop->isAncestor(producer);
            }))
          return failure();
      }
      return success();
    }
    if (!src || !dst || src == dst ||
        (guards[*src]->getParentOp() == guards[*dst]->getParentOp()) == cross)
      return failure();
    if (cross) {
      hasCrossCredit = true;
      auto tid = consumer->getAttrOfType<IntegerAttr>(kTransferId);
      hasTransferIds &=
          tid &&
          llvm::all_of(producers,
                       [&](Operation *op) {
                         return op->getAttrOfType<IntegerAttr>(kTransferId) ==
                                tid;
                       }) &&
          nativeGroups
              .try_emplace(tid.getInt(),
                           NativeCreditGroup{guards[*src], guards[*dst],
                                             producers.size()})
              .second;
    }
    // This is the capacity used by UpdateConditionInfo: one slot per
    // alternative producer operation, and one update per firing.
    edges.push_back({*src, *dst, 0});
    edges.push_back({*dst, *src, static_cast<int>(producers.size())});
    return success();
  };
  for (const auto &[consumer, groups] : info.crossCoreDependentMap)
    for (const auto &producers : groups)
      if (failed(addGroup(consumer, producers, /*cross=*/true)))
        return failure();
  for (Operation *loop : loops)
    if (auto deps = info.intraCoreDependentMap.find(loop);
        deps != info.intraCoreDependentMap.end())
      for (const auto &[consumer, producers] : deps->second)
        if (failed(addGroup(consumer, producers, /*cross=*/false)))
          return failure();
  if (!hasCrossCredit || failed(addCarryEdges(*sides, info, number, edges)))
    return failure();

  // FlowOpt reads the old source counter. Let the source execute at least
  // opt firings in earlier sweeps, or finish first when T is smaller.
  if (info.crossCoreBufferCount > CROSS_CORE_BUFFER_COUNT_THRESHOLD &&
      info.intraCoreBufferCount > INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    int opt =
        std::min(info.intraCoreBufferCount - 1, info.crossCoreBufferCount);
    for (const auto &[target, source] : info.flowOptIfOpPairs) {
      auto from = number.find(source), to = number.find(target);
      if (from == number.end() || to == number.end() ||
          from->second >= to->second ||
          source->getParentOp() != target->getParentOp())
        return failure();
      edges.push_back({from->second, to->second, -opt});
    }
  }
  auto offsets = solveOffsets(guards.size(), edges);
  if (failed(offsets))
    return failure();
  OuterEventStaticSchedule result;
  result.drain = *llvm::max_element(*offsets) + 1;
  // A two-sweep grant orders C[2n], C[2n+1], V[2n], V[2n+1] when C
  // is first. A dependency back to the first side must be ready even
  // across the worst alignment of the two-sweep boundary. All other
  // constraints stay identical. Use the same stage offsets and only group
  // grants when every reverse edge can afford the extra sweep.
  unsigned firstSideActors = (*sides)[0].guards.size();
  result.groupSweepGrants =
      hasTransferIds &&
      llvm::all_of(edges,
                   [&](const Edge &edge) {
                     return edge.from < firstSideActors ||
                            edge.to >= firstSideActors ||
                            (*offsets)[edge.to] + edge.lag >=
                                (*offsets)[edge.from] + 2;
                   }) &&
      hasNativeCreditProtocol(*sides, nativeGroups);

  for (auto [guard, offset] : llvm::zip(guards, *offsets))
    result.offsets[guard] = offset;
  return result;
}

LogicalResult mlir::triton::groupOuterEventSweepGrants(scf::ForOp loop) {
  Operation *wait = &loop.getBody()->front();
  Operation *set = loop.getBody()->getTerminator()->getPrevNode();
  auto waitSync = readSweep(wait), setSync = readSweep(set);
  if (!waitSync || !setSync || waitSync->set || !setSync->set ||
      waitSync->flag.getInt() != setSync->flag.getInt() ||
      waitSync->core != setSync->core || wait->getNumOperands() ||
      set->getNumOperands())
    return failure();

  Location loc = loop.getLoc();
  OpBuilder builder(wait);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 64);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  Value iv = loop.getInductionVar();
  Value lowBit = builder.create<arith::AndIOp>(loc, iv, one);
  Value firstInGrant = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lowBit, zero);
  auto waitIf =
      builder.create<scf::IfOp>(loc, firstInGrant, /*withElse=*/false);
  wait->moveBefore(waitIf.thenYield());
  if (waitSync->core == hivm::TCoreType::CUBE)
    waitIf->setAttr(kPairedCubeWait, builder.getUnitAttr());

  builder.setInsertionPoint(set);
  Value secondInGrant = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, lowBit, zero);
  Value next = builder.create<arith::AddIOp>(loc, iv, one);
  Value last = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             next, loop.getUpperBound());
  Value release = builder.create<arith::OrIOp>(loc, secondInGrant, last);
  auto setIf = builder.create<scf::IfOp>(loc, release, /*withElse=*/false);
  set->moveBefore(setIf.thenYield());
  return success();
}

namespace {

bool canIssueProducerBeforeWait(scf::IfOp stage) {
  bool publishes = false, waitsForReuse = false;
  auto result = stage.walk([&](Operation *op) {
    if (auto sync = readBlockSync(op)) {
      if (!sync->isNativeTransfer(op) || sync->core != hivm::TCoreType::CUBE)
        return WalkResult::interrupt();
      if (sync->set && sync->from == hivm::PIPE::PIPE_FIX &&
          sync->to == hivm::PIPE::PIPE_V) {
        publishes = true;
        return WalkResult::advance();
      }
      if (!sync->set && sync->from == hivm::PIPE::PIPE_V &&
          sync->to == hivm::PIPE::PIPE_FIX) {
        waitsForReuse = true;
        return WalkResult::advance();
      }
      return WalkResult::interrupt();
    }
    if (isMemoryEffectFree(op) || isa<scf::IfOp, scf::YieldOp>(op))
      return WalkResult::advance();
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects)
      return WalkResult::interrupt();
    SmallVector<MemoryEffects::EffectInstance> accesses;
    effects.getEffects(accesses);
    for (const auto &access : accesses) {
      if (isa<MemoryEffects::Allocate>(access.getEffect()))
        continue;
      Value storage = getOuterEventBufferRoot(access.getValue());
      if (!storage)
        return WalkResult::interrupt();

      auto alloc = storage.getDefiningOp<memref::AllocOp>();
      if (alloc && stage->isProperAncestor(alloc))
        continue;
      if (isa<MemoryEffects::Read>(access.getEffect())) {
        if (isa<TensorType>(storage.getType()))
          continue;
        if (auto arg = dyn_cast<BlockArgument>(storage)) {
          if (auto func =
                  dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
            auto kind = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                           "tt.tensor_kind");
            if (kind && kind.getInt() == TensorKind::INPUT)
              continue;
          }
        }
      }
      if (alloc && isa<MemoryEffects::Write>(access.getEffect()) &&
          isa<hivm::FixpipeOp>(op))
        continue;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted() && publishes && waitsForReuse;
}

bool collectProducerControl(Value value, scf::ForOp loop, Operation *wait,
                            llvm::SetVector<Operation *> &control) {
  Operation *op = value.getDefiningOp();
  if (!op || !loop->isProperAncestor(op))
    return true;
  if (op->getBlock() != loop.getBody())
    return false;
  if (op->isBeforeInBlock(wait) || control.contains(op))
    return true;
  if (op->getNumRegions() != 0 || !isMemoryEffectFree(op))
    return false;
  for (Value operand : op->getOperands())
    if (!collectProducerControl(operand, loop, wait, control))
      return false;
  control.insert(op);
  return true;
}

} // namespace

void mlir::triton::issueOuterProducerBeforeWait(ModuleOp module) {
  SmallVector<scf::IfOp> waits;
  module.walk([&](scf::IfOp branch) {
    if (branch->removeAttr(kPairedCubeWait))
      waits.push_back(branch);
  });
  for (scf::IfOp wait : waits) {
    auto loop = dyn_cast<scf::ForOp>(wait->getParentOp());
    if (!loop)
      continue;
    auto stages = loop.getBody()->getOps<scf::IfOp>();
    auto it = llvm::find_if(
        stages, [](scf::IfOp stage) { return stage->hasAttr(kIf); });
    if (it == stages.end())
      continue;
    scf::IfOp producer = *it;
    if (!wait->isBeforeInBlock(producer) ||
        !canIssueProducerBeforeWait(producer))
      continue;

    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(producer->getRegions(), captured);
    captured.insert(producer.getCondition());
    llvm::SetVector<Operation *> control;
    if (!llvm::all_of(captured, [&](Value value) {
          return collectProducerControl(value, loop, wait, control);
        }))
      continue;

    // The producer keeps its original tensor carry. Its FIXP credit gates
    // publication to Vector, while the Cube sweep grant still gates PV.
    for (Operation *op : control)
      op->moveBefore(wait);
    producer->moveBefore(wait);
  }
}
