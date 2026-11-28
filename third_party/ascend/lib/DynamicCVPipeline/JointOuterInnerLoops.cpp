/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/OuterEventLoop.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <array>

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

bool canCrossBranch(Type type) {
  if (type.isIntOrIndexOrFloat())
    return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape();
}

bool isScalarOrAddress(Type type) {
  return type.isIntOrIndexOrFloat() || isa<MemRefType>(type);
}

struct BodyPlan {
  scf::ForOp loop;
  SmallVector<linalg::MatmulOp, 2> matmuls;
  SmallVector<SmallVector<Operation *>, 3> segments;
  // Results of this segment that have uses after its following matmul.
  // Matmul results themselves are shared and need no branch-specific slot.
  SmallVector<SmallVector<Value>, 2> live;

  static FailureOr<BodyPlan> analyze(scf::ForOp loop) {
    BodyPlan plan;
    plan.loop = loop;
    plan.segments.emplace_back();
    DenseMap<Operation *, unsigned> positions;
    unsigned position = 0;
    for (Operation &op : loop.getBody()->without_terminator()) {
      positions[&op] = position++;
      // An earlier full-count read cannot observe a partial iteration's write.
      if (!hasOnlyLocalMemoryEffects(&op, loop, /*allowFree=*/false))
        return failure();
      bool unsupported = false;
      op.walk([&](Operation *nested) {
        if (isa<LoopLikeOpInterface>(nested) ||
            (nested != &op && isa<linalg::MatmulOp>(nested)))
          unsupported = true;
      });
      if (unsupported)
        return failure();
      if (auto matmul = dyn_cast<linalg::MatmulOp>(&op)) {
        if (!matmul.hasPureTensorSemantics())
          return failure();
        plan.matmuls.push_back(matmul);
        plan.segments.emplace_back();
      } else {
        plan.segments.back().push_back(&op);
      }
    }
    if (plan.matmuls.empty())
      return failure();
    positions[loop.getBody()->getTerminator()] = position;
    plan.live.resize(plan.matmuls.size());
    for (unsigned cut = 0; cut < plan.matmuls.size(); ++cut) {
      unsigned cutPosition = positions.lookup(plan.matmuls[cut]);
      for (Operation *op : plan.segments[cut]) {
        for (Value result : op->getResults()) {
          bool neededLater =
              llvm::any_of(result.getUses(), [&](OpOperand &use) {
                Operation *root =
                    loop.getBody()->findAncestorOpInBlock(*use.getOwner());
                return root && positions.lookup(root) > cutPosition;
              });
          if (!neededLater)
            continue;
          // Memref views, casts and allocation aliases may not escape their
          // branch. A to_tensor result may escape with its tensor semantics.
          if (!canCrossBranch(result.getType()))
            return failure();
          plan.live[cut].push_back(result);
        }
      }
      if (!llvm::all_of(plan.matmuls[cut]->getOperandTypes(), canCrossBranch) ||
          !llvm::all_of(plan.matmuls[cut]->getResultTypes(), canCrossBranch))
        return failure();
    }
    return plan;
  }
};

struct SharedInit {
  Value common;
  std::array<linalg::FillOp, 2> fills;
  std::array<tensor::EmptyOp, 2> empties;
  arith::ConstantOp constant;
};

struct LiveSlot {
  std::array<Value, 2> values;
  Type getType() const {
    return values[0] ? values[0].getType() : values[1].getType();
  }
};

struct CommonLoadSuffix {
  unsigned size = 0;
  SmallVector<std::array<Value, 2>> inputs;
  std::array<DenseSet<Value>, 2> results;
  DenseMap<Value, Value> equivalent;
  // When possible, select scalar metadata separately from the remaining
  // tensor/memory prefix. Cloning a scalar dependency must not clone a mask.
  bool separateScalarPrefix = false;
  std::array<unsigned, 2> scalarPrefixSize{0, 0};
};

// Split only a contiguous leading slice. In particular, do not hoist metadata
// reads across a preceding mask copy or tensor operation just to enable this.
void analyzeScalarPrefix(CommonLoadSuffix &suffix,
                         std::array<BodyPlan, 2> &bodies) {
  if (!suffix.size || suffix.inputs.empty())
    return;
  std::array<unsigned, 2> sizes{0, 0};
  for (unsigned branch = 0; branch < 2; ++branch) {
    auto loop = bodies[branch].loop;
    auto prefix = ArrayRef<Operation *>(bodies[branch].segments[0])
                      .drop_back(suffix.size);
    DenseSet<Value> captures;
    for (auto input : suffix.inputs) {
      Value value = input[branch];
      captures.insert(value);
      Operation *def = value.getDefiningOp();
      if (!def || !loop->isProperAncestor(def))
        continue;
      auto it = llvm::find(prefix, def);
      // Do not extract a value from a nested conditional or move its effects.
      if (it == prefix.end())
        return;
      sizes[branch] =
          std::max(sizes[branch], unsigned(it - prefix.begin()) + 1);
    }
    auto scalarOps = prefix.take_front(sizes[branch]);
    DenseSet<Operation *> roots(scalarOps.begin(), scalarOps.end());
    for (Operation *op : scalarOps) {
      if (op->getNumRegions() ||
          (!isMemoryEffectFree(op) && !isa<memref::LoadOp>(op)) ||
          !llvm::all_of(op->getOperandTypes(), isScalarOrAddress) ||
          !llvm::all_of(op->getResultTypes(), isScalarOrAddress))
        return;
      for (Value result : op->getResults()) {
        if (captures.contains(result))
          continue;
        // No hidden scalar or memref result may cross the split. The exact
        // scalar captures already have a paired, non-placeholder result slot.
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !roots.contains(
                  loop.getBody()->findAncestorOpInBlock(*user));
            }))
          return;
      }
    }
  }
  suffix.scalarPrefixSize = sizes;
  suffix.separateScalarPrefix = true;
}

// A load suffix remains in the original position after the selected prefix.
// Every write belongs to scratch allocated in the suffix, and none of that
// scratch (including aliases) escapes. Only tensor results may escape.
bool isClosedLoadSuffix(ArrayRef<Operation *> ops, Block *body) {
  DenseSet<Operation *> roots(ops.begin(), ops.end());
  bool hasAlloc = false;
  bool hasCopy = false;
  auto inside = [&](Operation *op) {
    return op && roots.contains(body->findAncestorOpInBlock(*op));
  };
  for (Operation *op : ops) {
    auto effects = getEffectsRecursively(op);
    if (!effects || llvm::any_of(*effects, [&](const auto &effect) {
          if (isa<MemoryEffects::Read>(effect.getEffect()))
            return false;
          return isa<MemoryEffects::Free>(effect.getEffect()) ||
                 !inside(getOuterEventScratchAllocation(effect.getValue()));
        }))
      return false;
    bool escapes = false;
    op->walk([&](Operation *nested) {
      hasAlloc |= isa<memref::AllocOp, memref::AllocaOp>(nested);
      hasCopy |= isa<memref::CopyOp>(nested);
      for (Value result : nested->getResults()) {
        if (auto tensor = dyn_cast<RankedTensorType>(result.getType()))
          escapes |= !tensor.hasStaticShape();
        if (isa<MemRefType>(result.getType()))
          escapes |= llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !inside(user);
          });
      }
    });
    if (escapes)
      return false;
  }
  return hasAlloc && hasCopy;
}

CommonLoadSuffix findCommonLoadSuffix(std::array<BodyPlan, 2> &bodies) {
  const auto &partial = bodies[0].segments[0];
  const auto &full = bodies[1].segments[0];
  unsigned largest = std::min(partial.size(), full.size());
  for (unsigned size = largest; size > 0; --size) {
    std::array<ArrayRef<Operation *>, 2> ops{
        ArrayRef<Operation *>(partial).take_back(size),
        ArrayRef<Operation *>(full).take_back(size)};
    CommonLoadSuffix candidate;
    candidate.size = size;
    std::array<DenseSet<Operation *>, 2> roots;
    for (unsigned branch = 0; branch < 2; ++branch) {
      roots[branch].insert(ops[branch].begin(), ops[branch].end());
      for (Operation *op : ops[branch])
        candidate.results[branch].insert(op->getResults().begin(),
                                         op->getResults().end());
    }
    auto isInternal = [&](Value value, unsigned branch) {
      Operation *def = value.getDefiningOp();
      if (!def)
        return false;
      return roots[branch].contains(
          bodies[branch].loop.getBody()->findAncestorOpInBlock(*def));
    };
    DenseMap<Value, Value> reverse;
    auto checkEquivalent = [&](Value left, Value right) {
      if (left == right)
        return success();
      auto known = candidate.equivalent.find(left);
      if (known != candidate.equivalent.end())
        return success(known->second == right);
      if (isInternal(left, 0) || isInternal(right, 1) ||
          !left.getType().isIntOrIndexOrFloat() ||
          left.getType() != right.getType() || reverse.contains(right))
        return failure();
      // Divergent scalar metadata is selected before the common load region;
      // divergent memory or tensor captures cannot be selected this way.
      candidate.equivalent[left] = right;
      reverse[right] = left;
      candidate.inputs.push_back({left, right});
      return success();
    };
    auto markEquivalent = [&](Value left, Value right) {
      candidate.equivalent[left] = right;
      reverse[right] = left;
    };
    bool matched = true;
    for (unsigned index = 0; index < size; ++index) {
      if (!llvm::equal(ops[0][index]->getOperandTypes(),
                       ops[1][index]->getOperandTypes()) ||
          !llvm::equal(ops[0][index]->getResultTypes(),
                       ops[1][index]->getResultTypes()) ||
          !OperationEquivalence::isEquivalentTo(
              ops[0][index], ops[1][index], checkEquivalent, markEquivalent,
              OperationEquivalence::IgnoreLocations)) {
        matched = false;
        break;
      }
    }
    if (!matched || !isClosedLoadSuffix(ops[0], bodies[0].loop.getBody()) ||
        !isClosedLoadSuffix(ops[1], bodies[1].loop.getBody()))
      continue;
    // An operand defined in the common suffix must name the same common
    // result on both paths. It cannot be selected in the preceding guard.
    for (unsigned input = 0; input < 2; ++input) {
      Value left = bodies[0].matmuls[0]->getOperand(input);
      Value right = bodies[1].matmuls[0]->getOperand(input);
      if ((candidate.results[0].contains(left) ||
           candidate.results[1].contains(right)) &&
          candidate.equivalent.lookup(left) != right)
        matched = false;
    }
    if (matched) {
      analyzeScalarPrefix(candidate, bodies);
      return candidate;
    }
  }
  return CommonLoadSuffix();
}

bool equivalentIgnoringOperands(Operation *lhs, Operation *rhs) {
  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, OperationEquivalence::ignoreValueEquivalence,
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

// Keep matmul C out of a VECTOR scf.if result. An identical external SSA value
// needs no selection. Otherwise only an equivalent, private, pure zero fill is
// safe to materialize once beside the shared CUBE matmul.
FailureOr<SharedInit> analyzeSharedInit(linalg::MatmulOp lhs,
                                        linalg::MatmulOp rhs) {
  if (lhs->getNumOperands() != 3 || rhs->getNumOperands() != 3)
    return failure();
  SharedInit init;
  Value left = lhs->getOperand(2);
  Value right = rhs->getOperand(2);
  if (left == right) {
    init.common = left;
    return init;
  }
  auto leftArg = dyn_cast<BlockArgument>(left);
  auto rightArg = dyn_cast<BlockArgument>(right);
  // The two loops have matching iter_args. Both map this accumulator to
  // the same argument of the joined loop.
  if (leftArg && rightArg && leftArg.getOwner() == lhs->getBlock() &&
      rightArg.getOwner() == rhs->getBlock() && leftArg.getArgNumber() != 0 &&
      leftArg.getArgNumber() == rightArg.getArgNumber()) {
    init.common = left;
    return init;
  }
  std::array<Value, 2> values{left, right};
  std::array<arith::ConstantOp, 2> constants;
  for (unsigned branch = 0; branch < 2; ++branch) {
    auto fill = values[branch].getDefiningOp<linalg::FillOp>();
    if (!fill || !fill.hasPureTensorSemantics() ||
        !values[branch].hasOneUse() || fill->getNumOperands() != 2 ||
        (!matchPattern(fill->getOperand(0), m_Zero()) &&
         !matchPattern(fill->getOperand(0), m_AnyZeroFloat())))
      return failure();
    auto empty = fill->getOperand(1).getDefiningOp<tensor::EmptyOp>();
    auto constant = fill->getOperand(0).getDefiningOp<arith::ConstantOp>();
    if (!empty || !constant || !empty.getResult().hasOneUse() ||
        !empty.getDynamicSizes().empty() || !isMemoryEffectFree(fill))
      return failure();
    init.fills[branch] = fill;
    init.empties[branch] = empty;
    constants[branch] = constant;
  }
  if (!equivalentIgnoringOperands(init.fills[0], init.fills[1]) ||
      !equivalentIgnoringOperands(init.empties[0], init.empties[1]) ||
      !equivalentIgnoringOperands(constants[0], constants[1]) ||
      !OperationEquivalence::isRegionEquivalentTo(
          &init.fills[0].getRegion(), &init.fills[1].getRegion(),
          OperationEquivalence::IgnoreLocations))
    return failure();
  init.constant = constants[0];
  return init;
}

struct JointPlan {
  std::array<BodyPlan, 2> bodies;
  CommonLoadSuffix loadSuffix;
  struct Cut {
    SharedInit init;
    SmallVector<LiveSlot> liveSlots;
  };
  SmallVector<Cut, 2> cuts;

  static FailureOr<JointPlan> analyze(scf::ForOp partial, scf::ForOp full) {
    auto ivType = dyn_cast<IntegerType>(partial.getInductionVar().getType());
    // Two zero-based <=32-bit trip counts fit exactly in signed i64, including
    // unsigned loops. Index and wider bounds need a separate overflow proof.
    if (!ivType || ivType.getWidth() > 32 ||
        !llvm::equal(partial.getResultTypes(), full.getResultTypes()))
      return failure();
    for (scf::ForOp loop : {partial, full}) {
      // Dynamic trip counts retain zero-trip behavior. Cube scope annotations
      // are rebuilt when the scheduler splits the joined body.
      if (llvm::any_of(loop->getAttrs(), [](NamedAttribute attr) {
            StringRef name = attr.getName().strref();
            return name != "unsignedCmp" && name != kMayNotExecNPU &&
                   name != kHIVMMatmulLimitedInCubeAttr;
          }))
        return failure();
    }
    for (auto [result, init] :
         llvm::zip(partial.getResults(), full.getInitArgs()))
      if (result != init || !result.hasOneUse())
        return failure();

    // The intervening metadata slice moves before both phases. It must neither
    // depend on partial results nor observe a write from a partial iteration.
    for (Operation *op = partial->getNextNode(); op != full.getOperation();
         op = op->getNextNode()) {
      if (op->getNumRegions() ||
          (!isMemoryEffectFree(op) && !isa<memref::LoadOp>(op)) ||
          !llvm::all_of(op->getOperandTypes(), isScalarOrAddress) ||
          !llvm::all_of(op->getResultTypes(), isScalarOrAddress))
        return failure();
    }
    auto partialBody = BodyPlan::analyze(partial);
    auto fullBody = BodyPlan::analyze(full);
    if (failed(partialBody) || failed(fullBody) ||
        partialBody->matmuls.size() != fullBody->matmuls.size())
      return failure();
    JointPlan plan;
    plan.bodies = {std::move(*partialBody), std::move(*fullBody)};
    plan.loadSuffix = findCommonLoadSuffix(plan.bodies);
    plan.cuts.resize(plan.bodies[0].matmuls.size());
    for (unsigned cut = 0; cut < plan.cuts.size(); ++cut) {
      auto lhs = plan.bodies[0].matmuls[cut];
      auto rhs = plan.bodies[1].matmuls[cut];
      if (!llvm::equal(lhs->getOperandTypes(), rhs->getOperandTypes()) ||
          !llvm::equal(lhs->getResultTypes(), rhs->getResultTypes()) ||
          !OperationEquivalence::isEquivalentTo(
              lhs, rhs, OperationEquivalence::ignoreValueEquivalence,
              /*markEquivalent=*/nullptr,
              OperationEquivalence::IgnoreLocations) ||
          !OperationEquivalence::isRegionEquivalentTo(
              &lhs.getRegion(), &rhs.getRegion(),
              OperationEquivalence::IgnoreLocations))
        return failure();
      auto init = analyzeSharedInit(lhs, rhs);
      if (failed(init))
        return failure();
      plan.cuts[cut].init = *init;

      // The branches are mutually exclusive and every later read remains in
      // its matching branch. Equal-type values can share a result slot even
      // when their computations differ; unmatched slots alone need empties.
      auto &slots = plan.cuts[cut].liveSlots;
      auto isSharedLoadValue = [&](Value value, unsigned branch) {
        if (cut != 0)
          return false;
        if (plan.loadSuffix.results[branch].contains(value))
          return true;
        return plan.loadSuffix.separateScalarPrefix &&
               llvm::any_of(plan.loadSuffix.inputs,
                            [&](auto input) { return input[branch] == value; });
      };
      for (Value value : plan.bodies[0].live[cut]) {
        if (isSharedLoadValue(value, 0))
          continue;
        slots.push_back(LiveSlot{{value, Value()}});
      }
      for (Value value : plan.bodies[1].live[cut]) {
        if (isSharedLoadValue(value, 1))
          continue;
        auto slot = llvm::find_if(slots, [&](const LiveSlot &slot) {
          return !slot.values[1] && slot.getType() == value.getType();
        });
        if (slot == slots.end())
          slots.push_back(LiveSlot{{Value(), value}});
        else
          slot->values[1] = value;
      }
    }
    return plan;
  }

  scf::ForOp rewrite();
};

Value integer(OpBuilder &b, Location loc, Type type, int64_t value) {
  return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(type, value));
}

Value tripCount(OpBuilder &b, scf::ForOp loop) {
  Value bound = loop.getUpperBound();
  Location loc = loop.getLoc();
  if (!loop->hasAttr("unsignedCmp"))
    bound = b.create<arith::MaxSIOp>(loc, bound,
                                     integer(b, loc, bound.getType(), 0));
  return b.create<arith::ExtUIOp>(loc, b.getI64Type(), bound);
}

// These slots are read only in the branch that provided the real value.
// A placeholder is never selected as an operand of a shared matmul.
Value placeholder(OpBuilder &b, Location loc, Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return b.create<tensor::EmptyOp>(loc, tensor.getShape(),
                                     tensor.getElementType(), ValueRange{},
                                     tensor.getEncoding());
  return b.create<arith::ConstantOp>(loc, b.getZeroAttr(type));
}

// Operations can be shared only after their branch-local operands have become
// exactly the same SSA values. The temporary maps also check region-local SSA
// correspondences; an unsuccessful comparison cannot change the real maps.
bool sameMappedPureOp(Operation *lhs, Operation *rhs,
                      const std::array<IRMapping, 2> &mappings) {
  if (!isMemoryEffectFree(lhs) || !isMemoryEffectFree(rhs) ||
      !llvm::equal(lhs->getOperandTypes(), rhs->getOperandTypes()) ||
      !llvm::equal(lhs->getResultTypes(), rhs->getResultTypes()))
    return false;
  auto compareMaps = mappings;
  return OperationEquivalence::isEquivalentTo(
      lhs, rhs,
      [&](Value left, Value right) {
        return success(compareMaps[0].lookupOrDefault(left) ==
                       compareMaps[1].lookupOrDefault(right));
      },
      [&](Value left, Value right) {
        compareMaps[0].map(left, left);
        compareMaps[1].map(right, left);
      },
      OperationEquivalence::IgnoreLocations);
}

unsigned hoistCommonPrefix(ArrayRef<Operation *> partial,
                           ArrayRef<Operation *> full, OpBuilder &builder,
                           std::array<IRMapping, 2> &mappings) {
  unsigned count = 0;
  while (count < partial.size() && count < full.size() &&
         sameMappedPureOp(partial[count], full[count], mappings)) {
    Operation *shared = builder.clone(*partial[count], mappings[0]);
    mappings[1].map(full[count]->getResults(), shared->getResults());
    ++count;
  }
  return count;
}

// Carry the row scale across PV instead of a full scaled accumulator. Keep
// both FP operations and their operand order, and apply the move symmetrically.
static void sinkPairedAccumulatorScale(ArrayRef<scf::ForOp> loops) {
  struct Scale {
    linalg::BroadcastOp broadcast;
    arith::MulFOp multiply;
    linalg::MatmulOp matmul;
  };
  SmallVector<Scale, 2> scales;
  for (scf::ForOp loop : loops) {
    SmallVector<Scale, 1> candidates;
    for (auto add : loop.getBody()->getOps<arith::AddFOp>()) {
      for (unsigned i : {0u, 1u}) {
        auto matmul = add->getOperand(i).getDefiningOp<linalg::MatmulOp>();
        auto mul = add->getOperand(1 - i).getDefiningOp<arith::MulFOp>();
        if (!matmul || !mul || matmul->getBlock() != loop.getBody() ||
            mul->getBlock() != loop.getBody() ||
            !matmul.hasPureTensorSemantics() || !mul.getResult().hasOneUse() ||
            !mul->isBeforeInBlock(matmul) || !isMemoryEffectFree(mul))
          continue;
        for (unsigned j : {0u, 1u}) {
          auto accumulator = dyn_cast<BlockArgument>(mul->getOperand(j));
          auto broadcast =
              mul->getOperand(1 - j).getDefiningOp<linalg::BroadcastOp>();
          if (!accumulator || accumulator.getOwner() != loop.getBody() ||
              accumulator == loop.getInductionVar() || !broadcast ||
              broadcast->getBlock() != loop.getBody() ||
              !broadcast.hasPureTensorSemantics() ||
              !broadcast->getResult(0).hasOneUse() ||
              !isMemoryEffectFree(broadcast))
            continue;
          auto input =
              dyn_cast<RankedTensorType>(broadcast->getOperand(0).getType());
          auto output = dyn_cast<RankedTensorType>(mul.getType());
          if (input && output && input.getRank() < output.getRank())
            candidates.push_back({broadcast, mul, matmul});
        }
      }
    }
    if (candidates.size() != 1)
      return;
    scales.push_back(candidates.front());
  }
  for (Scale scale : scales) {
    scale.broadcast->moveAfter(scale.matmul);
    scale.multiply->moveAfter(scale.broadcast);
  }
}

// Bring independent tensor matmuls together to keep their VECTOR
// consumers in one stage. Move only pure SSA producers, in source order.
static void hoistReadyMatmuls(scf::ForOp loop) {
  Block *body = loop.getBody();
  Operation *previous = nullptr;
  auto matmuls = llvm::to_vector(body->getOps<linalg::MatmulOp>());
  for (auto matmul : matmuls) {
    Operation *current = matmul.getOperation();
    if (!previous || !isMemoryEffectFree(current)) {
      previous = current;
      continue;
    }
    Operation *first = previous->getNextNode();
    DenseSet<Operation *> producers;
    SmallVector<Value> operands(current->getOperands());
    bool movable = true;
    while (!operands.empty() && movable) {
      Operation *def = operands.pop_back_val().getDefiningOp();
      if (!def || def->getBlock() != body || def->isBeforeInBlock(previous))
        continue;
      if (def == previous || !isMemoryEffectFree(def) ||
          (def->getNumRegions() && !isa<linalg::TransposeOp>(def))) {
        movable = false;
        break;
      }
      if (producers.insert(def).second)
        llvm::append_range(operands, def->getOperands());
    }
    if (movable) {
      while (first != current && producers.contains(first))
        first = first->getNextNode();
      for (Operation *op = first; op != current;) {
        Operation *next = op->getNextNode();
        if (producers.contains(op))
          op->moveBefore(first);
        op = next;
      }
      current->moveBefore(first);
    }
    previous = current;
  }
}

scf::ForOp JointPlan::rewrite() {
  auto partial = bodies[0].loop;
  auto full = bodies[1].loop;
  OpBuilder b(full);
  Location loc = partial.getLoc();
  Value partialCount = tripCount(b, partial);
  Value fullCount = tripCount(b, full);
  Value total = b.create<arith::AddIOp>(loc, partialCount, fullCount);
  Value zero = integer(b, loc, b.getI64Type(), 0);
  Value one = integer(b, loc, b.getI64Type(), 1);
  auto joint =
      b.create<scf::ForOp>(loc, zero, total, one, partial.getInitArgs());
  OpBuilder body = createOuterEventBodyBuilder(*joint.getBody());
  Value isPartial = body.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, joint.getInductionVar(), partialCount);
  Value fullOffset =
      body.create<arith::SubIOp>(loc, joint.getInductionVar(), partialCount);
  Type originalIVType = partial.getInductionVar().getType();
  std::array<Value, 2> localIV{
      body.create<arith::TruncIOp>(loc, originalIVType,
                                   joint.getInductionVar()),
      body.create<arith::TruncIOp>(loc, originalIVType, fullOffset)};
  std::array<IRMapping, 2> mappings;
  for (unsigned branch = 0; branch < 2; ++branch) {
    mappings[branch].map(bodies[branch].loop.getInductionVar(),
                         localIV[branch]);
    mappings[branch].map(bodies[branch].loop.getRegionIterArgs(),
                         joint.getRegionIterArgs());
  }

  for (unsigned cut = 0; cut < cuts.size(); ++cut) {
    Cut &stage = cuts[cut];
    SharedInit &init = stage.init;
    // In particular, scale(QK) must have one shared consumer. Keeping the
    // identical operation in sibling regions forces duplicate transfer users.
    unsigned commonPrefix =
        cut ? hoistCommonPrefix(bodies[0].segments[cut],
                                bodies[1].segments[cut], body, mappings)
            : 0;
    auto partialMatmul = bodies[0].matmuls[cut];
    unsigned suffixSize = cut == 0 ? loadSuffix.size : 0;
    bool separateScalars = suffixSize && loadSuffix.separateScalarPrefix;
    if (separateScalars) {
      SmallVector<Type> scalarTypes;
      for (auto input : loadSuffix.inputs)
        scalarTypes.push_back(input[0].getType());
      auto scalarGuard = createOuterEventIf(
          body, loc, isPartial, scalarTypes,
          [&](OpBuilder &bb, unsigned branch) {
            for (Operation *op :
                 ArrayRef<Operation *>(bodies[branch].segments[cut])
                     .take_front(loadSuffix.scalarPrefixSize[branch]))
              bb.clone(*op, mappings[branch]);
            return llvm::map_to_vector(loadSuffix.inputs, [&](auto input) {
              return mappings[branch].lookupOrDefault(input[branch]);
            });
          });
      for (auto [index, input] : llvm::enumerate(loadSuffix.inputs))
        for (unsigned branch = 0; branch < 2; ++branch)
          mappings[branch].map(input[branch], scalarGuard.getResult(index));
    }
    // Select only operands that still differ. Q and the shared V tensor, for
    // example, must not acquire an artificial VECTOR if-result ownership.
    std::array<int, 2> selectedInput{-1, -1};
    SmallVector<Type> types;
    for (unsigned input = 0; input < 2; ++input) {
      Value left = partialMatmul->getOperand(input);
      Value right = bodies[1].matmuls[cut]->getOperand(input);
      bool shared = mappings[0].lookupOrDefault(left) ==
                    mappings[1].lookupOrDefault(right);
      shared |= suffixSize && loadSuffix.results[0].contains(left) &&
                loadSuffix.equivalent.lookup(left) == right;
      if (!shared) {
        selectedInput[input] = types.size();
        types.push_back(left.getType());
      }
    }
    unsigned selectedCount = types.size();
    unsigned capturedCount =
        suffixSize && !separateScalars ? loadSuffix.inputs.size() : 0;
    if (capturedCount)
      for (auto input : loadSuffix.inputs)
        types.push_back(input[0].getType());
    for (const LiveSlot &slot : stage.liveSlots)
      types.push_back(slot.getType());
    auto guard = createOuterEventIf(
        body, loc, isPartial, types, [&](OpBuilder &bb, unsigned branch) {
          unsigned skippedPrefix =
              commonPrefix +
              (separateScalars ? loadSuffix.scalarPrefixSize[branch] : 0);
          for (Operation *op :
               ArrayRef<Operation *>(bodies[branch].segments[cut])
                   .drop_front(skippedPrefix)
                   .drop_back(suffixSize)) {
            if (op == init.fills[branch].getOperation() ||
                op == init.empties[branch].getOperation())
              continue;
            bb.clone(*op, mappings[branch]);
          }
          SmallVector<Value> values;
          for (unsigned input = 0; input < 2; ++input)
            if (selectedInput[input] >= 0)
              values.push_back(mappings[branch].lookupOrDefault(
                  bodies[branch].matmuls[cut]->getOperand(input)));
          if (capturedCount)
            for (auto input : loadSuffix.inputs)
              values.push_back(mappings[branch].lookupOrDefault(input[branch]));
          for (const LiveSlot &slot : stage.liveSlots) {
            Value value = slot.values[branch];
            values.push_back(value ? mappings[branch].lookupOrDefault(value)
                                   : placeholder(bb, loc, slot.getType()));
          }
          return values;
        });
    for (unsigned index = 0; index < capturedCount; ++index)
      for (unsigned branch = 0; branch < 2; ++branch)
        mappings[branch].map(loadSuffix.inputs[index][branch],
                             guard.getResult(selectedCount + index));
    for (auto [index, slot] : llvm::enumerate(stage.liveSlots))
      for (unsigned branch = 0; branch < 2; ++branch)
        if (Value value = slot.values[branch])
          mappings[branch].map(
              value, guard.getResult(selectedCount + capturedCount + index));

    if (suffixSize) {
      auto partialOps =
          ArrayRef<Operation *>(bodies[0].segments[cut]).take_back(suffixSize);
      auto fullOps =
          ArrayRef<Operation *>(bodies[1].segments[cut]).take_back(suffixSize);
      for (unsigned index = 0; index < suffixSize; ++index) {
        Operation *shared = body.clone(*partialOps[index], mappings[0]);
        mappings[1].map(fullOps[index]->getResults(), shared->getResults());
      }
    }

    Value commonInit = mappings[0].lookupOrDefault(init.common);
    if (!commonInit) {
      IRMapping initMapping;
      body.clone(*init.constant, initMapping);
      body.clone(*init.empties[0], initMapping);
      Operation *fill = body.clone(*init.fills[0], initMapping);
      commonInit = fill->getResult(0);
    }
    // Set by operand position, not a Value-to-Value map: one original value
    // can occupy two operands that select different values on the other path.
    Operation *shared = body.clone(*partialMatmul);
    SmallVector<Value> operands;
    for (unsigned input = 0; input < 2; ++input)
      operands.push_back(
          selectedInput[input] >= 0
              ? guard.getResult(selectedInput[input])
              : mappings[0].lookupOrDefault(partialMatmul->getOperand(input)));
    operands.push_back(commonInit);
    shared->setOperands(operands);
    for (unsigned branch = 0; branch < 2; ++branch)
      mappings[branch].map(bodies[branch].matmuls[cut]->getResults(),
                           shared->getResults());
  }

  unsigned commonTail = hoistCommonPrefix(
      bodies[0].segments.back(), bodies[1].segments.back(), body, mappings);
  auto mappedYield = [&](unsigned branch) {
    return mapOuterEventValues(bodies[branch].loop.getYieldedValues(),
                               mappings[branch]);
  };
  SmallVector<Value> yielded = mappedYield(0);
  if (commonTail != bodies[0].segments.back().size() ||
      commonTail != bodies[1].segments.back().size() ||
      yielded != mappedYield(1)) {
    auto tail = createOuterEventIf(
        body, loc, isPartial, partial.getResultTypes(),
        [&](OpBuilder &bb, unsigned branch) {
          for (Operation *op :
               ArrayRef<Operation *>(bodies[branch].segments.back())
                   .drop_front(commonTail))
            bb.clone(*op, mappings[branch]);
          return mappedYield(branch);
        });
    yielded.assign(tail.result_begin(), tail.result_end());
  }
  body.create<scf::YieldOp>(loc, yielded);
  full.replaceAllUsesWith(joint.getResults());
  full.erase();
  partial.erase();
  return joint;
}

} // namespace

LogicalResult
mlir::CVPipeline::tryJoinSparseInnerLoops(OuterEventLoopPlan &plan,
                                          bool allowTaskLocalAccumulation) {
  assert(plan.inner.size() == 2 && "expected a validated partial/full pair");
  if (llvm::any_of(plan.inner,
                   [](scf::ForOp loop) { return !loop.getNumResults(); }))
    return failure();

  // Build and re-analyze on a detached clone. Optional factoring must not
  // leave a partially rewritten nest when the event analysis cannot accept it.
  IRMapping mapping;
  OwningOpRef<scf::ForOp> probe(cast<scf::ForOp>(plan.outer->clone(mapping)));
  SmallVector<scf::ForOp> loops;
  for (scf::ForOp loop : plan.inner)
    loops.push_back(
        cast<scf::ForOp>(mapping.lookup(loop.getResult(0)).getDefiningOp()));
  bool taskLocalAccumulation = llvm::any_of(loops, [](scf::ForOp loop) {
    Block *body = loop.getBody();
    return llvm::any_of(
        body->getOps<linalg::MatmulOp>(), [body](linalg::MatmulOp matmul) {
          auto init = dyn_cast<BlockArgument>(matmul.getDpsInits()[0]);
          return init && init.getOwner() == body;
        });
  });
  if (taskLocalAccumulation && !allowTaskLocalAccumulation)
    return failure();
  sinkPairedAccumulatorScale(loops);
  if (taskLocalAccumulation)
    for (scf::ForOp loop : loops)
      hoistReadyMatmuls(loop);
  FailureOr<OuterEventLoopPlan> joined = failure();
  {
    ScopedDiagnosticHandler suppress(plan.outer.getContext(),
                                     [](Diagnostic &) { return success(); });
    if (auto joint = JointPlan::analyze(loops[0], loops[1]); succeeded(joint))
      joined = OuterEventLoopPlan::analyze(*probe, {joint->rewrite()});
  }
  if (failed(joined))
    return failure();
  scf::ForOp outer = plan.outer;
  if (taskLocalAccumulation)
    joined->serialFallback =
        OwningOpRef<scf::ForOp>(cast<scf::ForOp>(outer->clone()));
  outer.getRegion().takeBody(probe->getRegion());
  joined->outer = outer;
  plan = std::move(*joined);
  return success();
}
