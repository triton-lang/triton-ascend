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

#include "TritonToGraph/GraphOptimizationRule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// A load the modulo result addresses, together with the dimension of that load
// along which the wrapped coordinate varies.  Recording the dimension per load
// instead of once per candidate is what lets one index address differently
// shaped loads: in a block-quantised matmul the same column index addresses a
// rank-two weight tile and the rank-one scale vector that scales it.
struct AddressedLoad {
  triton::LoadOp load;
  int64_t dim = 0;
};

// A modulo whose result is only ever used to compute load addresses, and whose
// wrapped lanes are already discarded by a store mask.  Such a modulo can be
// dropped so the addresses become linear again, provided the loads that consume
// them get a boundary mask so the wrapped lanes read in-bounds zeros instead of
// the data the wrap used to fetch.
struct ModuloCandidate {
  // The op defining the modulo result: either arith.remsi, or the subi of the
  // divsi/muli/subi expansion that canonicalization leaves behind.
  Operation *anchor = nullptr;
  Value dividend;
  Value divisor;
  Value boundScalar;
  Value result;
  // anchor plus, for the expanded form, the muli and divsi that die with it.
  SmallVector<Operation *, 3> patternOps;
  SmallVector<AddressedLoad, 4> loads;
  int64_t tileSize = 0;
};

Value getSplatScalar(Value value) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return splat.getSrc();
  return nullptr;
}

// Reports the extent of the tt.make_range that the tile offset is built from.
// The offset is `splat(pid * BLOCK) + make_range(0, BLOCK)`, so the range is
// reached through the addi.
int64_t getMakeRangeExtent(Value value) {
  if (auto range = value.getDefiningOp<triton::MakeRangeOp>())
    return static_cast<int64_t>(range.getEnd()) -
           static_cast<int64_t>(range.getStart());
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (int64_t lhs = getMakeRangeExtent(add.getLhs()))
      return lhs;
    return getMakeRangeExtent(add.getRhs());
  }
  return 0;
}

bool isCompileTimeConstant(Value value) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant));
}

// Reports whether every lane of `value` holds the same number.  An elementwise
// op with such an operand leaves each result lane a function of that same lane
// of the tracked index, so one boundary comparison still describes exactly the
// lanes the wrap used to fold back.
bool isUniformOperand(Value value) {
  if (getSplatScalar(value))
    return true;
  DenseElementsAttr constant;
  return matchPattern(value, m_Constant(&constant)) && constant.isSplat();
}

// Where the tracked dimension ends up after tt.expand_dims inserted a unit
// dimension.  Ranks above two are rejected: the boundary mask this rule builds
// is one comparison expanded once, which only covers a rank-two consumer.
std::optional<int64_t> mapDimThroughExpand(int64_t dim,
                                           triton::ExpandDimsOp expand) {
  auto type = dyn_cast<RankedTensorType>(expand.getResult().getType());
  if (!type || type.getRank() > 2)
    return std::nullopt;
  int64_t axis = expand.getAxis();
  int64_t mapped = dim >= axis ? dim + 1 : dim;
  if (mapped >= type.getRank())
    return std::nullopt;
  return mapped;
}

// Where the tracked dimension ends up after tt.trans permuted the dimensions.
// Result dimension i holds source dimension order[i].
std::optional<int64_t> mapDimThroughTrans(int64_t dim,
                                          triton::TransOp transpose) {
  for (auto [position, source] : llvm::enumerate(transpose.getOrder())) {
    if (static_cast<int64_t>(source) == dim)
      return static_cast<int64_t>(position);
  }
  return std::nullopt;
}

// tt.broadcast only stretches dimensions of extent one, so it moves no lane.
// Stretching the tracked dimension itself would leave the boundary mask
// describing fewer lanes than the result has, so that case is rejected.
bool broadcastKeepsDim(triton::BroadcastOp broadcast, int64_t dim) {
  auto source = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
  auto result = dyn_cast<RankedTensorType>(broadcast.getResult().getType());
  return source && result && dim < source.getRank() && dim < result.getRank() &&
         source.getShape()[dim] == result.getShape()[dim];
}

// The dimensions a store's mask discards the wrapped lanes along, as a bit per
// dimension.  A store may be guarded on more than one of them, and a tile
// wrapped along any one of those is safe to rewrite.
using GuardedStores = DenseMap<Operation *, unsigned>;

// Collects the tt.store ops that `mask` guards, tracking which dimension of the
// stored shape the comparison varies along.  Shape ops and arith.andi are
// transparent because a store mask is normally the conjunction of one
// comparison per output axis, each broadcast to the output shape.
void collectMaskedStores(Value mask, int64_t dim, GuardedStores &stores,
                         DenseMap<Value, unsigned> &visited) {
  unsigned bit = 1u << dim;
  unsigned &seen = visited[mask];
  if (seen & bit)
    return;
  seen |= bit;

  for (OpOperand &use : mask.getUses()) {
    Operation *user = use.getOwner();
    if (auto store = dyn_cast<triton::StoreOp>(user)) {
      if (store.getMask() == mask)
        stores[store.getOperation()] |= bit;
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(user)) {
      if (std::optional<int64_t> next = mapDimThroughExpand(dim, expand))
        collectMaskedStores(expand.getResult(), *next, stores, visited);
      continue;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(user)) {
      if (broadcastKeepsDim(broadcast, dim))
        collectMaskedStores(broadcast.getResult(), dim, stores, visited);
      continue;
    }
    // A conjunction only ever drops more lanes, so the dimension this operand
    // discards is still discarded by the result.
    if (isa<arith::AndIOp>(user))
      collectMaskedStores(user->getResult(0), dim, stores, visited);
  }
}

// Collects the stores guarded by a `cmpi slt` of the un-wrapped tile offset
// against the same bound, which is where the kernel itself discards the wrapped
// lanes.  Which stores they are matters, and along which dimension: without the
// dimension a square tile whose store mask drops rows would look like a proof
// about the columns the injected mask zeros.
bool collectStoreMaskGuard(Value offset, Value boundScalar,
                           GuardedStores &stores) {
  // The comparison is written either on the bare tile offset or on an already
  // expanded form of it, so both shapes have to be considered.
  SmallVector<std::pair<Value, int64_t>, 4> forms = {{offset, 0}};
  for (OpOperand &use : offset.getUses()) {
    auto expand = dyn_cast<triton::ExpandDimsOp>(use.getOwner());
    if (!expand)
      continue;
    if (std::optional<int64_t> dim = mapDimThroughExpand(0, expand))
      forms.emplace_back(expand.getResult(), *dim);
  }

  for (auto [form, dim] : forms) {
    for (OpOperand &use : form.getUses()) {
      auto compare = dyn_cast<arith::CmpIOp>(use.getOwner());
      if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt ||
          compare.getLhs() != form)
        continue;
      Value bound = getSplatScalar(compare.getRhs());
      if (bound != boundScalar)
        continue;
      DenseMap<Value, unsigned> visited;
      collectMaskedStores(compare.getResult(), dim, stores, visited);
    }
  }
  return !stores.empty();
}

// Returns true when two distinct loop-carried paths cross nested scf.for ops.
// An address passed through one loop may still be lowered as a direct access,
// but the current structured lowering does not preserve every component of a
// nested loop-carried pointer state.  Keep the modulo in that shape until the
// lowerer can prove the relay lossless.
bool crossesNestedCarriedFor(ArrayRef<Operation *> carriedForChain,
                             scf::ForOp nextFor) {
  Operation *next = nextFor.getOperation();
  return llvm::any_of(carriedForChain, [next](Operation *carriedFor) {
    if (carriedFor == next)
      return false;
    return carriedFor->isProperAncestor(next) ||
           next->isProperAncestor(carriedFor);
  });
}

// Reports whether `assertOp` is one the overflow sanitizer inserted.  A user
// device_assert may carry the same message, so only the marker proves it.
bool isAutomaticOverflowAssert(triton::AssertOp assertOp) {
  if (!assertOp->hasAttr("tt.auto_overflow_assert"))
    return false;
  auto message = dyn_cast<StringAttr>(assertOp.getMessageAttr());
  return message &&
         message.getValue().contains("overflow detected for operation");
}

// Reports whether `op` can only make the kernel trap: every path forward ends
// in an overflow assert, so nothing it computes is observed.  In debug builds
// the sanitizer hangs such a cone off every tile offset, which would otherwise
// count as a consumer of the un-wrapped index.  The rewrite keeps the check
// honest because the emitted arithmetic really does become un-wrapped, and that
// argument is why the sanitizer's provenance is required here.
bool onlyFeedsOverflowAssert(Operation *op,
                             SmallPtrSetImpl<Operation *> &visited) {
  if (auto assertOp = dyn_cast<triton::AssertOp>(op))
    return isAutomaticOverflowAssert(assertOp);
  if (!visited.insert(op).second)
    return true;
  // Anything else without results ends the walk before reaching an assert.
  if (op->getNumResults() == 0)
    return false;
  if (!isa<arith::ExtSIOp, arith::MulIOp, arith::AddIOp, arith::SubIOp,
           arith::CmpIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
           triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp>(op))
    return false;
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (!onlyFeedsOverflowAssert(use.getOwner(), visited))
        return false;
    }
  }
  return true;
}

// Walks forward from the modulo result and collects the loads it addresses,
// carrying the dimension the tile coordinate varies along.  Returns false as
// soon as the value reaches anything else, which is what makes dropping the
// wrap unobservable: no other consumer can see the widened index.
bool collectAddressedLoads(Value value, int64_t dim,
                           DenseMap<Value, int64_t> &visited,
                           SmallVectorImpl<AddressedLoad> &loads,
                           ArrayRef<Operation *> carriedForChain) {
  auto known = visited.find(value);
  if (known != visited.end())
    // Two paths reached the same value on different dimensions, so the index
    // addresses both and no single mask orientation describes it.
    return known->second == dim;
  visited[value] = dim;

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();

    if (auto load = dyn_cast<triton::LoadOp>(user)) {
      // Reject a value that is the `other` operand rather than the address:
      // rewriting it would change the loaded data itself.
      if (load.getPtr() != value)
        return false;
      loads.push_back({load, dim});
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(user)) {
      std::optional<int64_t> next = mapDimThroughExpand(dim, expand);
      if (!next || !collectAddressedLoads(expand.getResult(), *next, visited,
                                          loads, carriedForChain))
        return false;
      continue;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(user)) {
      if (!broadcastKeepsDim(broadcast, dim) ||
          !collectAddressedLoads(broadcast.getResult(), dim, visited, loads,
                                 carriedForChain))
        return false;
      continue;
    }
    // These keep operand and result shapes identical, so they move no lane.
    if (isa<triton::AddPtrOp, arith::MulIOp, arith::AddIOp>(user)) {
      for (Value result : user->getResults()) {
        if (!collectAddressedLoads(result, dim, visited, loads,
                                   carriedForChain))
          return false;
      }
      continue;
    }
    // A uniform divisor leaves each lane of the derived index a function of
    // that same lane of the tracked one, so the boundary mask still zeros
    // exactly the lanes the wrap used to fold back.  The derived index itself
    // may leave the array it indexes, but only on lanes the mask removes, so
    // the load never reads them.
    if (isa<arith::DivSIOp, arith::RemSIOp>(user)) {
      if (user->getOperand(0) != value ||
          !isUniformOperand(user->getOperand(1)))
        return false;
      if (!collectAddressedLoads(user->getResult(0), dim, visited, loads,
                                 carriedForChain))
        return false;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      // An address carried around a loop stays an address only if the iter arg
      // and the loop result are both used as one.
      for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg != value)
          continue;
        if (crossesNestedCarriedFor(carriedForChain, forOp))
          return false;
        SmallVector<Operation *, 2> nextChain(carriedForChain);
        nextChain.push_back(forOp.getOperation());
        if (!collectAddressedLoads(forOp.getRegionIterArg(index), dim, visited,
                                   loads, nextChain))
          return false;
      }
      continue;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!forOp)
        return false;
      unsigned index = use.getOperandNumber();
      if (index >= forOp.getNumResults())
        return false;
      if (crossesNestedCarriedFor(carriedForChain, forOp))
        return false;
      SmallVector<Operation *, 2> nextChain(carriedForChain);
      nextChain.push_back(forOp.getOperation());
      if (!collectAddressedLoads(forOp.getResult(index), dim, visited, loads,
                                 nextChain))
        return false;
      continue;
    }

    // A consumer that can only trap never observes the index in a value.
    SmallPtrSet<Operation *, 16> assertCone;
    if (onlyFeedsOverflowAssert(user, assertCone))
      continue;

    return false;
  }
  return true;
}

// Reports whether a boundary mask of `tileSize` lanes can be shaped to fit
// every load, on the dimension that load's address varies along.  Checking it
// up front means the mask ops this rule creates are valid by construction.
bool areLoadsMaskable(ArrayRef<AddressedLoad> loads, int64_t tileSize) {
  if (loads.empty())
    return false;
  for (const AddressedLoad &addressed : loads) {
    triton::LoadOp load = addressed.load;
    auto type = dyn_cast<RankedTensorType>(load.getResult().getType());
    if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
        type.getRank() > 2 || addressed.dim >= type.getRank() ||
        type.getShape()[addressed.dim] != tileSize)
      return false;
    // A load carrying `other` without a mask reads it nowhere, so injecting a
    // mask would suddenly make that operand observable.
    if (!load.getMask() && load.getOther())
      return false;
    if (!load.getBoundaryCheck().empty() || load.getIsVolatile())
      return false;
  }
  return true;
}

// Reports whether `value` still carries the wrapped coordinate on dimension
// `dim`, with the extent the injected mask has.
bool keepsWrappedDim(Value value, int64_t dim, int64_t tileSize) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getRank() >= 1 &&
         type.getRank() <= 2 && dim < type.getRank() &&
         type.getShape()[dim] == tileSize;
}

// Reports whether the tile a load produced reaches memory only through stores
// that discard the wrapped lanes, on the dimension those lanes ended up on.
// The guard alone proves some store drops them, not that the store writing this
// tile does; without that, a kernel whose guard belongs to an unrelated store
// gets a deliberate circular index rewritten and the injected zeros written
// out.  Requiring the dimension to agree is what makes the proof hold for a
// square tile, where a mask on rows and a mask on columns have the same shape.
bool tileReachesOnlyGuardedStores(Value value, int64_t dim, int64_t tileSize,
                                  const GuardedStores &guarded,
                                  DenseMap<Value, int64_t> &visited) {
  auto known = visited.find(value);
  if (known != visited.end())
    return known->second == dim;
  visited[value] = dim;

  if (!keepsWrappedDim(value, dim, tileSize))
    return false;

  auto reaches = [&](Value next, int64_t nextDim) {
    return tileReachesOnlyGuardedStores(next, nextDim, tileSize, guarded,
                                        visited);
  };

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();

    if (auto store = dyn_cast<triton::StoreOp>(user)) {
      // As an address or a mask the tile is no longer the data reasoned about.
      if (store.getValue() != value)
        return false;
      auto guard = guarded.find(store.getOperation());
      if (guard == guarded.end() || !(guard->second & (1u << dim)))
        return false;
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(user)) {
      std::optional<int64_t> next = mapDimThroughExpand(dim, expand);
      if (!next || !reaches(expand.getResult(), *next))
        return false;
      continue;
    }
    if (auto transpose = dyn_cast<triton::TransOp>(user)) {
      std::optional<int64_t> next = mapDimThroughTrans(dim, transpose);
      if (!next || !reaches(transpose.getResult(), *next))
        return false;
      continue;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(user)) {
      if (!broadcastKeepsDim(broadcast, dim) ||
          !reaches(broadcast.getResult(), dim))
        return false;
      continue;
    }
    if (auto dot = dyn_cast<triton::DotOp>(user)) {
      // Zeros on a contracted axis would spread into every output element, so
      // an operand has to carry the wrapped coordinate on the axis the result
      // keeps: rows for A, columns for B.
      if (dot.getA() == value && dim != 0)
        return false;
      if (dot.getB() == value && dim != 1)
        return false;
      if (!reaches(dot.getResult(), dim))
        return false;
      continue;
    }
    // Arith ops on tensors are elementwise, so they move no lane.
    Dialect *dialect = user->getDialect();
    if (dialect &&
        dialect->getNamespace() == arith::ArithDialect::getDialectNamespace()) {
      for (Value result : user->getResults())
        if (!reaches(result, dim))
          return false;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg != value)
          continue;
        if (!reaches(forOp.getRegionIterArg(index), dim) ||
            !reaches(forOp.getResult(index), dim))
          return false;
      }
      continue;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!forOp)
        return false;
      unsigned index = use.getOperandNumber();
      if (index >= forOp.getNumResults())
        return false;
      if (!reaches(forOp.getResult(index), dim))
        return false;
      continue;
    }

    return false;
  }
  return true;
}

// Matches the divsi/muli/subi expansion of a modulo.  It is equivalent to
// `x % d` by the signed-division identity x == (x / d) * d + x % d.
bool matchExpandedModulo(arith::SubIOp subtract, Value &dividend,
                         Value &divisor,
                         SmallVectorImpl<Operation *> &patternOps) {
  auto multiply = subtract.getRhs().getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;

  arith::DivSIOp divide;
  Value candidateDivisor;
  if (auto lhs = multiply.getLhs().getDefiningOp<arith::DivSIOp>()) {
    divide = lhs;
    candidateDivisor = multiply.getRhs();
  } else if (auto rhs = multiply.getRhs().getDefiningOp<arith::DivSIOp>()) {
    divide = rhs;
    candidateDivisor = multiply.getLhs();
  } else {
    return false;
  }

  if (divide.getLhs() != subtract.getLhs() ||
      divide.getRhs() != candidateDivisor)
    return false;

  dividend = subtract.getLhs();
  divisor = candidateDivisor;
  patternOps.assign({subtract.getOperation(), multiply.getOperation(),
                     divide.getOperation()});
  return true;
}

std::optional<ModuloCandidate> analyzeModulo(Operation *op) {
  ModuloCandidate candidate;
  if (auto remainder = dyn_cast<arith::RemSIOp>(op)) {
    candidate.dividend = remainder.getLhs();
    candidate.divisor = remainder.getRhs();
    candidate.patternOps.push_back(op);
  } else if (auto subtract = dyn_cast<arith::SubIOp>(op)) {
    if (!matchExpandedModulo(subtract, candidate.dividend, candidate.divisor,
                             candidate.patternOps))
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  candidate.anchor = op;
  candidate.result = op->getResult(0);

  // A 1-D i32 index is the shape every tile offset has at this point in the
  // pipeline.  Anything else is left alone rather than reasoned about.
  auto type = dyn_cast<RankedTensorType>(candidate.result.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 1 ||
      !type.getElementType().isInteger(32) || type.getEncoding())
    return std::nullopt;

  candidate.tileSize = getMakeRangeExtent(candidate.dividend);
  if (candidate.tileSize <= 0 || candidate.tileSize != type.getShape()[0])
    return std::nullopt;

  candidate.boundScalar = getSplatScalar(candidate.divisor);
  if (!candidate.boundScalar)
    return std::nullopt;

  // A constant divisor belongs to TritonToStructured, whose visitOperandRem
  // keeps the wrap and re-expresses it as a strided access.  That is exactly
  // equivalent, so it is always preferable to discarding the wrap here.
  if (isCompileTimeConstant(candidate.boundScalar))
    return std::nullopt;

  GuardedStores guardedStores;
  if (!collectStoreMaskGuard(candidate.dividend, candidate.boundScalar,
                             guardedStores))
    return std::nullopt;

  DenseMap<Value, int64_t> visited;
  SmallVector<Operation *, 2> carriedForChain;
  if (!collectAddressedLoads(candidate.result, /*dim=*/0, visited,
                             candidate.loads, carriedForChain))
    return std::nullopt;
  if (!areLoadsMaskable(candidate.loads, candidate.tileSize))
    return std::nullopt;

  for (const AddressedLoad &addressed : candidate.loads) {
    triton::LoadOp load = addressed.load;
    DenseMap<Value, int64_t> tileVisited;
    if (!tileReachesOnlyGuardedStores(load.getResult(), addressed.dim,
                                      candidate.tileSize, guardedStores,
                                      tileVisited))
      return std::nullopt;
  }

  return candidate;
}

bool matchesCandidate(const ModuloCandidate &candidate,
                      const ModuloCandidate &current) {
  if (candidate.anchor != current.anchor ||
      candidate.dividend != current.dividend ||
      candidate.divisor != current.divisor ||
      candidate.boundScalar != current.boundScalar ||
      candidate.result != current.result ||
      candidate.tileSize != current.tileSize ||
      candidate.patternOps.size() != current.patternOps.size() ||
      !std::equal(candidate.patternOps.begin(), candidate.patternOps.end(),
                  current.patternOps.begin()) ||
      candidate.loads.size() != current.loads.size())
    return false;
  for (auto [planned, found] :
       llvm::zip_equal(candidate.loads, current.loads)) {
    if (planned.load != found.load || planned.dim != found.dim)
      return false;
  }
  return true;
}

void eraseCreatedOperations(IRRewriter &rewriter,
                            ArrayRef<Operation *> created) {
  for (Operation *operation : llvm::reverse(created))
    rewriter.eraseOp(operation);
}

bool recordVerifiedOperation(Operation *operation,
                             SmallVectorImpl<Operation *> &created) {
  if (!operation)
    return false;
  created.push_back(operation);
  return succeeded(mlir::verify(operation));
}

// The mask that a single load needs, plus the zero fill for a load that had no
// mask before.  Nothing is attached to the load until every load's operands
// have been built and verified.
struct LoadMask {
  triton::LoadOp load;
  Value mask;
  Value other;
};

LogicalResult applyCandidate(IRRewriter &rewriter,
                             const ModuloCandidate &candidate) {
  SmallVector<Operation *, 16> created;
  auto fail = [&]() {
    eraseCreatedOperations(rewriter, created);
    return failure();
  };

  Location loc = candidate.anchor->getLoc();
  rewriter.setInsertionPointAfter(candidate.anchor);
  auto boundary = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, candidate.dividend, candidate.divisor);
  if (!recordVerifiedOperation(boundary.getOperation(), created))
    return fail();

  SmallVector<LoadMask, 4> loadMasks;
  for (const AddressedLoad &addressed : candidate.loads) {
    triton::LoadOp load = addressed.load;
    auto loadType = dyn_cast<RankedTensorType>(load.getResult().getType());
    if (!loadType)
      return fail();

    rewriter.setInsertionPoint(load);
    Location loadLoc = load.getLoc();

    // The comparison already has one lane per tile element.  Putting the unit
    // dimension where the load does not vary leaves it varying along the same
    // dimension as the address it masks.
    Value mask = boundary.getResult();
    if (loadType.getRank() == 2) {
      auto expanded = rewriter.create<triton::ExpandDimsOp>(
          loadLoc, mask, static_cast<int32_t>(1 - addressed.dim));
      if (!recordVerifiedOperation(expanded.getOperation(), created))
        return fail();
      mask = expanded.getResult();
    }
    if (cast<RankedTensorType>(mask.getType()).getShape() !=
        loadType.getShape()) {
      auto maskType =
          RankedTensorType::get(loadType.getShape(), rewriter.getI1Type());
      auto broadcast =
          rewriter.create<triton::BroadcastOp>(loadLoc, maskType, mask);
      if (!recordVerifiedOperation(broadcast.getOperation(), created))
        return fail();
      mask = broadcast.getResult();
    }

    LoadMask loadMask{load, mask, nullptr};
    if (Value existing = load.getMask()) {
      auto combined = rewriter.create<arith::AndIOp>(loadLoc, existing, mask);
      if (!recordVerifiedOperation(combined.getOperation(), created))
        return fail();
      loadMask.mask = combined.getResult();
    } else {
      Type elementType = loadType.getElementType();
      Attribute zero;
      if (isa<FloatType>(elementType))
        zero = rewriter.getFloatAttr(elementType, 0.0);
      else if (isa<IntegerType>(elementType))
        zero = rewriter.getIntegerAttr(elementType, 0);
      else
        return fail();
      auto fill = rewriter.create<arith::ConstantOp>(
          loadLoc, DenseElementsAttr::get(loadType, zero));
      if (!recordVerifiedOperation(fill.getOperation(), created))
        return fail();
      loadMask.other = fill.getResult();
    }
    loadMasks.push_back(loadMask);
  }
  if (loadMasks.empty())
    return fail();

  // Everything needed has been built and verified, so the observable rewrite
  // can now run without any step that could still fail.
  for (LoadMask &loadMask : loadMasks) {
    loadMask.load.getMaskMutable().assign(loadMask.mask);
    if (loadMask.other)
      loadMask.load.getOtherMutable().assign(loadMask.other);
  }

  rewriter.replaceAllUsesWith(candidate.result, candidate.dividend);
  for (Operation *operation : candidate.patternOps) {
    if (operation->use_empty())
      rewriter.eraseOp(operation);
  }
  return success();
}

class ConvertModuloToMaskPlan final : public RewritePlan {
public:
  ConvertModuloToMaskPlan(ModuloCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::ConvertModuloToMask;
  }

  // Every load that stops wrapping can become a contiguous transfer.
  unsigned getBenefit() const override {
    return static_cast<unsigned>(std::min<size_t>(
        candidate.loads.size(), std::numeric_limits<unsigned>::max()));
  }

  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (candidate.anchor->getParentOfType<triton::FuncOp>() !=
        context.getFunction())
      return failure();
    std::optional<ModuloCandidate> current = analyzeModulo(candidate.anchor);
    return current && matchesCandidate(candidate, *current) ? success()
                                                            : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // Re-prove locally so that a stale plan can never mutate the IR.
    std::optional<ModuloCandidate> current = analyzeModulo(candidate.anchor);
    if (!current || !matchesCandidate(candidate, *current))
      return failure();
    return applyCandidate(rewriter, *current);
  }

private:
  ModuloCandidate candidate;
  unsigned epoch;
};

class ConvertModuloToMaskRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::ConvertModuloToMask;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](Operation *op) {
      if (!isa<arith::RemSIOp, arith::SubIOp>(op))
        return;
      if (std::optional<ModuloCandidate> candidate = analyzeModulo(op))
        plans.push_back(std::make_unique<ConvertModuloToMaskPlan>(
            std::move(*candidate), context.getEpoch()));
    });
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createConvertModuloToMaskRule() {
  return std::make_unique<ConvertModuloToMaskRule>();
}
