// SPDX-License-Identifier: MIT
#include "TritonToGraph/GraphOptimizationContext.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cfg;

namespace {

std::optional<int64_t> integer(Value v) {
  APInt n;
  if (v && matchPattern(v, m_ConstantInt(&n)) && n.getBitWidth() <= 64)
    return n.getSExtValue();
  return std::nullopt;
}

bool isConstant(Value v, int64_t n) { return integer(v) == n; }

bool equalTo(arith::CmpIOp cmp, Value value, int64_t n) {
  return cmp && cmp.getPredicate() == arith::CmpIPredicate::eq &&
         ((cmp.getLhs() == value && isConstant(cmp.getRhs(), n)) ||
          (cmp.getRhs() == value && isConstant(cmp.getLhs(), n)));
}

// Flatten addptr/splat chains without conflating entry pointer arguments.
struct Address {
  BlockArgument base;
  SmallVector<Value, 4> offsets;
};

void addOffset(Value value, SmallVectorImpl<Value> &offsets) {
  if (auto splat = value.getDefiningOp<SplatOp>())
    addOffset(splat.getSrc(), offsets);
  else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    addOffset(add.getLhs(), offsets);
    addOffset(add.getRhs(), offsets);
  } else if (!isConstant(value, 0))
    offsets.push_back(value);
}

Address address(Value pointer) {
  Address result;
  while (true) {
    if (auto add = pointer.getDefiningOp<AddPtrOp>()) {
      addOffset(add.getOffset(), result.offsets);
      pointer = add.getPtr();
    } else if (auto splat = pointer.getDefiningOp<SplatOp>()) {
      pointer = splat.getSrc();
    } else {
      result.base = dyn_cast<BlockArgument>(pointer);
      return result;
    }
  }
}

bool sameAddress(Value lhs, Value rhs) {
  Address a = address(lhs), b = address(rhs);
  return a.base && a.base == b.base &&
         std::is_permutation(a.offsets.begin(), a.offsets.end(),
                             b.offsets.begin(), b.offsets.end());
}

// Return the stride in an offset of the form index * stride.
Value indexStride(Value offset, Value index) {
  if (auto mul = offset.getDefiningOp<arith::MulIOp>()) {
    if (mul.getLhs() == index)
      return mul.getRhs();
    if (mul.getRhs() == index)
      return mul.getLhs();
  }
  return {};
}

Value vectorStride(const Address &a, Value index, Value column) {
  if (!a.base || a.offsets.size() != 2)
    return {};
  if (a.offsets[0] == column)
    return indexStride(a.offsets[1], index);
  if (a.offsets[1] == column)
    return indexStride(a.offsets[0], index);
  return {};
}

struct LockSlot {
  BlockArgument base;
  int64_t stride;
  int64_t offset;
};

std::optional<LockSlot> lockSlot(Value ptr, Value group, int64_t groups) {
  Address a = address(ptr);
  if (!a.base)
    return std::nullopt;
  int64_t stride = 0, offset = 0;
  for (Value v : a.offsets) {
    if (auto constant = integer(v)) {
      // Addresses for this pattern use nonnegative i32 offsets.
      if (*constant < 0 || *constant > INT32_MAX - offset)
        return std::nullopt;
      offset += *constant;
      continue;
    }
    if (stride)
      return std::nullopt;
    stride = v == group ? 1 : integer(indexStride(v, group)).value_or(0);
    if (stride <= 0 || stride > INT32_MAX)
      return std::nullopt;
  }
  if (!stride || stride * (groups - 1) + offset > INT32_MAX)
    return std::nullopt;
  return LockSlot{a.base, stride, offset};
}

bool isExchange(AtomicRMWOp op, Value pointer, int64_t value) {
  return op && op.getAtomicRmwOp() == RMWOp::XCHG &&
         op.getResult().use_empty() && isConstant(op.getVal(), value) &&
         (!op.getMask() || matchPattern(op.getMask(), m_One())) &&
         sameAddress(op.getPtr(), pointer);
}

bool isCast(Operation *op) {
  return isa_and_nonnull<FpToFpOp, arith::ExtFOp, arith::TruncFOp>(op);
}

Value stripCast(Value v) {
  auto *op = v.getDefiningOp();
  return isCast(op) ? op->getOperand(0) : v;
}

bool plainLoad(LoadOp load) {
  if (!load || load.getIsVolatile() || load.getCache() != CacheModifier::NONE ||
      !load.getBoundaryCheck().empty())
    return false;
  auto tensor = dyn_cast<RankedTensorType>(load.getType());
  return !tensor || tensor.getElementType().isIntOrFloat();
}

struct Partial {
  StoreOp store;
  Value contribution;
  arith::AddFOp add;
  LoadOp previous;
  Value previousOperand;
};

struct Match {
  FuncOp function;
  GetProgramIdOp row;
  arith::RemSIOp group;
  int64_t groups;
  Value columns;
  int64_t width;
  Value n;
  scf::WhileOp acquire;
  SmallVector<Partial, 2> partials;
  SmallVector<ReduceOp, 2> reductions;
};

// Match one entry block with the CAS loop and Count branch as its only
// top-level regions other than the row's two scalar reductions.
// The operator provides private, zeroed lock/count buffers, disjoint
// memory ranges and valid offsets, and a one-dimensional grid with one
// original program per row and a column tile covering the full row.
std::optional<Match> match(FuncOp function) {
  auto module = function->getParentOfType<ModuleOp>();
  if (!module || module->hasAttr(kProgramGridTransformsAttr) ||
      module->hasAttr("hacc.coalesce_factor") ||
      !llvm::hasSingleElement(*module.getBody()) ||
      !llvm::hasSingleElement(function.getBody()) || !function.isPublic() ||
      function.getFunctionType().getNumResults())
    return std::nullopt;

  Match m{};
  m.function = function;
  Block &entry = function.getBody().front();
  StoreOp dx;
  for (Operation &op : entry) {
    if (auto row = dyn_cast<GetProgramIdOp>(&op)) {
      if (m.row || static_cast<int>(row.getAxis()) != 0)
        return std::nullopt;
      m.row = row;
    } else if (auto rem = dyn_cast<arith::RemSIOp>(&op)) {
      if (m.group || !m.row || rem.getLhs() != m.row.getResult())
        return std::nullopt;
      m.group = rem;
    } else if (auto columns = dyn_cast<MakeRangeOp>(&op)) {
      if (m.columns || columns.getStart() != 0)
        return std::nullopt;
      m.columns = columns;
      m.width = columns.getEnd();
    } else if (auto store = dyn_cast<StoreOp>(&op)) {
      if (dx)
        return std::nullopt;
      dx = store;
    } else if (auto load = dyn_cast<LoadOp>(&op)) {
      if (!plainLoad(load))
        return std::nullopt;
    } else if (auto reduce = dyn_cast<ReduceOp>(&op)) {
      if (!isa_and_nonnull<arith::AddFOp>(reduce.getSingleCombiner()) ||
          reduce.getAxis() != 0 || !reduce->getResult(0).getType().isF32())
        return std::nullopt;
      m.reductions.push_back(reduce);
    } else if (auto acquire = dyn_cast<scf::WhileOp>(&op)) {
      m.acquire = acquire;
      break;
    } else if (!isa<AddPtrOp, SplatOp, FpToFpOp>(&op) &&
               !(op.getDialect() &&
                 op.getDialect()->getNamespace() == "arith" &&
                 isMemoryEffectFree(&op) && !op.getNumRegions())) {
      return std::nullopt;
    }
  }
  if (!m.row || !m.group || !m.columns || !dx || !m.acquire ||
      !llvm::isPowerOf2_64(m.width) || m.reductions.size() != 2)
    return std::nullopt;
  auto groups = integer(m.group.getRhs());
  if (!groups || *groups <= 0)
    return std::nullopt;
  m.groups = *groups;

  auto mask = dx.getMask() ? dx.getMask().getDefiningOp<arith::CmpIOp>()
                           : arith::CmpIOp();
  if (!mask || mask.getPredicate() != arith::CmpIPredicate::slt ||
      mask.getLhs() != m.columns)
    return std::nullopt;
  auto n = mask.getRhs().getDefiningOp<SplatOp>();
  if (!n)
    return std::nullopt;
  m.n = n.getSrc();
  if (!m.n.getType().isInteger(32) || !isa<BlockArgument>(m.n) ||
      !vectorStride(address(dx.getPtr()), m.row, m.columns) ||
      !dx.getBoundaryCheck().empty())
    return std::nullopt;

  if (m.acquire->getNumOperands() || m.acquire->getNumResults() ||
      !llvm::hasSingleElement(m.acquire.getAfter().front()))
    return std::nullopt;
  Block &test = m.acquire.getBefore().front();
  auto cas = dyn_cast<AtomicCASOp>(test.front());
  if (!cas || !isConstant(cas.getCmp(), 0) || !isConstant(cas.getVal(), 1) ||
      !cas.getResult().hasOneUse() || test.getOperations().size() != 3)
    return std::nullopt;
  auto condition = cast<scf::ConditionOp>(test.getTerminator());
  auto retry = condition.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!equalTo(retry, cas, 1) || retry->getBlock() != &test)
    return std::nullopt;
  auto lock = lockSlot(cas.getPtr(), m.group, m.groups);
  auto count = dyn_cast_or_null<LoadOp>(m.acquire->getNextNode());
  if (!lock || !plainLoad(count) || count.getMask())
    return std::nullopt;
  auto counter = lockSlot(count.getPtr(), m.group, m.groups);
  if (!counter || counter->base != lock->base ||
      counter->stride != lock->stride)
    return std::nullopt;
  int64_t delta = std::abs(counter->offset - lock->offset);
  if (delta % lock->stride == 0 && delta / lock->stride < m.groups)
    return std::nullopt;

  auto cmp = dyn_cast_or_null<arith::CmpIOp>(count->getNextNode());
  auto branch =
      cmp ? dyn_cast_or_null<scf::IfOp>(cmp->getNextNode()) : scf::IfOp();
  if (!equalTo(cmp, count, 0) || !branch || branch.getCondition() != cmp ||
      branch->getNumResults() != 2 ||
      branch.getThenRegion().front().getOperations().size() != 2 ||
      !isExchange(dyn_cast<AtomicRMWOp>(branch.getThenRegion().front().front()),
                  count.getPtr(), 1))
    return std::nullopt;

  llvm::SmallPtrSet<Operation *, 8> updates;
  llvm::SmallPtrSet<void *, 8> writtenRoots;
  writtenRoots.insert(lock->base.getAsOpaquePointer());
  if (!writtenRoots.insert(address(dx.getPtr()).base.getAsOpaquePointer())
           .second)
    return std::nullopt;
  Operation *next = branch->getNextNode();
  auto first = cast<scf::YieldOp>(branch.thenBlock()->getTerminator());
  auto later = cast<scf::YieldOp>(branch.elseBlock()->getTerminator());
  for (unsigned i = 0; i < 2; ++i) {
    Value result = branch.getResult(i);
    if (!result.hasOneUse())
      return std::nullopt;
    Operation *user = *result.getUsers().begin();
    if (isCast(user) && user->getResult(0).hasOneUse() && user == next) {
      next = user->getNextNode();
      user = *user->getResult(0).getUsers().begin();
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store != next || store.getMask() != dx.getMask() ||
        !store.getBoundaryCheck().empty())
      return std::nullopt;
    next = store->getNextNode();

    Address slot = address(store.getPtr());
    Value step = vectorStride(slot, m.group, m.columns);
    auto padding = step ? step.getDefiningOp<arith::AddIOp>() : arith::AddIOp();
    if (!step ||
        !(step == m.n || integer(step).value_or(0) >= m.width ||
          (padding && padding.getLhs() == m.n &&
           integer(padding.getRhs()).value_or(-1) >= 0)) ||
        !writtenRoots.insert(slot.base.getAsOpaquePointer()).second)
      return std::nullopt;

    Value contribution = first.getOperand(i);
    auto add = later.getOperand(i).getDefiningOp<arith::AddFOp>();
    if (!contribution.getDefiningOp() ||
        contribution.getDefiningOp()->getBlock() != &entry ||
        !contribution.getDefiningOp()->isBeforeInBlock(m.acquire) || !add ||
        add->getBlock() != branch.elseBlock())
      return std::nullopt;
    Value old = add.getLhs() == contribution   ? add.getRhs()
                : add.getRhs() == contribution ? add.getLhs()
                                               : Value();
    if (!old)
      return std::nullopt;
    auto load = stripCast(old).getDefiningOp<LoadOp>();
    if (!plainLoad(load) || load->getBlock() != branch.elseBlock() ||
        !sameAddress(load.getPtr(), store.getPtr()) ||
        load.getMask() != store.getMask())
      return std::nullopt;
    updates.insert(load);
    updates.insert(add);
    if (old != load.getResult()) {
      if (old.getDefiningOp()->getBlock() != branch.elseBlock())
        return std::nullopt;
      updates.insert(old.getDefiningOp());
    }
    m.partials.push_back({store, contribution, add, load, old});
  }
  for (Operation &op : branch.elseBlock()->without_terminator())
    if (!updates.contains(&op))
      return std::nullopt;
  if (!isExchange(dyn_cast_or_null<AtomicRMWOp>(next), cas.getPtr(), 0) ||
      !isa_and_nonnull<ReturnOp>(next->getNextNode()))
    return std::nullopt;

  for (Operation &op : entry) {
    if (&op == m.acquire.getOperation())
      break;
    if (auto load = dyn_cast<LoadOp>(&op)) {
      BlockArgument root = address(load.getPtr()).base;
      if (!root || root.getOwner() != &entry ||
          writtenRoots.contains(root.getAsOpaquePointer()))
        return std::nullopt;
    }
  }
  return m;
}

RankedTensorType tileType(RankedTensorType type, int64_t width) {
  return RankedTensorType::get({width}, type.getElementType(),
                               type.getEncoding());
}

Operation *cloneTiled(IRRewriter &rw, Operation &op, IRMapping &map,
                      int64_t width) {
  Operation *copy = rw.clone(op, map);
  for (auto [result, old] : llvm::zip(copy->getResults(), op.getResults()))
    if (auto type = dyn_cast<RankedTensorType>(old.getType()))
      result.setType(tileType(type, width));
  if (auto dense = copy->getAttrOfType<DenseElementsAttr>("value"))
    copy->setAttr("value",
                  DenseElementsAttr::get(
                      cast<RankedTensorType>(copy->getResult(0).getType()),
                      dense.getSplatValue<Attribute>()));
  return copy;
}

// Clone the entry-block calculation needed by a result in the current tile.
Value cloneValue(IRRewriter &rw, Value value, IRMapping &map, int64_t width) {
  if (auto mapped = map.lookupOrNull(value))
    return mapped;
  auto *op = value.getDefiningOp();
  if (!op)
    return value;
  for (Value operand : op->getOperands())
    cloneValue(rw, operand, map, width);
  cloneTiled(rw, *op, map, width);
  return map.lookup(value);
}

bool slotWidthsAreSupported(Match &m) {
  bool valid = true;
  for (Operation &op : m.function.getBody().front()) {
    if (&op == m.acquire.getOperation())
      break;
    for (Type type : op.getResultTypes()) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        valid &= tensor.getRank() == 1 && tensor.getDimSize(0) == m.width &&
                 !tensor.getEncoding();
    }
    if (auto dense = op.getAttrOfType<DenseElementsAttr>("value"))
      valid &= dense.isSplat();
  }
  return valid;
}

bool reductionsCanBeRetiled(const Match &match) {
  // A later sum may consume the full-row result of the earlier sum. The
  // tile-local clone would see only that tile's contribution.
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  SetVector<Operation *> dependencies;
  return succeeded(
             getBackwardSlice(match.reductions[1], &dependencies, options)) &&
         !dependencies.contains(match.reductions[0]);
}

Value makeColumns(IRRewriter &rw, Location loc, Value original, Value offset,
                  int64_t width) {
  auto type = tileType(cast<RankedTensorType>(original.getType()), width);
  auto lanes = rw.create<MakeRangeOp>(loc, type, 0, width);
  if (!offset)
    return lanes;
  auto base = rw.create<SplatOp>(loc, type, offset);
  return rw.create<arith::AddIOp>(loc, base, lanes);
}

LogicalResult rewrite(ModuleOp module, Match &m, int64_t width) {
  FuncOp function = m.function;
  if (failed(addProgramGridHiddenExtentArguments(function)))
    return failure();
  Block &entry = function.getBody().front();
  SmallVector<Operation *> old;
  for (Operation &op : entry.without_terminator())
    old.push_back(&op);

  IRRewriter rw(function.getContext());
  rw.setInsertionPointToStart(&entry);
  Location loc = function.getLoc();
  Value group = rw.create<GetProgramIdOp>(loc, m.row.getAxis());
  Value M = function.getArgument(function.getNumArguments() - 2);
  Value group64 = rw.create<arith::ExtSIOp>(loc, rw.getI64Type(), group);
  Value rows64 = rw.create<arith::ExtSIOp>(loc, rw.getI64Type(), M);
  Value step64 = rw.create<arith::ConstantIntOp>(loc, m.groups, 64);
  Value zero = rw.create<arith::ConstantIntOp>(loc, 0, 32);
  Value end = rw.create<arith::ConstantIntOp>(loc, m.width, 32);
  Value step = rw.create<arith::ConstantIntOp>(loc, width, 32);

  auto processTile = [&](Value offset) {
    Value cols = makeColumns(rw, loc, m.columns, offset, width);
    SmallVector<Value, 2> init;
    for (Partial &p : m.partials) {
      auto type =
          tileType(cast<RankedTensorType>(p.store.getValue().getType()), width);
      init.push_back(rw.create<arith::ConstantOp>(
          loc,
          DenseElementsAttr::get(type, rw.getZeroAttr(type.getElementType()))));
    }
    auto rows = rw.create<scf::ForOp>(
        loc, group64, rows64, step64, init,
        [&](OpBuilder &, Location, Value iv, ValueRange previous) {
          Value row = rw.create<arith::TruncIOp>(loc, rw.getI32Type(), iv);
          IRMapping map;
          map.map(m.row.getResult(), row);
          map.map(m.group.getResult(), group);
          map.map(m.columns, cols);

          // For a narrower tile, recompute the two scalar row reductions
          // over all column tiles before using them in the row's dx formula.
          if (offset) {
            Value fzero =
                rw.create<arith::ConstantOp>(loc, rw.getF32FloatAttr(0));
            auto stats = rw.create<scf::ForOp>(
                loc, zero, end, step, ValueRange{fzero, fzero},
                [&](OpBuilder &, Location, Value off, ValueRange sums) {
                  IRMapping slice;
                  slice.map(m.row.getResult(), row);
                  slice.map(m.group.getResult(), group);
                  slice.map(m.columns,
                            makeColumns(rw, loc, m.columns, off, width));
                  SmallVector<Value, 2> updated;
                  for (auto [i, reduce] : llvm::enumerate(m.reductions)) {
                    Value partial =
                        cloneValue(rw, reduce->getResult(0), slice, width);
                    updated.push_back(
                        rw.create<arith::AddFOp>(loc, sums[i], partial));
                  }
                  rw.create<scf::YieldOp>(loc, updated);
                });
            for (auto [i, reduce] : llvm::enumerate(m.reductions))
              map.map(reduce->getResult(0), stats.getResult(i));
          }
          for (Operation *op : old) {
            if (op == m.acquire.getOperation())
              break;
            if (op == m.row.getOperation() || op == m.group.getOperation() ||
                op == m.columns.getDefiningOp() ||
                (offset && isa<ReduceOp>(op)))
              continue;
            cloneTiled(rw, *op, map, width);
          }

          Value isFirst = rw.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, iv, group64);
          SmallVector<Value, 2> updated;
          for (auto [i, p] : llvm::enumerate(m.partials)) {
            Value contribution = map.lookup(p.contribution);
            // Materialize the first write and the later accumulation on
            // separate paths. A select between the two tensor buffers can
            // alias the carried sum when A5 multibuffering is enabled.
            auto update = rw.create<scf::IfOp>(
                loc, TypeRange{contribution.getType()}, isFirst, true);
            {
              OpBuilder::InsertionGuard guard(rw);
              rw.setInsertionPointToStart(update.thenBlock());
              rw.create<scf::YieldOp>(loc, contribution);
              rw.setInsertionPointToStart(update.elseBlock());
              map.map(p.previous.getResult(), previous[i]);
              if (p.previousOperand != p.previous.getResult())
                cloneTiled(rw, *p.previousOperand.getDefiningOp(), map, width);
              Value sum = cloneTiled(rw, *p.add.getOperation(), map, width)
                              ->getResult(0);
              rw.create<scf::YieldOp>(loc, sum);
            }
            Value value = update.getResult(0);
            if (isCast(p.store.getValue().getDefiningOp())) {
              auto *castOp = p.store.getValue().getDefiningOp();
              map.map(castOp->getOperand(0), value);
              value = cloneTiled(rw, *castOp, map, width)->getResult(0);
            }
            updated.push_back(value);
          }
          rw.create<scf::YieldOp>(loc, updated);
        });

    IRMapping out;
    out.map(m.group.getResult(), group);
    out.map(m.columns, cols);
    for (auto [i, p] : llvm::enumerate(m.partials)) {
      cloneValue(rw, p.store.getPtr(), out, width);
      cloneValue(rw, p.store.getMask(), out, width);
      out.map(p.store.getValue(), rows.getResult(i));
      rw.clone(*p.store.getOperation(), out);
    }
  };

  if (width == m.width)
    processTile({});
  else
    rw.create<scf::ForOp>(loc, zero, end, step, ValueRange{},
                          [&](OpBuilder &, Location, Value offset, ValueRange) {
                            processTile(offset);
                            rw.create<scf::YieldOp>(loc);
                          });

  for (Operation *op : llvm::reverse(old))
    rw.eraseOp(op);

  ProgramGridTransformContract contract;
  contract.transforms.push_back({0, 0, m.groups, false, false, true});
  return setProgramGridTransformContract(module, contract);
}

// TTIR liveness does not account for every buffer that A5 bufferization
// materializes. Reserve two slots for row DMA, row output, and loop carries,
// including FP32 staging for narrow floating-point inputs and partials.
uint64_t estimateStagingBytes(Match &m, int64_t width) {
  uint64_t slots = 0;
  auto reserve = [&](Value value) {
    if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
      unsigned bits = type.getElementType().getIntOrFloatBitWidth();
      slots += std::max(4u, (bits + 7) / 8);
    }
  };
  for (Operation &op : m.function.getBody().front()) {
    if (&op == m.acquire.getOperation())
      break;
    if (auto load = dyn_cast<LoadOp>(&op))
      reserve(load.getResult());
    if (auto store = dyn_cast<StoreOp>(&op))
      reserve(store.getValue());
  }
  for (Partial &partial : m.partials)
    reserve(partial.store.getValue());
  return slots * width * kUBMultiBufferReplication;
}

class LockRemovalPlan final : public RewritePlan {
public:
  LockRemovalPlan(FuncOp function, unsigned epoch,
                  OwningOpRef<ModuleOp> prepared)
      : function(function), epoch(epoch), prepared(std::move(prepared)) {}
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::CoalescAtomicLockRemoval;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return function; }
  unsigned getCreationEpoch() const override { return epoch; }
  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    return success(context.getFunction() == function);
  }
  LogicalResult apply(IRRewriter &) override {
    auto replacement = cast<FuncOp>(&prepared->getBody()->front());
    if (failed(commitProgramGridFunctionFromSandbox(function, replacement)))
      return failure();
    function->getParentOfType<ModuleOp>()->setAttr(
        kProgramGridTransformsAttr,
        prepared->getOperation()->getAttr(kProgramGridTransformsAttr));
    return success();
  }

private:
  FuncOp function;
  unsigned epoch;
  OwningOpRef<ModuleOp> prepared;
};

class LockRemovalRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::CoalescAtomicLockRemoval;
  }
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::ResourceCost;
  }
  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    FuncOp function = context.getFunction();
    auto candidate = match(function);
    auto budget = context.getResourceCostAnalysis()
                      .getResourceSnapshot()
                      .getSafeUBBudget();
    if (!candidate || !budget || !slotWidthsAreSupported(*candidate))
      return success();
    auto module = function->getParentOfType<ModuleOp>();
    bool checkedRetile = false;
    for (int64_t width = candidate->width; width > 0; width /= 2) {
      if (estimateStagingBytes(*candidate, width) > *budget)
        continue;
      if (width < candidate->width && !checkedRetile) {
        if (!reductionsCanBeRetiled(*candidate))
          break;
        checkedRetile = true;
      }
      OwningOpRef<ModuleOp> sandbox(ModuleOp::create(function.getLoc()));
      sandbox->getOperation()->setAttrs(module->getAttrs());
      sandbox->getBody()->push_back(function->clone());
      auto cloned = cast<FuncOp>(&sandbox->getBody()->front());
      auto current = match(cloned);
      if (!current || failed(rewrite(*sandbox, *current, width)) ||
          failed(verify(sandbox->getOperation())))
        continue;
      PassManager cleanup(function.getContext());
      cleanup.addPass(createCanonicalizerPass());
      cleanup.addPass(createCSEPass());
      if (failed(cleanup.run(*sandbox)) ||
          failed(verify(sandbox->getOperation())))
        continue;
      LiveByteEstimate estimate = estimatePeakLiveBytes(cloned.getOperation());
      if (estimate.known && !estimate.overflow &&
          estimate.peakPipelinedLiveBytes <= *budget) {
        plans.push_back(std::make_unique<LockRemovalPlan>(
            function, context.getEpoch(), std::move(sandbox)));
        break;
      }
    }
    return success();
  }
};
} // namespace

std::unique_ptr<GraphOptimizationRule>
mlir::triton::cfg::createCoalescAtomicLockRemovalRule() {
  return std::make_unique<LockRemovalRule>();
}
