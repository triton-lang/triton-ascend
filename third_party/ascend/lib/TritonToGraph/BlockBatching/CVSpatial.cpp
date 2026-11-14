// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/CVSpatial.h"
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::cfg {
namespace {
// Coordinate concatenation, rather than a batch of cloned rank-two dots:
//   [A0; ...; AF-1] B = [A0 B; ...; AF-1 B]
//   A [B0 ... BF-1] = [A B0 ... A BF-1].
// The K dimension/order and every original mask/elementwise operation remain
// unchanged. The guard checks the union of ORIGINAL program memory domains.
// A full group makes that union exactly the enlarged tile's coordinates.
class SpatialBuilder {
  ModuleOp module;
  triton::FuncOp function;
  int axis;
  int64_t factor;
  IRRewriter rw;
  IRMapping mapped;
  SmallVector<Operation *> originals;
  SmallVector<Operation *> nested;
  llvm::DenseMap<Value, int> expanded;
  llvm::DenseMap<Block *, llvm::DenseMap<std::pair<Value, int>, Value>>
      uniformCaches;
  std::string &reason;
  std::string &lowering;
  triton::GetProgramIdOp pid;
  arith::AddIOp coordinate;
  int64_t tile = 0;

  LogicalResult reject(llvm::StringRef why) {
    reason = why.str();
    return failure();
  }
  int dim(Value v) const {
    auto it = expanded.find(v);
    return it == expanded.end() ? -1 : it->second;
  }
  RankedTensorType wider(Type old, int d) {
    auto t = dyn_cast<RankedTensorType>(old);
    if (!t || d < 0 || d >= t.getRank())
      return {};
    SmallVector<int64_t> shape(t.getShape());
    shape[d] *= factor; // caller's tensor budget already bounds this product
    return RankedTensorType::get(shape, t.getElementType());
  }
  bool elementwise(Operation *op) {
    auto desc = lookupBatchOp(op->getName().getStringRef());
    return desc &&
           (desc->model == BatchModel::Elementwise ||
            desc->model == BatchModel::IndexCast || isa<triton::AddPtrOp>(op));
  }
  Operation *clone(Operation *op, ValueRange operands, Type result = {}) {
    IRMapping args;
    for (auto [old, next] : llvm::zip(op->getOperands(), operands))
      args.map(old, next);
    auto copy = rw.clone(*op, args);
    if (result)
      copy->getResult(0).setType(result);
    return copy;
  }

  // Independent of pid is insufficient: arange(M) may still vary WITHIN a
  // tile. Only resize values proved invariant along the concatenated axis.
  // Specializations are per use; a CSE-shared value can also feed unchanged K
  // or the other matrix operand. Never change its original mapping globally.
  Value widenUniform(Value v, int d, unsigned depth = 0) {
    // A value materialized in one region does not dominate a sibling region.
    auto &uniformCache = uniformCaches[rw.getInsertionBlock()];
    auto key = std::make_pair(v, d);
    if (auto it = uniformCache.find(key); it != uniformCache.end())
      return it->second;
    auto type = wider(v.getType(), d);
    Operation *op = v.getDefiningOp();
    if (!type || !op || dim(v) != -1 || depth > 128)
      return {};
    Value result;
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto dense = dyn_cast<DenseElementsAttr>(c.getValue());
      if (dense && dense.isSplat())
        result = rw.create<arith::ConstantOp>(
            op->getLoc(), type,
            DenseElementsAttr::get(type, dense.getSplatValue<Attribute>()));
    } else if (isa<triton::SplatOp>(op)) {
      result =
          clone(op, {mapped.lookup(op->getOperand(0))}, type)->getResult(0);
    } else if (auto b = dyn_cast<triton::BroadcastOp>(op)) {
      auto src = cast<RankedTensorType>(b.getSrc().getType());
      Value operand = src.getShape()[d] == 1
                          ? mapped.lookup(b.getSrc())
                          : widenUniform(b.getSrc(), d, depth + 1);
      if (operand)
        result = clone(op, {operand}, type)->getResult(0);
    } else if (auto e = dyn_cast<triton::ExpandDimsOp>(op)) {
      if (d == e.getAxis()) {
        result = rw.create<triton::BroadcastOp>(op->getLoc(), type,
                                                mapped.lookup(v));
      } else if (Value src = widenUniform(e.getSrc(), d - (d > e.getAxis()),
                                          depth + 1)) {
        result = clone(op, {src}, type)->getResult(0);
      }
    } else if (auto t = dyn_cast<triton::TransOp>(op)) {
      if (Value src = widenUniform(t.getSrc(), t.getOrder()[d], depth + 1))
        result = clone(op, {src}, type)->getResult(0);
    } else if (elementwise(op) && op->getNumResults() == 1) {
      SmallVector<Value> operands;
      for (Value a : op->getOperands()) {
        if (!isa<RankedTensorType>(a.getType())) {
          operands.push_back(mapped.lookup(a));
          continue;
        }
        auto at = cast<RankedTensorType>(a.getType());
        if (at.getShape() != cast<RankedTensorType>(v.getType()).getShape())
          return {};
        Value next = widenUniform(a, d, depth + 1);
        if (!next)
          return {};
        operands.push_back(next);
      }
      result = clone(op, operands, type)->getResult(0);
    }
    if (result)
      uniformCache[key] = result;
    return result;
  }

  LogicalResult findCoordinate() {
    unsigned dots = 0;
    for (Operation *op : nested) {
      dots += isa<triton::DotOp>(op);
      if (auto p = dyn_cast<triton::GetProgramIdOp>(op);
          p && p.getAxisAsInt() == axis) {
        if (pid)
          return reject("CV_SPATIAL_MULTIPLE_PROGRAM_IDS");
        pid = p;
      }
    }
    if (dots != 1 || !pid)
      return reject("CV_SPATIAL_REQUIRES_SINGLE_DOT");
    llvm::DenseSet<Operation *> prefix{pid.getOperation()};
    for (Operation *op : nested) {
      auto add = dyn_cast<arith::AddIOp>(op);
      if (!add)
        continue;
      for (unsigned side = 0; side != 2; ++side) {
        auto s = add->getOperand(side).getDefiningOp<triton::SplatOp>();
        auto r = add->getOperand(1 - side).getDefiningOp<triton::MakeRangeOp>();
        auto mul =
            s ? s.getSrc().getDefiningOp<arith::MulIOp>() : arith::MulIOp{};
        if (!mul || !r || r.getStart() != 0)
          continue;
        Value c = mul.getLhs() == pid.getResult()   ? mul.getRhs()
                  : mul.getRhs() == pid.getResult() ? mul.getLhs()
                                                    : Value{};
        auto constant = c ? c.getDefiningOp<arith::ConstantIntOp>()
                          : arith::ConstantIntOp{};
        if (!constant || constant.value() != r.getEnd())
          continue;
        if (coordinate)
          return reject("CV_SPATIAL_MULTIPLE_COORDINATES");
        coordinate = add;
        tile = r.getEnd();
        prefix.insert(mul.getOperation());
        prefix.insert(s.getOperation());
      }
    }
    if (!coordinate || tile <= 0 || tile > (1 << 20) / factor)
      return reject("CV_SPATIAL_COORDINATE_NOT_PROVEN");
    // No pid-dependent scalar bias, indirect batch offset, side effect, or
    // second interpretation of this pid may escape the recognized chain.
    for (Operation *op : prefix)
      for (Operation *user : op->getUsers())
        if (user != coordinate.getOperation() && !prefix.contains(user))
          return reject("CV_SPATIAL_PROGRAM_USE_NOT_PROVEN");
    return success();
  }

  // Infer one concatenated dimension through SSA and region recurrences
  // BEFORE rewriting. Joining init/yield at a loop is a monotone fixed point:
  // a uniform zero init may feed a lane-varying accumulator on later trips.
  // It must therefore be widened even on the first (or zero) iteration.
  LogicalResult inferDimensions() {
    expanded[coordinate.getResult()] = 0;
    bool changed = false;
    auto mark = [&](Value value, int d) -> LogicalResult {
      if (d < 0)
        return success();
      if (!wider(value.getType(), d))
        return reject("CV_SPATIAL_SCALAR_RESULT_NOT_SUPPORTED");
      int old = dim(value);
      if (old >= 0 && old != d)
        return reject("CV_SPATIAL_MIXED_AXES");
      if (old < 0) {
        expanded[value] = d;
        changed = true;
      }
      return success();
    };
    auto join = [&](ValueRange values, int &d) -> LogicalResult {
      for (Value value : values)
        if (dim(value) >= 0) {
          if (d >= 0 && d != dim(value))
            return reject("CV_SPATIAL_MIXED_AXES");
          d = dim(value);
        }
      return success();
    };
    for (unsigned iteration = 0; iteration != 64; ++iteration) {
      changed = false;
      for (Operation *op : nested) {
        if (op == coordinate.getOperation() || isa<scf::YieldOp>(op))
          continue;
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          if (dim(loop.getLowerBound()) >= 0 ||
              dim(loop.getUpperBound()) >= 0 || dim(loop.getStep()) >= 0)
            return reject("CV_SPATIAL_VARYING_CONTROL");
          auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
          for (auto [init, arg, end, result] :
               llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                         yield.getOperands(), loop.getResults())) {
            int d = -1;
            if (failed(join({init, arg, end, result}, d)) ||
                failed(mark(arg, d)) || failed(mark(result, d)))
              return failure();
          }
          continue;
        }
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
          if (dim(branch.getCondition()) >= 0)
            return reject("CV_SPATIAL_VARYING_CONTROL");
          for (auto [i, result] : llvm::enumerate(branch.getResults())) {
            int d = dim(result);
            for (Region &region : branch->getRegions())
              if (!region.empty()) {
                Value end = region.front().getTerminator()->getOperand(i);
                if (failed(join({end}, d)))
                  return failure();
              }
            if (failed(mark(result, d)))
              return failure();
          }
          continue;
        }
        if (auto dot = dyn_cast<triton::DotOp>(op)) {
          int a = dim(dot.getA()), b = dim(dot.getB());
          if (a < 0 && b < 0)
            continue; // An outer recurrence may become known next iteration.
          if (!((a == 0 && b == -1) || (a == -1 && b == 1)))
            return reject("CV_SPATIAL_DOT_AXIS_NOT_PROVEN");
          int d = a == 0 ? 0 : 1;
          if (dim(dot.getC()) >= 0 && dim(dot.getC()) != d)
            return reject("CV_SPATIAL_ACCUMULATOR_AXIS_NOT_PROVEN");
          if (failed(mark(dot.getResult(), d)))
            return failure();
          continue;
        }
        int d = -1;
        if (failed(join(op->getOperands(), d)))
          return failure();
        if (d < 0)
          continue;
        if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
          d += expand.getAxis() <= d;
        else if (auto trans = dyn_cast<triton::TransOp>(op)) {
          auto order = trans.getOrder();
          d = std::find(order.begin(), order.end(), d) - order.begin();
        } else if (auto broadcast = dyn_cast<triton::BroadcastOp>(op)) {
          if (cast<RankedTensorType>(broadcast.getSrc().getType())
                  .getDimSize(d) != broadcast.getType().getDimSize(d))
            return reject("CV_SPATIAL_BROADCAST_AXIS_NOT_PROVEN");
        } else if (!(elementwise(op) ||
                     isa<triton::LoadOp, triton::StoreOp>(op)))
          return reject("CV_SPATIAL_OPERATION_NOT_SUPPORTED");
        for (Value result : op->getResults())
          if (failed(mark(result, d)))
            return failure();
      }
      if (!changed)
        return success();
    }
    return reject("CV_SPATIAL_RECURRENCE_BUDGET_EXCEEDED");
  }

  Value materialize(Value old, int d) {
    return d < 0 || dim(old) == d ? mapped.lookupOrNull(old)
                                  : widenUniform(old, d);
  }

  LogicalResult rewriteRegion(Block &source, Block &target) {
    OpBuilder::InsertionGuard guard(rw);
    rw.setInsertionPointToEnd(&target);
    for (Operation &op : source)
      if (failed(rewrite(&op)))
        return failure();
    return success();
  }

  LogicalResult rewriteControl(Operation *op) {
    OpBuilder::InsertionGuard guard(rw);
    SmallVector<Type> types;
    for (Value result : op->getResults())
      types.push_back(dim(result) < 0
                          ? result.getType()
                          : Type(wider(result.getType(), dim(result))));
    Operation *replacement;
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      auto created = rw.create<scf::IfOp>(op->getLoc(), types,
                                          mapped.lookup(branch.getCondition()),
                                          !branch.getElseRegion().empty());
      created->setAttrs(op->getAttrs());
      for (auto [from, to] :
           llvm::zip(op->getRegions(), created->getRegions())) {
        if (from.empty())
          continue;
        while (!to.front().empty())
          rw.eraseOp(&to.front().back());
        if (failed(rewriteRegion(from.front(), to.front())))
          return failure();
      }
      replacement = created;
    } else {
      auto loop = cast<scf::ForOp>(op);
      SmallVector<Value> init;
      for (auto [value, result] :
           llvm::zip(loop.getInitArgs(), loop.getResults())) {
        Value next = materialize(value, dim(result));
        if (!next)
          return reject("CV_SPATIAL_UNIFORM_INIT_NOT_PROVEN");
        init.push_back(next);
      }
      Value lower = mapped.lookup(loop.getLowerBound());
      Value upper = mapped.lookup(loop.getUpperBound());
      Value step = mapped.lookup(loop.getStep());
      // Wrapping an accumulator recurrence forces the consumer to move its
      // f32 result across a new if boundary (L0C -> UB). Only invariant or
      // unclassified yields need the explicit zero-trip workaround.
      scf::IfOp gate;
      if (cvLoopNeedsZeroTripGuard(loop)) {
        Value nonempty = rw.create<arith::CmpIOp>(
            op->getLoc(), arith::CmpIPredicate::slt, lower, upper);
        gate = rw.create<scf::IfOp>(op->getLoc(), types, nonempty, true);
        for (Region &region : gate->getRegions())
          while (!region.front().empty())
            rw.eraseOp(&region.front().back());
        rw.setInsertionPointToEnd(&gate.getThenRegion().front());
      }
      auto created =
          rw.create<scf::ForOp>(op->getLoc(), lower, upper, step, init);
      created->setAttrs(op->getAttrs());
      while (!created.getBody()->empty())
        rw.eraseOp(&created.getBody()->back());
      mapped.map(loop.getInductionVar(), created.getInductionVar());
      for (auto [from, to] :
           llvm::zip(loop.getRegionIterArgs(), created.getRegionIterArgs()))
        mapped.map(from, to);
      if (failed(rewriteRegion(*loop.getBody(), *created.getBody())))
        return failure();
      if (gate) {
        rw.setInsertionPointToEnd(&gate.getThenRegion().front());
        rw.create<scf::YieldOp>(op->getLoc(), created.getResults());
        rw.setInsertionPointToEnd(&gate.getElseRegion().front());
        rw.create<scf::YieldOp>(op->getLoc(), init);
        replacement = gate;
      } else {
        replacement = created;
      }
    }
    for (auto [from, to] :
         llvm::zip(op->getResults(), replacement->getResults()))
      mapped.map(from, to);
    return success();
  }

  LogicalResult rewrite(Operation *op) {
    if (op == coordinate.getOperation()) {
      auto loc = op->getLoc();
      auto t = wider(coordinate.getType(), 0);
      Value step = rw.create<arith::ConstantIntOp>(loc, tile * factor, 32);
      Value base =
          rw.create<arith::MulIOp>(loc, mapped.lookup(pid.getResult()), step);
      Value start = rw.create<triton::SplatOp>(loc, t, base);
      Value range = rw.create<triton::MakeRangeOp>(loc, t, 0, tile * factor);
      mapped.map(coordinate.getResult(),
                 rw.create<arith::AddIOp>(loc, start, range));
      expanded[coordinate.getResult()] = 0;
      return success();
    }
    if (auto count = dyn_cast<triton::GetNumProgramsOp>(op)) {
      mapped.map(count.getResult(),
                 function.getArgument(function.getNumArguments() - 3 +
                                      count.getAxisAsInt()));
      return success();
    }
    if (isa<scf::ForOp, scf::IfOp>(op))
      return rewriteControl(op);
    if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      Operation *parent = op->getParentOp();
      SmallVector<Value> operands;
      for (auto [value, result] :
           llvm::zip(yield.getOperands(), parent->getResults())) {
        Value next = materialize(value, dim(result));
        if (!next)
          return reject("CV_SPATIAL_UNIFORM_YIELD_NOT_PROVEN");
        operands.push_back(next);
      }
      rw.create<scf::YieldOp>(op->getLoc(), operands);
      return success();
    }
    int d = -1;
    for (Value v : op->getOperands())
      if (dim(v) != -1) {
        if (d != -1 && d != dim(v))
          return reject("CV_SPATIAL_MIXED_AXES");
        d = dim(v);
      }
    if (auto dot = dyn_cast<triton::DotOp>(op)) {
      int a = dim(dot.getA()), b = dim(dot.getB());
      if (!((a == 0 && b == -1) || (a == -1 && b == 1)))
        return reject("CV_SPATIAL_DOT_AXIS_NOT_PROVEN");
      d = a == 0 ? 0 : 1;
      if (dim(dot.getC()) != -1 && dim(dot.getC()) != d)
        return reject("CV_SPATIAL_ACCUMULATOR_AXIS_NOT_PROVEN");
      if (dim(dot.getC()) == -1) {
        auto c = dot.getC().getDefiningOp<arith::ConstantOp>();
        auto fp = c ? dyn_cast<DenseFPElementsAttr>(c.getValue())
                    : DenseFPElementsAttr{};
        auto integer = c ? dyn_cast<DenseIntElementsAttr>(c.getValue())
                         : DenseIntElementsAttr{};
        bool zero =
            (fp && fp.isSplat() && fp.getSplatValue<APFloat>().isZero()) ||
            (integer && integer.isSplat() &&
             integer.getSplatValue<APInt>().isZero());
        if (!zero)
          return reject("CV_SPATIAL_REQUIRES_ZERO_ACCUMULATOR");
      }
      Value acc = materialize(dot.getC(), d);
      if (!acc)
        return reject("CV_SPATIAL_UNIFORM_VALUE_NOT_PROVEN");
      auto copy =
          clone(op, {mapped.lookup(dot.getA()), mapped.lookup(dot.getB()), acc},
                wider(dot.getType(), d));
      mapped.map(dot.getResult(), copy->getResult(0));
      expanded[dot.getResult()] = d;
      lowering = d == 0 ? "spatial_m_v1" : "spatial_n_v1";
      return success();
    }
    if (auto store = dyn_cast<triton::StoreOp>(op);
        store && dim(store.getPtr()) == -1)
      return reject("CV_SPATIAL_STORE_NOT_CONCATENATED");
    if (d == -1) {
      rw.clone(*op, mapped);
      return success();
    }
    int resultDim = d;
    bool shape = false;
    if (auto e = dyn_cast<triton::ExpandDimsOp>(op)) {
      resultDim += e.getAxis() <= d;
      shape = true;
    } else if (auto t = dyn_cast<triton::TransOp>(op)) {
      auto order = t.getOrder();
      resultDim = std::find(order.begin(), order.end(), d) - order.begin();
      shape = true;
    } else if (auto b = dyn_cast<triton::BroadcastOp>(op)) {
      if (cast<RankedTensorType>(b.getSrc().getType()).getShape()[d] !=
          b.getType().getShape()[d])
        return reject("CV_SPATIAL_BROADCAST_AXIS_NOT_PROVEN");
      shape = true;
    } else if (!(elementwise(op) || isa<triton::LoadOp, triton::StoreOp>(op))) {
      // In particular, reductions must not silently cross original programs.
      return reject("CV_SPATIAL_OPERATION_NOT_SUPPORTED");
    }
    if (auto load = dyn_cast<triton::LoadOp>(op);
        load && dim(load.getPtr()) != d)
      return reject("CV_SPATIAL_LOAD_AXIS_NOT_PROVEN");
    SmallVector<Value> operands;
    for (Value v : op->getOperands()) {
      Value next = mapped.lookup(v);
      if (!shape && isa<RankedTensorType>(v.getType()) && dim(v) == -1)
        next = widenUniform(v, d);
      if (!next)
        return reject("CV_SPATIAL_UNIFORM_VALUE_NOT_PROVEN");
      operands.push_back(next);
    }
    if (op->getNumResults() > 1)
      return reject("CV_SPATIAL_OPERATION_NOT_SUPPORTED");
    Type result = op->getNumResults()
                      ? wider(op->getResult(0).getType(), resultDim)
                      : Type{};
    if (op->getNumResults() && !result)
      return reject("CV_SPATIAL_SCALAR_RESULT_NOT_SUPPORTED");
    auto copy = clone(op, operands, result);
    if (result) {
      mapped.map(op->getResult(0), copy->getResult(0));
      expanded[op->getResult(0)] = resultDim;
    }
    return success();
  }

public:
  SpatialBuilder(ModuleOp module, triton::FuncOp function, int axis,
                 int64_t factor, std::string &lowering, std::string &reason)
      : module(module), function(function), axis(axis), factor(factor),
        rw(module.getContext()), reason(reason), lowering(lowering) {}
  LogicalResult run() {
    for (Operation &op : function.getBody().front())
      originals.push_back(&op);
    function.walk([&](Operation *op) {
      if (op != function.getOperation())
        nested.push_back(op);
    });
    if (failed(findCoordinate()) || failed(inferDimensions()))
      return failure();
    if (failed(addProgramGridHiddenExtentArguments(function, 3)))
      return reject("HIDDEN_ARGUMENT_CONFLICT");
    for (Value arg : function.getArguments())
      mapped.map(arg, arg);
    rw.setInsertionPointToStart(&function.getBody().front());
    for (Operation *op : originals)
      if (failed(rewrite(op)))
        return failure();
    for (Operation *op : llvm::reverse(originals))
      rw.eraseOp(op);
    ProgramGridTransformContract contract;
    contract.version = 3;
    contract.transforms.push_back({0, axis, factor, false, false});
    if (failed(setProgramGridTransformContract(module, contract)))
      return reject("CV_SPATIAL_GRID_CONTRACT_FAILED");
    module->setAttr("hacc.block_mapping_owner",
                    rw.getStringAttr("block_batching"));
    function->setAttr("hacc.block_mapping_owner",
                      rw.getStringAttr("block_batching"));
    // This is an internal verification error, not a modeled eligibility miss.
    return verify(module);
  }
};
} // namespace

LogicalResult rewriteSpatialCV(ModuleOp module, triton::FuncOp function,
                               int axis, int64_t factor, std::string &lowering,
                               std::string &reason) {
  return SpatialBuilder(module, function, axis, factor, lowering, reason).run();
}
} // namespace mlir::triton::cfg
