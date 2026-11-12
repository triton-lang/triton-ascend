// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/BlockBatching/ElementwiseModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"
#include <functional>
namespace mlir::triton::cfg {
namespace {
using K = BatchExpr;
bool isConstant(BatchExprRef expression, int64_t value) {
  return expression && expression->kind == K::Constant &&
         expression->value == value;
}

// 逐节点比对两个表达式。表达式不做 hash-cons，相同的式子会是两个不同的对象，
// 所以指针相等问不出"是不是同一个量"。
bool sameExpression(BatchExprRef a, BatchExprRef b) {
  if (a == b)
    return true;
  if (!a || !b)
    return false;
  return a->kind == b->kind && a->bits == b->bits && a->value == b->value &&
         a->extent == b->extent && sameExpression(a->lhs, b->lhs) &&
         sameExpression(a->rhs, b->rhs);
}

// Describe the newly emitted wide arithmetic over the FULL physical tile,
// including inactive lanes. The source guard cannot authorize these new
// intermediates. Only actually rewritten plans publish this extra guard.
std::string layoutRuntimeGuard(const BatchLayoutPlan &plan,
                               const BatchAccessFact &fact,
                               ArrayRef<int64_t> targetShape,
                               ArrayRef<int64_t> coordinateMap,
                               unsigned fusedTarget, int axis, int64_t factor) {
  using E = BatchExprRef;
  auto constant = [](int64_t n) { return K::leaf(K::Constant, n, 64); };
  auto add = [](E a, E b) { return K::node(K::Add, 64, a, b); };
  auto mul = [](E a, E b) { return K::node(K::Mul, 64, a, b); };
  E base =
      mul(K::leaf(K::Program, axis, 32), constant(factor * plan.tileExtent));
  E range = K::leaf(K::Coordinate, fusedTarget, 32, targetShape[fusedTarget]);
  E logicalBase = plan.origin ? add(base, constant(plan.origin)) : base;
  E endpoint = mul(K::leaf(K::Grid, axis, 32), constant(plan.tileExtent));
  if (plan.origin)
    endpoint = add(endpoint, constant(plan.origin));
  E offsetBase = base, offset = range;
  if (!isConstant(plan.elementStride, 1)) {
    offsetBase = mul(offsetBase, plan.elementStride);
    offset = mul(offset, plan.elementStride);
  }
  if (!isConstant(plan.elementOrigin, 0))
    offsetBase = add(offsetBase, plan.elementOrigin);
  for (const auto &term : plan.restTerms) {
    E value;
    bool tensor = term.kind == K::Coordinate;
    if (tensor) {
      int64_t id = coordinateMap[term.id];
      value = K::leaf(K::Coordinate, id, 32, targetShape[id]);
    } else {
      // 循环变量的位宽按它那条循环来，不能写死 32。源码只要写过
      // tl.program_id(0).cast(tl.int64)，整条循环就是 i64，运行期守卫会拿这一位
      // 和循环域里登记的位宽对照，对不上直接报 INVALID_LOOP_DOMAIN。
      unsigned bits = 32;
      if (term.kind == K::Induction)
        for (const auto &loop : fact.loops)
          if (loop.id == term.id)
            bits = loop.bits;
      value = K::leaf(term.kind, term.id, bits);
    }
    if (!isConstant(term.coefficient, 1))
      value = mul(value, term.coefficient);
    if (tensor)
      offset = add(offset, value);
    else
      offsetBase = add(offsetBase, value);
  }
  E byteBase = mul(offsetBase, constant(fact.bytes));
  E byteOffset = mul(add(offsetBase, offset), constant(fact.bytes));
  llvm::json::Array shape, checks, loops;
  for (int64_t n : targetShape)
    shape.push_back(n);
  // 桥上的地址含圈内偏移，运行期要解释 iv 就得有循环域。和原访存守卫共用同一份
  // 序列化与同一段解释代码（ByteDomain.__init__），不另立一套语义。没有循环的
  // 访存这里是空数组，运行期行为一个字不变。
  for (const auto &loop : fact.loops)
    loops.push_back(batchLoopJSON(loop));
  for (E expression : {logicalBase, add(logicalBase, range), endpoint,
                       K::node(K::MinSI, 64, plan.upper, endpoint), offsetBase,
                       offset, byteBase, byteOffset})
    checks.push_back(batchExprJSON(expression));
  return llvm::formatv(
             "{0}", llvm::json::Value(llvm::json::Object{
                        {"version", 1},
                        {"axis", axis},
                        {"factor", factor},
                        {"access_id", plan.accessID},
                        {"root", fact.root},
                        {"bytes", fact.bytes},
                        {"shape", std::move(shape)},
                        {"loops", std::move(loops)},
                        {"positive",
                         llvm::json::Array{batchExprJSON(plan.elementStride)}},
                        {"checks", std::move(checks)},
                        {"pointer_checks",
                         llvm::json::Array{batchExprJSON(byteBase),
                                           batchExprJSON(byteOffset)}}}))
      .str();
}
// Upper bounds must remain scalar. Rebuild only exact signed arithmetic
// already covered by the original guard, never an unmodeled predicate.
bool supportedUpper(BatchExprRef e) {
  if (!e)
    return false;
  if (e->kind == K::Constant || e->kind == K::Argument || e->kind == K::Grid)
    return true;
  if (e->kind == K::ExtSI)
    return supportedUpper(e->lhs);
  if (e->kind == K::Add || e->kind == K::Sub || e->kind == K::Mul ||
      e->kind == K::MinSI || e->kind == K::MaxSI)
    return supportedUpper(e->lhs) && supportedUpper(e->rhs);
  return false;
}
bool supportedPredicate(BatchExprRef e, int axis, ArrayRef<int64_t> fused) {
  if (!e)
    return false;
  if (e->kind == K::Unknown || e->kind == K::Induction || e->kind == K::ExtUI ||
      e->kind == K::Trunc || e->kind == K::ULT)
    return false;
  if (e->kind == K::Program && e->value == axis)
    return false;
  if (e->kind == K::Coordinate && llvm::is_contained(fused, e->value))
    return false;
  return (!e->lhs || supportedPredicate(e->lhs, axis, fused)) &&
         (!e->rhs || supportedPredicate(e->rhs, axis, fused));
}
struct Emitter {
  IRRewriter &r;
  Location loc;
  triton::FuncOp function;
  unsigned originalArgs;
  SmallVector<int64_t> targetShape;
  SmallVector<int64_t> coordinateMap;
  // 循环编号到循环变量。布局分析把圈内偏移留成一个 K::Induction 的 rest term，
  // 这里是把它还原成 SSA 值的唯一依据。按访存逐条填，见 inductionVariables()。
  // 显式写出初值：DenseMap 的默认构造函数是
  // explicit，聚合初始化省略它会编不过， 而现有的三个构造点都不传这一项。
  llvm::DenseMap<int64_t, Value> inductionVars{};
  Value constant(int64_t n) {
    return r.create<arith::ConstantIntOp>(loc, n, 64);
  }
  Value wide(Value value) {
    if (value.getType().isInteger(64))
      return value;
    return r.create<arith::ExtSIOp>(loc, r.getI64Type(), value);
  }
  Value upper(BatchExprRef e) {
    switch (e->kind) {
    case K::Constant:
      return constant(e->value);
    case K::Argument:
      return wide(function.getArgument(e->value));
    case K::Grid:
      return wide(function.getArgument(originalArgs + e->value));
    case K::Program:
      return wide(r.create<triton::GetProgramIdOp>(
          loc, static_cast<triton::ProgramIDDim>(e->value)));
    case K::Induction:
      // 查不到就是编号配不上，宁可崩在这里也不能发一条地址错了的访存。调用方在
      // 填表时已经把配不上的情形整条拒掉了，所以走到这里说明两边的约定破了。
      assert(inductionVars.count(e->value) && "unmapped loop induction value");
      return wide(inductionVars.lookup(e->value));
    case K::ExtSI:
      return upper(e->lhs);
    case K::Add:
      return r.create<arith::AddIOp>(loc, upper(e->lhs), upper(e->rhs));
    case K::Sub:
      return r.create<arith::SubIOp>(loc, upper(e->lhs), upper(e->rhs));
    case K::Mul:
      return r.create<arith::MulIOp>(loc, upper(e->lhs), upper(e->rhs));
    case K::MinSI:
      return r.create<arith::MinSIOp>(loc, upper(e->lhs), upper(e->rhs));
    case K::MaxSI:
      return r.create<arith::MaxSIOp>(loc, upper(e->lhs), upper(e->rhs));
    default:
      llvm_unreachable("unproved layout upper bound");
    }
  }
  Value splat(Value v) {
    if (isa<RankedTensorType>(v.getType()))
      return v;
    return r.create<triton::SplatOp>(
        loc, RankedTensorType::get(targetShape, v.getType()), v);
  }
  Value expandAxis(Value value, unsigned axis) {
    Type element = cast<RankedTensorType>(value.getType()).getElementType();
    for (unsigned dim = 0; dim < targetShape.size(); ++dim)
      if (dim != axis)
        value = r.create<triton::ExpandDimsOp>(loc, value, dim);
    auto target = RankedTensorType::get(targetShape, element);
    if (value.getType() != target)
      value = r.create<triton::BroadcastOp>(loc, target, value);
    return value;
  }
  Value range(unsigned axis) {
    auto type = RankedTensorType::get({targetShape[axis]}, r.getI32Type());
    Value value =
        r.create<triton::MakeRangeOp>(loc, type, 0, targetShape[axis]);
    value = r.create<arith::ExtSIOp>(
        loc, RankedTensorType::get({targetShape[axis]}, r.getI64Type()), value);
    return expandAxis(value, axis);
  }
  Value axisMask(Value base, Value limit, unsigned axis) {
    // inductionVars 要跟着传：局部 Emitter 也会走 expression()，拿不到映射的话
    // 一个圈内偏移叶子就会在 upper() 里取到空值。
    Emitter local{r,  loc,          function, originalArgs, {targetShape[axis]},
                  {}, inductionVars};
    Value index =
        r.create<arith::AddIOp>(loc, local.splat(base), local.range(0));
    Value mask = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, index,
                                         local.splat(limit));
    // Complete the one-dimensional comparison before introducing broadcast
    // axes. Adding a scalar after expand/broadcast would assign its offset to
    // those synthetic axes in the existing MaskState representation.
    return expandAxis(mask, axis);
  }
  Value predicate(BatchExprRef expr) {
    llvm::DenseSet<int64_t> axes;
    std::function<void(BatchExprRef)> collect = [&](BatchExprRef node) {
      if (!node)
        return;
      if (node->kind == K::Coordinate)
        axes.insert(coordinateMap[node->value]);
      collect(node->lhs);
      collect(node->rhs);
    };
    collect(expr);
    if (axes.size() != 1)
      return splat(expression(expr));
    unsigned axis = *axes.begin();
    SmallVector<int64_t> projection(coordinateMap.size(), 0);
    Emitter local{
        r,          loc,          function, originalArgs, {targetShape[axis]},
        projection, inductionVars};
    return expandAxis(local.expression(expr), axis);
  }
  Value expression(BatchExprRef e) {
    if (e->kind == K::Coordinate)
      return range(coordinateMap[e->value]);
    // Induction 和其余标量叶子一样转给 upper()。漏掉它的话这里会掉进下面的二元
    // 分支，对空的 e->lhs 解引用——不是报错，是直接崩。圈内偏移正是以这种叶子的
    // 形式出现在 rest term 里的。
    if (e->kind == K::Constant || e->kind == K::Argument ||
        e->kind == K::Grid || e->kind == K::Program || e->kind == K::Induction)
      return upper(e);
    if (e->kind == K::ExtSI)
      return expression(e->lhs);
    if (e->kind == K::True || e->kind == K::False)
      return r.create<arith::ConstantIntOp>(loc, e->kind == K::True, 1);
    Value a = expression(e->lhs), b = expression(e->rhs);
    if (a.getType() != b.getType()) {
      a = splat(a);
      b = splat(b);
    }
    switch (e->kind) {
    case K::Add:
      return r.create<arith::AddIOp>(loc, a, b);
    case K::Sub:
      return r.create<arith::SubIOp>(loc, a, b);
    case K::Mul:
      return r.create<arith::MulIOp>(loc, a, b);
    case K::MinSI:
      return r.create<arith::MinSIOp>(loc, a, b);
    case K::MaxSI:
      return r.create<arith::MaxSIOp>(loc, a, b);
    case K::And:
      return r.create<arith::AndIOp>(loc, a, b);
    case K::SLT:
      return r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, a, b);
    case K::EQ:
      return r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, a, b);
    default:
      llvm_unreachable("unproved residual mask");
    }
  }
  Value reshape(Value v, ArrayRef<int64_t> shape) {
    auto type = cast<RankedTensorType>(v.getType());
    return r.create<triton::ReshapeOp>(
        loc, RankedTensorType::get(shape, type.getElementType()), v);
  }
};
// A deliberately closed, single-block pointwise region. All nonuniform
// operands use one proven coordinate map. Smaller-domain computations and
// effect/region boundaries stay semantic; they never get expanded by cloning.
struct PointwiseRegion {
  Block *block = nullptr;
  SmallVector<int64_t> semanticShape, targetShape;
  llvm::DenseSet<Operation *> loads, stores, oldOperations;
  llvm::DenseMap<Value, Value> values;
  llvm::DenseSet<Value> checked;
  unsigned budget = 4096;
  bool tensorShape(Value value) const {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.getShape() == ArrayRef<int64_t>(semanticShape);
  }
  Attribute uniformConstant(Value value) const {
    for (unsigned depth = 0; depth < 64; ++depth) {
      Operation *op = value.getDefiningOp();
      if (!op)
        return {};
      if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        if (auto dense = dyn_cast<DenseElementsAttr>(c.getValue()))
          return dense.isSplat() ? dense.getSplatValue<Attribute>()
                                 : Attribute{};
        return c.getValue();
      }
      if (!isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
               triton::ReshapeOp>(op))
        return {};
      value = op->getOperand(0);
    }
    return {};
  }
  Value uniformScalar(Value value) const {
    for (unsigned depth = 0; depth < 64; ++depth) {
      if (!isa<RankedTensorType>(value.getType()))
        return value;
      Operation *op = value.getDefiningOp();
      if (!op || !isa<triton::SplatOp, triton::ExpandDimsOp,
                      triton::BroadcastOp, triton::ReshapeOp>(op))
        return {};
      value = op->getOperand(0);
    }
    return {};
  }
  bool supports(Value value) {
    if (!isa<RankedTensorType>(value.getType()))
      return true;
    if (checked.contains(value))
      return true;
    if (!budget || !tensorShape(value))
      return false;
    --budget;
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != block)
      return false;
    if (loads.contains(op)) {
      checked.insert(value);
      return true;
    }
    if (uniformConstant(value) || uniformScalar(value)) {
      // Keep uniform computation at its original scalar domain.
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto data = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!data || !data.isSplat())
        return false;
    } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      if (isa<RankedTensorType>(splat.getSrc().getType()))
        return false;
    } else {
      const auto *descriptor = lookupBatchOp(op->getName().getStringRef());
      if (!descriptor || descriptor->model != BatchModel::Elementwise ||
          op->getNumRegions() || op->getNumResults() != 1)
        return false;
      for (Value input : op->getOperands())
        if (!supports(input))
          return false;
    }
    checked.insert(value);
    oldOperations.insert(op);
    return true;
  }
  bool closedUsers(Value value, llvm::DenseSet<Value> &visited) {
    if (!visited.insert(value).second)
      return true;
    for (Operation *user : value.getUsers()) {
      if (stores.contains(user)) {
        if (cast<triton::StoreOp>(user).getValue() != value)
          return false;
        continue;
      }
      if (user->getNumResults() != 1 || !supports(user->getResult(0)))
        return false;
      if (!closedUsers(user->getResult(0), visited))
        return false;
    }
    return true;
  }
  bool verify() {
    for (Operation *op : loads) {
      if (!supports(cast<triton::LoadOp>(op).getOther()))
        return false;
      llvm::DenseSet<Value> visited;
      if (!closedUsers(op->getResult(0), visited))
        return false;
    }
    for (Operation *op : stores)
      if (!supports(cast<triton::StoreOp>(op).getValue()))
        return false;
    return true;
  }
  Value materialize(Value value, IRRewriter &r) {
    if (!isa<RankedTensorType>(value.getType()))
      return value;
    if (auto found = values.find(value); found != values.end())
      return found->second;
    Operation *op = value.getDefiningOp();
    OpBuilder::InsertionGuard guard(r);
    r.setInsertionPoint(op);
    auto type = RankedTensorType::get(
        targetShape, cast<RankedTensorType>(value.getType()).getElementType());
    Value result;
    if (Attribute attr = uniformConstant(value)) {
      result = r.create<arith::ConstantOp>(
          op->getLoc(), type,
          DenseElementsAttr::get(type, cast<TypedAttr>(attr)));
    } else if (Value scalar = uniformScalar(value)) {
      result = r.create<triton::SplatOp>(op->getLoc(), type, scalar);
    } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      result = r.create<triton::SplatOp>(op->getLoc(), type, splat.getSrc());
    } else {
      OperationState state(op->getLoc(), op->getName());
      for (Value input : op->getOperands())
        state.addOperands(materialize(input, r));
      state.addTypes(type);
      state.addAttributes(op->getAttrs());
      result = r.create(state)->getResult(0);
    }
    values[value] = result;
    return result;
  }
};
} // namespace
bool hasBatchColumnPointwiseRegion(triton::FuncOp function,
                                   const BatchAccessReport &accesses, int axis,
                                   int64_t factor) {
  auto plans = planBatchLayouts(accesses, axis, factor);
  if (plans.empty())
    return false;
  PointwiseRegion region;
  region.block = &function.getBody().front();
  region.semanticShape = plans.front().shape;
  for (const auto &plan : plans) {
    if (plan.action != BatchLayoutPlan::PointwiseRegion ||
        !plan.linearMemoryView || plan.shape != region.semanticShape ||
        plan.localAxes != SmallVector<int64_t>{1} ||
        !isConstant(plan.elementStride, 1) || !supportedUpper(plan.upper) ||
        llvm::any_of(plan.residualMasks, [&](BatchExprRef predicate) {
          return !supportedPredicate(predicate, axis, plan.localAxes);
        }))
      return false;
    const auto &access = accesses.accesses[plan.accessID];
    if (access.deviceAddress || !access.loops.empty() ||
        access.operation->getBlock() != region.block)
      return false;
    if (isa<triton::LoadOp>(access.operation))
      region.loads.insert(access.operation);
    else if (isa<triton::StoreOp>(access.operation))
      region.stores.insert(access.operation);
    else
      return false;
  }
  if (region.loads.empty() || region.stores.empty() || !region.verify())
    return false;
  auto control = analyzeBatchControlFlow(function, axis);
  if (!control.reason.empty())
    return false;
  // A scalar program-id expression becomes lane-varying after batching;
  // unlike a shared scalar parameter, it cannot be splatted into this map.
  for (Value value : region.checked)
    if (Value scalar = region.uniformScalar(value))
      if (control.isVarying(scalar))
        return false;
  // Newly introduced lane masks are not part of the closed source region.
  // Leave poison-sensitive arithmetic on the existing axis-selection path.
  for (Operation *op : region.oldOperations)
    if (const auto *descriptor = lookupBatchOp(op->getName().getStringRef()))
      if (descriptor->model == BatchModel::Elementwise &&
          needsBatchActiveMask(op, *descriptor))
        return false;
  return true;
}

LogicalResult rewriteBatchMemoryViews(ModuleOp module,
                                      const BatchAccessReport &accesses,
                                      SmallVectorImpl<BatchLayoutPlan> &plans,
                                      ArrayRef<Operation *> semanticAccesses,
                                      Value group, unsigned originalArgCount,
                                      int axis, int64_t factor) {
  IRMapping mapping;
  OwningOpRef<ModuleOp> sandbox(cast<ModuleOp>(module->clone(mapping)));
  auto function = *sandbox->getOps<triton::FuncOp>().begin();
  IRRewriter rewriter(module.getContext());
  PointwiseRegion region;
  bool columnRegion = llvm::any_of(plans, [](const BatchLayoutPlan &plan) {
    return plan.linearMemoryView &&
           plan.action == BatchLayoutPlan::PointwiseRegion;
  });
  if (columnRegion) {
    region.block = &function.getBody().front();
    const auto first = llvm::find_if(plans, [](const BatchLayoutPlan &plan) {
      return plan.action == BatchLayoutPlan::PointwiseRegion;
    });
    bool compatible = first != plans.end();
    for (const auto &plan : plans) {
      if (plan.action != BatchLayoutPlan::PointwiseRegion)
        continue;
      compatible &= plan.linearMemoryView && supportedUpper(plan.upper) &&
                    llvm::all_of(plan.residualMasks,
                                 [&](BatchExprRef predicate) {
                                   return supportedPredicate(predicate, axis,
                                                             plan.localAxes);
                                 }) &&
                    plan.action == BatchLayoutPlan::PointwiseRegion &&
                    plan.shape == first->shape &&
                    plan.localAxes == SmallVector<int64_t>{1};
      Operation *op = mapping.lookupOrNull(semanticAccesses[plan.accessID]);
      if (!op || op->getBlock() != region.block) {
        compatible = false;
        continue;
      }
      if (isa<triton::LoadOp>(op))
        region.loads.insert(op);
      else
        region.stores.insert(op);
    }
    if (compatible) {
      region.semanticShape = {factor, first->shape[0], first->shape[1]};
      region.targetShape = {first->shape[0], factor * first->shape[1]};
      compatible = region.verify();
    }
    if (!compatible) {
      for (auto &plan : plans)
        if (plan.action == BatchLayoutPlan::PointwiseRegion) {
          plan.action = BatchLayoutPlan::Keep;
          plan.reason = "EXPENSIVE_LAYOUT_BRIDGE";
          plan.linearMemoryView = false;
        }
      columnRegion = false;
    }
  }
  bool changed = false;
  for (auto &plan : plans) {
    if (!plan.linearMemoryView || !supportedUpper(plan.upper) ||
        llvm::any_of(plan.residualMasks, [&](BatchExprRef predicate) {
          return !supportedPredicate(predicate, axis, plan.localAxes);
        }))
      continue;
    bool inColumnRegion =
        columnRegion && plan.action == BatchLayoutPlan::PointwiseRegion;
    const auto &fact = accesses.accesses[plan.accessID];
    Operation *semantic = semanticAccesses[plan.accessID];
    if (!semantic)
      return module.emitError("missing semantic memory mapping"), failure();
    Operation *op = mapping.lookupOrNull(semantic);
    if (!op)
      return module.emitError("missing cloned memory mapping"), failure();
    rewriter.setInsertionPoint(op);
    int64_t count = factor * plan.tileExtent;
    SmallVector<int64_t> targetShape, coordinateMap(plan.shape.size(), -1);
    unsigned fusedTarget = 0;
    for (auto [local, extent] : llvm::enumerate(plan.shape)) {
      if (static_cast<int64_t>(local) == plan.localAxes.front()) {
        fusedTarget = targetShape.size();
        targetShape.push_back(count);
      } else if (!llvm::is_contained(plan.localAxes, local)) {
        coordinateMap[local] = targetShape.size();
        targetShape.push_back(extent);
      }
    }
    SmallVector<int64_t> semanticShape{factor};
    llvm::append_range(semanticShape, plan.shape);
    // 采事实时循环是沿父链自内向外记的（BatchAccessAnalysis.cpp:409-426），
    // 这里用同样的次序走回来，按位置配上编号。两边对不上就整条放弃——那说明
    // 克隆改变了嵌套结构，继续发射只会得到一条地址错了的访存。
    //
    // 还原的是**圈内偏移**（0, step, 2*step,
    // …）而不是循环变量本身：布局分析对每 一条循环都做过代换 Induction(id) ≡
    // lower + Induction(id)' （BatchLayoutAnalysis.cpp:249，经
    // normalizeBatchAccessOnLogicalAxis），剩下
    // 的这个节点代表的是等号右边那一项。还整个循环变量的话下界会被算两遍——程序轴
    // 循环上这正是 g*F*R 那一项，地址直接错掉。
    SmallVector<Value> enclosing;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp())
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());
        enclosing.push_back(rewriter.create<arith::SubIOp>(
            loop.getLoc(), loop.getInductionVar(), loop.getLowerBound()));
      }
    if (enclosing.size() != fact.loops.size()) {
      plan.linearMemoryView = false;
      plan.action = BatchLayoutPlan::Keep;
      plan.reason = "LOOP_NESTING_MISMATCH";
      continue;
    }
    Emitter e{rewriter,         op->getLoc(), function,
              originalArgCount, targetShape,  coordinateMap};
    for (auto [loop, inductionVar] : llvm::zip(fact.loops, enclosing))
      e.inductionVars[loop.id] = inductionVar;
    Value q = e.wide(mapping.lookup(group));
    Value base = rewriter.create<arith::MulIOp>(e.loc, q, e.constant(count));
    Value t = e.range(fusedTarget);
    Value logicalBase = base;
    if (plan.origin)
      logicalBase =
          rewriter.create<arith::AddIOp>(e.loc, base, e.constant(plan.origin));
    // 掩码必须界住地址真正落到的那一行。logicalBase 只含程序号那一项，圈内偏移
    // 在下面以 rest term 的形式加进 offsetBase——两边不同步的话第 k 圈会拿第 0
    // 圈 的掩码，越界就拦不住了。没有循环时 rest term 里不会有
    // Induction，这一段是 空转。
    //
    // 只在元素步长为 1、元素起点为 0 时成立：那时 offsetBase = base，元素单位和
    // 逻辑单位是同一个量，圈内偏移可以直接加到 logicalBase 上。别的比例关系要另
    // 行证明，没证之前整条劝退，不发一条掩码错了的访存。
    Value shifted;
    for (const auto &term : plan.restTerms) {
      if (term.kind != K::Induction)
        continue;
      // logicalBase 是**行单位**，而 term.coefficient
      // 是**元素单位**。两者的换算 系数必须精确知道，不能估。认两种情形：
      //   元素步长为 1  两种单位本来就是同一个，照搬系数。
      //   系数恰是元素步长  循环变量就是行号，换算系数精确为 1，行偏移即 iv。
      //      正例 #7/#8 的二维按行访存全落在这一支；少了它，它们会静悄悄地退回
      //      语义体——照样 REWRITTEN、照样算对，只是宽访存桥没搭上，比基线慢。
      bool unit = isConstant(plan.elementStride, 1);
      if ((!unit && !sameExpression(term.coefficient, plan.elementStride)) ||
          !isConstant(plan.elementOrigin, 0)) {
        plan.linearMemoryView = false;
        plan.action = BatchLayoutPlan::Keep;
        plan.reason = "LOOP_MASK_UNIT_MISMATCH";
        break;
      }
      Value shift = e.expression(K::leaf(term.kind, term.id, 32));
      if (unit && !isConstant(term.coefficient, 1))
        shift = rewriter.create<arith::MulIOp>(e.loc, shift,
                                               e.upper(term.coefficient));
      shifted = rewriter.create<arith::AddIOp>(
          e.loc, shifted ? shifted : logicalBase, shift);
    }
    if (!plan.linearMemoryView)
      continue;
    Value grid = e.wide(function.getArgument(originalArgCount + axis));
    Value endpoint = rewriter.create<arith::MulIOp>(
        e.loc, grid, e.constant(plan.tileExtent));
    if (plan.origin)
      endpoint = rewriter.create<arith::AddIOp>(e.loc, endpoint,
                                                e.constant(plan.origin));
    // 没有圈内偏移时两条界融成一条 min，这是已发布的形状，原样保留。
    //
    // 有圈内偏移就不能融：两条界管的不是一回事。plan.upper
    // 界的是这条访存真正落到 的行，圈内偏移要算进去；endpoint = G*tileExtent
    // 界的是 lane 活没活，它说的是 "原网格只有 G
    // 份"，和走到第几圈无关。把圈内偏移塞进 min 里，第 k 圈的活 lane
    // 会被当成越界挡掉——少读，不是多读，但一样是算错。
    Value mask;
    if (shifted) {
      mask = rewriter.create<arith::AndIOp>(
          e.loc, e.axisMask(shifted, e.upper(plan.upper), fusedTarget),
          e.axisMask(logicalBase, endpoint, fusedTarget));
    } else {
      Value limit =
          rewriter.create<arith::MinSIOp>(e.loc, e.upper(plan.upper), endpoint);
      mask = e.axisMask(logicalBase, limit, fusedTarget);
    }
    for (auto predicate : plan.residualMasks)
      mask =
          rewriter.create<arith::AndIOp>(e.loc, mask, e.predicate(predicate));
    Value offsetBase = base, offsetRange = t;
    if (!isConstant(plan.elementStride, 1)) {
      Value stride = e.upper(plan.elementStride);
      offsetBase = rewriter.create<arith::MulIOp>(e.loc, offsetBase, stride);
      offsetRange =
          rewriter.create<arith::MulIOp>(e.loc, offsetRange, e.splat(stride));
    }
    if (!isConstant(plan.elementOrigin, 0))
      offsetBase = rewriter.create<arith::AddIOp>(e.loc, offsetBase,
                                                  e.upper(plan.elementOrigin));
    Value offset = offsetRange;
    for (const auto &term : plan.restTerms) {
      Value value = e.expression(K::leaf(term.kind, term.id, 32));
      Value scale = e.upper(term.coefficient);
      if (isa<RankedTensorType>(value.getType()))
        scale = e.splat(scale);
      if (!isConstant(term.coefficient, 1))
        value = rewriter.create<arith::MulIOp>(e.loc, value, scale);
      if (isa<RankedTensorType>(value.getType()))
        offset = rewriter.create<arith::AddIOp>(e.loc, offset, value);
      else
        offsetBase = rewriter.create<arith::AddIOp>(e.loc, offsetBase, value);
    }
    Value oldPointer = isa<triton::LoadOp>(op)
                           ? cast<triton::LoadOp>(op).getPtr()
                           : cast<triton::StoreOp>(op).getPtr();
    auto targetPointer =
        cast<RankedTensorType>(oldPointer.getType()).getElementType();
    Value root = function.getArgument(fact.root);
    bool normalizedBool = op->hasAttr("was_bool_to_int8");
    // Preserve the frontend bool->byte contract and its direct load marker.
    if (!normalizedBool && root.getType() != targetPointer)
      root = rewriter.create<triton::BitcastOp>(e.loc, targetPointer, root);
    // Keep the uniform address base scalar as well. This exposes a normal
    // strided tensor offset to the existing structured pointer analysis.
    root = rewriter.create<triton::AddPtrOp>(e.loc, root.getType(), root,
                                             offsetBase);
    auto pointerType = RankedTensorType::get(targetShape, root.getType());
    Value pointer = rewriter.create<triton::AddPtrOp>(e.loc, pointerType,
                                                      e.splat(root), offset);
    if (normalizedBool)
      pointer = rewriter.create<triton::BitcastOp>(
          e.loc, RankedTensorType::get(targetShape, targetPointer), pointer);
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      auto originalType = cast<RankedTensorType>(load.getType());
      if (originalType.getShape() != ArrayRef<int64_t>(semanticShape))
        return module.emitError("invalid semantic load shape for memory view"),
               failure();
      Value other = inColumnRegion
                        ? region.materialize(load.getOther(), rewriter)
                        : e.reshape(load.getOther(), targetShape);
      auto replacement = rewriter.create<triton::LoadOp>(
          e.loc, pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      if (auto marker = load->getAttr("was_bool_to_int8"))
        replacement->setAttr("was_bool_to_int8", marker);
      if (inColumnRegion)
        region.values[load.getResult()] = replacement.getResult();
      else
        rewriter.replaceOp(op, e.reshape(replacement, originalType.getShape()));
    } else {
      auto store = cast<triton::StoreOp>(op);
      auto originalType = cast<RankedTensorType>(store.getValue().getType());
      if (originalType.getShape() != ArrayRef<int64_t>(semanticShape))
        return module.emitError("invalid semantic store shape for memory view"),
               failure();
      Value stored = inColumnRegion
                         ? region.materialize(store.getValue(), rewriter)
                         : e.reshape(store.getValue(), targetShape);
      rewriter.create<triton::StoreOp>(e.loc, pointer, stored, mask,
                                       store.getBoundaryCheck(),
                                       store.getCache(), store.getEvict());
      rewriter.eraseOp(op);
    }
    if (plan.symbolic)
      plan.runtimeGuardJSON = layoutRuntimeGuard(
          plan, fact, targetShape, coordinateMap, fusedTarget, axis, factor);
    plan.rewritten = true;
    changed = true;
  }
  if (!changed)
    return success();
  if (columnRegion) {
    SmallVector<Operation *> dead;
    for (Operation &op : *region.block)
      if (region.oldOperations.contains(&op) || region.loads.contains(&op))
        dead.push_back(&op);
    for (Operation *op : llvm::reverse(dead))
      if (op->use_empty())
        rewriter.eraseOp(op);
    for (Operation &op : *region.block)
      if (region.loads.contains(&op))
        return module.emitError(
                   "pointwise layout left an unbridged semantic load"),
               failure();
  }
  if (failed(verify(*sandbox)))
    return module.emitError("block batching memory view failed verification"),
           failure();
  module->setAttrs((*sandbox)->getAttrs());
  module.getBodyRegion().takeBody(sandbox->getBodyRegion());
  return success();
}
} // namespace mlir::triton::cfg
