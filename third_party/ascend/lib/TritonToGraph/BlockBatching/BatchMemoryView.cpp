// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "TritonToGraph/BlockBatching/BatchMemoryRewriter.h"
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "TritonToGraph/BlockBatching/BatchRewriteFinalizer.h"
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/BlockBatching/ElementwiseModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SaveAndRestore.h"
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
                               const BatchFacts::Access &fact,
                               const BatchFacts &facts,
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
        for (unsigned id : fact.loopIDs)
          if (id == term.id)
            bits = facts.findLoop(id)->bits;
      value = K::leaf(term.kind, term.id, bits, term.divisor);
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
  for (unsigned id : fact.loopIDs)
    loops.push_back(batchLoopJSON(*facts.findLoop(id)));
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
  // 批处理后的 get_program_id 返回组号，per-lane 的程序号是 group*F + lane。
  // 商/余数叶子同样由批处理轴派生，整组 lane 只会得到一个谓词值，掩码对
  // lane>=1 就是错的。地址侧已在 BatchAffineForm 拒绝它们，残差掩码必须一致。
  if ((e->kind == K::Program || e->kind == K::ProgramQuotient ||
       e->kind == K::ProgramRemainder) &&
      e->value == axis)
    return false;
  if (e->kind == K::Coordinate && llvm::is_contained(fused, e->value))
    return false;
  return (!e->lhs || supportedPredicate(e->lhs, axis, fused)) &&
         (!e->rhs || supportedPredicate(e->rhs, axis, fused));
}
// The same closed map covers [N] and [M, N]. Only the final contiguous
// dimension is widened; other dimensions and scalar row origins are kept.
bool pointwiseTile(const BatchLayoutPlan &plan) {
  return (plan.shape.size() == 1 &&
          plan.localAxes == SmallVector<int64_t>{0}) ||
         (plan.shape.size() == 2 && plan.localAxes == SmallVector<int64_t>{1});
}
} // namespace
Value BatchMemoryRewriter::constant(int64_t n) {
  return r.create<arith::ConstantIntOp>(loc, n, 64);
}
Value BatchMemoryRewriter::wide(Value value) {
  if (!value || !isa<IntegerType>(value.getType()) ||
      cast<IntegerType>(value.getType()).getWidth() > 64)
    return {};
  if (value.getType().isInteger(64))
    return value;
  return r.create<arith::ExtSIOp>(loc, r.getI64Type(), value);
}
Value BatchMemoryRewriter::upper(BatchExprRef e) {
  if (!e)
    return {};
  switch (e->kind) {
  case K::Constant:
    return constant(e->value);
  case K::Argument:
    return e->value >= 0 && static_cast<uint64_t>(e->value) < originalArgs
               ? wide(function.getArgument(e->value))
               : Value{};
  case K::Grid:
    return e->value >= 0 && e->value < 3 &&
                   originalArgs + e->value < function.getNumArguments()
               ? wide(function.getArgument(originalArgs + e->value))
               : Value{};
  case K::Program:
    if (e->value < 0 || e->value >= 3)
      return {};
    return wide(r.create<triton::GetProgramIdOp>(
        loc, static_cast<triton::ProgramIDDim>(e->value)));
  case K::ProgramQuotient:
  case K::ProgramRemainder: {
    if (e->extent <= 0 || e->value < 0 || e->value >= 3)
      return {};
    Value pid = wide(r.create<triton::GetProgramIdOp>(
        loc, static_cast<triton::ProgramIDDim>(e->value)));
    Value divisor = constant(e->extent);
    if (e->kind == K::ProgramQuotient)
      return r.create<arith::DivSIOp>(loc, pid, divisor);
    return r.create<arith::RemSIOp>(loc, pid, divisor);
  }
  case K::Induction:
    return wide(inductionVars.lookup(e->value));
  case K::ExtSI:
    return upper(e->lhs);
  case K::Add:
  case K::Sub:
  case K::Mul:
  case K::MinSI:
  case K::MaxSI:
    break;
  default:
    return {};
  }
  // Preserve the native builder's operand evaluation order. binary() checks
  // both emitted values before constructing the operation.
  return binary(e->kind, upper(e->lhs), upper(e->rhs));
}
Value BatchMemoryRewriter::splat(Value v) {
  if (!v || isa<RankedTensorType>(v.getType()))
    return v;
  return r.create<triton::SplatOp>(
      loc, RankedTensorType::get(targetShape, v.getType()), v);
}
Value BatchMemoryRewriter::expandAxis(Value value, unsigned axis) {
  Type element = cast<RankedTensorType>(value.getType()).getElementType();
  for (unsigned dim = 0; dim < targetShape.size(); ++dim)
    if (dim != axis)
      value = r.create<triton::ExpandDimsOp>(loc, value, dim);
  auto target = RankedTensorType::get(targetShape, element);
  if (value.getType() != target)
    value = r.create<triton::BroadcastOp>(loc, target, value);
  return value;
}
Value BatchMemoryRewriter::range(unsigned axis) {
  auto type = RankedTensorType::get({targetShape[axis]}, r.getI32Type());
  Value value = r.create<triton::MakeRangeOp>(loc, type, 0, targetShape[axis]);
  value = r.create<arith::ExtSIOp>(
      loc, RankedTensorType::get({targetShape[axis]}, r.getI64Type()), value);
  return expandAxis(value, axis);
}
Value BatchMemoryRewriter::axisMask(Value base, Value limit, unsigned axis) {
  Value mask;
  {
    llvm::SaveAndRestore<SmallVector<int64_t>> shape(targetShape,
                                                     {targetShape[axis]});
    Value index = r.create<arith::AddIOp>(loc, splat(base), range(0));
    mask = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, index,
                                   splat(limit));
  }
  // Complete the one-dimensional comparison before introducing broadcast
  // axes. Adding a scalar after expand/broadcast would assign its offset to
  // those synthetic axes in the existing MaskState representation.
  return expandAxis(mask, axis);
}
std::optional<unsigned> BatchMemoryRewriter::mappedAxis(int64_t id) const {
  if (id < 0 || static_cast<uint64_t>(id) >= coordinateMap.size())
    return {};
  int64_t dim = coordinateMap[id];
  if (dim < 0 || static_cast<uint64_t>(dim) >= targetShape.size())
    return {};
  return static_cast<unsigned>(dim);
}

Value BatchMemoryRewriter::predicate(BatchExprRef expr) {
  llvm::DenseSet<int64_t> axes;
  std::function<bool(BatchExprRef)> collect = [&](BatchExprRef node) {
    if (!node)
      return true;
    if (node->kind == K::Coordinate) {
      auto axis = mappedAxis(node->value);
      if (!axis)
        return false;
      axes.insert(*axis);
    }
    return collect(node->lhs) && collect(node->rhs);
  };
  if (!collect(expr))
    return {};
  if (axes.size() != 1)
    return splat(expression(expr));
  unsigned axis = *axes.begin();
  Value comparison;
  {
    llvm::SaveAndRestore<SmallVector<int64_t>> shape(targetShape,
                                                     {targetShape[axis]});
    llvm::SaveAndRestore<SmallVector<int64_t>> projection(
        coordinateMap, SmallVector<int64_t>(coordinateMap.size(), 0));
    comparison = expression(expr);
  }
  return comparison ? expandAxis(comparison, axis) : Value{};
}
Value BatchMemoryRewriter::expression(BatchExprRef e) {
  if (!e)
    return {};
  if (e->kind == K::Coordinate) {
    auto axis = mappedAxis(e->value);
    return axis ? range(*axis) : Value{};
  }
  // Induction 和其余标量叶子一样转给 upper()。漏掉它的话这里会掉进下面的二元
  // 分支，对空的 e->lhs 解引用——不是报错，是直接崩。圈内偏移正是以这种叶子的
  // 形式出现在 rest term 里的。
  if (e->kind == K::Constant || e->kind == K::Argument || e->kind == K::Grid ||
      e->kind == K::Program || e->kind == K::Induction ||
      e->kind == K::ProgramQuotient || e->kind == K::ProgramRemainder)
    return upper(e);
  if (e->kind == K::ExtSI)
    return expression(e->lhs);
  if (e->kind == K::True || e->kind == K::False)
    return r.create<arith::ConstantIntOp>(loc, e->kind == K::True, 1);
  Value a = expression(e->lhs), b = expression(e->rhs);
  if (!a || !b)
    return {};
  if (a.getType() != b.getType()) {
    a = splat(a);
    b = splat(b);
  }
  return binary(e->kind, a, b);
}
Value BatchMemoryRewriter::binary(K::Kind kind, Value a, Value b) {
  if (!a || !b)
    return {};
  switch (kind) {
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
    return {};
  }
}
Value BatchMemoryRewriter::reshape(Value v, ArrayRef<int64_t> shape) {
  auto type = cast<RankedTensorType>(v.getType());
  return r.create<triton::ReshapeOp>(
      loc, RankedTensorType::get(shape, type.getElementType()), v);
}
bool BatchMemoryRewriter::tensorShape(Value value) const {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == ArrayRef<int64_t>(semanticShape);
}
Attribute BatchMemoryRewriter::uniformConstant(Value value) const {
  for (unsigned depth = 0; depth < 64; ++depth) {
    Operation *op = value.getDefiningOp();
    if (!op)
      return {};
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto dense = dyn_cast<DenseElementsAttr>(c.getValue()))
        return dense.isSplat() ? dense.getSplatValue<Attribute>() : Attribute{};
      return c.getValue();
    }
    if (!isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
             triton::ReshapeOp>(op))
      return {};
    value = op->getOperand(0);
  }
  return {};
}
Value BatchMemoryRewriter::uniformScalar(Value value) const {
  for (unsigned depth = 0; depth < 64; ++depth) {
    if (!isa<RankedTensorType>(value.getType()))
      return value;
    Operation *op = value.getDefiningOp();
    if (!op || !isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
                    triton::ReshapeOp>(op))
      return {};
    value = op->getOperand(0);
  }
  return {};
}
bool BatchMemoryRewriter::supports(Value value) {
  if (!value)
    return false;
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
bool BatchMemoryRewriter::closedUsers(Value value,
                                      llvm::DenseSet<Value> &visited) {
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
bool BatchMemoryRewriter::proveClosedRegion() {
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
Value BatchMemoryRewriter::materializeRegion(Value value) {
  if (!value || !isa<RankedTensorType>(value.getType()))
    return value;
  if (Value bound = bindings.lookupGenerated(value))
    return bound;
  Operation *op = value.getDefiningOp();
  // Every tensor from a load must have been bound at its rewrite. Recreating
  // it as an elementwise op would lose its effect identity.
  if (!op || !checked.contains(value) || loads.contains(op))
    return {};
  OpBuilder::InsertionGuard guard(r);
  r.setInsertionPoint(op);
  auto type = RankedTensorType::get(
      regionShape, cast<RankedTensorType>(value.getType()).getElementType());
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
    for (Value input : op->getOperands()) {
      Value mapped = materializeRegion(input);
      if (!mapped)
        return {};
      state.addOperands(mapped);
    }
    state.addTypes(type);
    state.addAttributes(op->getAttrs());
    result = r.create(state)->getResult(0);
  }
  bindings.bindGenerated(value, result, true);
  return result;
}

bool BatchMemoryRewriter::probeColumnRegion(triton::FuncOp function,
                                            const BatchRewriteWorkspace &scope,
                                            int axis, int64_t factor) {
  auto plans = planBatchLayouts(scope, axis, factor);
  if (plans.empty())
    return false;
  BatchMemoryRewriter edit(function);
  edit.block = &function.getBody().front();
  edit.semanticShape = plans.front().shape;
  for (const auto &plan : plans) {
    if (plan.action != BatchLayoutPlan::PointwiseRegion ||
        !plan.linearMemoryView || plan.shape != edit.semanticShape ||
        !pointwiseTile(plan) || !isConstant(plan.elementStride, 1) ||
        !supportedUpper(plan.upper) ||
        llvm::any_of(plan.residualMasks, [&](BatchExprRef predicate) {
          return !supportedPredicate(predicate, axis, plan.localAxes);
        }))
      return false;
    const auto &access = scope.facts->accesses()[plan.accessID];
    if (access.deviceAddress || !access.loopIDs.empty() ||
        scope.sourceAccesses()[plan.accessID]->getBlock() != edit.block)
      return false;
    if (isa<triton::LoadOp>(scope.sourceAccesses()[plan.accessID]))
      edit.loads.insert(scope.sourceAccesses()[plan.accessID]);
    else if (isa<triton::StoreOp>(scope.sourceAccesses()[plan.accessID]))
      edit.stores.insert(scope.sourceAccesses()[plan.accessID]);
    else
      return false;
  }
  if (edit.loads.empty() || edit.stores.empty() || !edit.proveClosedRegion())
    return false;
  BatchRewriteWorkspace probe;
  BatchAnalyzer analyzer(function, 0, probe);
  if (failed(analyzer.analyzeVariation(axis)))
    return false;
  // A scalar program-id expression becomes lane-varying after batching;
  // unlike a shared scalar parameter, it cannot be splatted into this map.
  for (Value value : edit.checked)
    if (Value scalar = edit.uniformScalar(value))
      if (probe.isVarying(scalar))
        return false;
  // Newly introduced lane masks are not part of the closed source region.
  // Leave poison-sensitive arithmetic on the existing axis-selection path.
  for (Operation *op : edit.oldOperations)
    if (const auto *descriptor = lookupBatchOp(op->getName().getStringRef()))
      if (descriptor->model == BatchModel::Elementwise &&
          needsBatchActiveMask(op, *descriptor))
        return false;
  return true;
}

LogicalResult BatchMemoryRewriter::rewriteViews(
    ModuleOp module, const BatchRewriteWorkspace &scope,
    ArrayRef<BatchLayoutPlan> plans,
    SmallVectorImpl<BatchLayoutResult> &results,
    ArrayRef<Operation *> semanticAccesses, Value group,
    unsigned originalArgCount, int axis, int64_t factor) {
  if (!scope.facts || plans.size() != scope.facts->accesses().size() ||
      plans.size() != semanticAccesses.size())
    return module.emitError("incomplete memory view access mapping"), failure();
  for (auto [id, plan] : llvm::enumerate(plans))
    if (plan.accessID != id || !semanticAccesses[id])
      return module.emitError("invalid memory view access mapping"), failure();
  auto attempts = initialBatchLayoutResults(plans);
  IRMapping mapping;
  OwningOpRef<ModuleOp> sandbox(cast<ModuleOp>(module->clone(mapping)));
  auto entries = sandbox->getOps<triton::FuncOp>();
  if (entries.empty() || (*entries.begin()).getBody().empty())
    return module.emitError("missing memory rewrite entry"), failure();
  auto function = *entries.begin();
  BatchMemoryRewriter edit(function, originalArgCount);
  bool columnRegion = llvm::any_of(plans, [](const BatchLayoutPlan &plan) {
    return plan.linearMemoryView &&
           plan.action == BatchLayoutPlan::PointwiseRegion;
  });
  if (columnRegion) {
    edit.block = &function.getBody().front();
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
                    plan.localAxes == first->localAxes && pointwiseTile(plan);
      Operation *op = mapping.lookupOrNull(semanticAccesses[plan.accessID]);
      if (!op || op->getBlock() != edit.block) {
        compatible = false;
        continue;
      }
      if (isa<triton::LoadOp>(op))
        edit.loads.insert(op);
      else
        edit.stores.insert(op);
    }
    if (compatible) {
      edit.semanticShape = {factor};
      llvm::append_range(edit.semanticShape, first->shape);
      edit.regionShape = first->shape;
      edit.regionShape.back() *= factor;
      compatible = edit.proveClosedRegion();
    }
    if (!compatible) {
      for (auto &result : attempts) {
        const auto &plan = plans[result.accessID];
        if (plan.action == BatchLayoutPlan::PointwiseRegion) {
          result.action = BatchLayoutPlan::Keep;
          result.reason = "EXPENSIVE_LAYOUT_BRIDGE";
        }
      }
      columnRegion = false;
    }
  }
  bool changed = false;
  Value mappedGroup = mapping.lookupOrNull(group);
  if (!mappedGroup)
    return module.emitError("missing cloned batch group"), failure();
  for (auto &result : attempts) {
    const auto &plan = plans[result.accessID];
    if (result.action != plan.action || !plan.linearMemoryView ||
        !supportedUpper(plan.upper) ||
        llvm::any_of(plan.residualMasks, [&](BatchExprRef predicate) {
          return !supportedPredicate(predicate, axis, plan.localAxes);
        }))
      continue;
    Operation *op = mapping.lookupOrNull(semanticAccesses[plan.accessID]);
    if (!op || !isa<triton::LoadOp, triton::StoreOp>(op))
      return module.emitError("invalid cloned memory mapping"), failure();
    if (failed(edit.rewriteMemoryView(
            op, *scope.facts, plan, result, mappedGroup,
            columnRegion && plan.action == BatchLayoutPlan::PointwiseRegion,
            axis, factor)))
      return failure();
    changed |= result.rewritten;
  }
  if (changed) {
    if (failed(edit.finish(*sandbox, columnRegion)))
      return failure();
    module->setAttrs((*sandbox)->getAttrs());
    module.getBodyRegion().takeBody(sandbox->getBodyRegion());
  }
  results = std::move(attempts);
  return success();
}

LogicalResult BatchMemoryRewriter::rewriteMemoryView(
    Operation *op, const BatchFacts &facts, const BatchLayoutPlan &plan,
    BatchLayoutResult &result, Value mappedGroup, bool inColumnRegion, int axis,
    int64_t factor) {
  const auto &fact = facts.access(plan.accessID);
  r.setInsertionPoint(op);
  int64_t count;
  if (plan.localAxes.empty() || plan.shape != fact.shape || factor <= 1 ||
      plan.tileExtent <= 0 ||
      llvm::MulOverflow(factor, plan.tileExtent, count) || count > INT32_MAX ||
      axis < 0 || axis >= 3 ||
      originalArgs + axis >= function.getNumArguments() || fact.root < 0 ||
      static_cast<unsigned>(fact.root) >= originalArgs ||
      llvm::any_of(
          plan.localAxes,
          [&](int64_t dim) {
            return dim < 0 || static_cast<uint64_t>(dim) >= plan.shape.size();
          }))
    return function.emitError("invalid planned memory shape or ABI"), failure();
  targetShape.clear();
  coordinateMap.assign(plan.shape.size(), -1);
  loc = op->getLoc();
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
  SmallVector<int64_t> expectedShape{factor};
  llvm::append_range(expectedShape, plan.shape);
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
      OpBuilder::InsertionGuard guard(r);
      r.setInsertionPointToStart(loop.getBody());
      enclosing.push_back(r.create<arith::SubIOp>(
          loop.getLoc(), loop.getInductionVar(), loop.getLowerBound()));
    }
  if (enclosing.size() != fact.loopIDs.size()) {
    result.action = BatchLayoutPlan::Keep;
    result.reason = "LOOP_NESTING_MISMATCH";
    return success();
  }
  inductionVars.clear();
  for (auto [loop, inductionVar] : llvm::zip(fact.loopIDs, enclosing))
    inductionVars[loop] = inductionVar;
  Value q = wide(mappedGroup);
  if (!q)
    return function.emitError("invalid cloned batch group type"), failure();
  Value base = r.create<arith::MulIOp>(loc, q, constant(count));
  Value t = range(fusedTarget);
  Value logicalBase = base;
  if (plan.origin)
    logicalBase = r.create<arith::AddIOp>(loc, base, constant(plan.origin));
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
      result.action = BatchLayoutPlan::Keep;
      result.reason = "LOOP_MASK_UNIT_MISMATCH";
      break;
    }
    Value shift = expression(K::leaf(term.kind, term.id, 32));
    if (!shift)
      return function.emitError("unmapped planned loop induction"), failure();
    if (unit && !isConstant(term.coefficient, 1)) {
      Value scale = upper(term.coefficient);
      if (!scale)
        return function.emitError("invalid planned loop coefficient"),
               failure();
      shift = r.create<arith::MulIOp>(loc, shift, scale);
    }
    shifted =
        r.create<arith::AddIOp>(loc, shifted ? shifted : logicalBase, shift);
  }
  if (result.action != plan.action)
    return success();
  Value grid = wide(function.getArgument(originalArgs + axis));
  if (!grid)
    return function.emitError("invalid planned grid extent"), failure();
  Value endpoint =
      r.create<arith::MulIOp>(loc, grid, constant(plan.tileExtent));
  if (plan.origin)
    endpoint = r.create<arith::AddIOp>(loc, endpoint, constant(plan.origin));
  // 没有圈内偏移时两条界融成一条 min，这是已发布的形状，原样保留。
  //
  // 有圈内偏移就不能融：两条界管的不是一回事。plan.upper
  // 界的是这条访存真正落到 的行，圈内偏移要算进去；endpoint = G*tileExtent
  // 界的是 lane 活没活，它说的是 "原网格只有 G
  // 份"，和走到第几圈无关。把圈内偏移塞进 min 里，第 k 圈的活 lane
  // 会被当成越界挡掉——少读，不是多读，但一样是算错。
  Value limitValue = upper(plan.upper);
  if (!limitValue)
    return function.emitError("invalid planned layout upper bound"), failure();
  Value mask;
  if (shifted) {
    mask =
        r.create<arith::AndIOp>(loc, axisMask(shifted, limitValue, fusedTarget),
                                axisMask(logicalBase, endpoint, fusedTarget));
  } else {
    Value limit = r.create<arith::MinSIOp>(loc, limitValue, endpoint);
    mask = axisMask(logicalBase, limit, fusedTarget);
  }
  for (auto residual : plan.residualMasks) {
    Value test = predicate(residual);
    if (!test)
      return function.emitError("invalid planned residual predicate"),
             failure();
    mask = r.create<arith::AndIOp>(loc, mask, test);
  }
  Value offsetBase = base, offsetRange = t;
  if (!isConstant(plan.elementStride, 1)) {
    Value stride = upper(plan.elementStride);
    if (!stride)
      return function.emitError("invalid planned element stride"), failure();
    offsetBase = r.create<arith::MulIOp>(loc, offsetBase, stride);
    offsetRange = r.create<arith::MulIOp>(loc, offsetRange, splat(stride));
  }
  if (!isConstant(plan.elementOrigin, 0)) {
    Value origin = upper(plan.elementOrigin);
    if (!origin)
      return function.emitError("invalid planned element origin"), failure();
    offsetBase = r.create<arith::AddIOp>(loc, offsetBase, origin);
  }
  Value offset = offsetRange;
  for (const auto &term : plan.restTerms) {
    Value value = expression(K::leaf(term.kind, term.id, 32, term.divisor));
    Value scale = upper(term.coefficient);
    if (!value || !scale)
      return function.emitError("invalid planned layout address term"),
             failure();
    if (isa<RankedTensorType>(value.getType()))
      scale = splat(scale);
    if (!isConstant(term.coefficient, 1))
      value = r.create<arith::MulIOp>(loc, value, scale);
    if (isa<RankedTensorType>(value.getType()))
      offset = r.create<arith::AddIOp>(loc, offset, value);
    else
      offsetBase = r.create<arith::AddIOp>(loc, offsetBase, value);
  }
  Value oldPointer = isa<triton::LoadOp>(op)
                         ? cast<triton::LoadOp>(op).getPtr()
                         : cast<triton::StoreOp>(op).getPtr();
  auto tensorPointer = dyn_cast<RankedTensorType>(oldPointer.getType());
  if (!tensorPointer)
    return function.emitError("invalid semantic pointer type"), failure();
  auto targetPointer = tensorPointer.getElementType();
  Value root = function.getArgument(fact.root);
  bool normalizedBool = op->hasAttr("was_bool_to_int8");
  // Preserve the frontend bool->byte contract and its direct load marker.
  if (!normalizedBool && root.getType() != targetPointer)
    root = r.create<triton::BitcastOp>(loc, targetPointer, root);
  // Keep the uniform address base scalar as well. This exposes a normal
  // strided tensor offset to the existing structured pointer analysis.
  root = r.create<triton::AddPtrOp>(loc, root.getType(), root, offsetBase);
  auto pointerType = RankedTensorType::get(targetShape, root.getType());
  Value pointer =
      r.create<triton::AddPtrOp>(loc, pointerType, splat(root), offset);
  if (normalizedBool)
    pointer = r.create<triton::BitcastOp>(
        loc, RankedTensorType::get(targetShape, targetPointer), pointer);
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    auto originalType = dyn_cast<RankedTensorType>(load.getType());
    if (!originalType ||
        originalType.getShape() != ArrayRef<int64_t>(expectedShape))
      return function.emitError("invalid semantic load shape for memory view"),
             failure();
    if (!load.getOther() ||
        (!inColumnRegion && !isa<RankedTensorType>(load.getOther().getType())))
      return function.emitError("invalid semantic load fallback type"),
             failure();
    Value other = inColumnRegion ? materializeRegion(load.getOther())
                                 : reshape(load.getOther(), targetShape);
    if (!other)
      return function.emitError("missing pointwise load fallback mapping"),
             failure();
    auto replacement = r.create<triton::LoadOp>(
        loc, pointer, mask, other, load.getBoundaryCheck(), load.getPadding(),
        load.getCache(), load.getEvict(), load.getIsVolatile());
    copyBatchEffectIdentity(op, replacement);
    if (auto marker = load->getAttr("was_bool_to_int8"))
      replacement->setAttr("was_bool_to_int8", marker);
    if (inColumnRegion)
      bindings.bindGenerated(load.getResult(), replacement.getResult(), true);
    else
      r.replaceOp(op, reshape(replacement, originalType.getShape()));
  } else {
    auto store = cast<triton::StoreOp>(op);
    auto originalType = dyn_cast<RankedTensorType>(store.getValue().getType());
    if (!originalType ||
        originalType.getShape() != ArrayRef<int64_t>(expectedShape))
      return function.emitError("invalid semantic store shape for memory view"),
             failure();
    Value stored = inColumnRegion ? materializeRegion(store.getValue())
                                  : reshape(store.getValue(), targetShape);
    if (!stored)
      return function.emitError("missing pointwise store value mapping"),
             failure();
    auto replacement = r.create<triton::StoreOp>(
        loc, pointer, stored, mask, store.getBoundaryCheck(), store.getCache(),
        store.getEvict());
    copyBatchEffectIdentity(op, replacement);
    r.eraseOp(op);
  }
  if (plan.symbolic)
    result.runtimeGuardJSON =
        layoutRuntimeGuard(plan, fact, facts, targetShape, coordinateMap,
                           fusedTarget, axis, factor);
  result.rewritten = true;
  return success();
}

LogicalResult BatchMemoryRewriter::finish(ModuleOp sandbox, bool columnRegion) {
  if (columnRegion) {
    SmallVector<Operation *> dead;
    for (Operation &op : *block)
      if (oldOperations.contains(&op) || loads.contains(&op))
        dead.push_back(&op);
    for (Operation *op : llvm::reverse(dead))
      if (op->use_empty())
        r.eraseOp(op);
    for (Operation &op : *block)
      if (loads.contains(&op))
        return sandbox.emitError(
                   "pointwise layout left an unbridged semantic load"),
               failure();
  }
  if (failed(verify(sandbox)))
    return sandbox.emitError("block batching memory view failed verification"),
           failure();
  return success();
}
} // namespace mlir::triton::cfg
