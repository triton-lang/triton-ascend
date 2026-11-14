// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchAffine.h"
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>
#include <tuple>
namespace mlir::triton::cfg {
namespace {
using E = BatchExprRef;
using K = BatchExpr;
using Key = BatchAffineForm::Key;
using Coefficient = BatchAffineForm::Coefficient;
bool contains(E e, K::Kind kind, int64_t id = -1) {
  if (!e)
    return false;
  return (e->kind == kind && (id < 0 || e->value == id)) ||
         contains(e->lhs, kind, id) || contains(e->rhs, kind, id);
}
void conjuncts(E mask, SmallVectorImpl<E> &out) {
  if (!mask)
    return;
  if (mask->kind == K::And) {
    conjuncts(mask->lhs, out);
    conjuncts(mask->rhs, out);
  } else
    out.push_back(mask);
}
// Block-pointer boundary checks include x >= 0, represented by -1 < x.
// Program ids and local coordinates are nonnegative. Remove this predicate
// from layout matching only when exact affine algebra proves it redundant.
// The original access mask and its no-wrap guard remain unchanged; arbitrary
// scalar parameters, induction variables and negative origins are not proven.
bool redundantNonnegativeBound(E predicate) {
  if (predicate->kind != K::SLT || !predicate->lhs ||
      predicate->lhs->kind != K::Constant || predicate->lhs->value != -1)
    return false;
  unsigned budget = 256;
  auto form = BatchAffineForm::fromExpr(
      predicate->rhs, budget,
      batchLeaf(K::Program) | batchLeaf(K::Coordinate) | batchLeaf(K::Grid),
      BatchAffineForm::Profile::Integer, -1);
  return form && form->offset.constant >= 0 &&
         llvm::all_of(form->terms, [](const auto &term) {
           return term.second.constant >= 0;
         });
}
void coordinateIDs(E e, SmallVectorImpl<int64_t> &ids) {
  if (!e)
    return;
  if (e->kind == K::Coordinate && !llvm::is_contained(ids, e->value))
    ids.push_back(e->value);
  coordinateIDs(e->lhs, ids);
  coordinateIDs(e->rhs, ids);
}
bool rangeGroup(E e, ArrayRef<int64_t> axes) {
  if (!e)
    return true;
  if (e->rangeOrigin) {
    SmallVector<int64_t> ids;
    coordinateIDs(e, ids);
    llvm::sort(ids);
    return ArrayRef<int64_t>(ids) == axes;
  }
  if (e->kind == K::Coordinate)
    return false;
  return rangeGroup(e->lhs, axes) && rangeGroup(e->rhs, axes);
}
// A bound on a coordinate, before byte scaling, associates that coordinate's
// local projections with the selected program direction. It excludes all
// other program directions, even if their strides happen to be equal.
// 名字不叫 bind：实参里有 std::shared_ptr，ADL 会把 std::bind 拉进候选集。
bool bindProgramDirection(E predicate, const BatchFacts::Access &fact,
                          BatchLayoutPlan &plan) {
  if (predicate->kind != K::SLT || contains(predicate->rhs, K::Coordinate) ||
      contains(predicate->rhs, K::Program))
    return false;
  unsigned budget = 256;
  auto form =
      BatchAffineForm::fromExpr(predicate->lhs, budget, kBatchLoopLeaves,
                                BatchAffineForm::Profile::Integer, -1);
  if (!form)
    return false;
  SmallVector<int64_t> axes;
  for (const auto &[key, coefficient] : form->terms) {
    int64_t value = coefficient.constant;
    if (!value)
      continue;
    if (std::get<0>(key) == K::Coordinate && value > 0)
      axes.push_back(std::get<1>(key));
    // 圈内偏移对整块是一个各份相同的平移，不参与方向绑定，放行但不记轴。
    else if (std::get<0>(key) == K::Induction)
      continue;
    else if (key != Key{K::Program, plan.gridAxis, 0})
      return false;
  }
  if (axes.empty())
    return false;
  int64_t extent = 1;
  for (int64_t axis : llvm::reverse(axes)) {
    if (axis < 0 || axis >= static_cast<int64_t>(fact.shape.size()) ||
        form->terms[{K::Coordinate, axis, 0}].constant != extent ||
        fact.shape[axis] <= 0 ||
        llvm::MulOverflow(extent, fact.shape[axis], extent))
      return false;
  }
  if (form->terms[{K::Program, plan.gridAxis, 0}].constant != extent)
    return false;
  plan.localAxes = axes;
  plan.tileExtent = extent;
  plan.origin = form->offset.constant;
  plan.coordinate = predicate->lhs;
  plan.upper = predicate->rhs;
  return true;
}
} // namespace
SmallVector<BatchLayoutPlan, 0>
planBatchLayouts(const BatchRewriteWorkspace &scope, int axis, int64_t factor) {
  SmallVector<BatchLayoutPlan, 0> plans;
  for (auto [id, originalFact] : llvm::enumerate(scope.facts->accesses())) {
    // 把循环变量接回程序号，再往下全程用归一后的这一份。不接的话，一条按行走
    // 的访存地址里只有归纳变量、没有 K::Program，下面第一个分支就会判它
    // SHARED_ACCESS——字面正确但结论错，程序号只是藏在循环下界里。
    //
    // 地址与掩码必须一起换、而且只换一次：下面要拿 plan.coordinate 和谓词做
    // 指针相等比较，两边来源不同就对不上。
    BatchFacts::Access fact = originalFact;
    bool normalized = true;
    for (unsigned id : originalFact.loopIDs) {
      const auto &loop = *scope.facts->findLoop(id);
      fact.byteOffset =
          normalizeBatchAccessOnLogicalAxis(fact.byteOffset, loop);
      fact.mask = normalizeBatchAccessOnLogicalAxis(fact.mask, loop);
      normalized &= bool(fact.byteOffset);
    }
    BatchLayoutPlan plan;
    plan.accessID = id;
    plan.gridAxis = axis;
    plan.shape = fact.shape;
    // Reuse the access proof: nested execution, device-derived addresses and
    // unknown expressions do not establish control-flow-independent memory.
    plan.controlFlowIndependent =
        isa<triton::FuncOp>(scope.sourceAccesses()[id]->getParentOp()) &&
        !fact.deviceAddress && fact.byteOffset &&
        !contains(fact.byteOffset, K::Unknown) &&
        !contains(fact.mask, K::Unknown);
    llvm::raw_string_ostream stream(plan.location);
    scope.sourceAccesses()[id]->getLoc().print(stream);
    if (!normalized) {
      // 归一不下去就按证不出处理，绝不退回原式：那等于把"程序号不在地址里"这个
      // 错误结论静默放行。
      plan.reason = "LOOP_ORIGIN_UNPROVEN";
      plans.push_back(std::move(plan));
      continue;
    }
    SmallVector<E> predicates;
    conjuncts(fact.mask, predicates);
    llvm::erase_if(predicates, redundantNonnegativeBound);
    bool bound = false, unknown = false;
    for (auto predicate : predicates) {
      if (predicate->kind == K::Unknown)
        unknown = true;
      if (contains(predicate, K::Program, axis)) {
        if (bound || !bindProgramDirection(predicate, fact, plan)) {
          unknown = true;
          continue;
        }
        bound = true;
      }
    }
    if (!contains(fact.byteOffset, K::Program, axis)) {
      plan.reason = "SHARED_ACCESS";
      plan.maskClass = "shared";
    } else if (!bound && !unknown && !contains(fact.mask, K::Program, axis)) {
      plan.reason = "ALREADY_STRUCTURED";
      plan.maskClass = "independent_axes";
    } else if (bound && !unknown) {
      // Direction is established above. Check the byte-stride chain only now.
      unsigned budget = 512;
      auto address = BatchAffineForm::fromExpr(
          fact.byteOffset, budget,
          kBatchLoopLeaves | batchLeaf(K::ProgramQuotient) |
              batchLeaf(K::ProgramRemainder),
          BatchAffineForm::Profile::Parameter, axis);
      bool strides = bool(address);
      if (address) {
        int64_t weight = 1;
        Coefficient stride =
            address->terms[{K::Coordinate, plan.localAxes.back(), 0}];
        // Positive dynamic stride is checked at launch, never sampled here.
        strides = !stride.parameters.empty() || stride.constant > 0;
        for (int64_t local : llvm::reverse(plan.localAxes)) {
          auto expected = stride.multiply(Coefficient{weight, {}});
          if (!expected ||
              !(address->terms[{K::Coordinate, local, 0}] == *expected))
            strides = false;
          if (llvm::MulOverflow(weight, fact.shape[local], weight))
            strides = false;
        }
        auto expected = stride.multiply(Coefficient{weight, {}});
        if (!expected || !(address->terms[{K::Program, axis, 0}] == *expected))
          strides = false;
      }
      bool extraMask = false;
      for (auto predicate : predicates) {
        if (predicate->lhs == plan.coordinate && predicate->rhs == plan.upper)
          continue;
        for (int64_t local : plan.localAxes)
          if (contains(predicate, K::Coordinate, local))
            extraMask = true;
      }
      int64_t total;
      if (llvm::MulOverflow(plan.tileExtent, factor, total) ||
          total > (1 << 20))
        plan.reason = "ANALYSIS_BUDGET_EXCEEDED";
      else if (plan.localAxes.size() > 1 &&
               !rangeGroup(plan.coordinate, plan.localAxes))
        plan.reason = "MULTI_AXIS_PROVENANCE_UNPROVEN";
      else if (!strides)
        plan.reason = "STRIDE_CHAIN_UNPROVEN";
      else if (extraMask)
        plan.reason = "MASK_NOT_INTERVAL_OR_RECTANGLE";
      else {
        plan.action = plan.localAxes.front() == 0
                          ? BatchLayoutPlan::MemoryView
                          : BatchLayoutPlan::PointwiseRegion;
        plan.maskClass = "linear_prefix";
        plan.reason = "PROVEN_DIRECTION_AND_STRIDE_CHAIN";
        // Memory-only collapse is free of permutations only for a leading
        // group. Nonadjacent groups need a proved pointwise region/cost plan.
        bool leading = true;
        for (auto [position, local] : llvm::enumerate(plan.localAxes))
          leading &= local == static_cast<int64_t>(position);
        bool columnRegion =
            fact.shape.size() == 2 && plan.localAxes == SmallVector<int64_t>{1};
        // A rank-one tile may still be relative to a separate grid row.
        // Preserve that uniform row origin just as for [1, N] tiles; the
        // closed-region verifier below the planner proves the computation.
        bool rowVector =
            fact.shape.size() == 1 && fact.loopIDs.empty() &&
            plan.localAxes == SmallVector<int64_t>{0} &&
            llvm::any_of(address->terms, [&](const auto &term) {
              auto [kind, id, divisor] = term.first;
              return kind == K::Program && id != axis && !term.second.isZero();
            });
        // 这里原本还带一条 fact.loopIDs.empty()。它挡住的不只是程序轴循环——块内
        // 循环今天就放行，这类访存一路证得出
        // PROVEN_DIRECTION_AND_STRIDE_CHAIN， 却因为这一条而 linearMemoryView
        // 为假，于是 BatchMemoryView.cpp:451
        // 一律跳过：证出来了从不改写。删掉之后它们第一次进入改写路径，这是本步
        // 唯一一处改变已发布行为的地方，干跑 diff 要逐行过目。
        //
        // 地址里真的用到循环变量时，圈内偏移会作为 K::Induction 的 rest term
        // 留下来，由下面两处和发射器一起接住。
        bool simple = (leading || columnRegion) && fact.shape.size() <= 3;
        auto origin = address->offset.divide(fact.bytes);
        auto stride =
            address->terms[{K::Coordinate, plan.localAxes.back(), 0}].divide(
                fact.bytes);
        if (!origin || !stride) {
          plan.action = BatchLayoutPlan::Keep;
          plan.reason = "NON_ELEMENT_ALIGNED_LAYOUT";
          plans.push_back(std::move(plan));
          continue;
        }
        plan.symbolic = address->symbolic();
        plan.elementOrigin = origin->expression();
        plan.elementStride = stride->expression();
        // Unit-stride columns can use the same closed pointwise-region map
        // with dynamic row pitches. The emitter preserves those pitches and
        // publishes guards for every new wide address intermediate. Keep
        // non-unit symbolic column strides on the original semantic path.
        bool contiguousColumn = (columnRegion || rowVector) &&
                                stride->parameters.empty() &&
                                stride->constant == 1;
        if (rowVector && contiguousColumn)
          plan.action = BatchLayoutPlan::PointwiseRegion;
        if (plan.symbolic && !leading && !contiguousColumn) {
          plan.action = BatchLayoutPlan::Keep;
          plan.reason = "SYMBOLIC_LAYOUT_BRIDGE_UNSUPPORTED";
          plans.push_back(std::move(plan));
          continue;
        }
        __int128 maxCoordinate =
            (__int128(INT32_MAX) + factor) * plan.tileExtent;
        __int128 magnitude = maxCoordinate * stride->constant;
        magnitude += origin->constant < 0 ? -__int128(origin->constant)
                                          : origin->constant;
        for (const auto &[key, coefficient] : address->terms) {
          auto [kind, id, divisor] = key;
          bool uniformDivision =
              kind == K::ProgramQuotient || kind == K::ProgramRemainder;
          if (coefficient.isZero() || key == Key{K::Program, axis, 0} ||
              (kind == K::Coordinate && llvm::is_contained(plan.localAxes, id)))
            continue;
          auto scaled = coefficient.divide(fact.bytes);
          if (!scaled) {
            simple = false;
            continue;
          }
          // Induction 与前三者同级：它代表圈内偏移，发射器从外层 scf.for 取回
          // 循环变量（BatchMemoryView.cpp 的 Emitter::upper）。
          if (kind != K::Coordinate && kind != K::Program && kind != K::Grid &&
              kind != K::Induction && !uniformDivision)
            simple = false;
          // A closed unit-column region retains other coordinates and their
          // parameter-affine pitches, without permuting its computation.
          // Other layouts still require the previously supported loop term.
          if (!scaled->parameters.empty() && kind != K::Induction &&
              !contiguousColumn && !uniformDivision)
            simple = false;
          // New wide scalar operations use the same layout guard contract.
          plan.symbolic |= uniformDivision;
          int64_t c = scaled->constant;
          plan.restTerms.push_back({kind, id, scaled->expression(), divisor});
          magnitude += (__int128(INT32_MAX) + 1) * (c < 0 ? -__int128(c) : c);
        }
        for (auto predicate : predicates)
          if (predicate->lhs != plan.coordinate || predicate->rhs != plan.upper)
            plan.residualMasks.push_back(predicate);
        __int128 maxIndex = maxCoordinate + plan.origin;
        bool endpoints =
            (plan.symbolic || magnitude * fact.bytes <= INT64_MAX) &&
            maxIndex <= INT64_MAX && maxIndex >= INT64_MIN;
        simple &= endpoints;
        if (!endpoints) {
          plan.action = BatchLayoutPlan::Keep;
          plan.reason = "NEW_BYTE_ENDPOINT_UNPROVEN";
        }
        plan.linearMemoryView = simple;
        if (simple && plan.symbolic)
          plan.reason = "PROVEN_SYMBOLIC_STRIDE_CHAIN";
        else if (plan.symbolic && !simple && endpoints) {
          plan.action = BatchLayoutPlan::Keep;
          plan.reason = "SYMBOLIC_LAYOUT_BRIDGE_UNSUPPORTED";
        }
      }
    } else if (unknown)
      plan.reason = "MASK_NOT_INTERVAL_OR_RECTANGLE";
    plans.push_back(std::move(plan));
  }
  return plans;
}
SmallVector<BatchLayoutResult, 0>
initialBatchLayoutResults(ArrayRef<BatchLayoutPlan> plans) {
  SmallVector<BatchLayoutResult, 0> results;
  for (const auto &plan : plans)
    results.push_back({plan.accessID, plan.action, plan.reason, false, {}});
  return results;
}
std::string serializeBatchLayouts(ArrayRef<BatchLayoutPlan> plans) {
  return serializeBatchLayoutResults(plans, initialBatchLayoutResults(plans));
}
std::string serializeBatchLayoutResults(ArrayRef<BatchLayoutPlan> plans,
                                        ArrayRef<BatchLayoutResult> results) {
  llvm::json::Array rows;
  for (auto [plan, result] : llvm::zip_equal(plans, results)) {
    llvm::json::Array axes, shape, effectiveShape;
    for (auto axis : plan.localAxes)
      axes.push_back(axis);
    for (auto extent : plan.shape) {
      shape.push_back(extent);
      if (extent != 1)
        effectiveShape.push_back(extent);
    }
    rows.push_back(llvm::json::Object{
        {"access_id", plan.accessID},
        {"action", result.action == BatchLayoutPlan::Keep ? "Keep"
                   : result.action == BatchLayoutPlan::MemoryView
                       ? "MemoryView"
                       : "PointwiseRegion"},
        {"selected_grid_axis", plan.gridAxis},
        {"local_axes", std::move(axes)},
        {"original_shape", std::move(shape)},
        {"effective_shape", std::move(effectiveShape)},
        {"control_flow_independent", plan.controlFlowIndependent},
        {"tile_extent", plan.tileExtent},
        {"origin", plan.origin},
        {"reason", result.reason},
        {"mask_class", plan.maskClass},
        {"source_location", plan.location},
        {"rewritten", result.rewritten},
        {"bridge", !result.rewritten ? "not_applied"
                   : result.action == BatchLayoutPlan::PointwiseRegion
                       ? "direct_pointwise"
                       : "ordered_view"},
        {"expected_memory_form",
         result.rewritten ? "strided_block_candidate" : "semantic"},
        {"symbolic_stride", plan.symbolic},
        {"element_stride", batchExprJSON(plan.elementStride)},
        {"proof_scope", plan.symbolic ? "original_and_layout_runtime_guard"
                                      : "existing_guard_no_wrap"}});
  }
  return llvm::formatv("{0}",
                       llvm::json::Value(llvm::json::Object{
                           {"planner_version", kBatchLayoutPlannerVersion},
                           {"accesses", std::move(rows)}}))
      .str();
}
} // namespace mlir::triton::cfg
