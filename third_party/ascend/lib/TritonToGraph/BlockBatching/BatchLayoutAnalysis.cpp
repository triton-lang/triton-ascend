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
using Key = std::pair<K::Kind, int64_t>;
// The exact integer affine decomposition now lives in BatchAffine.h, shared
// with the loop judgment layer. This file keeps its historical leaf set
// (kBatchLayoutLeaves): induction variables still fail the decomposition here,
// so a loop-dependent bound still falls through to KEEP exactly as before.

// Parameter-affine addresses: coordinates vary inside the batch; scalar
// arguments/grid extents are uniform parameters of their coefficients. Keep
// scalar algebra bounded to affine parameter expressions (no parameter products
// or solver). This proves (p*M+r)*S structurally without specializing S.
struct Scalar {
  using Atom = std::tuple<K::Kind, int64_t, unsigned>;
  int64_t constant = 0;
  std::map<Atom, int64_t> parameters;
  bool isZero() const { return constant == 0 && parameters.empty(); }
  bool operator==(const Scalar &other) const {
    return constant == other.constant && parameters == other.parameters;
  }
  bool addScaled(const Scalar &other, int64_t scale) {
    int64_t value;
    if (llvm::MulOverflow(other.constant, scale, value) ||
        llvm::AddOverflow(constant, value, constant))
      return false;
    for (auto [atom, coefficient] : other.parameters) {
      if (llvm::MulOverflow(coefficient, scale, value) ||
          llvm::AddOverflow(parameters[atom], value, parameters[atom]))
        return false;
      if (!parameters[atom])
        parameters.erase(atom);
    }
    return parameters.size() <= 32;
  }
  std::optional<Scalar> multiply(const Scalar &other) const {
    if (!parameters.empty() && !other.parameters.empty())
      return std::nullopt;
    Scalar result;
    bool ok = parameters.empty() ? result.addScaled(other, constant)
                                 : result.addScaled(*this, other.constant);
    return ok ? std::optional<Scalar>(result) : std::nullopt;
  }
  std::optional<Scalar> divide(int64_t divisor) const {
    if (constant % divisor)
      return std::nullopt;
    Scalar result{constant / divisor, {}};
    for (auto [atom, coefficient] : parameters) {
      if (coefficient % divisor)
        return std::nullopt;
      result.parameters[atom] = coefficient / divisor;
    }
    return result;
  }
  E expression() const {
    E result = K::leaf(K::Constant, constant, 64);
    for (auto [atom, coefficient] : parameters) {
      auto [kind, id, bits] = atom;
      E term = K::leaf(kind, id, bits);
      if (bits != 64)
        term = K::node(K::ExtSI, 64, term, {}, bits);
      if (coefficient != 1)
        term = K::node(K::Mul, 64, term, K::leaf(K::Constant, coefficient, 64));
      if (result->kind == K::Constant && result->value == 0)
        result = term;
      else
        result = K::node(K::Add, 64, result, term);
    }
    return result;
  }
};
struct ParameterAffine {
  Scalar offset;
  std::map<Key, Scalar> terms;
  bool symbolic() const {
    return !offset.parameters.empty() ||
           llvm::any_of(terms, [](const auto &term) {
             return !term.second.parameters.empty();
           });
  }
};
std::optional<ParameterAffine> parameterAffine(E e, unsigned &budget) {
  if (!e || !budget)
    return std::nullopt;
  --budget;
  if (e->kind == K::Constant)
    return ParameterAffine{Scalar{e->value, {}}, {}};
  if (e->kind == K::Argument) {
    if (e->bits != 32 && e->bits != 64)
      return std::nullopt;
    Scalar value;
    value.parameters[{e->kind, e->value, e->bits}] = 1;
    return ParameterAffine{value, {}};
  }
  // Induction 在这里和坐标同级：normalizeBatchAccessOnLogicalAxis 已经把循环
  // 变量换成「下界 + 圈内偏移」，剩下的这个叶子代表圈内偏移，各份相同。
  if (e->kind == K::Coordinate || e->kind == K::Program || e->kind == K::Grid ||
      e->kind == K::Induction) {
    ParameterAffine value;
    value.terms[{e->kind, e->value}] = Scalar{1, {}};
    return value;
  }
  if (e->kind == K::ExtSI)
    return parameterAffine(e->lhs, budget);
  if (e->kind != K::Add && e->kind != K::Sub && e->kind != K::Mul)
    return std::nullopt;
  auto a = parameterAffine(e->lhs, budget);
  auto b = parameterAffine(e->rhs, budget);
  if (!a || !b)
    return std::nullopt;
  if (e->kind == K::Mul) {
    if (!a->terms.empty() && !b->terms.empty())
      return std::nullopt;
    if (b->terms.empty())
      std::swap(a, b);
    auto offset = b->offset.multiply(a->offset);
    if (!offset)
      return std::nullopt;
    b->offset = *offset;
    for (auto &[key, coefficient] : b->terms) {
      auto product = coefficient.multiply(a->offset);
      if (!product)
        return std::nullopt;
      coefficient = *product;
    }
    for (auto it = b->terms.begin(); it != b->terms.end();)
      if (it->second.isZero())
        it = b->terms.erase(it);
      else
        ++it;
    return b;
  }
  int64_t sign = e->kind == K::Sub ? -1 : 1;
  if (!a->offset.addScaled(b->offset, sign))
    return std::nullopt;
  for (const auto &[key, coefficient] : b->terms) {
    if (!a->terms[key].addScaled(coefficient, sign))
      return std::nullopt;
    if (a->terms[key].isZero())
      a->terms.erase(key);
  }
  return a->terms.size() <= 32 ? a : std::nullopt;
}
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
bool bindProgramDirection(E predicate, const BatchAccessFact &fact,
                          BatchLayoutPlan &plan) {
  if (predicate->kind != K::SLT || contains(predicate->rhs, K::Coordinate) ||
      contains(predicate->rhs, K::Program))
    return false;
  unsigned budget = 256;
  auto form = batchLinearForm(predicate->lhs, budget, kBatchLoopLeaves);
  if (!form)
    return false;
  SmallVector<int64_t> axes;
  for (auto [key, value] : form->terms) {
    if (!value)
      continue;
    if (key.first == K::Coordinate && value > 0)
      axes.push_back(key.second);
    // 圈内偏移对整块是一个各份相同的平移，不参与方向绑定，放行但不记轴。
    else if (key.first == K::Induction)
      continue;
    else if (key != Key{K::Program, plan.gridAxis})
      return false;
  }
  if (axes.empty())
    return false;
  int64_t extent = 1;
  for (int64_t axis : llvm::reverse(axes)) {
    if (axis < 0 || axis >= static_cast<int64_t>(fact.shape.size()) ||
        form->terms[{K::Coordinate, axis}] != extent || fact.shape[axis] <= 0 ||
        llvm::MulOverflow(extent, fact.shape[axis], extent))
      return false;
  }
  if (form->terms[{K::Program, plan.gridAxis}] != extent)
    return false;
  plan.localAxes = axes;
  plan.tileExtent = extent;
  plan.origin = form->constant;
  plan.coordinate = predicate->lhs;
  plan.upper = predicate->rhs;
  return true;
}
} // namespace
SmallVector<BatchLayoutPlan, 0>
planBatchLayouts(const BatchAccessReport &accesses, int axis, int64_t factor) {
  SmallVector<BatchLayoutPlan, 0> plans;
  for (auto [id, originalFact] : llvm::enumerate(accesses.accesses)) {
    // 把循环变量接回程序号，再往下全程用归一后的这一份。不接的话，一条按行走
    // 的访存地址里只有归纳变量、没有 K::Program，下面第一个分支就会判它
    // SHARED_ACCESS——字面正确但结论错，程序号只是藏在循环下界里。
    //
    // 地址与掩码必须一起换、而且只换一次：下面要拿 plan.coordinate 和谓词做
    // 指针相等比较，两边来源不同就对不上。
    BatchAccessFact fact = originalFact;
    bool normalized = true;
    for (const BatchLoopFact &loop : originalFact.loops) {
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
        isa<triton::FuncOp>(fact.operation->getParentOp()) &&
        !fact.deviceAddress && fact.byteOffset &&
        !contains(fact.byteOffset, K::Unknown) &&
        !contains(fact.mask, K::Unknown);
    llvm::raw_string_ostream stream(plan.location);
    fact.operation->getLoc().print(stream);
    if (!normalized) {
      // 归一不下去就按证不出处理，绝不退回原式：那等于把"程序号不在地址里"这个
      // 错误结论静默放行。
      plan.reason = "LOOP_ORIGIN_UNPROVEN";
      plans.push_back(std::move(plan));
      continue;
    }
    SmallVector<E> predicates;
    conjuncts(fact.mask, predicates);
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
      auto address = parameterAffine(fact.byteOffset, budget);
      bool strides = bool(address);
      if (address) {
        int64_t weight = 1;
        Scalar stride = address->terms[{K::Coordinate, plan.localAxes.back()}];
        // Positive dynamic stride is checked at launch, never sampled here.
        strides = !stride.parameters.empty() || stride.constant > 0;
        for (int64_t local : llvm::reverse(plan.localAxes)) {
          auto expected = stride.multiply(Scalar{weight, {}});
          if (!expected ||
              !(address->terms[{K::Coordinate, local}] == *expected))
            strides = false;
          if (llvm::MulOverflow(weight, fact.shape[local], weight))
            strides = false;
        }
        auto expected = stride.multiply(Scalar{weight, {}});
        if (!expected || !(address->terms[{K::Program, axis}] == *expected))
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
        // 这里原本还带一条 fact.loops.empty()。它挡住的不只是程序轴循环——块内
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
            address->terms[{K::Coordinate, plan.localAxes.back()}].divide(
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
        bool contiguousColumn =
            columnRegion && stride->parameters.empty() && stride->constant == 1;
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
          if (coefficient.isZero() || key == Key{K::Program, axis} ||
              (key.first == K::Coordinate &&
               llvm::is_contained(plan.localAxes, key.second)))
            continue;
          auto scaled = coefficient.divide(fact.bytes);
          if (!scaled) {
            simple = false;
            continue;
          }
          // Induction 与前三者同级：它代表圈内偏移，发射器从外层 scf.for 取回
          // 循环变量（BatchMemoryView.cpp 的 Emitter::upper）。
          if (key.first != K::Coordinate && key.first != K::Program &&
              key.first != K::Grid && key.first != K::Induction)
            simple = false;
          // A closed unit-column region retains other coordinates and their
          // parameter-affine pitches, without permuting its computation.
          // Other layouts still require the previously supported loop term.
          if (!scaled->parameters.empty() && key.first != K::Induction &&
              !contiguousColumn)
            simple = false;
          int64_t c = scaled->constant;
          plan.restTerms.push_back(
              {key.first, key.second, scaled->expression()});
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
std::string serializeBatchLayouts(ArrayRef<BatchLayoutPlan> plans) {
  llvm::json::Array rows;
  for (const auto &plan : plans) {
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
        {"action", plan.action == BatchLayoutPlan::Keep ? "Keep"
                   : plan.action == BatchLayoutPlan::MemoryView
                       ? "MemoryView"
                       : "PointwiseRegion"},
        {"selected_grid_axis", plan.gridAxis},
        {"local_axes", std::move(axes)},
        {"original_shape", std::move(shape)},
        {"effective_shape", std::move(effectiveShape)},
        {"control_flow_independent", plan.controlFlowIndependent},
        {"tile_extent", plan.tileExtent},
        {"origin", plan.origin},
        {"reason", plan.reason},
        {"mask_class", plan.maskClass},
        {"source_location", plan.location},
        {"rewritten", plan.rewritten},
        {"bridge", !plan.rewritten ? "not_applied"
                   : plan.action == BatchLayoutPlan::PointwiseRegion
                       ? "direct_pointwise"
                       : "ordered_view"},
        {"expected_memory_form",
         plan.rewritten ? "strided_block_candidate" : "semantic"},
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
