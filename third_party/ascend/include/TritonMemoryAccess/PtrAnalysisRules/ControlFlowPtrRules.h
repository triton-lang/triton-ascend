/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_RULES_CONTROL_FLOW_PTR_RULES_H
#define TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_RULES_CONTROL_FLOW_PTR_RULES_H
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/PtrInfoUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <functional>
#include <map>
#include <memory>
#include <variant>
namespace mlir::triton::ptr {

/// A source/field label, not necessarily an independently available SSA value.
/// An empty source denotes canonical zero. Compare both identity and type.
struct ComponentIdentity {
  Value source;
  unsigned field = 0;
  bool operator==(const ComponentIdentity &other) const {
    return source == other.source && field == other.field;
  }
};
struct ComponentExpression;
struct ComponentInfo {
  Type type;
  ComponentIdentity identity;
  std::optional<OpFoldResult> concrete;
  std::shared_ptr<const ComponentExpression> expression;
  static ComponentInfo boundary(Value source, unsigned field, Type type);
  bool isZero() const { return !identity.source; }
  bool sameIdentity(const ComponentInfo &other) const {
    return type == other.type && identity == other.identity;
  }
};
/// An immutable numeric recipe recorded during planning. Rewriting evaluates
/// this recipe, without recognizing the original pointer expression again.
struct ComponentExpression {
  enum Kind {
    Add,
    Subtract,
    Multiply,
    SignedCast,
    UnsignedCast,
    Splat,
    Broadcast,
    ExpandDims
  } kind;
  SmallVector<ComponentInfo, 2> inputs;
  unsigned axis = 0;
};
struct SymbolicOffsetInfo {
  Type valueType;
  SmallVector<int64_t> shape;
  ComponentInfo fullOffsets;
  // Exact equality in valueType: full = origin + sum(k[d]*coefficient[d])
  // + irregular. origin is the scalar contribution, not the value at k=0.
  OffsetTerms<ComponentInfo> terms;
  bool isUniform() const;
};
struct SymbolicBlockInfo {
  SmallVector<ComponentInfo> parentShape, coordinates, strides;
  SmallVector<int32_t> order;
};
struct SymbolicPtrInfo {
  Type ptrType;
  ComponentInfo base; // A scalar base OR an opaque tensor pointer component.
  SymbolicOffsetInfo offsets;
  std::optional<SymbolicBlockInfo> block;
};

/// Representative CFO expression rules. No SCF traversal, joins, signature
/// planning or rewriting: consumers bind those boundaries and compare fields.
/// Planning creates attributes and recipes only, never helper operations.
class ControlFlowPtrRules {
public:
  using Result =
      std::variant<std::monostate, SymbolicOffsetInfo, SymbolicPtrInfo>;
  explicit ControlFlowPtrRules(AnalysisOptions options = {})
      : options(options) {}
  template <typename Context>
  FailureOr<Result> parse(Value value, Context &context) const {
    return parseOperation(value,
                          [&](Value input) { return context.analyze(input); });
  }
  LogicalResult validateBinding(Value value, const Result &info) const;
  void clearCaches() const {}

private:
  using Analyze = llvm::function_ref<FailureOr<Result>(Value)>;
  FailureOr<Result> parseOperation(Value value, Analyze analyze) const;
  FailureOr<Result> parseArithmetic(Value value, Analyze analyze) const;
  FailureOr<Result> parseShape(Value value, Analyze analyze) const;
  FailureOr<Result> parseCast(Value value, Analyze analyze) const;
  FailureOr<Result> parseAddPtr(triton::AddPtrOp op, Analyze analyze) const;
  FailureOr<Result> parseMakeBlock(triton::MakeTensorPtrOp op,
                                   Analyze analyze) const;
  FailureOr<Result> parseAdvance(triton::AdvanceOp op, Analyze analyze) const;
  AnalysisOptions options;
};

/// One rewrite session before a fixed operation result anchor. The resolver
/// maps identities to newly created region arguments; known concrete leaves can
/// be used directly. Cache lifetime ends before callers erase or replace those
/// SSA values. All generated operations are pure numeric/shape operations.
class ComponentMaterializer {
public:
  using Resolver = std::function<std::optional<OpFoldResult>(
      const ComponentIdentity &, Type)>;
  ComponentMaterializer(OpBuilder &builder, Value anchor,
                        Resolver resolver = {}, AnalysisOptions options = {})
      : builder(builder), anchor(anchor), resolver(std::move(resolver)),
        options(options) {}
  FailureOr<OpFoldResult> materialize(const ComponentInfo &info);

private:
  OpBuilder &builder;
  Value anchor;
  Resolver resolver;
  AnalysisOptions options;
  // Own recipes so temporary returned Results cannot leave dangling keys.
  std::map<std::shared_ptr<const ComponentExpression>, OpFoldResult,
           std::owner_less<std::shared_ptr<const ComponentExpression>>>
      cache;
};
} // namespace mlir::triton::ptr
#endif
