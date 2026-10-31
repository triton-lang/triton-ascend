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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_RULES_STRUCTURED_PTR_RULES_H
#define TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_RULES_STRUCTURED_PTR_RULES_H
#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
namespace mlir::triton::ptr {

/// Representative Structured rules, independent of the production converter.
/// Proven nonnegative constant decompositions and positive divisors create
/// new splits. Shared numerical combiners propagate existing splits through
/// arithmetic, shapes, casts and addptr; unsupported divisions stay exact.
class StructuredPtrRules {
public:
  using Result = CommonPtrRules::Result;
  explicit StructuredPtrRules(AnalysisOptions options = {}) : common(options) {}
  const AnalysisOptions &getOptions() const { return common.getOptions(); }
  LogicalResult validateBinding(Value value, const Result &info) const {
    return common.validateBinding(value, info);
  }
  void clearCaches() const { common.clearCaches(); }
  template <typename Context>
  FailureOr<Result> parse(Value value, Context &context) const {
    auto analyze = [&](Value input) { return context.analyze(input); };
    return parseOperation(value, analyze, context.getBuilder());
  }

private:
  using Analyze = CommonPtrRules::Analyze;
  FailureOr<Result> parseOperation(Value value, Analyze analyze,
                                   OpBuilder &builder) const;
  FailureOr<Result> parseDivRem(Value value, Value lhs, Value rhs, bool divide,
                                Analyze analyze, OpBuilder &builder) const;
  FailureOr<Result> refineDivRem(Value value, const Result &lhsInfo,
                                 const Result &rhsInfo, const Result &fallback,
                                 bool divide, OpBuilder &builder) const;
  CommonPtrRules common;
};
} // namespace mlir::triton::ptr
#endif
