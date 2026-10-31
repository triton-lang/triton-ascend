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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_RULES_COMMON_PTR_RULES_H
#define TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_RULES_COMMON_PTR_RULES_H

#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/PtrInfoUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <variant>

namespace mlir::triton::ptr {
/// monostate means this Rules view is unavailable, not a lifecycle error.
using CommonPtrResult = std::variant<std::monostate, IntegerInfo, PtrInfo>;
inline const IntegerInfo *getIntegerInfo(const CommonPtrResult &result) {
  return std::get_if<IntegerInfo>(&result);
}
inline const PtrInfo *getPtrInfo(const CommonPtrResult &result) {
  return std::get_if<PtrInfo>(&result);
}

/// Concrete integer/pointer rules. All recursive input queries use the current
/// context.analyze through a borrowed callback. Parsers live in the
/// implementation file and combine already analyzed inputs with shared Info
/// utilities.
class CommonPtrRules {
public:
  using Result = CommonPtrResult;
  using Analyze = llvm::function_ref<FailureOr<Result>(Value)>;
  explicit CommonPtrRules(AnalysisOptions options = {}) : options(options) {}
  const AnalysisOptions &getOptions() const { return options; }
  LogicalResult validateBinding(Value value, const Result &info) const;
  void clearCaches() const {}

  template <typename Context>
  FailureOr<Result> parse(Value value, Context &context) const {
    auto analyze = [&](Value input) { return context.analyze(input); };
    return parseOperation(value, analyze, context.getBuilder());
  }

  /// Structured reuses this parser with its own session's recursive callback.
  /// The callback is borrowed only for this synchronous call.
  FailureOr<Result> parseOperation(Value value, Analyze analyze,
                                   OpBuilder &builder) const;

  /// Combine already analyzed operands at the caller-selected insertion point.
  FailureOr<Result> combineElementwise(Value value, const IntegerInfo &lhs,
                                       const IntegerInfo &rhs,
                                       OpBuilder &builder) const;

private:
  FailureOr<Result> parseElementwise(Value value, Analyze analyze,
                                     OpBuilder &builder) const;
  FailureOr<Result> parseReshape(triton::ReshapeOp op, Analyze analyze,
                                 OpBuilder &builder) const;
  FailureOr<Result> parseExtractSlice(tensor::ExtractSliceOp op,
                                      Analyze analyze,
                                      OpBuilder &builder) const;
  FailureOr<Result> parseTensorCast(tensor::CastOp op, Analyze analyze,
                                    OpBuilder &builder) const;
  FailureOr<Result> parseAdd(arith::AddIOp op, Analyze analyze,
                             OpBuilder &builder) const;
  FailureOr<Result> parseSub(arith::SubIOp op, Analyze analyze,
                             OpBuilder &builder) const;
  FailureOr<Result> parseMul(arith::MulIOp op, Analyze analyze,
                             OpBuilder &builder) const;
  FailureOr<Result> parseSplat(triton::SplatOp op, Analyze analyze,
                               OpBuilder &builder) const;
  FailureOr<Result> parseBroadcast(triton::BroadcastOp op, Analyze analyze,
                                   OpBuilder &builder) const;
  FailureOr<Result> parseExpandDims(triton::ExpandDimsOp op, Analyze analyze,
                                    OpBuilder &builder) const;
  FailureOr<Result> parseCast(Value value, Value input, bool unsignedCast,
                              Analyze analyze, OpBuilder &builder) const;
  FailureOr<Result> parseBitcast(triton::BitcastOp op, Analyze analyze,
                                 OpBuilder &builder) const;
  FailureOr<Result> parseSelect(arith::SelectOp op, Analyze analyze,
                                OpBuilder &builder) const;
  FailureOr<Result> parseAddPtr(triton::AddPtrOp op, Analyze analyze,
                                OpBuilder &builder) const;
  FailureOr<Result> parseMakeTensorPtr(triton::MakeTensorPtrOp op,
                                       Analyze analyze,
                                       OpBuilder &builder) const;
  FailureOr<Result> parseAdvance(triton::AdvanceOp op, Analyze analyze,
                                 OpBuilder &builder) const;

  AnalysisOptions options;
};

} // namespace mlir::triton::ptr

#endif // TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_RULES_COMMON_PTR_RULES_H
