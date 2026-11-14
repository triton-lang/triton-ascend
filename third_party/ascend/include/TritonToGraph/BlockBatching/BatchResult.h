// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_RESULT_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_RESULT_H
#include <optional>
#include <string>
#include <utility>

namespace mlir::triton::cfg {
// The diagnostic must remain readable after the private analysis IR is erased.
struct BatchDiagnostic {
  std::string reason, stage, operation, location, context;
  std::optional<unsigned> sourceID;
};

template <typename T> class BatchResult {
public:
  enum class Status { Success, NotApplicable, InternalError };
  static BatchResult success(T value) {
    return BatchResult(Status::Success, std::move(value), {});
  }
  static BatchResult decline(BatchDiagnostic diagnostic) {
    return BatchResult(Status::NotApplicable, std::nullopt,
                       std::move(diagnostic));
  }
  static BatchResult error(BatchDiagnostic diagnostic) {
    return BatchResult(Status::InternalError, std::nullopt,
                       std::move(diagnostic));
  }
  Status status() const { return state; }
  bool hasValue() const { return payload.has_value(); }
  T takeValue() && { return std::move(payload).value(); }
  const BatchDiagnostic &diagnostic() const { return detail; }

private:
  BatchResult(Status state, std::optional<T> payload, BatchDiagnostic detail)
      : state(state), payload(std::move(payload)), detail(std::move(detail)) {}
  Status state;
  std::optional<T> payload;
  BatchDiagnostic detail;
};
} // namespace mlir::triton::cfg
#endif
