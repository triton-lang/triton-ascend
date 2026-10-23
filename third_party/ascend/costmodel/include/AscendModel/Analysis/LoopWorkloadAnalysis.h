//===- LoopWorkloadAnalysis.h - Describe runtime loop work --------*- C++
//-*-===//
#ifndef ASCEND_MODEL_ANALYSIS_LOOP_WORKLOAD_ANALYSIS_H
#define ASCEND_MODEL_ANALYSIS_LOOP_WORKLOAD_ANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <vector>

namespace mlir::ascend {

/// A read-only description of one invocation of an algorithm loop. Memory
/// expressions require a consistent readonly snapshot; they do not authorize
/// hoisting a load or synchronizing a device. Unknown expressions remain
/// unknown.
struct LoopWorkloadExpression {
  std::string loopId;
  Operation *operation = nullptr;
  std::optional<int64_t> constantTripCount;
  bool supported = false;
  llvm::json::Object description;
};

/// Loop IDs match StagePartitioner's function-local preorder, excluding
/// generated blockify shells (but including algorithm loops inside them).
std::vector<LoopWorkloadExpression>
analyzeLoopWorkloadExpressions(ModuleOp module);

} // namespace mlir::ascend
#endif
