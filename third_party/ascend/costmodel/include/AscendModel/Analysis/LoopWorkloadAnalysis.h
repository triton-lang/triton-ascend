//===- LoopWorkloadAnalysis.h - Describe runtime loop work --------*- C++
//-*-===//
#ifndef ASCEND_MODEL_ANALYSIS_LOOP_WORKLOAD_ANALYSIS_H
#define ASCEND_MODEL_ANALYSIS_LOOP_WORKLOAD_ANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <vector>

namespace mlir::ascend {

/// A read-only description of one invocation of an algorithm loop. Memory
/// expressions require a consistent readonly snapshot; they do not authorize
/// hoisting a load or synchronizing a device. Unknown expressions remain
/// unknown.
struct LoopTripCountInfo {
  std::string loopId;
  Operation *operation = nullptr;
  std::optional<int64_t> constantTripCount;
  /// Exact only for the scalar launch bindings carried by the analysis module.
  std::optional<int64_t> launchBoundTripCount;
  std::optional<int64_t> upperBoundTripCount;
  std::optional<int64_t> gridStrideTripCount;
  int64_t workloadTripCount = 1;
  bool workloadUsesUpperBound = false;
  bool workloadUsesFallback = false;
  bool supported = false;
  llvm::json::Object description;
};

/// The single authoritative parser for exact, bounded, launch-derived and
/// symbolic trip-count facts of one algorithm loop.
LoopTripCountInfo analyzeLoopTripCount(Operation *loop,
                                       int64_t logicalProgramCountHint = 0,
                                       llvm::StringRef loopId = {});

/// Loop IDs match StagePartitioner's function-local preorder, excluding
/// generated blockify shells (but including algorithm loops inside them).
std::vector<LoopTripCountInfo>
analyzeLoopTripCountInfos(ModuleOp module, int64_t logicalProgramCountHint = 0);

} // namespace mlir::ascend
#endif
