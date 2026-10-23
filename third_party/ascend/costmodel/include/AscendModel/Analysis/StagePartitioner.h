//===- StagePartitioner.h - Build semantic Stage IR ----------*- C++ -*-===//

#ifndef ASCENDMODEL_ANALYSIS_STAGEPARTITIONER_H
#define ASCENDMODEL_ANALYSIS_STAGEPARTITIONER_H

#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ascend {

/// Analysis-only counterfactual counts, keyed by prepared TTIR operations.
using LoopTripCountMap = llvm::DenseMap<Operation *, int64_t>;

struct StagePartitionerOptions {
  int64_t tinyDotFlopsMax = 16384;
  int64_t maximumSuperblockFactor = 1;
  int64_t maximumScopeSuperblockFactor = 4;
  int64_t logicalProgramCountHint = 0;
  bool scopeSuperblockMaterializable = false;
  const LoopTripCountMap *counterfactualLoopTripCounts = nullptr;
};

/// Ordered post-transform TTIR semantic roots.
struct ProgramStructure {
  std::vector<Operation *> rootOperations;
};

class ProgramStructureAnalysis {
public:
  llvm::Expected<ProgramStructure>
  analyze(ModuleOp module, const SimtAnchorPlan &anchorPlan) const;
};

/// Splits ordered semantic roots directly into single-kind Stages.
class StageBoundaryAnalysis {
public:
  llvm::Expected<StagePartition>
  analyze(const ProgramStructure &structure, const SimtAnchorPlan &anchorPlan,
          const LoopTripCountMap *loopTripCounts = nullptr) const;
};

/// Derives structural facts for every already-owned Stage.  It never chooses
/// a route and never reads a hardware throughput profile.
class StageFeatureAnalysis {
public:
  llvm::Error analyze(StagePartition &partition) const;
};

/// Verifies/classifies the one dominant resource semantics of every Stage.
class StageKindClassifier {
public:
  llvm::Error analyze(StagePartition &partition, int64_t tinyDotFlopsMax) const;
};

/// Verifies that operation/resource work is owned exactly once.
class StageWorkloadAnalysis {
public:
  llvm::Error analyze(StagePartition &partition,
                      int64_t logicalProgramCountHint = 0,
                      const LoopTripCountMap *loopTripCounts = nullptr) const;
};

/// Derives legal SIMD/SIMT implementations from structural Stage facts.
class StageModeLegalityAnalysis {
public:
  llvm::Error analyze(StagePartition &partition,
                      int64_t maximumSuperblockFactor = 4,
                      bool scopeSuperblockMaterializable = false,
                      int64_t maximumScopeSuperblockFactor = 4) const;
};

/// Applies profile-backed resource limits after the structural legality pass.
class StageResourceLegalityAnalysis {
public:
  llvm::Error applySimtUbCapacity(StagePartition &partition,
                                  int64_t usableUbBytes) const;
};

class StagePartitionVerifier {
public:
  llvm::Error verify(const StagePartition &partition) const;
};

/// Partitions post-layout/post-AutoBlockify-V1 TTIR directly into ordered,
/// single-mode Stages.
class StagePartitioner {
public:
  llvm::Expected<StagePartition>
  partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
            const StagePartitionerOptions &options) const;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_ANALYSIS_STAGEPARTITIONER_H
