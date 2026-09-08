//===- StagePartitioner.h - Build semantic Stage IR ----------*- C++ -*-===//

#ifndef ASCENDMODEL_ANALYSIS_STAGEPARTITIONER_H
#define ASCENDMODEL_ANALYSIS_STAGEPARTITIONER_H

#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ascend {

struct StagePartitionerOptions {
  int64_t tinyDotFlopsMax = 16384;
  int64_t maximumSuperblockFactor = 1;
  bool scopeSuperblockMaterializable = false;
  /// Anchor-free Stages may become StageOwnedScope local SIMT scopes only on
  /// targets whose backend can materialize local scope.scope regions.
  bool compileOn91095 = true;
  /// Expose the direct body operations of independent structured loops
  /// (no true loop-carried data dependency) as semantic roots, so the loop
  /// shell is charged only its backedge overhead while the body is partitioned
  /// with the normal Stage rules.  Loops carrying a real recurrence stay
  /// atomic.  Always on: independent loop bodies are semantic roots by
  /// default.
  bool splitIndependentLoopBody = true;
};

/// Ordered post-transform TTIR semantic roots.  AutoBlockify V1's outer loop
/// is represented as a scheduling shell while its direct body operations
/// remain semantic roots; this prevents double ownership.  Exact local SIMT
/// ownership is derived separately from the immutable SimtAnchorPlan.
struct ProgramStructure {
  std::vector<Operation *> rootOperations;
  /// Mirrors StagePartitionerOptions::splitIndependentLoopBody so every
  /// later analysis resolves roots with the same loop-body exposure rule.
  bool splitIndependentLoopBody = true;
};

class ProgramStructureAnalysis {
public:
  llvm::Expected<ProgramStructure>
  analyze(ModuleOp module, const SimtAnchorPlan &anchorPlan,
          bool splitIndependentLoopBody = true) const;
};

/// Splits ordered semantic roots directly into single-kind Stages.  It does
/// not evaluate cycles or choose SIMD/SIMT.  The anchor plan is used only as
/// exact ownership evidence for materializable local SIMT Stages; anchor-free
/// Stages may additionally become StageOwnedScope units when the target can
/// materialize local scopes.
class StageBoundaryAnalysis {
public:
  llvm::Expected<StagePartition>
  analyze(const ProgramStructure &structure,
          const SimtAnchorPlan &anchorPlan,
          bool compileOn91095 = true) const;
};

/// Derives structural facts for every already-owned Stage.  It never chooses
/// a route and never reads a hardware throughput profile.
class StageFeatureAnalysis {
public:
  llvm::Error analyze(StagePartition &partition) const;
};

/// Verifies/classifies the one dominant resource semantics of every Stage.
/// Strong structures (recurrence, reduction, dot, indirect/continuous
/// memory, conversion) are inferred from owned operations and must agree
/// with the boundary plan.  Scalar sub-kinds remain boundary semantics.
class StageKindClassifier {
public:
  llvm::Error analyze(StagePartition &partition, int64_t tinyDotFlopsMax) const;
};

/// Verifies that operation/resource work is owned exactly once.  Unlike the
/// retired implementation, this component never distributes a kernel total
/// using per-kind weights.
class StageWorkloadAnalysis {
public:
  llvm::Error analyze(StagePartition &partition) const;
};

/// Derives legal SIMD/SIMT implementations from structural Stage facts.
class StageModeLegalityAnalysis {
public:
  llvm::Error analyze(StagePartition &partition,
                      int64_t maximumSuperblockFactor = 4,
                      bool scopeSuperblockMaterializable = false) const;
};

class StagePartitionVerifier {
public:
  llvm::Error verify(const StagePartition &partition) const;
};

/// Partitions post-layout/post-AutoBlockify-V1 TTIR directly into ordered,
/// single-mode Stages.  A Stage may later be evaluated as SIMD or SIMT, but it
/// is never internally mixed.
class StagePartitioner {
public:
  llvm::Expected<StagePartition>
  partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
            const StagePartitionerOptions &options) const;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_ANALYSIS_STAGEPARTITIONER_H
