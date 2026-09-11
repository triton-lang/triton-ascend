//===- SimtAnchorAnalysis.h - Materializable SIMT anchors -------*- C++ -*-===//
//
// Shared TTIR pattern matching for the Route Model and scope materializer.
// Keeping this contract in one place guarantees that mixed-candidate costs
// describe the same operations that the selector can actually mark for SIMT.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_SIMTANCHORANALYSIS_H
#define ASCENDMODEL_ROUTEMODEL_SIMTANCHORANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

/// Hardware/lowering mechanism represented by one local SIMT anchor.  These
/// names describe what the backend will do; they are intentionally independent
/// of calibration workload names such as "tiny_irregular_dot".
enum class SimtAnchorKind {
  DirectGather,
  LoadedIndexDependentMemory,
  Histogram,
  PlainOneDimensionalCumsum,
  TensorAtomic,
  TriangularSolveLoop,
  /// Synthesized descriptor: no primitive anchor exists, so one whole
  /// LogicalStage's contiguous root range becomes the local SIMT scope.
  StageOwnedScope,
};

llvm::StringRef stringifySimtAnchorKind(SimtAnchorKind kind);

struct CandidateLowerability {
  bool allSimd = true;
  bool allSimtOnly = true;
  bool mixed = true;
};

/// Structural facts for a blockwise triangular recurrence such as solve_tril.
/// These are extracted from TTIR and deliberately avoid workload/function
/// names.  The dense dot tail is outside the SIMT anchor and must remain on the
/// SIMD/Cube side of a mixed route.
struct TriangularSolveFacts {
  int64_t blockRows = 0;
  int64_t blockColumns = 0;
  std::string accumulatorType = "unknown";
  int64_t recurrenceStartRow = 0;
  int64_t recurrenceLoopCount = 0;
  int64_t denseDotTailOps = 0;
  bool requiresCubeTailPartition = false;
};

struct SimtAnchorDescriptor {
  Operation *operation = nullptr;
  /// Exact top-level TTIR operations that will be moved into one local SIMT
  /// scope. A simple anchor contains only `operation`; compound mechanisms
  /// such as solve_tril may contain movable mask setup followed by the
  /// recurrence/final-update range. Feature extraction and materialization
  /// must consume this same set so the scored work is the work that the
  /// backend actually lowers.
  llvm::SmallVector<Operation *, 0> scopeOperations;
  /// Where to insert the scope before moving `scopeOperations`. This differs
  /// from scopeOperations.front() when pure mask setup is moved across the
  /// initial SIMD loads to reproduce the hand-written solve_tril scope.
  Operation *scopeInsertionPoint = nullptr;
  SimtAnchorKind kind = SimtAnchorKind::LoadedIndexDependentMemory;
  std::optional<TriangularSolveFacts> triangularSolve;
  CandidateLowerability lowerability;
  /// True only when the current target/materializer contract can turn this
  /// descriptor into a local SIMT scope.
  bool materializable = false;
};

/// One immutable analysis result shared by feature extraction, scoring, and
/// selector application.  Keeping Operation pointers here prevents those
/// stages from independently rediscovering different anchor sets.
struct SimtAnchorPlan {
  llvm::SmallVector<SimtAnchorDescriptor, 0> anchors;
  CandidateLowerability kernelLowerability;

  llvm::SmallVector<Operation *> materializableRoots() const;
};

/// Merge the materializable anchors owned by one LogicalStage into the exact
/// compound scope that will be scored and materialized.  When several anchor
/// operations share a block, every operation in the lexical interval between
/// the first and last anchor is included so operands are never captured from
/// after the new scope.  A failure means the Stage cannot be represented by
/// one local SIMT scope.
std::optional<SimtAnchorDescriptor>
mergeSimtStageAnchors(const SimtAnchorPlan &plan,
                      llvm::ArrayRef<unsigned> anchorIndices);

/// Materialize exactly the local SIMT regions described by `plan`.
///
/// This is deliberately a transform over the immutable analysis result: it
/// does not rediscover anchors, recompute features, or read per-operation
/// selection attributes.  The caller owns the final effective route decision.
LogicalResult materializeSimtAnchorPlan(ModuleOp module,
                                        const SimtAnchorPlan &plan,
                                        int64_t superblockFactor = 1);

/// True when a load/store pointer has an SSA backward slice that reaches a
/// loaded/gathered index.  This is a real data-dependence test and must not
/// be confused with the legacy rank-based laneDependentPointerOps proxy.
bool isLoadedIndexDependentMemoryOp(Operation *op);

/// Structural check: can `roots` be wrapped by one scope.scope region?  All
/// roots must live in one block and form a lexically contiguous range there
/// (no foreign operation between the first and last root), because the
/// materializer moves exactly this set into the scope.
bool isStageScopeWrappable(llvm::ArrayRef<Operation *> roots);

/// Synthesize a StageOwnedScope descriptor for an anchor-free Stage.  The
/// Stage's own root range becomes the local SIMT scope, so mixed remains
/// selectable for kernels with no primitive anchor.  Returns nullopt when
/// the range is not wrappable, would return pointer-like tensor state, or
/// when the target cannot materialize local scopes at all.
std::optional<SimtAnchorDescriptor>
buildStageOwnedScopeDescriptor(llvm::ArrayRef<Operation *> roots,
                               bool compileOn91095);

/// Build the non-overlapping shared plan in pre-order.
SimtAnchorPlan buildMixedSimtAnchorPlan(ModuleOp module, bool compileOn91095);

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ROUTEMODEL_SIMTANCHORANALYSIS_H
