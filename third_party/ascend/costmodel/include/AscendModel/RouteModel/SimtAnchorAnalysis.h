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
#include <variant>
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
};

llvm::StringRef stringifySimtAnchorKind(SimtAnchorKind kind);

/// Whether one physical route is available for an anchor.  BackendConditional
/// may be reported and scored offline, but it is not production-selectable.
enum class CandidateLoweringStatus {
  Unsupported,
  Native,
  BackendConditional,
  AliasesMixed,
};

llvm::StringRef
stringifyCandidateLoweringStatus(CandidateLoweringStatus status);

struct CandidateLowerability {
  CandidateLoweringStatus allSimd = CandidateLoweringStatus::Native;
  CandidateLoweringStatus allSimtOnly = CandidateLoweringStatus::Native;
  CandidateLoweringStatus mixed = CandidateLoweringStatus::Native;
  std::vector<std::string> allSimdReasons;
  std::vector<std::string> allSimtOnlyReasons;
  std::vector<std::string> mixedReasons;
};

struct TensorAtomicFacts {
  int64_t updateElements = 0;
  int64_t addressRank = 0;
  std::string valueType = "unknown";
  std::string offsetType = "unknown";
  std::string operation = "unknown";
  bool hasMask = false;
  std::optional<double> staticMaskActiveFraction;
  bool resultUsed = false;
  bool addressIsLaneVarying = false;
  bool addressDependsOnLoadedIndex = false;
  std::string contention = "unknown";
};

struct HistogramFacts {
  int64_t inputElements = 0;
  int64_t numBins = 0;
  std::string inputType = "unknown";
  std::string resultType = "unknown";
};

struct PlainCumsumFacts {
  int64_t axisExtent = 0;
  std::string elementType = "unknown";
  bool reverse = false;
};

using SimtAnchorFacts = std::variant<std::monostate, TensorAtomicFacts,
                                     HistogramFacts, PlainCumsumFacts>;

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
  SimtAnchorFacts facts;
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
  int64_t materializableCount() const;
};

/// Materialize exactly the local SIMT regions described by `plan`.
///
/// This is deliberately a transform over the immutable analysis result: it
/// does not rediscover anchors, recompute features, or read per-operation
/// selection attributes.  The caller owns the final effective route decision.
LogicalResult materializeSimtAnchorPlan(ModuleOp module,
                                        const SimtAnchorPlan &plan);

/// Classify a TTIR operation by SIMT mechanism, independently of whether the
/// selected target can currently materialize it.
std::optional<SimtAnchorKind> classifyMixedSimtAnchor(Operation *op);

/// True when a load/store pointer has an SSA backward slice that reaches a
/// loaded/gathered index.  This is a real data-dependence test and must not be
/// confused with the legacy rank-based laneDependentPointerOps proxy.
bool isLoadedIndexDependentMemoryOp(Operation *op);

/// Build the non-overlapping shared plan in pre-order.
SimtAnchorPlan buildMixedSimtAnchorPlan(ModuleOp module, bool compileOn91095);

/// Return true when `op` is a TTIR operation that the current local-scope
/// materializer can move into a SIMT scope on 910_95-class targets.
bool isMixedSimtAnchor(Operation *op, bool compileOn91095);

/// Collect non-overlapping anchors in pre-order.  Nested operations are not
/// returned when an enclosing operation already forms one materialized scope.
llvm::SmallVector<Operation *> collectMixedSimtAnchors(ModuleOp module,
                                                       bool compileOn91095);

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ROUTEMODEL_SIMTANCHORANALYSIS_H
