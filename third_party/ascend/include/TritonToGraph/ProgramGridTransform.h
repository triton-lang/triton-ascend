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
 */

#ifndef TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H
#define TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H

#include "TritonToGraph/ProgramGridSpecialization.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>

namespace mlir {
namespace triton {
namespace cfg {

struct ResourceSnapshot;

// This is an MLIR module attribute rather than a dialect attribute so it can
// be consumed by the Python compiler boundary and removed before a downstream
// backend that does not recognize hacc.* attributes.
inline constexpr llvm::StringLiteral kProgramGridTransformsAttr =
    "hacc.program_grid_transforms";
inline constexpr int64_t kProgramGridTransformsVersion = 1;

// A transform always uses ceil-div.  More transform kinds require a schema
// version bump so an older launcher rejects them rather than guessing.
struct ProgramGridTransform {
  int32_t order = 0;
  int32_t axis = 0;
  int64_t factor = 1;
  // The original, pre-transform extent.  It is retained by the rewrite for
  // tail masks and is checked by the launcher before it changes the grid.
  int64_t logicalExtent = 0;
  bool persistentCoverage = false;
  // A persistent cap is valid only if the rewritten kernel uses its actual
  // tt.get_num_programs(axis) as the grid-stride increment.
  bool gridStrideAbiVerified = false;
};

struct ProgramGridTransformContract {
  int64_t version = kProgramGridTransformsVersion;
  SmallVector<ProgramGridTransform> transforms;
};

// The exact launcher projection shared by program-mapping rules. ``logical``
// is the transformed grid before a persistent cap; ``physical`` is the actual
// Block Num emitted by the launcher. A nonpersistent transform intentionally
// disables legacy auto-map because it has no grid-stride coverage proof.
struct ProgramMappingLaunchProjection {
  std::array<uint64_t, 3> logicalGrid = {1, 1, 1};
  std::array<uint64_t, 3> physicalGrid = {1, 1, 1};
  uint64_t logicalPrograms = 0;
  uint64_t physicalPrograms = 0;
  uint64_t physicalWaves = 0;
  bool legacyAutoMap = false;
  bool persistentCoverage = false;
};

// Mirrors the generated launcher order: ceil-div transforms first, then the
// one proven persistent cap; otherwise legacy auto-map caps only a launch
// without any transform contract. Invalid or unprovable inputs fail closed.
std::optional<ProgramMappingLaunchProjection>
projectProgramMappingLaunch(const ProgramGridSpecialization &specialization,
                            llvm::ArrayRef<ProgramGridTransform> transforms,
                            const ResourceSnapshot &resources);

// Parse and validate the version-1 contract.  It is intentionally fail-closed:
// unknown keys, versions, transform kinds, dynamic/missing logical extents,
// and unverified persistent coverage all fail.
FailureOr<ProgramGridTransformContract>
parseProgramGridTransformContract(Attribute attribute);

// Serialize a validated contract to the canonical generic MLIR attribute
// representation used by hacc.program_grid_transforms.
DictionaryAttr serializeProgramGridTransformContract(
    MLIRContext *context, const ProgramGridTransformContract &contract);

LogicalResult
setProgramGridTransformContract(ModuleOp module,
                                const ProgramGridTransformContract &contract);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H
