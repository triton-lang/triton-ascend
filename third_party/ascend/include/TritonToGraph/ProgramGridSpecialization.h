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

#ifndef TRITON_TO_GRAPH_PROGRAM_GRID_SPECIALIZATION_H
#define TRITON_TO_GRAPH_PROGRAM_GRID_SPECIALIZATION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <string>

namespace mlir {
namespace triton {
namespace cfg {

// The full pre-transform grid is a static compiler input.  It is published on
// both the module and every tt.func so GraphOptimize rules can consume it
// without recovering extents from transformed tt.get_num_programs values.
inline constexpr llvm::StringLiteral kProgramGridSpecializationAttr =
    "hacc.grid_specialization";
inline constexpr int64_t kProgramGridSpecializationVersion = 1;

// Exact integer JIT arguments that are safe to substitute into a TTIR entry
// function before program-axis dependence analysis.  The JIT includes this
// contract in its cache key, so a compiled artifact can never be reused for a
// different runtime value.  It is consumed by GraphOptimize and never forms
// part of the launcher ABI.
inline constexpr llvm::StringLiteral kProgramMappingScalarSpecializationAttr =
    "hacc.program_mapping_scalar_specialization";
inline constexpr int64_t kProgramMappingScalarSpecializationLegacyVersion = 1;
inline constexpr int64_t kProgramMappingScalarSpecializationVersion = 2;

struct ProgramGridSpecialization {
  int64_t version = kProgramGridSpecializationVersion;
  std::array<int64_t, 3> grid = {1, 1, 1};
  uint32_t ruleMask = 0;
};

struct ProgramMappingScalarArgument {
  uint32_t index = 0;
  // Empty for the positional v1 contract.  Version 2 names the original JIT
  // parameter so a frontend-pruned TTIR signature cannot shift the target.
  std::string name;
  int64_t value = 0;
};

struct ProgramMappingScalarSpecialization {
  int64_t version = kProgramMappingScalarSpecializationVersion;
  SmallVector<ProgramMappingScalarArgument> arguments;
};

// Parse the fixed v1 generic attribute.  The bridge intentionally knows only
// the three stage-00 mapping bits (IAT/SPAF/PTSM), so unknown future bits fail
// rather than changing JIT cache timing accidentally.
FailureOr<ProgramGridSpecialization>
parseProgramGridSpecialization(Attribute attribute);

DictionaryAttr serializeProgramGridSpecialization(
    MLIRContext *context, const ProgramGridSpecialization &specialization);

// Set/clear module and function copies together.  Keeping this operation in
// C++ prevents a Python caller from accidentally leaving a hacc.* attr on a
// nested tt.func when the downstream vendor compiler is invoked.
LogicalResult
setProgramGridSpecialization(ModuleOp module,
                             const ProgramGridSpecialization &specialization);
void clearProgramGridSpecialization(ModuleOp module);

FailureOr<ProgramMappingScalarSpecialization>
parseProgramMappingScalarSpecialization(Attribute attribute);

DictionaryAttr serializeProgramMappingScalarSpecialization(
    MLIRContext *context,
    const ProgramMappingScalarSpecialization &specialization);

LogicalResult setProgramMappingScalarSpecialization(
    ModuleOp module, const ProgramMappingScalarSpecialization &specialization);
void clearProgramMappingScalarSpecialization(ModuleOp module);

// Replace only compatible public-entry integer block arguments with exact
// constants and then remove every copy of the transient contract.  Legacy v1
// uses an entry position; v2 uses the retained NameLoc and is safe if frontend
// canonicalization has removed earlier arguments.  Incompatible or ambiguous
// targets are deliberately left dynamic, so program mapping remains
// fail-closed through its ordinary dependence analysis.
LogicalResult applyProgramMappingScalarSpecialization(ModuleOp module);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_PROGRAM_GRID_SPECIALIZATION_H
