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

inline constexpr llvm::StringLiteral kProgramGridSpecializationAttr =
    "hacc.grid_specialization";
inline constexpr int64_t kProgramGridSpecializationVersion = 1;

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
  std::string name;
  int64_t value = 0;
};

struct ProgramMappingScalarSpecialization {
  int64_t version = kProgramMappingScalarSpecializationVersion;
  SmallVector<ProgramMappingScalarArgument> arguments;
};

FailureOr<ProgramGridSpecialization>
parseProgramGridSpecialization(Attribute attribute);

DictionaryAttr serializeProgramGridSpecialization(
    MLIRContext *context, const ProgramGridSpecialization &specialization);

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

LogicalResult applyProgramMappingScalarSpecialization(ModuleOp module);

}
}
}

#endif
