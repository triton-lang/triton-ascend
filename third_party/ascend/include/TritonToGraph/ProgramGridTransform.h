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
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H
#define TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace triton {
class FuncOp;
namespace cfg {

// This is an MLIR module attribute rather than a dialect attribute so it can
// be consumed by the Python compiler boundary and removed before a downstream
// backend that does not recognize hacc.* attributes.
inline constexpr llvm::StringLiteral kProgramGridTransformsAttr =
    "hacc.program_grid_transforms";
inline constexpr int64_t kProgramGridTransformsVersion = 2;

// A transform always uses ceil-div. Dynamic IAT/PTSM gets its extent from the
// two hidden i32 entry arguments; legacy SPAF retains logicalExtent as part of
// its fixed-grid contract.
struct ProgramGridTransform {
  int32_t order = 0;
  int32_t axis = 0;
  int64_t factor = 1;
  // Used only by the legacy fixed-grid SPAF contract. Dynamic IAT/PTSM leaves
  // it zero and receives its original extent through the hidden ABI instead.
  int64_t logicalExtent = 0;
  bool persistentCoverage = false;
  bool gridStrideAbiVerified = false;
};

struct ProgramGridTransformContract {
  int64_t version = kProgramGridTransformsVersion;
  // StaticProgramAxisFusion retains its pre-existing fixed-grid contract.
  // IAT/PTSM explicitly set this bit to select the version-2 dynamic ABI.
  bool dynamicOriginalGrid = false;
  SmallVector<ProgramGridTransform> transforms;
};

// Add/recognize the fixed internal ABI appended to the original TTIR entry
// arguments. They are raw launch-grid extents, not a user-visible DSL ABI.
LogicalResult addProgramGridHiddenExtentArguments(triton::FuncOp function);
bool hasProgramGridHiddenExtentArguments(triton::FuncOp function);

// Commit a verified program-mapping sandbox clone to its original function.
// The signature, complete argument-attribute array, and body form one hidden
// ABI transaction; callers publish the verified module contract afterwards.
LogicalResult commitProgramGridFunctionFromSandbox(triton::FuncOp destination,
                                                   triton::FuncOp source);

// Parse and validate only the three supported dynamic contracts:
// IAT16, IAT16+PTSM4, and PTSM64. Unknown keys or a fixed logical extent are
// rejected rather than silently retaining a specialization path.
FailureOr<ProgramGridTransformContract>
parseProgramGridTransformContract(Attribute attribute);

DictionaryAttr serializeProgramGridTransformContract(
    MLIRContext *context, const ProgramGridTransformContract &contract);

// The setter validates both the schema and the public entry's two trailing
// i32 arguments, then publishes the module contract atomically.
LogicalResult
setProgramGridTransformContract(ModuleOp module,
                                const ProgramGridTransformContract &contract);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H
