//===- PackInterleavedAccess.h -----------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_ASCEND_PACK_INTERLEAVED_ACCESS_H
#define TRITON_ASCEND_PACK_INTERLEAVED_ACCESS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton {
// A compile-time-only rewrite: preserve active addresses and reject motion
// across unknown pointer relationships. No runtime checks, second kernel, or
// synthetic noalias contracts are generated. Missing columns remain masked.
void packInterleavedAccesses(ModuleOp module);
std::unique_ptr<OperationPass<ModuleOp>> createPackInterleavedAccessPass();
} // namespace mlir::triton

#endif
