/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef TRITON_ASCEND_PROTECT_TCB_REUSE_UTILS_H
#define TRITON_ASCEND_PROTECT_TCB_REUSE_UTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::tcbReuse {
/// Query an outer loop using its first direct main loop as a shared anchor.
using LoopPredicate = llvm::function_ref<bool(scf::ForOp, Operation *)>;

/// Pair enclosing loops accepted by matchesCandidate and insert a sourcePipe
/// to FIX handshake before their first main loops. Validate all candidates and
/// flag resources before changing IR; unsupported input only logs a skip.
/// Preconditions: matchesCandidate is a pure query and does not modify IR.
/// Main loops are not nested; each logical ID identifies one C/V pair.
void protectTcbReuse(ModuleOp module, StringRef protectionName,
                     hivm::PIPE sourcePipe, LoopPredicate matchesCandidate);
} // namespace mlir::triton::tcbReuse

#endif
