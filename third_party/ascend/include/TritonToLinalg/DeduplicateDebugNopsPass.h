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

#ifndef TRITON_ASCEND_TRITONTOLINALG_DEDUPLICATE_DEBUG_NOPS_H
#define TRITON_ASCEND_TRITONTOLINALG_DEDUPLICATE_DEBUG_NOPS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

/// Creates a pass that deduplicates `llvm.inline_asm "nop"` debug-anchor ops
/// inserted by the converters during triton-to-linalg lowering.
///
/// Each source line typically has many NOPs scattered through the lowered IR
/// because converters insert one NOP per occurrence (e.g., every consumer of
/// an `offsets` value gets a NOP at its source line). This produces noisy
/// DWARF line tables with the same source line repeated at many PCs, which
/// makes debugger `next`-stepping bounce around.
///
/// This pass keeps only the FIRST NOP per unique source line (by `(filename,
/// line)`) within each function, in IR walk order. The first NOP gets a real
/// PC; subsequent duplicates are erased.
///
/// The pass is opt-in: it does nothing unless the env var
/// `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1` is set. Production compiles are
/// unaffected.
std::unique_ptr<Pass> createDeduplicateDebugNopsPass();

#define GEN_PASS_DECL_DEDUPLICATEDEBUGNOPS
#include "ascend/include/TritonToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_TRITONTOLINALG_DEDUPLICATE_DEBUG_NOPS_H
