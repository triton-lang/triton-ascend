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

#ifndef ASCEND_UTILS_DEBUGUTILS_H
#define ASCEND_UTILS_DEBUGUTILS_H

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Location.h>

namespace mlir {
class Operation;
class PatternRewriter;
class ConversionPatternRewriter;
} // namespace mlir

//===----------------------------------------------------------------------===//
// NOP insertion helpers (gated by TRITON_DEBUG). Definitions in DebugUtils.cpp.
//===----------------------------------------------------------------------===//

/// Unwrap CallSiteLoc (caller-preferring) and FusedLoc (last non-unknown) down
/// to a representative location for tagging a debug NOP.
mlir::Location unwrapFusedLocForDebug(mlir::Location loc);

/// Insert a side-effecting nop when TRITON_DEBUG=1 to preserve a source
/// location. Must be called before the op carrying the location is erased.
void insertDebugNop(mlir::Location loc, mlir::PatternRewriter &rewriter);

/// As insertDebugNop, but when `loc` is a FusedLoc, inserts one nop per unique
/// (line, column) across the fused sub-locations.
void insertDebugNopForAllLines(mlir::Location loc,
                               mlir::ConversionPatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// Shared location analysis / rewrite helpers used by the debug passes
// (CanonicalizeDebugLocationsPass, DeduplicateDebugNopsPass).
// Definitions in DebugUtils.cpp. These do NOT self-gate; the passes gate.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace debug {

/// True if `filename` is an inlined library/stdlib file (under site-packages).
bool isForeignFile(llvm::StringRef filename);

/// True if `op` is one of our debug NOPs: llvm.inline_asm "nop", side effects,
/// no results and no operands.
bool isDebugNop(Operation *op);

/// Resolve `loc` to the FileLineColLoc the user should see -- caller frame for
/// call sites (via unwrapFusedLocForDebug), descending NameLoc. Returns a null
/// FileLineColLoc if none is reachable.
FileLineColLoc unwrapToUserFileLineCol(Location loc);

/// Rewrite call-site locations whose callee resolves to a foreign (stdlib)
/// file so they collapse to their caller (user) frame. Recurses through
/// NameLoc / FusedLoc / nested call sites.
Location collapseForeignCallsites(Location loc, unsigned depth = 0);

} // namespace debug
} // namespace triton
} // namespace mlir

#endif // ASCEND_UTILS_DEBUGUTILS_H
