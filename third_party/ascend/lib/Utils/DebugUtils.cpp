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
//===----------------------------------------------------------------------===//
// DebugUtils.cpp
//
// Implementations for DebugUtils.h:
//   * TRITON_DEBUG-gated NOP insertion helpers (unchanged behaviour), and
//   * the shared location analysis / rewrite helpers used by
//     CanonicalizeDebugLocationsPass and DeduplicateDebugNopsPass.
//===----------------------------------------------------------------------===//

#include "Utils/DebugUtils.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <triton/Tools/Sys/GetEnv.hpp>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <functional>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// NOP insertion helpers (gated by TRITON_DEBUG). Behaviour unchanged from the
// previous header-inline versions.
//===----------------------------------------------------------------------===//

Location unwrapFusedLocForDebug(Location loc) {
  if (auto cs = dyn_cast<CallSiteLoc>(loc))
    return unwrapFusedLocForDebug(cs.getCaller());
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (auto inner : llvm::reverse(fused.getLocations())) {
      if (!isa<UnknownLoc>(inner))
        return unwrapFusedLocForDebug(inner);
    }
  }
  return loc;
}

void insertDebugNop(Location loc, PatternRewriter &rewriter) {
  if (!mlir::triton::tools::getBoolEnv("LLVM_EXTRACT_DI_LOCAL_VARIABLES"))
    return;
  auto unwrapped = unwrapFusedLocForDebug(loc);

  auto ctx = rewriter.getContext();
  rewriter.create<LLVM::InlineAsmOp>(
      unwrapped,
      /*resultTypes=*/TypeRange(),
      /*operands=*/ValueRange(),
      /*asm_string=*/"nop",
      /*constraints=*/"",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT), ArrayAttr());
}

void insertDebugNopForAllLines(Location loc,
                               ConversionPatternRewriter &rewriter) {
  if (!mlir::triton::tools::getBoolEnv("LLVM_EXTRACT_DI_LOCAL_VARIABLES"))
    return;

  std::function<Location(Location)> deepUnwrap = [&](Location x) -> Location {
    if (auto cs = dyn_cast<CallSiteLoc>(x))
      return deepUnwrap(cs.getCaller());
    if (auto n = dyn_cast<NameLoc>(x))
      return deepUnwrap(n.getChildLoc());
    if (auto f = dyn_cast<FusedLoc>(x)) {
      for (auto inner : llvm::reverse(f.getLocations()))
        if (!isa<UnknownLoc>(inner))
          return deepUnwrap(inner);
    }
    return x;
  };

  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    llvm::SmallDenseSet<std::pair<unsigned, unsigned>> seen;
    for (auto inner : fused.getLocations()) {
      auto u = deepUnwrap(inner);
      if (auto flc = dyn_cast<FileLineColLoc>(u)) {
        if (seen.insert({flc.getLine(), flc.getColumn()}).second)
          insertDebugNop(u, rewriter);
      }
    }
    return;
  }
  insertDebugNop(loc, rewriter);
}

//===----------------------------------------------------------------------===//
// Shared location analysis / rewrite helpers (no self-gating; passes gate).
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace debug {

bool isForeignFile(llvm::StringRef filename) {
  // A library/stdlib file inlined into the kernel (e.g. triton/language/
  // standard.py). Heuristic: lives under a site-packages tree.
  return filename.contains("/site-packages/");
}

bool isDebugNop(Operation *op) {
  auto asmOp = dyn_cast<LLVM::InlineAsmOp>(op);
  if (!asmOp)
    return false;
  if (!asmOp.getHasSideEffects())
    return false;
  if (asmOp.getAsmString() != "nop")
    return false;
  // Our NOPs have no results and no operands.
  if (asmOp->getNumResults() != 0 || asmOp->getNumOperands() != 0)
    return false;
  return true;
}

FileLineColLoc unwrapToUserFileLineCol(Location loc) {
  // Reuse the existing caller-preferring unwrapper, then descend NameLoc to the
  // underlying FileLineColLoc. callsite(stdlib at user) -> the user's line.
  Location l = ::unwrapFusedLocForDebug(loc);
  while (auto named = dyn_cast<NameLoc>(l))
    l = named.getChildLoc();
  return dyn_cast<FileLineColLoc>(l);
}

Location collapseForeignCallsites(Location loc, unsigned depth) {
  if (depth > 16)
    return loc;

  if (auto named = dyn_cast<NameLoc>(loc)) {
    Location child = collapseForeignCallsites(named.getChildLoc(), depth + 1);
    return child == named.getChildLoc()
               ? loc
               : Location(NameLoc::get(named.getName(), child));
  }

  if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
    Location caller = collapseForeignCallsites(cs.getCaller(), depth + 1);
    // The callee is the inlined helper frame; resolve its own file. (The
    // callee is not itself a call site, so caller-preference is moot here.)
    if (FileLineColLoc calleeFlc = unwrapToUserFileLineCol(cs.getCallee()))
      if (isForeignFile(calleeFlc.getFilename().getValue()))
        return caller; // drop the inlined library frame
    Location callee = collapseForeignCallsites(cs.getCallee(), depth + 1);
    if (callee == cs.getCallee() && caller == cs.getCaller())
      return loc;
    return Location(CallSiteLoc::get(callee, caller));
  }

  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    llvm::SmallVector<Location> newLocs;
    bool changed = false;
    for (Location sub : fused.getLocations()) {
      Location c = collapseForeignCallsites(sub, depth + 1);
      changed |= (c != sub);
      newLocs.push_back(c);
    }
    return changed ? Location(FusedLoc::get(loc.getContext(), newLocs,
                                            fused.getMetadata()))
                   : loc;
  }

  return loc;
}

} // namespace debug
} // namespace triton
} // namespace mlir
