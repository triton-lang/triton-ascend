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

#ifndef ASCENDMODEL_ROUTEMODEL_SIMTSELECTION_H
#define ASCENDMODEL_ROUTEMODEL_SIMTSELECTION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::simt_selection {

inline constexpr llvm::StringLiteral kEffectiveExecutionAttr =
    "ascend.simt_costmodel.effective";
// Use one spelling from the Python API through TTIR and Route Model lowering.
inline constexpr llvm::StringLiteral kVectorModeAttr = "vector_mode";
inline constexpr llvm::StringLiteral kLegacyVectorModeAttr = "vector_type";

inline constexpr llvm::StringLiteral kAllSimd = "all_simd";
inline constexpr llvm::StringLiteral kAllSimtOnly = "all_simt_only";
inline constexpr llvm::StringLiteral kMixedSimdSimt = "mixed_simd_simt";
inline constexpr llvm::StringLiteral kBackendDefault = "backend_default";

/// Find the nearest function/module execution decision visible to `op`.
///
/// The cost model may attach the decision either to the kernel function or to
/// its containing module.  Looking from the operation outwards also lets a
/// function-level decision override a module-level default.
inline StringAttr getEffectiveExecution(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (auto attr =
            current->getAttrOfType<StringAttr>(kEffectiveExecutionAttr)) {
      return attr;
    }
  }
  return {};
}

inline bool isKnownExecutionKind(llvm::StringRef kind) {
  return kind == kAllSimd || kind == kAllSimtOnly || kind == kMixedSimdSimt ||
         kind == kBackendDefault;
}

/// A backend_default or absent decision keeps the legacy backend routing.
/// Only an admitted concrete model decision suppresses global legacy force
/// flags and makes the local scope authoritative.
inline bool isModelControlled(Operation *op) {
  auto effective = getEffectiveExecution(op);
  return effective && isKnownExecutionKind(effective.getValue()) &&
         effective.getValue() != kBackendDefault;
}

inline bool isMixedModelDecision(Operation *op) {
  auto effective = getEffectiveExecution(op);
  return effective && effective.getValue() == kMixedSimdSimt;
}

inline StringAttr getVectorMode(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kVectorModeAttr))
    return attr;
  return op->getAttrOfType<StringAttr>(kLegacyVectorModeAttr);
}

/// Return the nearest enclosing vector-mode annotation.  Nearest scope wins
/// for nested explicit scopes.
inline StringAttr getEnclosingVectorMode(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto attr = getVectorMode(parent))
      return attr;
  }
  return {};
}

inline bool hasEnclosingVectorMode(Operation *op, llvm::StringRef mode) {
  auto attr = getEnclosingVectorMode(op);
  return attr && attr.getValue() == mode;
}

/// Resolve whether an individual operation may take a SIMT lowering.
///
/// In a model-controlled compilation, the legacy global force flag is ignored:
/// only an explicit local SIMT scope may enable SIMT.  In
/// backend_default/legacy compilation the historical global behavior remains
/// unchanged.  An explicit nearest SIMD scope always wins.
inline bool shouldUseSimtTemplate(Operation *op, bool legacyForceSimt) {
  if (hasEnclosingVectorMode(op, "simd"))
    return false;

  const bool locallySelected = hasEnclosingVectorMode(op, "simt");
  if (isModelControlled(op))
    return locallySelected;
  return legacyForceSimt || locallySelected;
}

inline bool isVoidSimtScope(Operation *op) {
  if (op->getName().getStringRef() != "scope.scope" ||
      op->getNumResults() != 0 || op->getNumRegions() != 1)
    return false;
  auto mode = getVectorMode(op);
  if (!mode || mode.getValue() != "simt")
    return false;

  Region &region = op->getRegion(0);
  if (!region.hasOneBlock() || region.front().empty() ||
      region.front().getNumArguments() != 0)
    return false;
  Operation &terminator = region.front().back();
  return terminator.getName().getStringRef() == "scope.return" &&
         terminator.getNumOperands() == 0;
}

/// Return the sole void SIMT scope that owns a kernel's whole executable body.
///
/// Constants and the function return may stay outside the scope.  Inspecting
/// structured IR here replaces the former Python TTIR text heuristic.
inline Operation *findWholeBodyVoidSimtScope(Operation *root) {
  Operation *kernel = nullptr;
  for (Region &region : root->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block) {
        llvm::StringRef name = nested.getName().getStringRef();
        if (name != "tt.func" && name != "func.func")
          continue;
        auto visibility = nested.getAttrOfType<StringAttr>("sym_visibility");
        if (visibility && visibility.getValue() != "public")
          continue;
        if (kernel)
          return nullptr;
        kernel = &nested;
      }
    }
  }
  if (!kernel || kernel->getNumRegions() != 1 ||
      !kernel->getRegion(0).hasOneBlock())
    return nullptr;

  Block &body = kernel->getRegion(0).front();
  if (body.empty())
    return nullptr;
  Operation *terminator = body.getTerminator();
  llvm::StringRef terminatorName = terminator->getName().getStringRef();
  if ((terminatorName != "tt.return" && terminatorName != "func.return") ||
      terminator->getNumOperands() != 0)
    return nullptr;

  Operation *simtScope = nullptr;
  for (Operation &nested : body.without_terminator()) {
    llvm::StringRef name = nested.getName().getStringRef();
    if (name == "arith.constant")
      continue;
    if (isVoidSimtScope(&nested) && !simtScope) {
      simtScope = &nested;
      continue;
    }
    return nullptr;
  }
  return simtScope;
}

/// Inline result-free SIMT scopes before the pure-SIMT TTIR compiler.
///
/// Result-bearing scopes are intentionally left untouched because they need
/// SSA result remapping and belong to the mixed lowering path.
inline int64_t inlineVoidSimtScopesForPureSimt(Operation *root) {
  llvm::SmallVector<Operation *> scopes;
  root->walk<WalkOrder::PostOrder>([&](Operation *nested) {
    if (isVoidSimtScope(nested))
      scopes.push_back(nested);
  });

  int64_t inlined = 0;
  for (Operation *scope : scopes) {
    Block &body = scope->getRegion(0).front();
    Operation *terminator = &body.back();
    while (&body.front() != terminator)
      body.front().moveBefore(scope);
    terminator->erase();
    scope->erase();
    ++inlined;
  }
  return inlined;
}

} // namespace mlir::ascend::simt_selection

#endif // ASCENDMODEL_ROUTEMODEL_SIMTSELECTION_H
