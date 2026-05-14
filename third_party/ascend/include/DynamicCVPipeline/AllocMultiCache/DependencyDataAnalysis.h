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


#ifndef DYNAMIC_CV_PIPELINE_ALLOC_MULTI_CACHE_DEPENDENCY_DATA_ANALYSIS_H
#define DYNAMIC_CV_PIPELINE_ALLOC_MULTI_CACHE_DEPENDENCY_DATA_ANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

// ============================================================================
// DependencyDataAnalysis: Stores buffer dependency relationships for multi-buffer passes
//
// Data structure:
//   - crossCoreDependentMap: key = consumer buffer, value = producer buffers (inter-core)
//   - intraCoreDependentMap: key = ForOp, value = {consumer buffer -> producer buffers} (intra-core)
// ============================================================================

class DependencyDataAnalysis {
public:
    explicit DependencyDataAnalysis(mlir::Operation *op) {}

    // === cross-core dependency ===

    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>&
    getCrossCoreDependentMap() {
        return crossCoreDependentMap;
    }

    const llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>&
    getCrossCoreDependentMap() const {
        return crossCoreDependentMap;
    }

    void setCrossCoreDependentMap(
        const llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> &map) {
        crossCoreDependentMap = map;
    }

    // === intra-core dependency ===

    llvm::DenseMap<mlir::scf::ForOp, llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>>&
    getIntraCoreDependentMap() {
        return intraCoreDependentMap;
    }

    const llvm::DenseMap<mlir::scf::ForOp, llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>>&
    getIntraCoreDependentMap() const {
        return intraCoreDependentMap;
    }

    void setIntraCoreDependentMap(
        const llvm::DenseMap<mlir::scf::ForOp, llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>> &map) {
        intraCoreDependentMap = map;
    }

    void clear() {
        crossCoreDependentMap.clear();
        intraCoreDependentMap.clear();
    }

    bool empty() const {
        return crossCoreDependentMap.empty() && intraCoreDependentMap.empty();
    }

    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
        return false;
    }

private:
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> crossCoreDependentMap;
    llvm::DenseMap<mlir::scf::ForOp, llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>> intraCoreDependentMap;
};

} // namespace triton
} // namespace mlir

#endif // DYNAMIC_CV_PIPELINE_ALLOC_MULTI_CACHE_DEPENDENCY_DATA_ANALYSIS_H
