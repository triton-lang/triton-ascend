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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_ANALYSIS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_ANALYSIS_H

#include <cstddef>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/LogicalResult.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

namespace mlir::CVPipeline {

/// Address spaces a hardware pipe moves data between.
struct PipeAnalysis {
  static constexpr size_t kMaxNumSpaces = 4;

  // Small dense set, used only for its API.
  llvm::SmallDenseSet<hivm::AddressSpace, kMaxNumSpaces> from;
  llvm::SmallDenseSet<hivm::AddressSpace, kMaxNumSpaces> to;

  /// Fills `from` / `to` for `pipe`; an unlisted pipe leaves both empty.
  PipeAnalysis(hivm::PIPE pipe) {
    using namespace hivm;
    switch (pipe) {
    case PIPE::PIPE_MTE1:
      from = {AddressSpace::L1};
      to = {AddressSpace::L0A, AddressSpace::L0B, AddressSpace::BiasBUF,
            AddressSpace::UB};
      break;
    case PIPE::PIPE_MTE2:
      from = {AddressSpace::GM};
      to = {AddressSpace::L1, AddressSpace::UB};
      break;
    case PIPE::PIPE_MTE3:
      from = {AddressSpace::L1, AddressSpace::UB};
      to = {AddressSpace::GM, AddressSpace::L1};
      break;
    case PIPE::PIPE_FIX:
      from = {AddressSpace::L0C};
      to = {AddressSpace::L1, AddressSpace::GM, AddressSpace::UB};
      break;
    default:
      break;
    }
  }
};

/// Memory-effect view of a `hivm.hir.custom` op: the buffers it reads and
/// writes, the pipe it runs on, and the core type implied by that pipe.
class CustomOpAnalysis {
public:
  /// True when the custom op loads GM data into on-chip memory.
  bool isLoad = false;

  /// Unknown pipe -> conservatively treat every buffer as both read and write.
  bool isUnknownPipe = false;

  hivm::CustomOp customOp;

  llvm::TinyPtrVector<Value> gmBuffers;
  llvm::TinyPtrVector<Value> localBuffers;

  CoreType coreType = UNDETERMINED;
  hivm::PIPE pipe = hivm::PIPE::PIPE_UNASSIGNED;
  SmallVector<Operation *> relevantOps;

  /// Analyses `customOp`; fails when its buffers cannot be resolved.
  static llvm::FailureOr<CustomOpAnalysis> get(hivm::CustomOp customOp);

  /// Buffers the op consumes (all buffers when the pipe is unknown).
  llvm::TinyPtrVector<Value> getReads() {
    if (isUnknownPipe) {
      return getAllBuffers();
    }
    if (isLoad) {
      return gmBuffers;
    }
    return localBuffers;
  }

  /// Buffers the op produces (all buffers when the pipe is unknown).
  llvm::TinyPtrVector<Value> getWrites() {
    if (isUnknownPipe) {
      return getAllBuffers();
    }
    if (isLoad) {
      return localBuffers;
    }
    return gmBuffers;
  }

private:
  CustomOpAnalysis(hivm::CustomOp customOp)
      : customOp(customOp), pipe(customOp.getPipe()) {}

  void determineCoreType();
  llvm::LogicalResult collectBuffers();
  void collectRelevantOps();
  void determineLoadOrUnknown();

  // For internal usage only - when needed elsewhere, should consider
  // re-implement as SmallVector.
  llvm::TinyPtrVector<Value> getAllBuffers() {
    llvm::TinyPtrVector<Value> res(gmBuffers);
    for (auto buffer : localBuffers) {
      res.push_back(buffer);
    }
    return res;
  }
};

} // namespace mlir::CVPipeline

#endif
