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
struct PipeAnalysis {
  static constexpr size_t kMaxNumSpaces = 4;

  // use small dense set here simply for its api
  llvm::SmallDenseSet<hivm::AddressSpace, kMaxNumSpaces> from;
  llvm::SmallDenseSet<hivm::AddressSpace, kMaxNumSpaces> to;

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

struct CustomOpAnalysis {
  bool isLoad = false;
  bool isUnknownPipe = false; // unknown pipe -> treat all buffers as write

  hivm::CustomOp customOp;

  llvm::TinyPtrVector<Value> gmBuffers;
  llvm::TinyPtrVector<Value> localBuffers;

  CoreType coreType = UNDETERMINED;
  hivm::PIPE pipe = hivm::PIPE::PIPE_UNASSIGNED;
  SmallVector<Operation *> relaventOps;
  static llvm::FailureOr<CustomOpAnalysis> get(hivm::CustomOp customOp);

  llvm::TinyPtrVector<Value> getReads() {
    if (isUnknownPipe) {
      return getAllBuffers();
    }
    if (isLoad) {
      return gmBuffers;
    }
    return localBuffers;
  }

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
  void collectRelaventOps();
  void determineLoadOrUnknown();

  // for internal usage only - when needed elsewhere, should consider
  // re-implement as SmallVector
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
