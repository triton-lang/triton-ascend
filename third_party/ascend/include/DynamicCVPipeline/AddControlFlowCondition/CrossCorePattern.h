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

#ifndef TRITON_ASCEND_SSBUF_CROSSCOREPATTERN_H
#define TRITON_ASCEND_SSBUF_CROSSCOREPATTERN_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace triton {

/// The shape of the cross-core dependency graph of the blocks that carry a
/// condition.
enum class CrossCoreShape {
  /// The graph does not hold a shape the normal preload scheme models.
  None,
  /// A linear chain of four or five blocks: the mla and paged decode
  /// pipelines.
  LinearChain,
  /// Two linear chains of four blocks sharing no block and no buffer.
  TwoChains,
  /// A linear chain of three blocks: the hstu attention pipeline.
  ShortChain,
  /// A tree of six or seven blocks that may fork and merge: the nsa
  /// compression backward pipeline.
  Tree,
  /// A forest of four chains of three blocks sharing no block and no
  /// buffer: the flex attention backward pipeline.
  ShortChainForest,
};

/// The shape of the cross-core dependency graph of a module.
struct CrossCorePattern {
  /// The number of independent pipelines of the graph.
  unsigned pipelines = 0;
  /// The number of entry blocks of the pipelines: the blocks that no other
  /// block of the chains feeds. A merge has two entries.
  unsigned entries = 0;
  /// The shape the graph holds.
  CrossCoreShape shape = CrossCoreShape::None;
};

/// Analyzes the cross-core dependency graph of `module` and reports its
/// pipelines.
CrossCorePattern analyzeCrossCorePattern(ModuleOp module,
                                         unsigned chainLength = 4);

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_SSBUF_CROSSCOREPATTERN_H
