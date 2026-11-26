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

/// The shape of the cross-core dependency graph of a module.
struct CrossCorePattern {
  /// The number of independent pipelines of the graph.
  unsigned pipelines = 0;
  /// The number of entry blocks of the pipelines: the blocks of the
  /// qualifying chains that no other block of the chains feeds. A pipeline
  /// that is a single linear chain has one entry; two blocks feeding the
  /// same block is a merge, and the merge has two entries.
  unsigned entries = 0;
};

/// Analyzes the cross-core dependency graph of `module` and reports its
/// pipelines.
///
/// The nodes of the graph are the computation blocks, a block being the pair
/// of the scope that holds it and its `ssbuffer.block_id`; block ids are only
/// unique inside a scope. The edges are the cross-core dependencies, the
/// producer block of a dependency group feeding the consumer block of the
/// same group. Every edge joins two different cores, so a chain always
/// alternates the cores: CUBE -> VECTOR -> CUBE -> VECTOR or VECTOR -> CUBE
/// -> VECTOR -> CUBE.
///
/// A chain has to carry the tightly coupled buffers of the validated
/// pipeline: every transfer of the chain is double buffered and no buffer is
/// shared by two transfers of the chain, so the chain of `chainLength` blocks
/// uses 2 * (chainLength - 1) tightly coupled buffers.
///
/// One direction of the graph may be carried by several cross-core groups.
/// The buffers of the direction are the buffers of all of them, and the
/// direction has to carry the buffers of exactly one transfer, so that every
/// transfer of the pipeline carries the same number of buffers.
///
/// The conditions are computed for the blocks that carry a condition, so the
/// cross-core blocks of the module that carry a condition have to be the
/// blocks of one of the two validated pipelines: four of them for the mla
/// pipeline and five of them for the paged decode pipeline.
///
/// Two chains that share a block are two views of the same pipeline and count
/// once; the chains that share no block are independent pipelines.
CrossCorePattern analyzeCrossCorePattern(ModuleOp module,
                                         unsigned chainLength = 4);

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_SSBUF_CROSSCOREPATTERN_H
