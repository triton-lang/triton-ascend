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

#ifndef TRITON_ASCEND_PROCESS_LOOP_INVARIANT_READ_H
#define TRITON_ASCEND_PROCESS_LOOP_INVARIANT_READ_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton {

/// Sink a loop-invariant VECTOR -> CUBE read wait from the cube block head to
/// the first consumer inside the main loop.
///
/// Pattern: a VECTOR producer (its cbuf copy and `sync_block_set`) lives
/// outside the main loop, the matching `sync_block_wait` also lives outside the
/// loop (block head), and the consumer (the first op that actually reads the
/// received value) lives inside the loop. The wait is only moved when every
/// step of the receive chain has exactly one user; otherwise the insertion
/// point is ambiguous and the wait is left in place. The single producer token
/// must still be consumed exactly once, so the wait is re-emitted twice:
///   - a zero-trip fallback before the loop, guarded by the negated loop-entry
///     condition;
///   - the real wait in front of the consumer, guarded by
///     `blockCounter == blockCounterInit`, i.e. the first execution of the
///     consuming ssbuffer block.
std::unique_ptr<OperationPass<ModuleOp>> createProcessLoopInvariantReadPass();

} // namespace mlir::triton

#endif // TRITON_ASCEND_PROCESS_LOOP_INVARIANT_READ_H
