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

#ifndef TRITON_ASCEND_TRITONTOLINALG_ZERO_VALUE_PROPAGATION_H
#define TRITON_ASCEND_TRITONTOLINALG_ZERO_VALUE_PROPAGATION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {

/// Creates a TTIR-level pass that proves which tensors/scalars are all-zero and
/// folds the algebraic identities that follow (`x + 0 -> x`, `x - 0 -> x`,
/// `x * 1 -> x`, `reduce_add(all_zero) -> all_zero`), then removes the dead
/// operand chains. Float and integer arithmetic are both covered: the integer
/// rules (`arith.muli` / `arith.addi` / `arith.subi`) need no option and carry
/// no caveat, because integers have no NaN, no Inf and no signed zero. `i1` is
/// included, where zero is `false` and `arith.muli` is a logical and. Runs
/// before `triton-to-linalg`, so both the arithmetic and the UB tile of a dead
/// statement disappear.
///
/// NaN / Inf semantics:
///
///   * every fold above is bit-exact; none of them can turn a NaN or an Inf
///     into a finite number, and none can turn a finite number into NaN/Inf.
///   * `x * 0 -> 0` is *not* exact in general, because IEEE defines
///     `0 * inf = nan` and `0 * nan = nan`. It is therefore only applied when
///     the other operand is a compile-time finite constant, or when the caller
///     opts in through `allowFpMulZeroFold`. This is the only rule the option
///     controls; every rule listed above it runs unconditionally.
///
/// `allowFpMulZeroFold` is off by default so that the pass is safe for any
/// input, including arbitrary loads that may carry NaN/Inf. The Ascend backend
/// exposes it as the `allow_fp_mul_zero_fold` launch option, which defaults to
/// on, because collapsing the zero initial state (`zeros * exp(g)`) is where
/// the speedup on the recurrent gating kernels comes from; that is a deliberate
/// IEEE relaxation. Whenever a fold relies on it (i.e. the multiplicand was not
/// proven finite) the pass records the count in the `tt.fp_mul_zero_folds`
/// module attribute; `backend/compiler.py` turns that into a Python warning, so
/// the assumption does not stay silent. (A pass-emitted MLIR warning would be
/// filtered out by default in this stack.)
///
/// Signed zero: only the *sign* of a zero result can differ from IEEE.
/// `x + 0 -> x` maps `x = -0.0` to `+0.0` when the constant is `+0.0`
/// (`(-0) + (+0) = +0`) and is exact when it is `-0.0`; `x - 0 -> x` maps
/// `x = -0.0` to `+0.0` when the constant is `-0.0` (`(-0) - (-0) = +0`) and is
/// exact when it is `+0.0`. Both folds are applied for both constant signs: a
/// zero's sign is observable only through `1/x`, `signbit`, `copysign` or
/// `atan2`, and rejecting `-0.0` operands would also kill the zero-state
/// elimination this pass exists for. See the implementation for the measured
/// reasoning.
std::unique_ptr<Pass>
createZeroValuePropagationPass(bool allowFpMulZeroFold = false);

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_TRITONTOLINALG_ZERO_VALUE_PROPAGATION_H
