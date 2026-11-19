# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""TLA scalar clamp: kernel IR, target compilation, and NPU accuracy."""
import random
import sys
from pathlib import Path

# Do not let this category's select.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import triton.language as tl
import pytest
from triton.language.extra import tla
from utils.tla_test_utils import get_stages, matrix, compile_offline, lower_offline, check_kernel_binary


@tla.jit
def clamp_scalar(Out, x, lo, hi, INPUT_DTYPE: tl.constexpr, BOUND_DTYPE: tl.constexpr, DTYPE: tl.constexpr,
                 N: tl.constexpr, PROPAGATE_NAN: tl.constexpr, IN_VF: tl.constexpr):
    out = tla.view(Out, (N, ))
    local = tla.alloc((N, ), DTYPE)
    with tla.scope.vector():
        x = tla.cast(x, INPUT_DTYPE)
        lo = tla.cast(lo, BOUND_DTYPE)
        hi = tla.cast(hi, BOUND_DTYPE)
        if not IN_VF:
            result = tla.clamp(x, lo, hi, propagate_nan=PROPAGATE_NAN)
            tl.static_assert(result.dtype == DTYPE)
            tl.static_assert(len(result.shape) == 0)
        with tla.scope.simd():
            if IN_VF:
                result = tla.clamp(x, lo, hi, propagate_nan=PROPAGATE_NAN)
                tl.static_assert(result.dtype == DTYPE)
                tl.static_assert(len(result.shape) == 0)
            reg = tla.simd.vdup(result, DTYPE)
            tla.simd.vsts(local[0], reg, dist="NORM_B32")
        tla.store_ub_to_gm(local, out)


CONFIGS = [
    dict(name=f"{tag}_{policy}_outside", tag=tag, input_dtype=input_dtype, bound_dtype=bound_dtype, dtype=dtype,
         propagate_nan=propagate_nan, in_vf=False) for tag, input_dtype, bound_dtype, dtype in (
             ("f32", tl.float32, tl.float32, tl.float32),
             ("f32_i32", tl.float32, tl.int32, tl.float32),
         ) for policy, propagate_nan in (("none", tl.PropagateNan.NONE), ("all", tl.PropagateNan.ALL))
]


def constexprs(case):
    return dict(INPUT_DTYPE=case["input_dtype"], BOUND_DTYPE=case["bound_dtype"], DTYPE=case["dtype"],
                N=2048 // case["dtype"].primitive_bitwidth, PROPAGATE_NAN=case["propagate_nan"], IN_VF=case["in_vf"])


@matrix(CONFIGS)
@pytest.mark.parametrize("in_vf", [False, True], ids=["outside_vf", "inside_vf"])
def test_ir(case, in_vf, output_dir=None):
    # Current A5 codegen rejects floating scalar min/max inside SIMD VF.
    # Still check that both placements lower correctly at the TLA IR boundary.
    name = case["name"].removesuffix("_outside") + ("_inside" if in_vf else "_outside")
    case = dict(case, in_vf=in_vf, name=name)
    signature = dict(Out="*fp32", x="fp32", lo="i32" if case["bound_dtype"].is_int() else "fp32",
                     hi="i32" if case["bound_dtype"].is_int() else "fp32")
    dump = Path(output_dir) / case["name"] if output_dir is not None else None
    module, ctx = compile_offline(clamp_scalar, signature, constexprs(case), output_dir=dump)
    assert "tt.clampf" in str(module)
    text = str(lower_offline(module, ctx, output_dir=dump))
    suffix = "numf" if case["propagate_nan"] == tl.PropagateNan.NONE else "imumf"
    assert "arith.min" + suffix in text and "arith.max" + suffix in text
    assert "arith.cmpf" not in text and "arith.select" not in text
    assert "tensor<" not in text and "tt." not in text
    assert text.count("memref.alloc()") == 1, "clamp introduced an implicit buffer"
    assert "ave.hir.broadcast" in text and "hivm.hir.store" in text
    assert "ave.hir.masked_store <NORM_B32>" in text
    print("IR PASS: clamp", case["name"])


def _new_output(case):
    import torch
    import torch_npu  # noqa: F401

    return torch.empty(constexprs(case)["N"], dtype=torch.float32, device="npu")


@matrix(CONFIGS)
def test_compile(case):
    out = _new_output(case)
    bounds = (-2, 3) if case["bound_dtype"].is_int() else (-2.0, 3.0)
    compiled = clamp_scalar.warmup(out, 0.5, *bounds, **constexprs(case), grid=(1, ))
    check_kernel_binary(compiled)


def _values(case):
    inputs = [-float("inf"), -4.0, -2.0, -0.0, 0.0, 3.0, 4.0, float("inf")]
    # TODO: Restore NaN x for NONE after CCE fixes scalar maxnum/minnum lowering.
    if case["propagate_nan"] == tl.PropagateNan.ALL:
        inputs.append(float("nan"))
    if case["bound_dtype"].is_int():
        return [(x, -2, 3) for x in inputs] + [
            (256.0, 257, 259),
            (260.0, 255, 257),
            (1.0, 2, 2),
        ]
    triples = [(x, -2.0, 3.0) for x in inputs]
    triples += [(1.0, 2.0, 2.0), (0.5, -float("inf"), float("inf")), (-float("inf"), -float("inf"), float("inf")),
                (float("inf"), -float("inf"), float("inf"))]
    rng = random.Random(0)
    triples += [(rng.uniform(-10, 10), -1.25, 2.75) for _ in range(8)]
    return triples


@matrix(CONFIGS)
def test_run(case):
    import torch

    types = {tl.float32: torch.float32, tl.int32: torch.int32}
    out = _new_output(case)
    for x, lo, hi in _values(case):
        clamp_scalar[(1, )](out, x, lo, hi, **constexprs(case))
        torch.npu.synchronize()
        actual = out.cpu()
        # Quantize each operand before promotion, including mixed integer bounds.
        a = torch.tensor(x, dtype=types[case["input_dtype"]]).to(out.dtype)
        lower = torch.tensor(lo, dtype=types[case["bound_dtype"]]).to(out.dtype)
        upper = torch.tensor(hi, dtype=types[case["bound_dtype"]]).to(out.dtype)
        if case["propagate_nan"] == tl.PropagateNan.ALL:
            expected = torch.maximum(lower, torch.minimum(a, upper))
        else:
            # The shared Ascend clamp lowering performs min(x, upper) first.
            expected = torch.fmax(lower, torch.fmin(a, upper))
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0, atol=0, equal_nan=True,
            msg=lambda msg: f"{case['name']}: x={x}, min={lo}, max={hi}, "
            f"actual[0]={actual[0].item()}, expected={expected.item()}\n{msg}")
        # Interior signed zero must pass through without losing its sign.
        if x == 0.0 and lo < 0 < hi:
            assert torch.equal(torch.signbit(actual), torch.signbit(a).expand_as(actual))
    print("CLAMP OK:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case, in_vf=False)
            test_ir(case, in_vf=True)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
