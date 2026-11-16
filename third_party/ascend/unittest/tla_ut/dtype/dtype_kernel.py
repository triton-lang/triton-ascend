# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""TLA 3-in-1 test: scalar dtype interfaces used by real kernels.

Each case passes one ``tla`` dtype as a constexpr into two kernels:

- ``dtype_copy`` declares a UB buffer and a GM view with that dtype, so the
  dtype reaches ``tla.alloc``, ``tla.view`` and the DMA copy ops;
- ``dtype_splat`` builds a full SIMD register with ``tla.simd.vfull``, so the
  dtype also reaches ``tla.vector_type``.

The ``tla.dtype("fp32")`` constructor is covered as one more case value.  The
stages are the usual ir / compile / run split; the run stage keeps a subset of
dtypes (fp32/bf16/int32/uint8/int64) that exercises float, bf16, signed,
unsigned and 64-bit elements.

``int64`` and ``uint64`` are covered by ``dtype_copy`` but excluded from the
``vfull`` case: 64-bit vector broadcast currently crashes in the Bisheng
backend (``vbr_int64_t``), so that part is left to the backend fix.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# tla export -> (dtype, torch dtype, full-register lanes, MLIR element type)
DTYPES = {
    "bfloat16": (tla.bfloat16, torch.bfloat16, 128, "bf16"),
    "float16": (tla.float16, torch.float16, 128, "f16"),
    "float32": (tla.float32, torch.float32, 64, "f32"),
    "int8": (tla.int8, torch.int8, 256, "i8"),
    "int16": (tla.int16, torch.int16, 128, "i16"),
    "int32": (tla.int32, torch.int32, 64, "i32"),
    "int64": (tla.int64, torch.int64, 64, "i64"),
    "uint8": (tla.uint8, torch.uint8, 256, "i8"),
    "uint16": (tla.uint16, torch.uint16, 128, "i16"),
    "uint32": (tla.uint32, torch.uint32, 64, "i32"),
    "uint64": (tla.uint64, torch.uint64, 64, "i64"),
}
RUN_CASES = ("float32", "bfloat16", "int32", "uint8", "int64")
# 64-bit vector broadcast (vfull) is not supported by the current Bisheng
# backend; keep these dtypes in dtype_copy but skip the vfull case.
VFULL_UNSUPPORTED = frozenset({"int64", "uint64"})


@tla.jit
def dtype_copy(X, Y, DT: tl.constexpr, N: tl.constexpr):
    ub = tla.alloc((N, ), DT, space="ub")
    with tla.scope.vector():
        tla.copy(tla.view(X, (N, )), ub)
        tla.copy(ub, tla.view(Y, (N, )))


@tla.jit
def dtype_splat(O, DT: tl.constexpr, N: tl.constexpr):
    ub = tla.alloc((N, ), DT, space="ub")
    with tla.scope.vector():
        with tla.scope.simd():
            tla.simd.vsts(ub[0], tla.simd.vfull(0, DT, N))
        tla.copy(ub, tla.view(O, (N, )))


def signature(ty):
    return {"X": "*" + ty.mangle(), "Y": "*" + ty.mangle()}


def test_ir(output_dir=None):
    cases = list(DTYPES.items()) + [("dtype_ctor_fp32", (tla.dtype("fp32"), torch.float32, 64, "f32"))]
    for name, (ty, _, lanes, elem) in cases:
        constexprs = {"DT": ty, "N": lanes}
        module, ctx = compile_offline(dtype_copy, signature(ty), constexprs, output_dir=output_dir)
        text = str(module)
        assert f"memref<{lanes}x{elem}, #hivm.address_space<ub>>" in text, name
        assert f"!tt.ptr<{elem}>" in text, name
        lowered = str(lower_offline(module, ctx, output_dir=output_dir))
        check_lowered_ir(lowered, loads=1, stores=1, buffers=None)

        if name in VFULL_UNSUPPORTED:
            continue
        module, ctx = compile_offline(dtype_splat, {"O": "*" + ty.mangle()}, constexprs, output_dir=output_dir)
        text = str(module)
        assert f"vector<{lanes}x{elem}>" in text, name
        assert "ave.hir.scalar_broadcast" in text, name
        lower_offline(module, ctx, output_dir=output_dir)
    print("IR PASS: 11 tla dtypes plus tla.dtype('fp32') reach buffers; vfull skips int64/uint64")


def _launch(name):
    ty, torch_dtype, lanes, _ = DTYPES[name]
    x = torch.arange(lanes, dtype=torch.int64)[:lanes].to(torch_dtype)
    y = torch.zeros(lanes, dtype=torch_dtype, device="npu")
    dtype_copy[(1, )](x.npu(), y, DT=ty, N=lanes)
    torch.npu.synchronize()
    got = y.cpu()
    torch.testing.assert_close(got, x, rtol=0, atol=0)
    if name in VFULL_UNSUPPORTED:
        return
    out = torch.full((lanes, ), 3, dtype=torch_dtype, device="npu")
    dtype_splat[(1, )](out, DT=ty, N=lanes)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.zeros_like(x), rtol=0, atol=0)


def test_compile():
    for name in RUN_CASES:
        _launch(name)
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: dtype cases generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for name in RUN_CASES:
        _launch(name)
    vfull_cases = [name for name in RUN_CASES if name not in VFULL_UNSUPPORTED]
    print("DTYPE OK: copy for", ", ".join(RUN_CASES), "; vfull for", ", ".join(vfull_cases))


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
