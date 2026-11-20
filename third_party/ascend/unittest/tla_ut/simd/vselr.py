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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Register permutation through native vselr, including optional result masks."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, lower_offline, compile_native_kernel


@tla.jit
def permutation_kernel(X, I, M, O, DT: tl.constexpr, W: tl.constexpr, MASKED: tl.constexpr, MODE: tl.constexpr):
    N: tl.constexpr = 2048 // W
    IT: tl.constexpr = tl.uint8 if W == 8 else (tl.uint16 if W == 16 else tl.uint32)
    MT: tl.constexpr = tl.int8 if W == 8 else (tl.int16 if W == 16 else tl.int32)
    x = tla.alloc_buffer((N, ), DT, space="ub")
    i = tla.alloc_buffer((N, ), IT, space="ub")
    m = tla.alloc_buffer((N, ), MT, space="ub")
    out = tla.alloc_buffer((2, N), DT, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(I, (N, )), i)
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        if not MASKED:
            mask = None
        a = tla.simd.vselr(tla.simd.vlds(x[0]), tla.simd.vlds(i[0]), mask, mode=MODE)
        if MASKED:
            b = tla.simd.vsel(mask, a, tla.simd.vsplat(7, DT))
        else:
            b = a
        tla.simd.vsts(out[0, 0], a, dist="NORM_B8" if W == 8 else ("NORM_B16" if W == 16 else "NORM_B32"))
        tla.simd.vsts(out[1, 0], b, dist="NORM_B8" if W == 8 else ("NORM_B16" if W == 16 else "NORM_B32"))
    tla.copy(out, tla.view(O, (2, N)))


TYPES = [("int8", "i8", 8), ("uint8", "u8", 8), ("int16", "i16", 16), ("uint16", "u16", 16), ("int32", "i32", 32),
         ("uint32", "u32", 32), ("float16", "fp16", 16), ("bfloat16", "bf16", 16), ("float32", "fp32", 32),
         ("float8e4nv", "fp8e4nv", 8), ("float8e5", "fp8e5", 8)]


def constants(typ, masked, mode):
    name, _, w = typ
    return dict(DT=getattr(tl, name), W=w, MASKED=masked, MODE=mode)


def launch(typ, pattern, index_kind, mode="ZEROING"):
    name, _, w = typ
    n = 2048 // w
    dtype = getattr(torch, {"float8e4nv": "float8_e4m3fn", "float8e5": "float8_e5m2"}.get(name, name))
    lane = torch.arange(n)
    raw = (lane * 7 + 3) % 29 - (0 if name.startswith("uint") else 14)
    data = raw.to(dtype)
    bit_dtype = {8: torch.int8, 16: torch.int16, 32: torch.int32}[w]
    if pattern == "bits":
        # Preserve arbitrary payloads, NaNs, infinities and signed zero.
        bits = (lane * 2654435761 + 12345).to(bit_dtype)
        bits[:4] = torch.tensor([0, -(1 << (w - 1)), -1, (1 << (w - 1)) - 1], dtype=bit_dtype)
        data = bits.view(dtype)
    idx = lane.flip(0) if index_kind == "reverse" else (lane * 13 + 7) % (n // 2)
    mask = {
        "unmasked": lane >= 0, "all": lane >= 0, "none": lane < 0, "tail": lane < n - 5, "mixed": (lane * 7 + 3) % 11
        < 5, "bits": (lane % 3) != 0
    }[pattern]
    out = torch.empty((2, n), dtype=dtype, device="npu")
    compiled = permutation_kernel[(1, )](data.npu(), idx.to({8: torch.uint8, 16: torch.uint16, 32:
                                                             torch.uint32}[w]).npu(),
                                         mask.to({8: torch.int8, 16: torch.int16, 32: torch.int32}[w]).npu(), out,
                                         **constants(typ, pattern != "unmasked", mode))
    torch.npu.synchronize()
    selected = data.view(bit_dtype).to(torch.int64)[idx]
    old_bits = torch.tensor(7, dtype=dtype).view(bit_dtype).to(torch.int64)
    ref = torch.stack((torch.where(mask, selected, 0), torch.where(mask, selected, old_bits)))
    return out.cpu().view(bit_dtype).to(torch.int64), ref, compiled


@pytest.mark.parametrize("typ", TYPES)
@pytest.mark.parametrize("masked", (False, True))
def test_ir(typ, masked):
    _, sig, w = typ
    module, ctx = compile_offline(permutation_kernel, dict(X=f"*{sig}", I=f"*u{w}", M=f"*i{w}", O=f"*{sig}"),
                                  constants(typ, masked, "ZEROING"))
    text = str(module)
    assert text.count('ave.hir.vselr ') == 1
    assert 'llvm.call_intrinsic' not in text
    assert 'vector.bitcast' not in text
    assert "ave.hir.vgather" not in text
    assert text.count("ave.hir.vsel ") == (1 if masked else 0)
    lowered = str(lower_offline(module, ctx))
    assert 'ave.hir.vselr' in lowered
    assert 'llvm.call_intrinsic' not in lowered


@pytest.mark.parametrize("typ", TYPES)
def test_compile(typ):
    _, sig, w = typ
    compile_native_kernel(permutation_kernel, dict(X=f"*{sig}", I=f"*u{w}", M=f"*i{w}", O=f"*{sig}"),
                          constants(typ, True, "ZEROING"))


@pytest.mark.parametrize("typ", TYPES)
@pytest.mark.parametrize("pattern", ("unmasked", "all", "none", "tail", "mixed", "bits"))
@pytest.mark.parametrize("index_kind", ("reverse", "repeated"))
@pytest.mark.parametrize("mode", ("ZEROING", "MERGING"))
def test_run(typ, pattern, index_kind, mode):
    actual, expected, _ = launch(typ, pattern, index_kind, mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
