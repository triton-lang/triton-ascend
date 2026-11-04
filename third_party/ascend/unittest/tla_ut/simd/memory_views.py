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
"""Regression coverage for SIMD memory aliasing and predicate byte views."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, lower_offline


@tla.jit
def alias_reload(X, I, O, OFFSET, W: tl.constexpr, VIEW: tl.constexpr, GATHER: tl.constexpr, LOOP: tl.constexpr):
    N: tl.constexpr = 2048 // W
    DT: tl.constexpr = tl.int8 if W == 8 else (tl.int16 if W == 16 else tl.int32)
    IT: tl.constexpr = tl.uint32 if W == 32 else tl.uint16
    a = tla.alloc_buffer((4 * N, ), DT, space="ub")
    indices = tla.alloc_buffer((N, ), IT, space="ub")
    out = tla.alloc_buffer((2 * N, ), DT, space="ub")
    tla.copy(tla.view(X, (4 * N, )), a)
    tla.copy(tla.view(I, (N, )), indices)
    if VIEW == "static":
        target = tla.subview(a, offsets=(N, ), sizes=(N, ))
    elif VIEW == "dynamic":
        target = tla.subview(a, offsets=(OFFSET, ), sizes=(N, ))
    elif VIEW == "nested":
        view = tla.subview(a, offsets=(OFFSET - 32, ), sizes=(2 * N, ))
        target = tla.subview(view, offsets=(32, ), sizes=(N, ))
    elif VIEW == "selected":
        if OFFSET == N:
            target = tla.subview(a, offsets=(N, ), sizes=(N, ))
        else:
            target = tla.subview(a, offsets=(2 * N, ), sizes=(N, ))
    with tla.scope.simd():
        if VIEW == "internal":
            target = tla.subview(a, offsets=(OFFSET, ), sizes=(N, ))
        if W == 8:
            idx = (tla.simd.vlds(indices[0]), tla.simd.vlds(indices[128]))
        else:
            idx = tla.simd.vlds(indices[0])
        data = tla.simd.vlds(a[0])
        if GATHER:
            old = tla.simd.vgather(a[OFFSET], idx)
        else:
            old = tla.simd.vlds(a[OFFSET])
        tla.simd.vsts(out[0], old, dist="NORM_B8" if W == 8 else ("NORM_B16" if W == 16 else "NORM_B32"))
        if LOOP:
            for iteration in range(3):
                tla.simd.vscatter(target[0], idx, data)
                tla.simd.membar("VST_VLD")
                data = tla.simd.vadds(tla.simd.vlds(a[OFFSET]), 1)
        else:
            tla.simd.vscatter(target[0], idx, data)
            tla.simd.membar("VST_VLD")
        if GATHER:
            new = tla.simd.vgather(a[OFFSET], idx)
        else:
            new = tla.simd.vlds(a[OFFSET])
        tla.simd.vsts(out[N], new, dist="NORM_B8" if W == 8 else ("NORM_B16" if W == 16 else "NORM_B32"))
    tla.copy(out, tla.view(O, (2 * N, )))


ALIAS_CASES = [(w, view, gather, loop)
               for w in (8, 16, 32)
               for view, gather, loop in (("static", False, False), ("dynamic", True, False), ("nested", False, True),
                                          ("internal", True, True), ("selected", False, False))]


def alias_signature(width):
    return dict(X=f"*i{width}", I="*u32" if width == 32 else "*u16", O=f"*i{width}", OFFSET="i32")


@pytest.mark.parametrize("case", ALIAS_CASES)
def test_ir_alias(case):
    width, view, gather, loop = case
    module, ctx = compile_offline(alias_reload, alias_signature(width),
                                  dict(W=width, VIEW=view, GATHER=gather, LOOP=loop))
    text = str(lower_offline(module, ctx))
    signature = next(line for line in text.splitlines() if "func.func private @tla_native_simd(" in line)
    # a, indices and out are distinct allocations; target must be rebuilt from a.
    assert signature.count("memref<") == 3
    assert "memref.subview" in text
    assert "llvm.launder.invariant.group" in text
    assert module.verify()


@pytest.mark.parametrize("case", ALIAS_CASES)
@pytest.mark.skipif(not get_stages()[2], reason="runtime stage disabled")
def test_run_alias(case):
    width, view, gather, loop = case
    n = 2048 // width
    dtype = {8: torch.int8, 16: torch.int16, 32: torch.int32}[width]
    data = ((torch.arange(4 * n) * 13 + 17) % 97 - 48).to(dtype)
    indices = torch.arange(n).to(torch.uint32 if width == 32 else torch.uint16)
    for offset in ((n, 2 * n) if view == "selected" else (n, )):
        out = torch.empty(2 * n, dtype=dtype, device="npu")
        alias_reload[(1, )](data.npu(), indices.npu(), out, offset, W=width, VIEW=view, GATHER=gather, LOOP=loop)
        torch.npu.synchronize()
        expected = torch.cat((data[offset:offset + n], data[:n] + (2 if loop else 0)))
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@tla.jit
def static_vci_alias_reload(X, O):
    """Keep the regression sensitive to the vci/scatter address optimizer."""
    a = tla.alloc_buffer((256, ), tl.int32, space="ub")
    out = tla.alloc_buffer((128, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (256, )), a)
    target = tla.subview(a, offsets=(64, ), sizes=(64, ))
    with tla.scope.simd():
        index = tla.simd.vbitcast(tla.simd.vci(0), "u32")
        data = tla.simd.vlds(a[0])
        old = tla.simd.vlds(a[64])
        tla.simd.vsts(out[0], old, dist="NORM_B32")
        tla.simd.vscatter(target[0], index, data)
        tla.simd.membar("VST_VLD")
        new = tla.simd.vlds(a[64])
        tla.simd.vsts(out[64], new, dist="NORM_B32")
    tla.copy(out, tla.view(O, (128, )))


def test_ir_static_vci_alias():
    module, ctx = compile_offline(static_vci_alias_reload, dict(X="*i32", O="*i32"), {})
    text = str(lower_offline(module, ctx))
    signature = next(line for line in text.splitlines() if "func.func private @tla_native_simd(" in line)
    assert signature.count("memref<") == 2
    assert "memref.subview" in text
    assert module.verify()


@pytest.mark.skipif(not get_stages()[2], reason="runtime stage disabled")
def test_run_static_vci_alias():
    data = torch.arange(256, dtype=torch.int32) * 131 + 93
    out = torch.empty(128, dtype=torch.int32, device="npu")
    static_vci_alias_reload[(1, )](data.npu(), out)
    torch.npu.synchronize()
    expected = torch.cat((data[64:128], data[:64]))
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@tla.jit
def opaque_alias_capture(X, O, COUNT):
    a = tla.alloc_buffer((256, ), tl.int32, space="ub")
    out = tla.alloc_buffer((64, ), tl.int32, space="ub")
    tla.copy(tla.view(X, (256, )), a)
    target = tla.subview(a, offsets=(64, ), sizes=(64, ))
    for j in range(COUNT):
        target = tla.subview(a, offsets=(j * 64, ), sizes=(64, ))
    with tla.scope.simd():
        index = tla.simd.vbitcast(tla.simd.vci(0), "u32")
        tla.simd.vscatter(target[0], index, tla.simd.vlds(a[0]))
        tla.simd.membar("VST_VLD")
        tla.simd.vsts(out[0], tla.simd.vlds(a[64]), dist="NORM_B32")
    tla.copy(out, tla.view(O, (64, )))


def test_ir_opaque_alias_capture():
    module, ctx = compile_offline(opaque_alias_capture, dict(X="*i32", O="*i32", COUNT="i32"), {})
    # A loop-carried memref and its base must not become independent noalias
    # arguments. Until rebasing supports it, reject this case explicitly.
    with pytest.raises(ValueError, match="memory captures may alias"):
        lower_offline(module, ctx)


@tla.jit
def predicate_views(X, O, LOAD_OFFSET, STORE_OFFSET, W: tl.constexpr, DIST: tl.constexpr, VIEW: tl.constexpr):
    x = tla.alloc_buffer((512, ), tl.uint8, space="ub")
    out = tla.alloc_buffer((512, ), tl.uint8, space="ub")
    tla.copy(tla.view(X, (512, )), x)
    tla.copy(tla.view(X, (512, )), out)
    if VIEW == "static":
        src = tla.subview(x, offsets=(64, ), sizes=(256, ))
        dst = tla.subview(out, offsets=(128, ), sizes=(256, ))
    elif VIEW == "dynamic":
        src = tla.subview(x, offsets=(LOAD_OFFSET, ), sizes=(256, ))
        dst = tla.subview(out, offsets=(STORE_OFFSET, ), sizes=(256, ))
    else:
        src_outer = tla.subview(x, offsets=(LOAD_OFFSET - 32, ), sizes=(320, ))
        dst_outer = tla.subview(out, offsets=(STORE_OFFSET - 32, ), sizes=(320, ))
        src = tla.subview(src_outer, offsets=(32, ), sizes=(256, ))
        dst = tla.subview(dst_outer, offsets=(32, ), sizes=(256, ))
    with tla.scope.simd():
        pred = tla.simd.pld(src[32], width=W, dist=DIST)
        tla.simd.pst(dst[64], pred)
    tla.copy(out, tla.view(O, (512, )))


PREDICATE_CASES = [(width, dist, view)
                   for width in (8, 16, 32)
                   for dist in ("NORM", "US", "DS")
                   for view in ("static", "dynamic", "nested")]


@pytest.mark.parametrize("case", PREDICATE_CASES)
def test_ir_predicate_view(case):
    width, dist, view = case
    module, _ = compile_offline(predicate_views, dict(X="*u8", O="*u8", LOAD_OFFSET="i32", STORE_OFFSET="i32"),
                                dict(W=width, DIST=dist, VIEW=view))
    assert module.verify()


@pytest.mark.parametrize("case", PREDICATE_CASES)
@pytest.mark.skipif(not get_stages()[2], reason="runtime stage disabled")
def test_run_predicate_view(case):
    width, dist, view = case
    raw = torch.randint(0, 256, (512, ), generator=torch.Generator().manual_seed(331), dtype=torch.uint8)
    out = torch.empty_like(raw, device="npu")
    predicate_views[(1, )](raw.npu(), out, 64, 128, W=width, DIST=dist, VIEW=view)
    torch.npu.synchronize()
    bits = torch.arange(2048 // width)
    if dist == "US":
        bits //= 2
    elif dist == "DS":
        bits *= 2
    logical = (raw.to(torch.int64)[96 + bits // 8] >> (bits % 8)) & 1
    packed = (logical.reshape(-1, 8) << torch.arange(8)).sum(1).to(torch.uint8)
    expected = raw.clone()
    expected[192:192 + packed.numel()] = packed
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
