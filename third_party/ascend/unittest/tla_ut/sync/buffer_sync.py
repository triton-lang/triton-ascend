"""TLA 3-in-1 test: A5 buffer-token synchronization.

``mutex_lock`` / ``mutex_unlock`` emit ``hivm.hir.get_buf`` / ``hivm.hir.rls_buf``
with the frontend pipe, blocking or nonblocking mode, and constant or runtime
IDs. PR4478 infers the core type from the pipe, so the ops no longer carry a
separate ``TCoreTypeAttr``.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR structure, core scopes, modes and dynamic IDs
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch (requires an NPU)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_kernel_binary)

MAX_BUF = 31
N = 64
SIGNATURE = {"A_ptr": "*fp32", "B_ptr": "*fp32", "O_ptr": "*fp32"}


@tla.jit
def buffer_token_add(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a_ub = tla.alloc((N, ), tl.float32, space="ub")
    b_ub = tla.alloc((N, ), tl.float32, space="ub")
    o_ub = tla.alloc((N, ), tl.float32, space="ub")
    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.sync.mutex_lock(0, tla.sync.PIPE.MTE2)
    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    tla.sync.mutex_unlock(0, tla.sync.PIPE.MTE2)
    tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
    tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)

    tla.sync.mutex_lock(1, tla.sync.PIPE.V)
    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")
    tla.sync.mutex_unlock(1, tla.sync.PIPE.V)
    tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)
    tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)

    tla.sync.mutex_lock(2, tla.sync.PIPE.MTE3)
    tla.copy(o_ub, o)
    tla.sync.mutex_unlock(2, tla.sync.PIPE.MTE3)


@tla.jit
def buffer_tokens():
    tla.sync.mutex_lock(31, tla.sync.PIPE.MTE2)
    tla.sync.mutex_unlock(31, tla.sync.PIPE.MTE2)
    tla.sync.mutex_lock(0, tla.sync.PIPE.V, True)
    tla.sync.mutex_unlock(0, tla.sync.PIPE.V, True)
    with tla.scope.cube():
        tla.sync.mutex_lock(1, tla.sync.PIPE.M)
        tla.sync.mutex_unlock(1, tla.sync.PIPE.M)
        tla.sync.mutex_lock(31, tla.sync.PIPE.F, True)
        tla.sync.mutex_unlock(31, tla.sync.PIPE.F, True)


@tla.jit
def buffer_dynamic(bufid):
    tla.sync.mutex_lock(bufid, tla.sync.PIPE.MTE2)
    tla.sync.mutex_unlock(bufid, tla.sync.PIPE.MTE2, True)


def test_ir(output_dir=None):
    module, ctx = compile_offline(buffer_tokens, {}, {}, output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    expected = (
        f"hivm.hir.get_buf[<PIPE_MTE2>, {MAX_BUF}, false]",
        f"hivm.hir.rls_buf[<PIPE_MTE2>, {MAX_BUF}, false]",
        "hivm.hir.get_buf[<PIPE_V>, 0, true]",
        "hivm.hir.rls_buf[<PIPE_V>, 0, true]",
        "hivm.hir.get_buf[<PIPE_M>, 1, false]",
        "hivm.hir.rls_buf[<PIPE_M>, 1, false]",
        f"hivm.hir.get_buf[<PIPE_FIX>, {MAX_BUF}, true]",
        f"hivm.hir.rls_buf[<PIPE_FIX>, {MAX_BUF}, true]",
    )
    for probe in expected:
        assert probe in text, probe
    assert text.count("hivm.hir.get_buf") == 4, text
    assert text.count("hivm.hir.rls_buf") == 4, text

    module, ctx = compile_offline(buffer_dynamic, {"bufid": "i64"}, {})
    text = str(lower_offline(module, ctx))
    assert "hivm.hir.get_buf[<PIPE_MTE2>, %" in text, text
    assert "hivm.hir.rls_buf[<PIPE_MTE2>, %" in text, text
    print("IR PASS: buffer tokens, core scopes, modes, boundary and dynamic IDs")


def test_rejects_invalid_arguments():

    @tla.jit
    def bufid_too_large():
        tla.sync.mutex_lock(32, tla.sync.PIPE.MTE2)

    with pytest.raises(Exception, match="bufid must be in"):
        compile_offline(bufid_too_large, {}, {})

    @tla.jit
    def invalid_mode():
        tla.sync.mutex_unlock(0, tla.sync.PIPE.MTE2, 1)

    with pytest.raises(Exception, match="compile-time bool"):
        compile_offline(invalid_mode, {}, {})

    @tla.jit
    def wrong_pipe_for_core():
        tla.sync.mutex_lock(0, tla.sync.PIPE.MTE1)

    with pytest.raises(Exception, match="not supported by the enclosing vector core"):
        compile_offline(wrong_pipe_for_core, {}, {})


def _launch():
    import torch
    a = torch.randn(N, dtype=torch.float32, device="npu")
    b = torch.randn(N, dtype=torch.float32, device="npu")
    o = torch.full((N, ), -7.0, dtype=torch.float32, device="npu")
    buffer_token_add[(1, )](a, b, o, N=N)
    torch.npu.synchronize()
    return a, b, o


def _warmup():
    import torch
    a = torch.empty(N, dtype=torch.float32, device="npu")
    b = torch.empty(N, dtype=torch.float32, device="npu")
    o = torch.empty(N, dtype=torch.float32, device="npu")
    return buffer_token_add.warmup(a, b, o, grid=(1, ), N=N)


def test_compile():
    check_kernel_binary(_warmup())
    print("COMPILE PASS: buffer_token_add")


def test_run():
    import numpy as np
    a, b, o = _launch()
    ref = a.cpu().numpy() + b.cpu().numpy()
    got = o.cpu().numpy()
    ok = np.allclose(got, ref, atol=0.0, rtol=0.0)
    print("BUFFER_SYNC", "OK" if ok else "MISMATCH")
    assert ok, "buffer token accuracy mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_rejects_invalid_arguments()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: buffer_sync", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
