"""TLA 3-in-1 test: intra-core synchronization.

Real kernel: ``out = a + b`` on f32 with the pipeline fully hand-synchronized:

    GM -> UB (MTE2)  --set/wait_flag-->  vector add (V)  --barrier-->
    UB -> GM (MTE3)  --sync_all-->  next AIV block

The sync calls are inserted manually between the DMA and vector stages, so
the test verifies both that the emitted synchronization is correct and that a
real dependency chain still produces the right values.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (no NPU, no torch)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (SYNC_PIPE_CORE OK)

The IR case checks frontend expansion of ``sync_all`` for both the pure AIV
(``False``) and mixed AIC/AIV (``True``) forms. Device execution covers the
hand-synchronized data path and both global barriers.
"""
import os
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_kernel_binary)

N = 64  # f32 vector-register lane count (2048 bits)

SIGNATURE = {"A_ptr": "*fp32", "B_ptr": "*fp32", "O_ptr": "*fp32"}


@tla.jit
def sync_pipe(A_ptr, B_ptr, O_ptr, N: tl.constexpr, SYNC_ALL: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    # GM -> UB must finish before the vector stage reads the buffers.
    tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
    tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")

    # Vector register writes must be visible before the UB -> GM store.
    tla.sync.barrier(tla.sync.PIPE.ALL)
    tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)
    tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)
    tla.copy(o_ub, o)

    # Hardware barrier across the AIV blocks after the GM writes.
    if SYNC_ALL:
        tla.sync.sync_all(False)


@tla.jit
def sync_all_mix(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    with tla.scope.cube():
        tla.sync.sync_all(True)

    with tla.scope.vector():
        # GM -> UB copies are Vector-core operations.
        tla.copy(a, a_ub)
        tla.copy(b, b_ub)
        tla.sync.sync_all(True)
        with tla.scope.simd():
            av = tla.simd.vlds(a_ub[0], dist="NORM")
            bv = tla.simd.vlds(b_ub[0], dist="NORM")
            tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")

        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)
        tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)
        tla.copy(o_ub, o)


def constexprs(sync_all=True):
    return {"N": N, "SYNC_ALL": sync_all}


@tla.jit
def enum_pipes(PIPE: tl.constexpr, CUBE: tl.constexpr):
    if CUBE:
        with tla.scope.cube():
            tla.sync.barrier(PIPE)
            tla.sync.set_flag(tla.sync.PIPE.S, PIPE, 0)
            tla.sync.wait_flag(tla.sync.PIPE.S, PIPE, 0)
    else:
        tla.sync.barrier(PIPE)
        tla.sync.set_flag(tla.sync.PIPE.S, PIPE, 0)
        tla.sync.wait_flag(tla.sync.PIPE.S, PIPE, 0)


def test_enum_pipes():
    for cube, pipes in (
        (False, (tla.sync.PIPE.S, tla.sync.PIPE.V, tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE3)),
        (True, (tla.sync.PIPE.S, tla.sync.PIPE.M, tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE3,
                tla.sync.PIPE.F)),
    ):
        for pipe in pipes:
            module, ctx = compile_offline(enum_pipes, {}, {"PIPE": pipe, "CUBE": cube})
            text = str(lower_offline(module, ctx))
            assert f"hivm.hir.pipe_barrier[<{pipe.value}>]" in text, text
            for op in ("set_flag", "wait_flag"):
                assert f"hivm.hir.{op}[<PIPE_S>, <{pipe.value}>" in text, text
    for pipe in ("MTE2", "PIPE_S", 0):
        try:
            compile_offline(enum_pipes, {}, {"PIPE": pipe, "CUBE": False})
        except Exception as exc:
            assert "pipe must be a compile-time tla.sync.PIPE enum member" in str(exc), str(exc)
        else:
            raise AssertionError(f"non-enum pipe accepted: {pipe}")
    print("IR PASS: Vector/Cube pipeline enums and non-enum rejection")


def test_ir(output_dir=None):
    module, ctx = compile_offline(sync_pipe, SIGNATURE, constexprs(True), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None)
    for op in ("hivm.hir.set_flag", "hivm.hir.wait_flag", "hivm.hir.pipe_barrier"):
        assert text.count(op) == 2, text
    assert text.count("hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>") == 1, text
    assert text.count("hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>") == 1, text
    assert "<EVENT_ID0>" in text and "<EVENT_ID1>" in text, text
    assert "hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>" in text, text
    assert "hivm.hir.pipe_barrier[<PIPE_ALL>]" in text, text
    assert "hivm.hir.sync_block[" not in text, text
    assert "hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 14" in text, text
    assert "hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 14" in text, text
    assert text.count("ave.hir.vadd") == 1, text
    order = [
        text.index(op)
        for op in ("hivm.hir.set_flag", "hivm.hir.wait_flag", "ave.hir.vadd", "hivm.hir.pipe_barrier", "hivm.hir.store")
    ]
    assert order == sorted(order), "sync ops must stay between the stages"

    module, ctx = compile_offline(sync_all_mix, SIGNATURE, {"N": N}, output_dir=output_dir)
    mix = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(mix, loads=2, stores=1, buffers=None, extra=('mix_mode = "mix"', ))
    assert "hivm.hir.sync_block[" not in mix, mix
    assert mix.count("hivm.hir.sync_block_set") == 3, mix
    assert mix.count("hivm.hir.sync_block_wait") == 3, mix
    assert mix.count("hivm.hir.pipe_barrier[<PIPE_ALL>]") == 3, mix
    for probe in (
            "hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 13",
            "hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 11",
            "hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10",
            "hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10",
            "hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 13",
            "hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 11",
    ):
        assert probe in mix, mix
    print("IR PASS: sync_pipe (MTE2->V->MTE3 hand sync, barrier, sync_all AIV/MIX)")
    test_enum_pipes()


def _launch(sync_all=True):
    import torch
    A = torch.randn(N, dtype=torch.float32, device="npu")
    B = torch.randn(N, dtype=torch.float32, device="npu")
    O = torch.full((N, ), -7.0, dtype=torch.float32, device="npu")
    sync_pipe[(1, )](A, B, O, **constexprs(sync_all))
    torch.npu.synchronize()
    return A, B, O


def _launch_mix():
    import torch
    A = torch.arange(N, dtype=torch.float32, device="npu")
    B = 2 * A
    O = torch.full((N, ), -7.0, dtype=torch.float32, device="npu")
    sync_all_mix[(1, )](A, B, O, N=N)
    torch.npu.synchronize()
    return A, B, O


def test_compile():
    pytest.skip("skipped: NPUIR bd228fae regression (AffinityHIR read-before-write)")
    import torch
    A = torch.randn(N, dtype=torch.float32, device="npu")
    B = torch.randn(N, dtype=torch.float32, device="npu")
    O = torch.empty(N, dtype=torch.float32, device="npu")
    sync_pipe_kernel = sync_pipe.warmup(A, B, O, grid=(1, ), **constexprs(True))
    sync_mix_kernel = sync_all_mix.warmup(A, B, O, grid=(1, ), N=N)
    check_kernel_binary(sync_pipe_kernel)
    check_kernel_binary(sync_mix_kernel)
    print("COMPILE PASS: sync_pipe + sync_all_mix")


def test_run():
    pytest.skip("skipped: NPUIR bd228fae regression (AffinityHIR read-before-write)")
    import numpy as np
    A, B, O = _launch(True)
    ref = A.cpu().numpy() + B.cpu().numpy()
    ok = np.allclose(O.cpu().numpy(), ref, atol=0.0, rtol=0.0)
    print("SYNC_PIPE_CORE", "OK" if ok else "MISMATCH")
    assert ok, "sync_pipe accuracy mismatch"

    A, B, O = _launch_mix()
    ref = A.cpu().numpy() + B.cpu().numpy()
    ok = np.allclose(O.cpu().numpy(), ref, atol=0.0, rtol=0.0)
    print("SYNC_ALL_TRUE MIX", "OK" if ok else "MISMATCH")
    assert ok, "sync_all_mix accuracy mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: sync_pipe", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
