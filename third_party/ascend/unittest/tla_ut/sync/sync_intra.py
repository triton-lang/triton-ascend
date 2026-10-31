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
  run     : real-device launch + accuracy (SYNC_INTRA OK)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

N = 64  # f32 vector-register lane count (2048 bits)

SIGNATURE = {"A_ptr": "*fp32", "B_ptr": "*fp32", "O_ptr": "*fp32"}


@triton.jit
def sync_intra(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    # GM -> UB must finish before the vector stage reads the buffers.
    tla.sync.set_flag("MTE2", "V", 0)
    tla.sync.wait_flag("MTE2", "V", 0)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")

    # Vector register writes must be visible before the UB -> GM store.
    tla.sync.barrier("ALL")
    tla.sync.set_flag("V", "MTE3", 1)
    tla.sync.wait_flag("V", "MTE3", 1)
    tla.copy(o_ub, o)

    # Hardware barrier across the AIV blocks after the GM writes.
    tla.sync.sync_all(False)


def constexprs():
    return {"N": N}


def test_ir(output_dir=None):
    module, ctx = compile_offline(sync_intra, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=3)
    for op in ("hivm.hir.set_flag", "hivm.hir.wait_flag", "hivm.hir.pipe_barrier", "hivm.hir.sync_block"):
        assert text.count(op) == (2 if "flag" in op else 1), text
    assert text.count("hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>") == 1, text
    assert text.count("hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>") == 1, text
    assert "<EVENT_ID0>" in text and "<EVENT_ID1>" in text, text
    assert "hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>" in text, text
    assert "hivm.hir.pipe_barrier[<PIPE_ALL>]" in text, text
    assert "hivm.hir.sync_block[<ALL_VECTOR>]" in text, text
    assert text.count("ave.hir.vadd") == 1, text
    order = [
        text.index(op)
        for op in ("hivm.hir.set_flag", "hivm.hir.wait_flag", "ave.hir.vadd", "hivm.hir.pipe_barrier", "hivm.hir.store")
    ]
    assert order == sorted(order), "sync ops must stay between the stages"
    print("IR PASS: sync_intra (MTE2->V->MTE3 hand sync, barrier, sync_all)")


def _launch():
    import torch
    A = torch.randn(N, dtype=torch.float32, device="npu")
    B = torch.randn(N, dtype=torch.float32, device="npu")
    O = torch.full((N, ), -7.0, dtype=torch.float32, device="npu")
    sync_intra[(1, )](A, B, O, enable_tla_mode=True, **constexprs())
    torch.npu.synchronize()
    return A, B, O


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: sync_intra")


def test_run():
    import numpy as np
    A, B, O = _launch()
    ref = A.cpu().numpy() + B.cpu().numpy()
    ok = np.allclose(O.cpu().numpy(), ref, atol=0.0, rtol=0.0)
    print("SYNC_INTRA", "OK" if ok else "MISMATCH")
    assert ok, "sync_intra accuracy mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: sync_intra", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
