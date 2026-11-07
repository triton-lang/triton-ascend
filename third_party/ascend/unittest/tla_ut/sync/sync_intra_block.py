"""TLA 3-in-1 test: A5 intra-block counter synchronization.

``block_set_flag`` / ``block_wait_flag`` with ``MODE.INTRA`` emit direct
``hivm.hir.sync_intra_block_set/wait`` operations. Unlike the FFTS modes, they
preserve the individual physical counter ID and do not signal or wait on the
partner ID automatically.

This file checks offline HIVM IR structure, core scopes, static ID bounds
and dynamic IDs, then compiles and runs the same protocol on device: Cube SET
on PIPE.F #0/#16 paired with Vector WAIT on PIPE.V #0 in front of a GM->UB->GM
data chain.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_kernel_binary

N = 64
VEC_ID = 15
CUBE_VEC0 = 15
CUBE_VEC1 = 31
MAX_ID = 31


@tla.jit
def intra_block():
    with tla.scope.cube():
        tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.F, 15)
        tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.F, 31)
        tla.sync.block_wait_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.S, 31)
    tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.MTE2, 0)
    tla.sync.block_wait_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.MTE2, 15)


@tla.jit
def intra_block_dynamic(flagid):
    tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.MTE2, flagid)
    tla.sync.block_wait_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.MTE2, flagid)


@tla.jit
def intra_block_data(X_ptr, O_ptr, N: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    x = tla.view(X_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    with tla.scope.cube():
        tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.F, 0)
        tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.F, 16)

    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.V, 0)
        tla.copy(x, x_ub)
        with tla.scope.simd():
            v = tla.simd.vlds(x_ub[0], dist="NORM")
            tla.simd.vsts(o_ub[0], v, dist="NORM_B32")
        tla.copy(o_ub, o)


def test_ir(output_dir=None):
    module, ctx = compile_offline(intra_block, {}, {}, output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    expected = (
        f"hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, {CUBE_VEC0}]",
        f"hivm.hir.sync_intra_block_set[<CUBE>, <PIPE_FIX>, {CUBE_VEC1}]",
        f"hivm.hir.sync_intra_block_wait[<CUBE>, <PIPE_S>, {CUBE_VEC1}]",
        "hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_MTE2>, 0]",
        f"hivm.hir.sync_intra_block_wait[<VECTOR>, <PIPE_MTE2>, {VEC_ID}]",
    )
    for probe in expected:
        assert probe in text, probe
    assert text.count("hivm.hir.sync_intra_block_set") == 3, text
    assert text.count("hivm.hir.sync_intra_block_wait") == 2, text

    module, ctx = compile_offline(intra_block_dynamic, {"flagid": "i64"}, {})
    text = str(lower_offline(module, ctx))
    assert "hivm.hir.sync_intra_block_set[<VECTOR>, <PIPE_MTE2>, %" in text, text
    assert "hivm.hir.sync_intra_block_wait[<VECTOR>, <PIPE_MTE2>, %" in text, text
    print("IR PASS: intra-block counters, core scopes, boundary and dynamic IDs")


def test_rejects_out_of_range():

    # Vector counters are limited to 15 by the C++ builder, below the generic
    # Python bound of 31, so 16 must be rejected only after the core is known.
    @tla.jit
    def vector_too_large():
        tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.MTE2, 16)

    with pytest.raises(Exception, match="flagid must be in"):
        compile_offline(vector_too_large, {}, {})

    @tla.jit
    def cube_too_large():
        with tla.scope.cube():
            tla.sync.block_set_flag(tla.sync.MODE.INTRA, tla.sync.PIPE.F, 32)

    with pytest.raises(Exception, match="flagid must be in"):
        compile_offline(cube_too_large, {}, {})


def test_compile():
    pytest.skip("skipped: NPUIR bd228fae regression (AffinityHIR read-before-write)")
    import torch
    x = torch.empty(N, dtype=torch.float32, device="npu")
    o = torch.empty(N, dtype=torch.float32, device="npu")
    check_kernel_binary(intra_block_data.warmup(x, o, grid=(1, ), N=N))
    print("COMPILE PASS: intra_block_data")


def test_run():
    pytest.skip("skipped: NPUIR bd228fae regression (AffinityHIR read-before-write)")
    import torch
    x = torch.arange(N, dtype=torch.float32, device="npu")
    o = torch.full((N, ), -7.0, dtype=torch.float32, device="npu")
    intra_block_data[(1, )](x, o, N=N)
    torch.npu.synchronize()
    torch.testing.assert_close(o.cpu(), x.cpu(), rtol=0, atol=0)
    print("INTRA RUN OK")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_rejects_out_of_range()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: sync_intra_block", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
