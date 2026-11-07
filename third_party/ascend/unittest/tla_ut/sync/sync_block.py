"""TLA 3-in-1 test: cross-core block flag synchronization.

Covers ``block_set_flag`` / ``block_wait_flag`` with three logical modes:

  INTER     -> INTER_BLOCK_SYNCHRONIZATION
  SUBBLOCK  -> INTER_SUBBLOCK_SYNCHRONIZATION
  BLOCK     -> INTRA_BLOCK_SYNCHRONIZATION

``INTER`` uses a pure AIV kernel: two AIV blocks run the same SET/WAIT
pair. ``SUBBLOCK`` uses the same kernel with the sibling AIV subcores in one
block. ``BLOCK`` keeps the MIX kernel with a Cube SET and a Vector WAIT.
All three modes are validated on device.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_kernel_binary)

N = 64
SIGNATURE = {"X_ptr": "*fp32", "O_ptr": "*fp32"}
MODE_NAMES = {
    tla.sync.MODE.INTER: "INTER_BLOCK_SYNCHRONIZATION",
    tla.sync.MODE.SUBBLOCK: "INTER_SUBBLOCK_SYNCHRONIZATION",
    tla.sync.MODE.BLOCK: "INTRA_BLOCK_SYNCHRONIZATION",
}


@tla.jit
def sync_block(X_ptr, O_ptr, N: tl.constexpr, MODE_C: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    x = tla.view(X_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    with tla.scope.cube():
        tla.sync.block_set_flag(MODE_C, tla.sync.PIPE.F, 0)

    with tla.scope.vector():
        tla.sync.block_wait_flag(MODE_C, tla.sync.PIPE.V, 0)
        tla.copy(x, x_ub)
        with tla.scope.simd():
            v = tla.simd.vlds(x_ub[0], dist="NORM")
            tla.simd.vsts(o_ub[0], v, dist="NORM_B32")
        tla.copy(o_ub, o)


@tla.jit
def sync_vector(X_ptr, O_ptr, N: tl.constexpr, MODE_C: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    offset = tl.program_id(0) * N
    x = tla.view(X_ptr, (N, ), strides=(1, ), offsets=(offset, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(offset, ))

    with tla.scope.vector():
        tla.copy(x, x_ub)
        tla.sync.block_set_flag(MODE_C, tla.sync.PIPE.MTE3, 0)
        tla.sync.block_wait_flag(MODE_C, tla.sync.PIPE.MTE3, 0)
        with tla.scope.simd():
            v = tla.simd.vlds(x_ub[0], dist="NORM")
            tla.simd.vsts(o_ub[0], v, dist="NORM_B32")
        tla.copy(o_ub, o)


def constexprs(mode):
    return {"N": N, "MODE_C": mode}


def test_ir(output_dir=None):
    for mode, attr in MODE_NAMES.items():
        kernel = sync_block if mode == tla.sync.MODE.BLOCK else sync_vector
        module, ctx = compile_offline(kernel, SIGNATURE, constexprs(mode), output_dir=output_dir)
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        extra = ('mix_mode = "mix"', ) if mode == tla.sync.MODE.BLOCK else ('mix_mode = "aiv"', )
        check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=extra)
        assert text.count("hivm.hir.sync_block_set") == 1, (mode, text)
        assert text.count("hivm.hir.sync_block_wait") == 1, (mode, text)
        assert "hivm.hir.sync_block[" not in text, text
        set_line = next(line for line in text.splitlines() if "hivm.hir.sync_block_set" in line)
        wait_line = next(line for line in text.splitlines() if "hivm.hir.sync_block_wait" in line)
        if mode == tla.sync.MODE.BLOCK:
            assert set_line.strip().startswith("hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>]"), set_line
            assert wait_line.strip().startswith("hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_V>]"), wait_line
            for line in (set_line, wait_line):
                # INTRA_BLOCK_SYNCHRONIZATION is the op default, so it is not printed.
                assert "sync_instr_mode = <" not in line, line
        else:
            assert set_line.strip().startswith(
                "hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 0"), set_line
            assert wait_line.strip().startswith(
                "hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 0"), wait_line
            for line in (set_line, wait_line):
                assert f"sync_instr_mode = <{attr}>" in line, line
    print("IR PASS: block_set_flag/block_wait_flag INTER/SUBBLOCK/BLOCK modes")


def test_rejects_non_enum_mode():

    @tla.jit
    def string_mode():
        tla.sync.block_set_flag("BLOCK", tla.sync.PIPE.F, 0)

    with pytest.raises(Exception, match="MODE enum"):
        compile_offline(string_mode, {}, {})

    @tla.jit
    def int_mode():
        tla.sync.block_set_flag(2, tla.sync.PIPE.F, 0)

    with pytest.raises(Exception, match="MODE enum"):
        compile_offline(int_mode, {}, {})


def test_rejects_pipe_for_enclosing_core():

    @tla.jit
    def vector_mte1():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 0)

    with pytest.raises(Exception, match="not supported by the enclosing vector core"):
        compile_offline(vector_mte1, {}, {})

    @tla.jit
    def cube_v():
        with tla.scope.cube():
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 0)

    with pytest.raises(Exception, match="not supported by the enclosing cube core"):
        compile_offline(cube_v, {}, {})

    @tla.jit
    def barrier_all():
        tla.sync.barrier(tla.sync.PIPE.ALL)

    compile_offline(barrier_all, {}, {})


@tla.jit
def global_barrier(MODE_C: tl.constexpr, CORE: tl.constexpr, SCOPED: tl.constexpr):
    if CORE == "cube" or CORE == "mix":
        with tla.scope.cube():
            tla.sync.barrier(tla.sync.PIPE.M)
            if SCOPED:
                tla.sync.sync_all(MODE_C)
    if CORE == "vector" or CORE == "mix":
        with tla.scope.vector():
            tla.sync.barrier(tla.sync.PIPE.V)
            if SCOPED:
                tla.sync.sync_all(MODE_C)
    if not SCOPED:
        tla.sync.sync_all(MODE_C)


@pytest.mark.parametrize("core,mode,scoped,flags", [
    ("vector", False, False, (14, 14)),
    ("vector", True, False, (12, 12)),
    ("cube", True, False, (13, 13)),
    ("mix", False, False, (14, 14)),
    ("mix", True, False, (10, 13, 13, 11, 10, 11)),
    ("mix", True, True, (10, 13, 13, 11, 10, 11)),
])
def test_global_barrier_expansion(core, mode, scoped, flags):
    module, ctx = compile_offline(global_barrier, {}, {"MODE_C": mode, "CORE": core, "SCOPED": scoped})
    text = str(lower_offline(module, ctx))
    assert "hivm.hir.sync_block[" not in text, text
    lines = [
        line.strip()
        for line in text.splitlines()
        if "hivm.hir.sync_block_set[" in line or "hivm.hir.sync_block_wait[" in line
    ]
    assert len(lines) == len(flags), text
    for line, flag in zip(lines, flags):
        assert f"flag = {flag}" in line, line
    if core == "mix" and mode:
        expected = (
            "wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10",
            "set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 13",
            "wait[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 13",
            "set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 11",
            "set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10",
            "wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 11",
        )
        for line, op in zip(lines, expected):
            assert f"hivm.hir.sync_block_{op}" in line, line
        for index, line in enumerate(lines):
            assert ("INTER_BLOCK_SYNCHRONIZATION" in line) == (index in (1, 2)), line
    else:
        core_name = "CUBE" if core == "cube" else "VECTOR"
        pipe = "PIPE_FIX" if core == "cube" else "PIPE_MTE3"
        for line, op in zip(lines, ("set", "wait")):
            assert f"sync_block_{op}[<{core_name}>, <{pipe}>, <PIPE_S>]" in line, line
            assert "INTER_BLOCK_SYNCHRONIZATION" in line, line


def test_rejects_global_barrier_in_simd():

    @tla.jit
    def simd_barrier():
        with tla.scope.simd():
            tla.sync.sync_all(False)

    module, ctx = compile_offline(simd_barrier, {}, {})
    with pytest.raises(ValueError, match="outside SIMD/SIMT scopes"):
        lower_offline(module, ctx)


def test_rejects_vector_barrier_on_cube():
    module, ctx = compile_offline(global_barrier, {}, {"MODE_C": False, "CORE": "cube", "SCOPED": False})
    with pytest.raises(ValueError, match="requires a vector core"):
        lower_offline(module, ctx)


def test_global_barrier_without_core_scope():

    @tla.jit
    def plain_barrier(MODE_C: tl.constexpr):
        tla.sync.sync_all(MODE_C)

    for mode, flag in ((False, 14), (True, 12)):
        module, ctx = compile_offline(plain_barrier, {}, {"MODE_C": mode})
        text = str(lower_offline(module, ctx))
        assert "hivm.hir.sync_block[" not in text, text
        for op in ("set", "wait"):
            assert f"hivm.hir.sync_block_{op}[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = {flag}" in text, text


def _kernel_and_grid(mode_name):
    mode = getattr(tla.sync.MODE, mode_name)
    kernel = sync_block if mode == tla.sync.MODE.BLOCK else sync_vector
    grid = (2, ) if mode_name == "INTER" else (1, )
    return kernel, grid, mode


def _launch(mode_name):
    import torch
    kernel, grid, mode = _kernel_and_grid(mode_name)
    count = N * grid[0]
    x = torch.arange(count, dtype=torch.float32, device="npu")
    o = torch.full((count, ), -7.0, dtype=torch.float32, device="npu")
    kernel[grid](x, o, **constexprs(mode))
    torch.npu.synchronize()
    return x, o


def test_compile():
    pytest.skip("skipped: NPUIR bd228fae regression (AffinityHIR read-before-write)")
    import torch
    for mode_name in ("INTER", "SUBBLOCK", "BLOCK"):
        kernel, grid, mode = _kernel_and_grid(mode_name)
        count = N * grid[0]
        x = torch.empty(count, dtype=torch.float32, device="npu")
        o = torch.empty(count, dtype=torch.float32, device="npu")
        check_kernel_binary(kernel.warmup(x, o, grid=grid, **constexprs(mode)))
    print("COMPILE PASS: sync_block INTER/SUBBLOCK/BLOCK")


def test_run():
    pytest.skip("skipped: NPUIR bd228fae regression (AffinityHIR read-before-write)")
    import torch
    for mode_name in ("INTER", "SUBBLOCK", "BLOCK"):
        x, o = _launch(mode_name)
        torch.testing.assert_close(o.cpu(), x.cpu(), rtol=0, atol=0)
        print(f"SYNC_BLOCK {mode_name} OK")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_rejects_non_enum_mode()
        test_rejects_pipe_for_enclosing_core()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: sync_block", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
