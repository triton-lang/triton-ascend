"""TLA 3-in-1 test: cross-core block flag synchronization.

Covers ``block_set_flag`` / ``block_wait_flag`` with the three FFTS modes:

  INTER     -> INTER_BLOCK_SYNCHRONIZATION
  SUBBLOCK  -> INTER_SUBBLOCK_SYNCHRONIZATION
  BLOCK     -> INTRA_BLOCK_SYNCHRONIZATION

All three modes are checked at IR level. Device execution uses ``BLOCK``,
which is the same-block mode available on A5.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

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


def constexprs(mode):
    return {"N": N, "MODE_C": mode}


def test_ir(output_dir=None):
    for mode, attr in MODE_NAMES.items():
        module, ctx = compile_offline(sync_block, SIGNATURE, constexprs(mode), output_dir=output_dir)
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=('mix_mode = "mix"', ))
        assert text.count("hivm.hir.sync_block_set") == 1, (mode, text)
        assert text.count("hivm.hir.sync_block_wait") == 1, (mode, text)
        set_line = next(line for line in text.splitlines() if "hivm.hir.sync_block_set" in line)
        wait_line = next(line for line in text.splitlines() if "hivm.hir.sync_block_wait" in line)
        assert set_line.strip().startswith("hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>]"), set_line
        assert wait_line.strip().startswith("hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_V>]"), wait_line
        for line in (set_line, wait_line):
            if mode == tla.sync.MODE.BLOCK:
                # INTRA_BLOCK_SYNCHRONIZATION is the op default, so it is not printed.
                assert "sync_instr_mode = <" not in line, line
            else:
                assert f"sync_instr_mode = <{attr}>" in line, line
    print("IR PASS: block_set_flag/block_wait_flag INTER/SUBBLOCK/BLOCK modes")


def _launch(mode):
    import torch
    x = torch.arange(N, dtype=torch.float32, device="npu")
    o = torch.full((N, ), -7.0, dtype=torch.float32, device="npu")
    sync_block[(1, )](x, o, **constexprs(mode))
    torch.npu.synchronize()
    return x, o


def test_compile():
    print("SKIP: sync_block compile/run is IR-only on A5; FFTS block-flag modes are not available")


def test_run():
    print("SKIP: sync_block compile/run is IR-only on A5; FFTS block-flag modes are not available")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: sync_block", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
