#!/usr/bin/env python3
"""Triton arithmetic calibration draft for f32.add / sin / cos.

Method: use the SAME Triton kernel for both `simd` and `simt_only`.
The only thing changed between modes is `compile_mode`; the backend decides
how to lower the arithmetic (including any Taylor expansion automatically).

Run on Ascend server:
    source ~/env_ascend.sh
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=1
    python3 bench_arith_triton.py --modes simd simt_only --ops add sin cos
"""
from __future__ import annotations

import argparse
import statistics

import torch
import torch_npu
import triton
import triton.language as tl

# Match the CCE f32.add baselines:
#   - SIMD: tput.cce ILP sweep saturates at ILP>=4 (3.30 vadd/cyc).
#   - SIMT: tput.cce uses 32 warps x 32 lanes, with 8 independent scalar
#     add chains per thread (ILP=8, 140.8 adds/cyc).
# The Triton benchmark uses the same arithmetic ILP per mode.
DEFAULT_ILP = {"simd": 4, "simt_only": 8}

DEFAULT_GRID = 56  # one AIV per program
DEFAULT_BLOCK = {
    "simd": 1024,  # SIMD ILP=4 works at 1024 elements
    "simt_only": 512
}  # SIMT ILP=8 needs a smaller tile here


@triton.jit
def _apply_op(x, MODE: tl.constexpr, k):
    if MODE == 0:  # add
        return x + k
    elif MODE == 1:  # sin
        return tl.sin(x)
    else:  # cos
        return tl.cos(x)


@triton.jit
def arith_kernel(x, out, ITERS, MODE: tl.constexpr, BLOCK: tl.constexpr, ILP: tl.constexpr):
    pid = tl.program_id(0)
    k = 1.0000001
    base = pid * (BLOCK * ILP)
    offs0 = base + tl.arange(0, BLOCK)
    offs1 = offs0 + BLOCK
    offs2 = offs1 + BLOCK
    offs3 = offs2 + BLOCK

    x0 = tl.load(x + offs0)
    x1 = tl.load(x + offs1)
    x2 = tl.load(x + offs2)
    x3 = tl.load(x + offs3)
    if ILP >= 8:
        x4 = tl.load(x + offs0 + 4 * BLOCK)
        x5 = tl.load(x + offs0 + 5 * BLOCK)
        x6 = tl.load(x + offs0 + 6 * BLOCK)
        x7 = tl.load(x + offs0 + 7 * BLOCK)

    for _ in tl.range(0, ITERS):
        x0 = _apply_op(x0, MODE, k)
        x1 = _apply_op(x1, MODE, k)
        x2 = _apply_op(x2, MODE, k)
        x3 = _apply_op(x3, MODE, k)
        if ILP >= 8:
            x4 = _apply_op(x4, MODE, k)
            x5 = _apply_op(x5, MODE, k)
            x6 = _apply_op(x6, MODE, k)
            x7 = _apply_op(x7, MODE, k)

    tl.store(out + offs0, x0)
    tl.store(out + offs1, x1)
    tl.store(out + offs2, x2)
    tl.store(out + offs3, x3)
    if ILP >= 8:
        tl.store(out + offs0 + 4 * BLOCK, x4)
        tl.store(out + offs0 + 5 * BLOCK, x5)
        tl.store(out + offs0 + 6 * BLOCK, x6)
        tl.store(out + offs0 + 7 * BLOCK, x7)


def _opts(mode: str, grid: int):
    if mode == "simd":
        return {
            "num_warps": 1,
            "compile_mode": "simd",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": False,
            "superblock_factor": 1,
            "logical_program_count_hint": grid,
        }
    if mode == "simt_only":
        return {
            "num_warps": 1,
            "compile_mode": "simt_only",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": True,
            "superblock_factor": 1,
            "logical_program_count_hint": grid,
        }
    raise ValueError(mode)


def _median_launch_ms(launch, reps: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _tensor_for_kernel(grid: int, block: int, ilp: int):
    n = grid * block * ilp
    x = torch.rand(n, dtype=torch.float32, device="npu")
    out = torch.zeros(n, dtype=torch.float32, device="npu")
    return x, out


def measure(mode, op, ops, grid, block, reps):
    ilp = DEFAULT_ILP[mode]
    x, out = _tensor_for_kernel(grid, block, ilp)
    opts = _opts(mode, grid)

    def launch():
        arith_kernel[(grid, )](x, out, ITERS=ops, MODE=op, BLOCK=block, ILP=ilp, **opts)

    return _median_launch_ms(launch, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--ops", nargs="+", default=["add", "sin", "cos"], choices=["add", "sin", "cos"])
    ap.add_argument("--ops-list", type=int, nargs="+", default=[0, 64, 128, 256, 512])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    ap.add_argument("--block", type=int, default=None, help="override per-mode default block size")
    args = ap.parse_args()

    op_code = {"add": 0, "sin": 1, "cos": 2}

    print("mode,op,ops,median_ms")
    for mode in args.modes:
        block = args.block if args.block is not None else DEFAULT_BLOCK[mode]
        for op_name in args.ops:
            op = op_code[op_name]
            for ops in args.ops_list:
                try:
                    ms = measure(
                        mode,
                        op,
                        ops,
                        args.grid,
                        block,
                        args.reps,
                    )
                    print(f"{mode},{op_name},{ops},{ms:.6f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"{mode},{op_name},{ops},ERROR:{type(exc).__name__}:{exc}", flush=True)


if __name__ == "__main__":
    main()
