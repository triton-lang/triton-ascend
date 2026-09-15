#!/usr/bin/env python3
"""Triton marginal calibration for transpose vs f32.add.

Method: use the SAME Triton kernel for both `simd` and `simt_only`.
The kernel performs repeated `add` or repeated `trans` on a square tile.
The only thing changed between modes is `compile_mode`.

Run on Ascend server:
    source ~/env_ascend.sh
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=1
    python3 bench_transpose_triton.py --modes simd simt_only --ops add trans
"""
from __future__ import annotations

import argparse
import statistics

import torch
import torch_npu
import triton
import triton.language as tl

DEFAULT_GRID = 56
DEFAULT_BM = 32
DEFAULT_BN = 32


@triton.jit
def _apply_op(x, MODE: tl.constexpr, k):
    if MODE == 0:  # add
        return x + k
    else:  # trans
        return tl.trans(x)


@triton.jit
def tile_kernel(x_ptr, out_ptr, ITERS, MODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    k = 1.0000001
    offs_m = pid * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    x = tl.load(x_ptr + offs_m[:, None] * BN + offs_n[None, :])

    for _ in tl.range(0, ITERS):
        x = _apply_op(x, MODE, k)

    tl.store(out_ptr + offs_m[:, None] * BN + offs_n[None, :], x)


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


def _tensor_for_kernel(grid: int, bm: int, bn: int):
    rows = grid * bm
    x = torch.rand(rows, bn, dtype=torch.float32, device="npu")
    out = torch.zeros(rows, bn, dtype=torch.float32, device="npu")
    return x, out


def measure(mode, op, ops, grid, bm, bn, reps):
    x, out = _tensor_for_kernel(grid, bm, bn)
    opts = _opts(mode, grid)

    def launch():
        tile_kernel[(grid, )](x, out, ITERS=ops, MODE=op, BM=bm, BN=bn, **opts)

    return _median_launch_ms(launch, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--ops", nargs="+", default=["add", "trans"], choices=["add", "trans"])
    ap.add_argument("--ops-list", type=int, nargs="+", default=[0, 32, 64, 128, 256, 512])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    ap.add_argument("--bm", type=int, default=DEFAULT_BM)
    ap.add_argument("--bn", type=int, default=DEFAULT_BN)
    args = ap.parse_args()

    op_code = {"add": 0, "trans": 1}

    print("mode,op,ops,median_ms")
    for mode in args.modes:
        for op_name in args.ops:
            op = op_code[op_name]
            for ops in args.ops_list:
                try:
                    ms = measure(
                        mode,
                        op,
                        ops,
                        args.grid,
                        args.bm,
                        args.bn,
                        args.reps,
                    )
                    print(f"{mode},{op_name},{ops},{ms:.6f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"{mode},{op_name},{ops},ERROR:{type(exc).__name__}:{exc}", flush=True)


if __name__ == "__main__":
    main()
