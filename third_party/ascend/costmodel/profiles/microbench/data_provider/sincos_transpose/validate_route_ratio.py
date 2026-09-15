#!/usr/bin/env python3
"""Compare cost-model SIMD/SIMT ratio with measured Triton runtime ratio.

Runs only ONE operator per invocation, so it triggers the cost model once.

Examples:
    python3 validate_route_ratio.py add
    python3 validate_route_ratio.py sin
    python3 validate_route_ratio.py cos
    python3 validate_route_ratio.py trans
"""
import argparse
import json
import os
import pathlib
import statistics

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def unary_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    if OP == 0:
        y = x + 1.0
    elif OP == 1:
        y = tl.sin(x)
    else:
        y = tl.cos(x)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def trans_kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    y = tl.trans(x)
    tl.store(y_ptr + offs_n[:, None] * M + offs_m[None, :], y)


def _opts(mode, programs, report=None):
    if mode == "simd":
        return {
            "num_warps": 1,
            "compile_mode": "simd",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": False,
            "superblock_factor": 1,
            "logical_program_count_hint": programs,
        }
    if mode == "simt_only":
        return {
            "num_warps": 1,
            "compile_mode": "simt_only",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": True,
            "superblock_factor": 1,
            "logical_program_count_hint": programs,
        }
    if mode == "auto":
        if report.exists():
            report.unlink()
        return {
            "num_warps": 1,
            "compile_mode": "simd_simt",
            "auto_simt_scope_mode": "auto",
            "auto_simt_scope_dump": str(report),
            "enable_auto_blockify": True,
            "logical_program_count_hint": programs,
        }
    raise ValueError(mode)


def _median_ms(launch, reps=15):
    for _ in range(5):
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


def _report_path(op: str) -> pathlib.Path:
    env_path = os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP", "").strip()
    if not env_path:
        return pathlib.Path(f"/tmp/{op}_route_ratio.json")
    p = pathlib.Path(env_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix != ".json":
        p.mkdir(parents=True, exist_ok=True)
        p = p / f"{op}_route.json"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["add", "sin", "cos", "trans"], nargs="?", default="sin")
    args = ap.parse_args()

    report = _report_path(args.op)
    if args.op == "trans":
        # Larger tile: on this shape both the measured and predicted
        # SIMD/SIMT ratios are below 1 (SIMD faster than SIMT).
        M = N = 4096
        BM = BN = 64
        grid = (M // BM, N // BN)
        x = torch.rand(M, N, dtype=torch.float32, device="npu")
        y = torch.empty(N, M, dtype=torch.float32, device="npu")
        programs = grid[0] * grid[1]

        def launch(mode):
            trans_kernel[grid](x, y, M=M, N=N, BM=BM, BN=BN, **_opts(mode, programs))
    else:
        N = 4096
        BLOCK = 1024
        grid = (N // BLOCK, )
        x = torch.rand(N, dtype=torch.float32, device="npu")
        y = torch.empty(N, dtype=torch.float32, device="npu")
        programs = grid[0]
        opcode = {"add": 0, "sin": 1, "cos": 2}[args.op]

        def launch(mode):
            unary_kernel[grid](x, y, N=N, BLOCK=BLOCK, OP=opcode, **_opts(mode, programs))

    simd_ms = _median_ms(lambda: launch("simd"))
    simt_ms = _median_ms(lambda: launch("simt_only"))

    opts = _opts("auto", programs, report)
    if args.op == "trans":
        trans_kernel[grid](x, y, M=M, N=N, BM=BM, BN=BN, **opts)
    else:
        unary_kernel[grid](x, y, N=N, BLOCK=BLOCK, OP=opcode, **opts)
    torch.npu.synchronize()
    if not report.exists():
        raise RuntimeError(f"cost model did not write {report}")

    data = json.loads(report.read_text().strip().splitlines()[-1])
    routes = data["stage_model"]["routes"]
    simd_score = routes["all_simd"]["total_system_cycles"]
    simt_score = routes["all_simt_only"]["total_system_cycles"]
    print("op:", args.op)
    print("simd_ms:", f"{simd_ms:.6f}")
    print("simt_ms:", f"{simt_ms:.6f}")
    print("measured_simd_over_simt:", f"{simd_ms / simt_ms:.4f}")
    print("all_simd_score:", simd_score)
    print("all_simt_score:", simt_score)
    print("predicted_simd_over_simt:", f"{simd_score / simt_score:.4f}")
    print("effective:", data.get("effective_decision_kind"))
    print("unmodeled:", data.get("unmodeled_cost_terms"))
    print("report:", report)


if __name__ == "__main__":
    main()
