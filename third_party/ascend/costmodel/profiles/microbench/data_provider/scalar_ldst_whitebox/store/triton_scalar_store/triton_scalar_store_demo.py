#!/usr/bin/env python3
"""Minimal Triton scalar-store demo for CAModel MTE3-chain analysis.

Runs a single Triton kernel that performs N_ST scalar GM stores per program.
No torch NPU kernel is created: inputs are built on CPU and moved with .to("npu").

Intended to be launched under msopprof simulator, e.g.:
  TRITON_ASCEND_COMPILE_MODE=simd python3 triton_scalar_store_demo.py --n-st 1 --grid 4
"""
import argparse

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver as triton_driver
from triton.runtime.jit import JITFunction


@triton.jit
def _triton_scalar_store_demo(out_ptr, N_ST: tl.constexpr):
    pid = tl.program_id(0)
    v = pid.to(tl.float32) + 1.0
    for i in tl.static_range(N_ST):
        tl.store(out_ptr + pid * N_ST + i, v + i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-st", type=int, default=1)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    try:
        props = triton_driver.active.utils.get_device_properties(torch.npu.current_device())
        vec_cores = int(props.get("num_vectorcore", 0))
    except Exception:
        vec_cores = 0

    original_run = JITFunction.run

    def run_with_hint(self, *f_args, **f_kwargs):
        if vec_cores and "physical_vector_core_count_hint" not in f_kwargs:
            f_kwargs["physical_vector_core_count_hint"] = vec_cores
        return original_run(self, *f_args, **f_kwargs)

    JITFunction.run = run_with_hint

    out = torch.zeros(args.grid * args.n_st, dtype=torch.float32).to("npu")
    opts = {
        "num_warps": 1,
        "compile_mode": "simd",
        "auto_simt_scope_mode": "off",
        "enable_auto_blockify": False,
        "superblock_factor": 1,
        "logical_program_count_hint": args.grid,
    }
    _triton_scalar_store_demo[(args.grid, )](out, N_ST=args.n_st, **opts)
    torch_npu.npu.synchronize()
    print(f"LAUNCHED triton_scalar_store_demo n_st={args.n_st} grid={args.grid}", flush=True)


if __name__ == "__main__":
    main()
