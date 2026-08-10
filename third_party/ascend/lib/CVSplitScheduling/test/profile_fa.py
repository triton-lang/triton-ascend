"""Correctness and hardware-profiling driver for CVSplitScheduling FA.

This imports the exact kernel used by test_fa_accuracy.py.  Only compiler
switches differ between baseline, DCVP, and CVSplit, so the benchmark cannot
silently compare different Python kernels.
"""

import argparse
import math
import os

os.environ.setdefault("TRITON_ASCEND_SOC_VERSION", "Ascend950PR_9589")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_npu
import triton

import test_fa_accuracy as fa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("baseline", "dcvp", "cvsplit", "auto"),
        required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-m", type=int, required=True)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--core-num", type=int, default=28)
    parser.add_argument("--unroll-factor", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--active-blocks", type=int, default=0,
        help="zero launches the full grid; positive values isolate that many tiles")
    parser.add_argument("--skip-accuracy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_accuracy and (args.batch_size != 1 or args.num_heads != 1):
        raise ValueError("accuracy mode requires batch-size=num-heads=1")

    torch.manual_seed(42)
    shape = (
        args.batch_size, args.num_heads, args.sequence_length, args.head_dim)
    q_cpu = torch.empty(shape, dtype=torch.float16).normal_(mean=0.0, std=0.5)
    k_cpu = torch.empty(shape, dtype=torch.float16).normal_(mean=0.0, std=0.5)
    v_cpu = torch.empty(shape, dtype=torch.float16).normal_(mean=0.0, std=0.5)
    q, k, v = q_cpu.npu(), k_cpu.npu(), v_cpu.npu()
    out = torch.empty_like(q)
    m = torch.empty(
        (args.batch_size, args.num_heads, args.sequence_length),
        device="npu", dtype=torch.float32)
    scale = 1.0 / math.sqrt(args.head_dim)
    num_blocks_m = triton.cdiv(args.sequence_length, args.block_m)
    all_blocks = num_blocks_m * args.batch_size * args.num_heads
    num_blocks = min(all_blocks, args.active_blocks) if args.active_blocks else all_blocks
    compiler_options = fa.get_compile_options(args.variant, args.unroll_factor)

    def launch():
        fa._attn_fwd[(args.core_num,)](
            q, k, v, scale, m, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            args.batch_size, args.num_heads,
            N_CTX=args.sequence_length, HEAD_DIM=args.head_dim,
            BLOCK_M=args.block_m, BLOCK_N=args.block_n, STAGE=0,
            NUM_BLOCKS_M=num_blocks_m, NUM_BLOCKS=num_blocks,
            AICORE_NUM=args.core_num, **compiler_options)
        torch.npu.synchronize()

    for _ in range(args.warmup):
        launch()
    for _ in range(args.iterations):
        launch()

    if args.skip_accuracy:
        print(
            f"PASS-PERF variant={args.variant} BM={args.block_m} "
            f"blocks={num_blocks} measured_launches={args.iterations}")
        return

    ref_rows = (
        args.sequence_length if not args.active_blocks
        else args.active_blocks * args.block_m)
    reference = fa.reference_attention(q_cpu[:, :, :ref_rows], k_cpu, v_cpu, scale)
    actual = out[:, :, :ref_rows].cpu()
    max_abs = (actual.float() - reference.float()).abs().max().item()
    if not torch.allclose(actual.float(), reference.float(), atol=0.05, rtol=0.05):
        raise RuntimeError(f"accuracy failure: max_abs={max_abs}")
    print(
        f"PASS variant={args.variant} BM={args.block_m} blocks={num_blocks} "
        f"max_abs={max_abs:.8f}")


if __name__ == "__main__":
    main()
