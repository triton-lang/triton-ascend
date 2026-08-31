# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Standalone debug script for the test_nearest_resize failure on A5 (Ascend 950).
# Run on the NPU node:
#   source /usr/local/Ascend/cann/set_env.sh
#   export PATH="/usr/local/bisheng/tools/bishengir/bin:$PATH"
#   python third_party/ascend/unittest/pytest_ut/debug_nearest_resize.py
#
# It reproduces the failing kernel deterministically, dumps the exact mismatch
# pattern, and matches every wrong value against plausible causes:
#   - dst-init value  -> the store never happened (scheduling/mask issue)
#   - neighbour pixel -> the source index was computed wrong (floor/f64 issue)
#   - same position, different channel -> channel address computation bug
#   - no match + non-repeatable -> garbage / uninitialized memory (race)

import collections
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


# ---- V0: the original kernel, copied verbatim from test_resize_performance.py ----
@triton.jit
def nearest_resize_kernel_col_tile(
    img_src_ptr, img_dst_ptr,
    src_rows: tl.constexpr, src_cols: tl.constexpr,
    dst_rows: tl.constexpr, dst_cols: tl.constexpr,
    RR_H: tl.constexpr, RR_W: tl.constexpr,
    stride_in_h: tl.constexpr, stride_in_w: tl.constexpr, stride_in_c: tl.constexpr,
    stride_out_h: tl.constexpr, stride_out_w: tl.constexpr, stride_out_c: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id_c = tl.program_id(0)
    block_id_h = tl.program_id(1)
    block_id_w = tl.program_id(2)

    dest_w_offs = block_id_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dest_offs = (
        block_id_c[None, None] * stride_out_c
        + block_id_h[None, None] * stride_out_h
        + dest_w_offs[None, :] * stride_out_w
    )

    fx = dest_w_offs * RR_W
    sx = tl.floor(fx)

    new_col = block_id_h * RR_H
    src_offsets = (
        block_id_c[None, None] * stride_in_c
        + new_col[None, None].to(tl.int32) * stride_in_h
        + tl.clamp(sx, 0, src_cols - 1)[None, :].to(tl.int32) * stride_in_w)
    src_val = tl.load(img_src_ptr + src_offsets)
    dst_mask = dest_w_offs[None, :] < dst_cols
    tl.store(img_dst_ptr + dest_offs, src_val, mask=dst_mask)


# ---- V1: pure integer index math (no f64, no tl.floor) ----
# Isolates the f64-constexpr * int32 + tl.floor lowering on A5.
@triton.jit
def nearest_resize_kernel_int_idx(
    img_src_ptr, img_dst_ptr,
    src_rows: tl.constexpr, src_cols: tl.constexpr,
    dst_rows: tl.constexpr, dst_cols: tl.constexpr,
    RR_H: tl.constexpr, RR_W: tl.constexpr,
    stride_in_h: tl.constexpr, stride_in_w: tl.constexpr, stride_in_c: tl.constexpr,
    stride_out_h: tl.constexpr, stride_out_w: tl.constexpr, stride_out_c: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id_c = tl.program_id(0)
    block_id_h = tl.program_id(1)
    block_id_w = tl.program_id(2)

    dest_w_offs = block_id_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dest_offs = (
        block_id_c[None, None] * stride_out_c
        + block_id_h[None, None] * stride_out_h
        + dest_w_offs[None, :] * stride_out_w
    )

    sx = dest_w_offs * RR_W  # RR_W passed as int 2: pure int32 math
    sx = tl.minimum(tl.maximum(sx, 0), src_cols - 1)  # int clamp (tl.clamp is float-only)
    new_col = block_id_h * RR_H
    src_offsets = (
        block_id_c[None, None] * stride_in_c
        + new_col[None, None] * stride_in_h
        + sx[None, :] * stride_in_w)
    src_val = tl.load(img_src_ptr + src_offsets)
    dst_mask = dest_w_offs[None, :] < dst_cols
    tl.store(img_dst_ptr + dest_offs, src_val, mask=dst_mask)


# ---- V2: contiguous load (sx = dest_w_offs, stride 1) ----
# Isolates the gather (stride-2 computed-address) load from the rest.
@triton.jit
def nearest_resize_kernel_contig(
    img_src_ptr, img_dst_ptr,
    src_rows: tl.constexpr, src_cols: tl.constexpr,
    dst_rows: tl.constexpr, dst_cols: tl.constexpr,
    RR_H: tl.constexpr, RR_W: tl.constexpr,
    stride_in_h: tl.constexpr, stride_in_w: tl.constexpr, stride_in_c: tl.constexpr,
    stride_out_h: tl.constexpr, stride_out_w: tl.constexpr, stride_out_c: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id_c = tl.program_id(0)
    block_id_h = tl.program_id(1)
    block_id_w = tl.program_id(2)

    dest_w_offs = block_id_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dest_offs = (
        block_id_c[None, None] * stride_out_c
        + block_id_h[None, None] * stride_out_h
        + dest_w_offs[None, :] * stride_out_w
    )

    sx = dest_w_offs  # contiguous within the row, no gather
    sx = tl.minimum(tl.maximum(sx, 0), src_cols - 1)  # int clamp (tl.clamp is float-only)
    new_col = block_id_h * RR_H
    src_offsets = (
        block_id_c[None, None] * stride_in_c
        + new_col[None, None] * stride_in_h
        + sx[None, :] * stride_in_w)
    src_val = tl.load(img_src_ptr + src_offsets)
    dst_mask = dest_w_offs[None, :] < dst_cols
    tl.store(img_dst_ptr + dest_offs, src_val, mask=dst_mask)


# ---- CPU references ----
def nearest_resize_cpu(img_src, img_dst, dst_rows, dst_cols):
    """Same as the test: dst[i, j] = src[2i, 2j]."""
    N, C, src_rows, src_cols = img_src.shape
    RR_H = src_rows / float(dst_rows)
    RR_W = src_cols / float(dst_cols)
    for i in range(dst_rows):
        for j in range(dst_cols):
            fy = i * RR_H
            sy = int(fy)
            fx = j * RR_W
            sx = int(fx)
            img_dst[0, :, i, j] = img_src[0, :, sy, sx]


def nearest_resize_cpu_contig(img_src, img_dst, dst_rows, dst_cols):
    """Reference for V2: dst[i, j] = src[2i, j]."""
    N, C, src_rows, src_cols = img_src.shape
    for i in range(dst_rows):
        for j in range(dst_cols):
            img_dst[0, :, i, j] = img_src[0, :, 2 * i, j]


def run_case(kernel, dtype, ref_fn, seed, name, rr_h=2.0, rr_w=2.0, quiet=False):
    """rr_h/rr_w: 2.0 for the original f64+floor kernel, 2 for the int-index variants."""
    if seed is not None:
        torch.manual_seed(seed)
    n, c, h, w = 1, 4, 64, 64
    dst_rows, dst_cols = h // 2, w // 2

    img_src = torch.randint(0, 255, size=(n, c, h, w), dtype=dtype)
    img_dst_cpu = torch.randint(0, 255, size=(n, c, dst_rows, dst_cols), dtype=dtype)
    ref_fn(img_src, img_dst_cpu, dst_rows, dst_cols)

    img_src = img_src.npu()
    img_dst_npu = torch.randint(0, 255, size=(n, c, dst_rows, dst_cols), dtype=dtype).npu()
    init_dst = img_dst_npu.cpu().clone()  # pre-kernel content, for the skipped-store hypothesis

    stride_in_h, stride_in_w, stride_in_c = h, 1, h * w
    stride_out_h, stride_out_w, stride_out_c = dst_rows, 1, dst_rows * dst_cols
    kernel[(4, 32, 1)](
        img_src, img_dst_npu,
        h, w, dst_rows, dst_cols,
        rr_h, rr_w,
        stride_in_h, stride_in_w, stride_in_c,
        stride_out_h, stride_out_w, stride_out_c,
        32,
    )
    return analyze(name, img_dst_npu.cpu(), img_dst_cpu, img_src.cpu(), init_dst, quiet=quiet)


def analyze(name, actual, expected, src_cpu, init_dst, quiet=False):
    if torch.equal(actual, expected):
        if not quiet:
            print(f"[{name}] PASS", flush=True)
        return True
    diff = actual != expected
    idx = diff.nonzero()
    print(f"[{name}] FAIL: {idx.shape[0]} mismatched elements", flush=True)
    rows = collections.Counter(idx[:, 2].tolist())
    chans = collections.Counter(idx[:, 1].tolist())
    print(f"  mismatch rows (i):    {dict(sorted(rows.items()))}", flush=True)
    print(f"  mismatch channels (c):{dict(sorted(chans.items()))}", flush=True)

    for pos in idx[:5].tolist():
        n_, c_, i_, j_ = pos
        act = actual[n_, c_, i_, j_].item()
        exp = expected[n_, c_, i_, j_].item()
        hypotheses = {"dst-init": init_dst[n_, c_, i_, j_].item()}
        for cc in range(4):  # wrong-channel hypothesis
            hypotheses[f"src[c={cc},2i,2j]"] = src_cpu[0, cc, 2 * i_, 2 * j_].item()
        for di in (-2, -1, 1, 2):  # row off-by-one/two
            si = 2 * i_ + di
            if 0 <= si < 64:
                hypotheses[f"src[2i{di:+d},2j]"] = src_cpu[0, c_, si, 2 * j_].item()
        for dj in (-2, -1, 1, 2):  # column off-by-one/two
            sj = 2 * j_ + dj
            if 0 <= sj < 64:
                hypotheses[f"src[2i,2j{dj:+d}]"] = src_cpu[0, c_, 2 * i_, sj].item()
        matches = [k for k, v in hypotheses.items() if v == act]
        print(f"  pos={tuple(pos)} actual={act} expected={exp} -> {matches or 'NO MATCH (garbage)'}", flush=True)
    return False


if __name__ == "__main__":
    # 1) Reproduce the original failure, twice with different seeds (check determinism)
    run_case(nearest_resize_kernel_col_tile, torch.int64, nearest_resize_cpu, seed=0, name="V0-int64-seed0")
    run_case(nearest_resize_kernel_col_tile, torch.int64, nearest_resize_cpu, seed=1, name="V0-int64-seed1")

    # 2) Isolate the int64 data path
    run_case(nearest_resize_kernel_col_tile, torch.int32, nearest_resize_cpu, seed=0, name="V0-int32")
    run_case(nearest_resize_kernel_col_tile, torch.uint8, nearest_resize_cpu, seed=0, name="V0-uint8")

    # 3) Isolate the f64-constexpr + tl.floor index computation (rr passed as int: pure int math)
    run_case(nearest_resize_kernel_int_idx, torch.int64, nearest_resize_cpu, seed=0, name="V1-int-idx-int64", rr_h=2, rr_w=2)
    run_case(nearest_resize_kernel_int_idx, torch.int32, nearest_resize_cpu, seed=0, name="V1-int-idx-int32", rr_h=2, rr_w=2)

    # 4) Isolate the gather (stride-2) load (rr passed as int)
    run_case(nearest_resize_kernel_contig, torch.int64, nearest_resize_cpu_contig, seed=0, name="V2-contig-int64", rr_h=2, rr_w=2)

    # 5) Hunt the data-dependent failure: fixed-seed sweep + unseeded sweep (CI runs unseeded)
    fails = []
    for s in range(20):
        if not run_case(nearest_resize_kernel_col_tile, torch.int64, nearest_resize_cpu, seed=s,
                        name=f"V0-sweep-seed{s}", quiet=True):
            fails.append(("seed", s))
    for i in range(10):
        if not run_case(nearest_resize_kernel_col_tile, torch.int64, nearest_resize_cpu, seed=None,
                        name=f"V0-sweep-unseeded-{i}", quiet=True):
            fails.append(("unseeded", i))
    print(f"seed sweep: {len(fails)}/30 failed -> {fails}", flush=True)

    # 6) Concurrency stress: several processes hammering the same kernel on one NPU
    if "--stress" in sys.argv:
        import multiprocessing as mp

        n_procs, rounds = 8, 20

        def stress_worker(proc_id, rounds, q):
            fails = 0
            for r in range(rounds):
                if not run_case(nearest_resize_kernel_col_tile, torch.int64, nearest_resize_cpu,
                                seed=1000 * proc_id + r, name=f"stress-p{proc_id}-r{r}", quiet=True):
                    fails += 1
            q.put((proc_id, fails))

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=stress_worker, args=(i, rounds, q)) for i in range(n_procs)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        total = 0
        while not q.empty():
            pid, f = q.get()
            total += f
            print(f"proc {pid}: {f} failures", flush=True)
        print(f"stress total: {total} failures / {n_procs * rounds} runs", flush=True)
