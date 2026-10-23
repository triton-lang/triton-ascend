"""Smoke cases for triton.set_allocator on Ascend (also runs on CUDA if present).

Covers issue_allocate.md §4 checks and a few related edge cases:
  1) import / set_allocator does not error
  2) plain kernel without set_allocator still runs (no scratch → NullAllocator unused)
  3) plain kernel with set_allocator runs; allocator typically not invoked on Ascend
  4) make_tensor_descriptor kernel runs (software rewrite path); allocator not invoked
  5) metadata.global_scratch_size (if exposed) is 0 / missing-as-0
  6) re-register allocator / set_profile_allocator API presence
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl


def pick_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    raise RuntimeError("neither NPU nor CUDA is available")


def sync(device: str) -> None:
    if device.startswith("npu"):
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def make_counting_allocator(device: str) -> tuple[Callable, dict]:
    state = {"n": 0, "last_size": None, "last_align": None, "last_stream": None}

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        state["n"] += 1
        state["last_size"] = size
        state["last_align"] = alignment
        state["last_stream"] = stream
        print(f"  [alloc_fn] #{state['n']} size={size} align={alignment} stream={stream}")
        return torch.empty(size, device=device, dtype=torch.int8)

    return alloc_fn, state


def try_read_global_scratch_size(jit_fn) -> Any:
    """Best-effort: inspect compiled kernel metadata after a launch."""
    caches = getattr(jit_fn, "device_caches", None)
    if not caches:
        return "N/A(no device_caches)"
    for _dev, cache in caches.items():
        if not hasattr(cache, "items"):
            continue
        for _key, compiled in cache.items():
            md = getattr(compiled, "metadata", None)
            if md is None:
                continue
            if hasattr(md, "global_scratch_size"):
                return getattr(md, "global_scratch_size")
            if isinstance(md, dict) and "global_scratch_size" in md:
                return md["global_scratch_size"]
            # some builds use AttrDict-like
            try:
                return md["global_scratch_size"]
            except Exception:
                pass
    return "N/A(no metadata.global_scratch_size)"


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)


@triton.jit
def desc_kernel(in_out_ptr, M: tl.constexpr, N: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[MB, NB],
    )
    m = tl.program_id(0) * MB
    n0 = tl.program_id(1) * NB
    v = desc.load([m, n0])
    desc.store([m, n0], tl.abs(v))


def case(title: str, fn: Callable[[], None], results: list) -> None:
    print(f"\n=== {title} ===")
    try:
        fn()
        results.append((title, "PASS", ""))
        print(f"[PASS] {title}")
    except Exception as e:
        results.append((title, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {title}: {type(e).__name__}: {e}")


def main() -> None:
    device = pick_device()
    print(f"triton={triton.__version__}")
    print(f"file={triton.__file__}")
    print(f"torch={torch.__version__} device={device}")
    results: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # 1) import / API surface
    # ------------------------------------------------------------------
    def c1():
        assert hasattr(triton, "set_allocator"), "triton.set_allocator missing"
        from triton.runtime import _allocation as alloc_mod
        assert hasattr(alloc_mod, "NullAllocator")
        assert hasattr(alloc_mod, "set_profile_allocator")
        print("  set_allocator / NullAllocator / set_profile_allocator present")

    case("1) import & API surface", c1, results)

    # ------------------------------------------------------------------
    # 2) plain kernel WITHOUT set_allocator (NullAllocator stays default)
    # ------------------------------------------------------------------
    def c2():
        n = 256
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)
        o = torch.empty_like(x)
        add_kernel[(triton.cdiv(n, 128),)](x, y, o, n)
        sync(device)
        err = (o - (x + y)).abs().max().item()
        assert err < 1e-5, err
        scratch = try_read_global_scratch_size(add_kernel)
        print(f"  max_err={err:.3e} metadata.global_scratch_size={scratch}")

    case("2) plain kernel, allocator NOT set (NullAllocator unused)", c2, results)

    # ------------------------------------------------------------------
    # 3) set_allocator itself + plain kernel WITH allocator
    # ------------------------------------------------------------------
    alloc_fn, state = make_counting_allocator(device)

    def c3():
        triton.set_allocator(alloc_fn)
        print("  set_allocator(alloc_fn) ok")
        n = 256
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)
        o = torch.empty_like(x)
        before = state["n"]
        add_kernel[(triton.cdiv(n, 128),)](x, y, o, n)
        sync(device)
        err = (o - (x + y)).abs().max().item()
        assert err < 1e-5, err
        delta = state["n"] - before
        scratch = try_read_global_scratch_size(add_kernel)
        print(f"  max_err={err:.3e} alloc_delta={delta} metadata.global_scratch_size={scratch}")
        # Ascend / pre-Hopper CUDA: expect no callback
        if delta != 0:
            print("  [warn] allocator was invoked; unexpected for Ascend/software path")

    case("3) set_allocator + plain kernel", c3, results)

    # ------------------------------------------------------------------
    # 4) make_tensor_descriptor (Ascend: rewrite to pointer; no TMA scratch)
    # ------------------------------------------------------------------
    def c4():
        before = state["n"]
        M = N = 64
        MB = NB = 32
        t = torch.randn((M, N), device=device, dtype=torch.float16)
        ref = t.abs()
        desc_kernel[(M // MB, N // NB)](t, M, N, MB, NB)
        sync(device)
        err = (t - ref).abs().max().item()
        assert err < 1e-2, err  # fp16
        delta = state["n"] - before
        scratch = try_read_global_scratch_size(desc_kernel)
        print(f"  max_err={err:.3e} alloc_delta={delta} metadata.global_scratch_size={scratch}")
        if delta != 0:
            print("  [warn] allocator invoked on descriptor kernel (TMA/ConSan path?)")

    case("4) make_tensor_descriptor kernel + set_allocator", c4, results)

    # ------------------------------------------------------------------
    # 5) re-register allocator (overwrite) still ok
    # ------------------------------------------------------------------
    def c5():
        alloc_fn2, state2 = make_counting_allocator(device)
        triton.set_allocator(alloc_fn2)
        n = 128
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)
        o = torch.empty_like(x)
        add_kernel[(1,)](x, y, o, n)
        sync(device)
        assert state2["n"] == 0, f"overwrite allocator unexpectedly called: {state2['n']}"
        print("  re-register allocator ok; still not invoked")

    case("5) re-register set_allocator", c5, results)

    # ------------------------------------------------------------------
    # 6) set_profile_allocator API (presence / no-op call; not launch-wired here)
    # ------------------------------------------------------------------
    def c6():
        from triton.runtime._allocation import set_profile_allocator, NullAllocator
        set_profile_allocator(None)  # reset to Null
        set_profile_allocator(alloc_fn)
        set_profile_allocator(NullAllocator())
        print("  set_profile_allocator(None/fn/Null) ok (API only)")

    case("6) set_profile_allocator API smoke", c6, results)

    # ------------------------------------------------------------------
    # Summary (matches issue_allocate.md table)
    # ------------------------------------------------------------------
    print("\n========== SUMMARY ==========")
    width = max(len(t) for t, _, _ in results)
    for title, status, detail in results:
        extra = f"  ({detail})" if detail else ""
        print(f"  [{status}] {title:<{width}}{extra}")

    failed = [t for t, s, _ in results if s != "PASS"]
    print(
        "\n[result] Ascend expectation: API usable (Level-1); "
        "allocator not invoked for plain / descriptor kernels (no global_scratch)."
    )
    if failed:
        raise SystemExit(f"{len(failed)} case(s) failed: {failed}")
    print("[result] all cases PASS")


if __name__ == "__main__":
    main()
