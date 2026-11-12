"""Device acceptance: original kernels, tail sentinels, padding and fallback."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def policy_copy(x, out, n, B: tl.constexpr, CACHE: tl.constexpr, STORE_CACHE: tl.constexpr, VOLATILE: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    offsets = tl.max_contiguous(offsets, B)
    value = tl.load(x + offsets, offsets < n, other=0, cache_modifier=CACHE, eviction_policy="evict_last",
                    volatile=VOLATILE)
    tl.store(out + offsets, value + 1, offsets < n, cache_modifier=STORE_CACHE)


@pytest.mark.parametrize("factor", [3, 6])
@pytest.mark.parametrize("cache,store_cache", [(".ca", ".wb"), (".cg", ".cs")])
@pytest.mark.parametrize("volatile", [False, True])
def test_policy_and_hint_kernel_applies_with_both_tails(factor, cache, store_cache, volatile):
    n = (factor + 1) * 128 - 3
    x_cpu = torch.arange(n, dtype=torch.float32) / 8
    x = x_cpu.to("npu")
    storage = torch.full((n + 32, ), -123., device="npu")
    out = storage[16:-16]
    kernel = policy_copy[(triton.cdiv(n, 128), )](x, out, n, 128, cache, store_cache, volatile,
                                                  block_batching_factor=factor, block_batching_axis=0)
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
    torch.testing.assert_close(out.cpu(), x_cpu + 1, rtol=0, atol=0)
    assert (storage[:16].cpu() == -123).all() and (storage[-16:].cpu() == -123).all()


@triton.jit
def boundary_copy(x, out, n, B: tl.constexpr, PAD: tl.constexpr, SHIFT: tl.constexpr):
    p = tl.program_id(0)
    block = tl.make_block_ptr(x, shape=(n, ), strides=(1, ), offsets=(p * B - SHIFT, ), block_shape=(B, ), order=(0, ))
    value = tl.load(block, boundary_check=(0, ), padding_option=PAD, cache_modifier=".ca")
    # Observe all padded values as well, rather than masking away padding bugs.
    tl.store(out + p * B + tl.arange(0, B), value, cache_modifier=".wb")


@pytest.mark.parametrize("pad", ["zero", "nan"])
@pytest.mark.parametrize("shift", [0, 3])
def test_boundary_lower_upper_padding_and_batch_tail(pad, shift):
    n, block, factor = 509, 128, 3
    blocks = triton.cdiv(n + shift, block)
    x_cpu = torch.arange(n, dtype=torch.float32)
    x = x_cpu.to("npu")
    storage = torch.full((blocks * block + 32, ), -123., device="npu")
    out = storage[16:-16]
    kernel = boundary_copy[(blocks, )](x, out, n, block, pad, shift, block_batching_factor=factor,
                                       block_batching_axis=0)
    baseline = torch.empty_like(out)
    kernel.baseline[(blocks, 1, 1)](x, baseline, n, block, pad, shift)
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "APPLIED", kernel.last_launch
    index = torch.arange(blocks * block) - shift
    expected = torch.full((blocks * block, ), 0. if pad == "zero" else float("nan"))
    mask = (index >= 0) & (index < n)
    expected[mask] = x_cpu[index[mask]]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(baseline.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert (storage[:16].cpu() == -123).all() and (storage[-16:].cpu() == -123).all()


@triton.jit
def shared_volatile(x, out):
    value = tl.load(x, volatile=True)
    tl.store(out + tl.program_id(0), value)


def test_shared_volatile_reads_select_complete_baseline():
    x = torch.tensor([7.], device="npu")
    out = torch.full((5, ), -123., device="npu")
    kernel = shared_volatile[(5, )](x, out, block_batching_factor=3, block_batching_axis=0)
    torch.npu.synchronize()
    assert kernel.last_launch["runtime_status"] == "BASELINE_SELECTED", kernel.last_launch
    assert kernel.last_launch["reason"] == "OVERLAPPING_VOLATILE_READS"
    torch.testing.assert_close(out.cpu(), torch.full((5, ), 7.), rtol=0, atol=0)
