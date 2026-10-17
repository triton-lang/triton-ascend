import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_merge_kernel(input, out, N: tl.constexpr, descending: tl.constexpr):
    off = tl.arange(0, N)
    input1 = tl.load(input + off)
    # bitonic_merge
    merged = tl.bitonic_merge(input1, dim=0, descending=descending)
    tl.store(out + off, merged)


def test_bitonic_merge():
    N = 8
    input = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.float32, device='npu')
    out = torch.empty_like(input)
    bitonic_merge_kernel[(1, )](input, out, N=N, descending=True)
    expected = torch.sort(input, descending=True)[0]
    assert torch.allclose(out, expected), f"bitonic_merge error: {out} vs {expected}"
    print("PASS: test_bitonic_merge_basic")


if __name__ == "__main__":
    test_bitonic_merge()
