import torch
import triton
import triton.language as tl


@triton.jit
def reduce_or_kernel(in_ptr, out_ptr, N: tl.constexpr):
    a = tl.load(in_ptr + tl.arange(0, N))
    b = tl.reduce_or(a, axis=0)
    tl.store(out_ptr, b)


def test_reduce_or():
    N = 16
    x = torch.zeros(N, dtype=torch.int32, device='npu')
    x[3] = 1
    x[10] = 1
    out = torch.zeros(1, dtype=torch.int32, device='npu')
    reduce_or_kernel[(1, )](x, out, N=N)
    expected = (x != 0).any().to(torch.int32).reshape(1)
    assert out.item() == expected.item(), f"reduce_or 错误: {out} vs {expected}"
    print("PASS: test_reduce_or_basic")


if __name__ == "__main__":
    test_reduce_or()
