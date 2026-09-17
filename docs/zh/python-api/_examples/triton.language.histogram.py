import triton
import triton.language as tl
import torch


@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)


def test_histogram():
    dtype = 'int32'
    M = 2048
    N = 2

    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=(M, ), dtype=eval(f'torch.{dtype}')).npu()
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_kernel[(1, )](x, y_ref, M=M, N=N)
    assert torch.equal(y_cal, y_ref)


if __name__ == "__main__":
    test_histogram()
