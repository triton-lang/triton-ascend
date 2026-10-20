import triton
import triton.language as tl
import torch


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


def test_jit():
    N = 128
    x = torch.randn(N, dtype=torch.float32, device='npu')
    y = torch.randn(N, dtype=torch.float32, device='npu')
    out = torch.empty(N, dtype=torch.float32, device='npu')
    add_kernel[(1, )](x, y, out, N=N)
    torch.testing.assert_close(out.cpu(), (x + y).cpu())


if __name__ == "__main__":
    test_jit()
