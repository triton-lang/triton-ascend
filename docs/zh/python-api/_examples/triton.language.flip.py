import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def flip_kernel(X, Z, M: tl.constexpr, N: tl.constexpr, dim: tl.constexpr):
    offx = tl.arange(0, M) * N
    offy = tl.arange(0, N)
    off2d = offx[:, None] + offy[None, :]
    x = tl.load(X + off2d)
    x = tl.flip(x, dim)
    tl.store(Z + off2d, x)


def test_flip():
    M, N = 8, 64
    dtype_str = 'int32'
    dim = 1
    x = torch.randint(low=0, high=256, size=(M, N), dtype=eval(f'torch.{dtype_str}')).npu()
    y = torch.flip(x, (dim, ))
    z = torch.empty_like(x, device='npu')
    flip_kernel[(1, )](x, z, M, N, dim, num_warps=8)
    assert (y == z).all(), (y, z)


if __name__ == "main":
    test_flip()
