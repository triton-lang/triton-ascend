import triton
import pytest
import torch
import triton.language as tl


@triton.jit
def topk_kernel_2d(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid
    offs_n = tl.arange(0, N)
    offs = offs_m * N + offs_n
    x = tl.load(X + offs)
    z = tl.topk(x, K, dim=0)
    tl.store(Z + offs_m * K + tl.arange(0, K), z)


def test_topk_2d():
    shape = (4, 8)
    k = 4
    x = torch.randint(size=shape, low=0, high=2000, dtype=torch.float32).npu()
    triton_res = torch.empty((shape[0], k), dtype=torch.float32, device='npu')

    torch_res = torch.topk(x, k, dim=-1)[0]
    topk_kernel_2d[(shape[0], )](x, triton_res, shape[0], shape[1], k)
    assert torch.allclose(torch_res, triton_res, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    test_topk_2d()
