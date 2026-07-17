import triton
import pytest
import torch
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit
def sort_kernel_1d(X, Z, M: tl.constexpr, descending: tl.constexpr):
    off = tl.arange(0, M)
    x = tl.load(X + off)
    x = extension.sort(x, descending=descending, dim=0)
    tl.store(Z + off, x)


@triton.jit
def sort_call_kernel(X, Z, M: tl.constexpr, descending: tl.constexpr):
    sort_kernel_1d(X, Z, M, descending)


def test_sort_call():
    shape = (8, )
    descending = True
    x = torch.randn(size=shape, dtype=torch.float32).npu()
    triton_res = torch.empty_like(x)
    M = x.shape[0]
    sort_call_kernel[(1, )](x, triton_res, M, descending)

    torch_res = torch.sort(x, descending=descending)[0]
    assert (torch_res == triton_res).all(), (torch_res, triton_res)
