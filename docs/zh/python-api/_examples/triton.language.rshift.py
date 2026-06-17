import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_elementwise_unary(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    tmp = tl.cast(2, tl.int8)
    ret = x >> tmp
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)


def test_rshift():
    dtype, N = ['int32', 32]
    x0 = torch.randint(low=0, high=256, size=(N, ), dtype=eval(f'torch.{dtype}')).npu()
    ans = x0 >> 2
    out = torch.zeros((N, ), dtype=eval(f'torch.{dtype}')).npu()
    triton_elementwise_unary[1, 1, 1](x0, out, N=N, NUMEL=N)
    torch.testing.assert_close(ans, out)


if __name__ == "main":
    test_rshift()
