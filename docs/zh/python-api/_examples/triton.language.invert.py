import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_neg(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1 = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = ~tmp0
        tl.store(out_ptr0 + (x0), tmp1, None)


def test_invert():
    dtype, shape, ncore, xblock, xblock_sub = ['int32', (8, 8), 8, 8, 8]
    x0 = torch.randint(low=0, high=256, size=shape, dtype=eval(f'torch.{dtype}')).npu()
    ans = ~x0
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_neg[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    torch.testing.assert_close(ans, y_cal)


if __name__ == "main":
    test_invert()
