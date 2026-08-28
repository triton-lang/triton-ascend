import triton
import triton.language as tl
import torch


def torch_div(x0, x1):
    res = x0 / x1
    return res


@triton.jit
def triton_div(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 / tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)


def test_div():
    param_list = ['float32', (2, 4096, 8), 2, 32768, 1024]
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.randn(size=shape, dtype=eval('torch.' + dtype)).npu()

    torch_res = torch_div(x0, x1)
    triton_res = torch.empty_like(x0)
    triton_div[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)

    torch.testing.assert_close(torch_res, triton_res, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == '__main__':
    test_div()
