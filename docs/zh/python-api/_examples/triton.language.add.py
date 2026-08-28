import triton
import triton.language as tl
import torch


def torch_add(x0, x1):
    res = x0 + x1
    return res


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y  # equivalent to output = tl.add(x,y)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_add():
    param_list = ['float32', (2, 1024, 4), 2, 4096]
    dtype, shape, ncore, block_size = param_list
    x0 = torch.randn(size=shape, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.randn(size=shape, dtype=eval('torch.' + dtype)).npu()

    torch_res = torch_add(x0, x1)
    triton_res = torch.empty_like(x0)
    add_kernel[ncore, 1, 1](x0, x1, triton_res, x0.numel(), block_size)

    torch.testing.assert_close(torch_res, triton_res, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == '__main__':
    test_add()
