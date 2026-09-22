"""
Vector Addition on Specified NPU Device Index

Verifies that a Triton kernel can be launched on tensors created with an
explicit device index (e.g. "npu:2").
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_add_kernel_on_specified_device_index():
    torch.manual_seed(0)

    n_elements = 4096
    BLOCK_SIZE = 1024
    device = "npu:2"

    x = torch.rand(n_elements, device=device)
    y = torch.rand(n_elements, device=device)
    output = torch.empty_like(x)

    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    expected = x + y
    torch.testing.assert_close(output, expected, rtol=1e-04, atol=1e-04)
