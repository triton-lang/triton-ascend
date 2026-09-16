import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_binary_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    if OP == 0:
        output = x + y
    elif OP == 1:
        output = x - y
    elif OP == 2:
        output = x * y
    else:
        output = x / y

    tl.store(output_ptr + offsets, output, mask=mask)
def test_elementwise_binary_ops():
    size = 98432
    x = torch.rand(size, device='npu', dtype=torch.float32)
    y = torch.rand(size, device='npu', dtype=torch.float32)
    BLOCK_SIZE = 1024

    ops = [torch.add, torch.sub, torch.mul, torch.div]

    for i, op in enumerate(ops):
        output = torch.empty_like(x)
        elementwise_binary_kernel[(triton.cdiv(size, BLOCK_SIZE),)](
            x, y, output,
            size,
            BLOCK_SIZE=BLOCK_SIZE,
            OP=i,
        )
        expected_output = op(x, y)
        torch.testing.assert_close(output, expected_output, atol=1e-5, rtol=1e-5)
        print(f"Operation {i} passed.")

if __name__ == "__main__":
    test_elementwise_binary_ops()
