import torch
import triton
import triton.language as tl


@triton.jit
def kernel(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    """
    Test tl.advance for 3D block_ptr:
    1. Create block_ptr with initial offset (3, 1, 2)
    2. Use tl.advance to roll back offset to (0, 0, 0)
    3. Load full tensor and store to output
    """
    # Create 3D block pointer with initial offset (3, 1, 2)
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(3, 1, 2),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    # Roll back offset to origin (0, 0, 0) via tl.advance
    reset_ptr = tl.advance(block_ptr_in, (-3, -1, -2))
    data = tl.load(reset_ptr)

    # Output block pointer
    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, data)


def test_advance():
    XB, YB, ZB = 4, 4, 4
    x = torch.arange(XB * YB * ZB).reshape(XB, YB, ZB).npu()
    output = torch.zeros_like(x)

    # Run Triton kernel
    kernel[(1, )](output, x, XB, YB, ZB)

    assert torch.all(output.cpu() == x.cpu())


if __name__ == "__main__":
    test_advance()
