import torch
import triton
import triton.language as tl


@triton.jit
def kernel(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    """
    Test tl.make_block_ptr for 3D tensor:
    1. Create 3D block pointer with initial offset (1, 2, 3)
    2. Load data with boundary check and zero padding
    3. Store the loaded block to output tensor
    """
    # Create block pointer starting at offset (1, 2, 3)
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(1, 2, 3),
        block_shape=(2, 2, 2),
        order=(2, 1, 0),
    )
    # Load with boundary check and out-of-bounds zero padding
    data = tl.load(block_ptr_in, boundary_check=(0, 1, 2), padding_option="zero")

    # Create block pointer for output at origin (0, 0, 0)
    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(2, 2, 2),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, data)


def test_make_block_ptr():
    XB, YB, ZB = 4, 4, 4
    x = torch.arange(XB * YB * ZB, dtype=torch.float32).reshape(XB, YB, ZB).npu()
    output = torch.zeros_like(x)

    kernel[(1, )](output, x, XB, YB, ZB)

    ref = torch.zeros((2, 2, 2), dtype=torch.float32).npu()
    ref[:, :, :1] = x[1:3, 2:4, 3:4]
    assert torch.allclose(output[:2, :2, :2].cpu(), ref.cpu())


if __name__ == "__main__":
    test_make_block_ptr()
