import torch
import triton
import triton.language as tl


@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    """
    Load a block of data from a tensor descriptor and store the element-wise
    absolute value back in place.
    """
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )

    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    # 1.use tl.load_tensor_descriptor/tl.store_tensor_descriptor
    # value = tl.load_tensor_descriptor(desc, [moffset, noffset])
    # tl.store_tensor_descriptor(desc, [moffset, noffset], tl.abs(value))
    # 2.use desc.load/desc.store indirectly (suggested)
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))


def test_tensor_descriptor():
    M, N = 256, 256
    M_BLOCK, N_BLOCK = 32, 32
    x = torch.randn(M, N, dtype=torch.float32).npu()
    x_ref = x.abs()

    grid = (M // M_BLOCK, N // N_BLOCK)
    inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)

    assert torch.allclose(x.cpu(), x_ref.cpu())


if __name__ == "__main__":
    test_tensor_descriptor()
