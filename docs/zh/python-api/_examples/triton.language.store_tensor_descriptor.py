import torch
import triton
import triton.language as tl


@triton.jit
def abs_kernel(in_ptr, out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    """
    Load a block of data via a tensor descriptor, compute the element-wise
    absolute value, and store it back through a second descriptor.
    """
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )

    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], tl.abs(value))


def test_store_tensor_descriptor():
    M, N = 64, 128
    M_BLOCK, N_BLOCK = 16, 32
    x = torch.randn(M, N, dtype=torch.float32, device='npu')
    out = torch.empty(M, N, dtype=torch.float32, device='npu')

    grid = (M // M_BLOCK, N // N_BLOCK)
    abs_kernel[grid](x, out, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(out, x.abs())


if __name__ == "__main__":
    test_store_tensor_descriptor()
