import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(in_ptr, out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    """
    Copy a 2D tensor blockwise using the functional
    tl.load_tensor_descriptor / tl.store_tensor_descriptor interface.
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

    block = tl.load_tensor_descriptor(in_desc, [moffset, noffset])
    tl.store_tensor_descriptor(out_desc, [moffset, noffset], block)


def test_load_tensor_descriptor():
    M, N = 64, 128
    M_BLOCK, N_BLOCK = 16, 32
    inp = torch.arange(M * N, dtype=torch.float32, device='npu').reshape(M, N)
    out = torch.empty(M, N, dtype=torch.float32, device='npu')

    grid = (M // M_BLOCK, N // N_BLOCK)
    copy_kernel[grid](inp, out, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(out, inp)


if __name__ == "__main__":
    test_load_tensor_descriptor()
