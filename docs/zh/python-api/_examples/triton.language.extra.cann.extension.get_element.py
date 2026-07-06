import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
import torch


@triton.jit
def get_element_test_kernel(in_ptr, indices_ptr, out_ptr, dim, g_stride: tl.constexpr, indice_length: tl.constexpr,
                            g_block: tl.constexpr, g_block_sub: tl.constexpr, other_block: tl.constexpr):
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            # Manual gather: iterate over each index
            for i in range(0, g_block_sub):
                gather_offset = extension.get_element(indices, (i, )) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = extension.insert_slice(tmp_buf, val[None, :], offsets=(i, 0), sizes=(1, other_block),
                                                 strides=(1, 1))

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf,
                     g_mask[:, None] & other_mask[None, :])


def triton_get_element(x0, dim, indices, num_vec_core=48):
    sz = list(x0.shape)
    sz[dim] = len(indices)
    out = torch.empty(tuple(sz), dtype=x0.dtype).npu()

    g_stride = x0.stride(dim)
    indice_length = indices.numel()
    g_block = (indice_length - 1) // num_vec_core + 1

    # Calculate UB space allocation
    enable_multi_buffer = True
    ub_size = 125 * 1024 // (2 if enable_multi_buffer else 1)
    other_block = g_stride
    g_block_sub = ub_size // (
        # max memory consumption: arith.select + other (mask handling in auto)
        x0.element_size() * g_stride * 3 + indices.element_size())
    if g_block_sub < 1:
        other_block = (ub_size - indices.element_size()) // x0.element_size()
        g_block_sub = 1

    # Select kernel based on implementation
    get_element_test_kernel[num_vec_core, 1,
                            1](x0, indices, out, dim, g_stride=g_stride, indice_length=indice_length, g_block=g_block,
                               g_block_sub=g_block_sub, other_block=other_block, multibuffer=False)

    return out


def test_index_select_manual():
    dtype = 'float32'
    src_shape = (10, 16)
    dim = 0
    indice_shape = (1024, )

    x0 = torch.randn(size=src_shape, dtype=eval('torch.' + dtype)).npu()
    indices = torch.randint(0, src_shape[dim], size=indice_shape, dtype=torch.int32).npu()
    torch_ref = torch.index_select(x0, dim, indices)
    triton_cal = triton_get_element(x0, dim, indices, num_vec_core=40)
    assert torch.equal(torch_ref, triton_cal), \
        f"Manual implementation failed for shape={src_shape}, dim={dim}, indices={indice_shape}"


if __name__ == "__main__":
    test_index_select_manual()
