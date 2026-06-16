@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, dim, g_stride: tl.constexpr, indice_length: tl.constexpr,
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

            for i in range(0, g_block_sub):
                gather_offset = tl.get_element(indices, (i, )) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = tl.insert_slice(tmp_buf, val[None, :], offsets=(i, 0), sizes=(1, other_block), strides=(1, 1))

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf,
                     g_mask[:, None] & other_mask[None, :])
