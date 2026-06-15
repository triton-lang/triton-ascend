@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
                  src_stride0: tl.constexpr, src_stride1: tl.constexpr, idx_dim0: tl.constexpr, idx_dim1: tl.constexpr,
                  idx_stride0: tl.constexpr, idx_stride1: tl.constexpr, out_dim0: tl.constexpr, out_dim1: tl.constexpr,
                  out_stride0: tl.constexpr, out_stride1: tl.constexpr):
    src_offs = (tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1)
    src = tl.load(src_ptr + src_offs)

    idx_offs = (tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1)
    idx = tl.load(idx_ptr + idx_offs)

    out = tl.gather(src, idx, axis)

    out_offs = (tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1)
    tl.store(out_ptr + out_offs, out)
