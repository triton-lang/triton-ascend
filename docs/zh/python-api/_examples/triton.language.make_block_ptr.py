@triton.jit
def kernel(in0_ptr: tl.tensor, out0_ptr: tl.tensor, in0_stride0: int, in0_stride1: int, in0_stride2: int,
           in0_stride_order0: tl.constexpr, in0_stride_order1: tl.constexpr, in0_stride_order2: tl.constexpr,
           out0_stride0: int, out0_stride1: int, out0_stride2: int, out0_stride_order0: tl.constexpr,
           out0_stride_order1: tl.constexpr, out0_stride_order2: tl.constexpr, s0: int, s1: int, s2: int,
           tile_size0: tl.constexpr, tile_size1: tl.constexpr, tile_size2: tl.constexpr):
    tile_id0 = tl.program_id(axis=0)
    tile_id1 = tl.program_id(axis=1)
    tile_id2 = tl.program_id(axis=2)
    offset0 = (tile_id0 * tile_size0).to(tl.int32)
    offset1 = (tile_id1 * tile_size1).to(tl.int32)
    offset2 = (tile_id2 * tile_size2).to(tl.int32)
    in0_bptr = tl.make_block_ptr(in0_ptr, (s0, s1, s2), (in0_stride0, in0_stride1, in0_stride2),
                                 (offset0, offset1, offset2), (tile_size0, tile_size1, tile_size2),
                                 order=(in0_stride_order0, in0_stride_order1, in0_stride_order2))
    in0 = tl.load(in0_bptr,
                  boundary_check=(in0_stride_order0, in0_stride_order1, in0_stride_order2)).to(in0_ptr.type.element_ty)

    out0 = in0

    out0_bptr = tl.make_block_ptr(out0_ptr, (s0, s1, s2), (out0_stride0, out0_stride1, out0_stride2),
                                  (offset0, offset1, offset2), (tile_size0, tile_size1, tile_size2),
                                  order=(out0_stride_order0, out0_stride_order1, out0_stride_order2))
    tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty),
             boundary_check=(out0_stride_order0, out0_stride_order1, out0_stride_order2))
