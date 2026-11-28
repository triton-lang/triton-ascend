import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def simd_const_example(O_ptr, N: tl.constexpr):
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    with tla.scope.simd():
        # vsplat: infer the full-register lane count from the element type.
        zero = tla.simd.vsplat(0.0, tl.float32)

        # vfull: specify the lane count explicitly.
        ones = tla.simd.vfull(1.0, tl.float32, 64)

        # vci: lane i holds base + i; fixed to int32 / 64 lanes.
        index = tla.simd.vci(0)

        # pge: the first n lanes are true.
        prefix = tla.simd.pge(32, 64)

        selected = tla.simd.vsel(prefix, ones, zero)
        tla.simd.vsts(o_ub[0], selected, dist="NORM_B32")

    tla.copy(o_ub, o)
