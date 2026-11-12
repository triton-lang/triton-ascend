import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def simd_reduce_example(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a = tla.view(A_ptr, (N, ), strides=(1, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0])
        bv = tla.simd.vlds(b_ub[0])

        # Horizontal reduction; the result is placed in lane 0.
        total = tla.simd.vreduce_sum(av)

        # Multi-register global maximum; the result is broadcast to all lanes.
        global_max = tla.simd.vreduce_max([av, bv])

        # Broadcast the low half to the whole register.
        splat = tla.simd.vbroadcast(total, low=True)

        # The result can be used directly in vcmp.
        mask = tla.simd.vcmp(splat, global_max, "le")
        result = tla.simd.vsel(mask, splat, global_max)
        tla.simd.vsts(o_ub[0], result, dist="NORM_B32")

    tla.copy(o_ub, o)
