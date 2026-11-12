import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def simd_compare_example(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
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

        # vcmp returns an int1 predicate vector.
        mask = tla.simd.vcmp(av, bv, "gt")

        # The mask lane count must match both value vectors.
        # tla.where is equivalent to tla.simd.vsel; both arms are tla.vector.
        selected = tla.where(mask, av, bv)
        tla.simd.vsts(o_ub[0], selected, dist="NORM_B32")

    tla.copy(o_ub, o)
