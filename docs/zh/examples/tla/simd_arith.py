import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def simd_arith_example(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
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

        # Binary operations require both operands to have exactly the same type.
        add = tla.simd.vadd(av, bv)
        mul = tla.simd.vmul(av, bv)
        fma = tla.simd.vfma(add, av, bv)

        # Unary operations preserve the element type.
        root = tla.simd.vsqrt(tla.simd.vneg(fma))
        exp = tla.simd.vexp(root)

        tla.simd.vsts(o_ub[0], exp, dist="NORM_B32")

    tla.copy(o_ub, o)
