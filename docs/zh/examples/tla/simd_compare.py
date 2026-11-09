import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def simd_compare_example(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a = tla.view(A_ptr, (N, ), strides=(1, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.load_gm_to_ub(a, a_ub)
    tla.load_gm_to_ub(b, b_ub)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0])
        bv = tla.simd.vlds(b_ub[0])

        # vcmp 返回 int1 谓词向量。
        mask = tla.simd.vcmp(av, bv, "gt")

        # mask 的 lane 数必须与两个值向量一致。
        selected = tla.simd.vsel(mask, av, bv)
        tla.simd.vsts(o_ub[0], selected, dist="NORM_B32")

    tla.store_ub_to_gm(o_ub, o)
