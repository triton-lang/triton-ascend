import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def simd_const_example(O_ptr, N: tl.constexpr):
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    with tla.scope.simd():
        # vsplat：按元素类型推导满寄存器 lane 数。
        zero = tla.simd.vsplat(0.0, tl.float32)

        # vfull：显式指定 lane 数。
        ones = tla.simd.vfull(1.0, tl.float32, 64)

        # vci：lane i 的值为 base + i，固定 int32 / 64 lane。
        index = tla.simd.vci(0)

        # pge：前 n 个 lane 为真。
        prefix = tla.simd.pge(32, 64)

        selected = tla.simd.vsel(prefix, ones, zero)
        tla.simd.vsts(o_ub[0], selected, dist="NORM_B32")

    tla.copy(o_ub, o)
