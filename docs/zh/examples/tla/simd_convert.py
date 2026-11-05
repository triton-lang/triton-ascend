import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def simd_convert_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.bfloat16, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.bfloat16, space="ub")
    tmp_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    with tla.scope.simd():
        # bf16 -> f32 拓宽，结果为 64 lane 满寄存器。
        bf16_v = tla.simd.vlds(x_ub[0], dist="UNPK_B16")
        f32_v = tla.simd.vextf(bf16_v, tl.float32, part="part_even")

        # 按位重解释：总位宽不变，lane 数按目标元素宽度换算。
        as_i32 = tla.simd.vbitcast(f32_v, "i32")
        tla.simd.vsts(tmp_ub[0], as_i32, dist="NORM_B32")

        # f32 -> bf16 收窄，总位宽保持不变。
        bf16_out = tla.simd.vtruncf(f32_v, tl.bfloat16, part="part_even")
        tla.simd.vsts(o_ub[0], bf16_out, dist="NORM_B16")

    tla.copy(o_ub, o)
