import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def simd_convert_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    x_ub = tla.alloc_buffer((N, ), tl.bfloat16, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.bfloat16, space="ub")
    tmp_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(x, x_ub)

    with tla.scope.simd():
        # bf16 -> f32 widening; the result is a full 64-lane register.
        bf16_v = tla.simd.vlds(x_ub[0], dist="UNPK_B16")
        f32_v = tla.simd.vextf(bf16_v, tl.float32, part="part_even")

        # Bitwise reinterpretation: the total width is unchanged and the lane count follows the target element width.
        as_i32 = tla.simd.vbitcast(f32_v, "i32")
        tla.simd.vsts(tmp_ub[0], as_i32, dist="NORM_B32")

        # f32 -> bf16 narrowing while keeping the total width unchanged.
        bf16_out = tla.simd.vtruncf(f32_v, tl.bfloat16, part="part_even")
        tla.simd.vsts(o_ub[0], bf16_out, dist="NORM_B16")

    tla.copy(o_ub, o)
