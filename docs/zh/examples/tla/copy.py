import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def copy_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))

    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    # GM -> UB，然后 UB -> GM。
    tla.load_gm_to_ub(x, x_ub)
    tla.store_ub_to_gm(x_ub, o)
