import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def copy_example(X_ptr, O_ptr, N: tl.constexpr):
    x = tla.view(X_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))

    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    # GM -> UB，然后 UB -> GM。
    tla.copy(x, x_ub)
    tla.copy(x_ub, o)
