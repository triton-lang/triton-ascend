import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


# Kernel-definition fragment, not a complete runnable example. Launch it with a
# one-program grid; host launch and output validation are omitted.
@triton.jit
def verify_sub_vec_num_kernel(out_ptr):
    sub_num: tl.constexpr = al.sub_vec_num()
    tl.store(out_ptr, sub_num)
