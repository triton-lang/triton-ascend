@triton.jit
def fixpipe_kernel(
    A_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    row_matmul = tl.program_id(0)
    offs_i = tl.arange(0, M)[:, None]
    offs_k = tl.arange(0, K)
    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.fixpipe(a_vals, ub, dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL)
