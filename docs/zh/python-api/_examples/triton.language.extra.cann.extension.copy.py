@triton.jit
def copy_kernel(
    A_ptr,
    A1_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    offs_a = tl.arange(0, M)[:, None]
    offs_b = tl.arange(0, N)[None, :]
    offs_c = (offs_a) * M + (offs_b)
    a_ptr = A_ptr + offs_c
    a_val = tl.load(a_ptr)
    a1_ptr = A1_ptr + offs_c
    a1_val = tl.load(a1_ptr)
    add = tl.add(a_val, a1_val)
    add_ub = bl.to_buffer(add, al.ascend_address_space.UB)

    A_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)
    al.copy_from_ub_to_l1(add_ub, A_l1)

    A_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.copy(add_ub, A_ub)
