@triton.jit
def verify_sub_vec_id_kernel(out_ptr, N: tl.constexpr):
    with al.scope(core_mode="vector"):
        sub_id = al.sub_vec_id()
        offs = sub_id * N + tl.arange(0, N)
        out_ptrs = out_ptr + offs
        tl.store(out_ptrs, sub_id.to(tl.int32))
