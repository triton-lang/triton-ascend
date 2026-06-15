@triton.jit
def parallel_kernel(
    input_ptr,
    output_ptr0,
    output_ptr1,
    n_elements: tl.constexpr,
):
    x = tl.arange(0, n_elements)
    x0 = x // 4
    x1 = x % 4

    a_ptr = input_ptr + x0
    b_ptr = input_ptr + x0
    for i in tl.parallel(0, 3, 1, bind_sub_block=False):
        a_ptr += x0
        b_ptr += x0
    a_ptr += x1
    b_ptr += x1
    val = tl.load(a_ptr + 0)
    tl.store(output_ptr0 + x, val)
    val = tl.load(b_ptr)
    tl.store(output_ptr1 + x, val)
