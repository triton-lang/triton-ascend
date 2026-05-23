#!/usr/bin/env python3
import torch
import triton
import triton.language as tl


@triton.jit
def two_reduce_kernel(a_ptr, b_ptr, out_ptr):
    """Two tl.sum reductions produce scalar offsets used for pointer arithmetic.
    Regression test: the second reduce path must not crash with an
    unrealized_conversion_cast from empty inputs.
    """
    offs = tl.arange(0, 1)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)

    off1 = tl.sum(a - b // 2)
    tmp = tl.load(b_ptr + off1.to(tl.int64))

    off2 = tl.sum(a - b)
    tl.store(out_ptr + off2.to(tl.int64), tmp)


def test_two_reduce_scalar_offsets():
    d = "npu"
    t = torch.int64

    # a[0] = 5, b[0] = 2
    #   r1 = 5 - 2//2 = 4  ->  tmp = b[4] = 40
    #   r2 = 5 - 2   = 3  ->  out[3] = 40
    a = torch.tensor([5], dtype=t, device=d)
    b = torch.tensor([2, 10, 20, 30, 40], dtype=t, device=d)
    out = torch.zeros(10, dtype=t, device=d)

    two_reduce_kernel[(1,)](a, b, out)

    expected = torch.zeros(10, dtype=t, device=d)
    expected[3] = 40

    torch.testing.assert_close(out, expected)
