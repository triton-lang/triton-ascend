import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def range_kernel(x_ptr, out_ptr, N: tl.constexpr):
    acc = 0.0
    # Iterate from 0 to N (exclusive), accumulating each element
    for i in tl.range(0, N):
        acc += tl.load(x_ptr + i)
    tl.store(out_ptr, acc)


def test_range():
    N = 128
    x = torch.randn(N, device="npu", dtype=torch.float32)
    out = torch.empty(1, device="npu", dtype=torch.float32)

    range_kernel[(1, )](x, out, N=N)
    torch.npu.synchronize()

    assert_close(out, x.sum().reshape(1), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_range()
    print("test_range PASSED!")
