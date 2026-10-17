import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def condition_kernel(x_ptr, out_ptr, N: tl.constexpr):
    acc = 0.0
    i = 0
    # Annotate the while-loop condition with tl.condition so that the
    # compiler keeps loop-invariant code inside the loop (disable_licm).
    while tl.condition(i < N, disable_licm=True):
        acc += tl.load(x_ptr + i)
        i += 1
    tl.store(out_ptr, acc)


def test_condition():
    N = 128
    x = torch.randn(N, device="npu", dtype=torch.float32)
    out = torch.empty(1, device="npu", dtype=torch.float32)

    condition_kernel[(1, )](x, out, N=N)
    torch.npu.synchronize()

    assert_close(out, x.sum().reshape(1), rtol=1e-3, atol=1e-3)
    print("test_condition PASSED!")


if __name__ == "__main__":
    test_condition()
