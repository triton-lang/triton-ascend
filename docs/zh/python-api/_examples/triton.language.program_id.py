import triton
import triton.language as tl
import torch


@triton.jit
def kernel(out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * N + tl.arange(0, N)
    tl.store(out_ptr + offsets, pid)


def test_program_id():
    N = 64
    ncore = 4
    out = torch.empty(ncore * N, dtype=torch.int32, device='npu')
    kernel[ncore, 1, 1](out, N=N)
    expected = torch.cat([torch.full((N, ), i, dtype=torch.int32) for i in range(ncore)])
    assert torch.equal(out.cpu(), expected)


if __name__ == "__main__":
    test_program_id()
