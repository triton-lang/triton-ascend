import torch
import triton
import triton.language as tl


@triton.jit
def arange_kernel(output_ptr, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(output_ptr + off, val)


def test_arange():
    start = 0
    end = 128
    BLOCK = end - start
    output = torch.zeros(BLOCK, dtype=torch.int32).npu()
    arange_kernel[(1, )](output, BLOCK=BLOCK, START=start, END=end)
    expected = torch.arange(start, end, dtype=torch.int32)
    assert torch.allclose(output.cpu(), expected.cpu())


if __name__ == "__main__":
    test_arange()
