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
    out = torch.zeros(BLOCK, dtype=torch.int32).npu()
    arange_kernel[(1, )](out, BLOCK=BLOCK, START=start, END=end)
    ref = torch.arange(start, end, dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), ref.cpu())


if __name__ == "__main__":
    test_arange()
    print("test_arange passed!")
