import torch
import triton
import triton.language as tl


@triton.jit
def kernel(output_ptr, x_ptr, N: tl.constexpr):
    """
    Test tl.load/tl.store with a 1D tensor pointer:
    1. Load data via pointer + offset
    2. Store loaded data to output
    """
    pid = tl.program_id(0)
    offset = pid * N + tl.arange(0, N)[:]
    data = tl.load(x_ptr + offset)
    tl.store(output_ptr + offset, data)


def test_load_store():
    N = 128
    total = 4 * N
    x = torch.arange(total, dtype=torch.float32).npu()
    output = torch.zeros_like(x)

    # Run Triton kernel
    kernel[(4, )](output, x, N)

    assert torch.allclose(output.cpu(), x.cpu())


if __name__ == "__main__":
    test_load_store()
