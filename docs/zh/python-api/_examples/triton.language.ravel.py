import torch
import triton
import triton.language as tl


@triton.jit
def flatten_kernel(x_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr):
    # Flatten (M, N) to 1D
    offsets_m = tl.arange(0, M)[:, None]
    offsets_n = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + offsets_m * N + offsets_n)
    x_flat = tl.ravel(x)
    tl.store(output_ptr + tl.arange(0, M * N), x_flat)


def test_ravel():
    M, N = 2, 3
    x = torch.zeros([M, N], dtype=torch.float32).npu()
    out = torch.empty((M * N, ), dtype=torch.float32, device="npu")
    flatten_kernel[(1, )](x, out, M=M, N=N)

    ref = x.reshape(M * N)
    torch.testing.assert_close(out.cpu(), ref.cpu())


if __name__ == "__main__":
    test_ravel()
    print("test_ravel passed!")
