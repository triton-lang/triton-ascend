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
    flatten_kernel[(1, )](out, x, M=M, N=N)

    assert out.shape == (M * N, ), f"Shape mismatch: expected {(M*N,)} but got {out.shape}."
    print("Test passed.")


if __name__ == "__main__":
    test_ravel()
