import torch
import triton
import triton.language as tl


@triton.jit
def interleave_2d_kernel(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # (M, N) interleave (M, N) -> (M, 2 * N)
    offsets_m = tl.arange(0, M)[:, None]
    offsets_n = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + offsets_m * N + offsets_n)
    y = tl.load(y_ptr + offsets_m * N + offsets_n)
    z = tl.interleave(x, y)
    offsets_n2 = tl.arange(0, 2 * N)[None, :]
    tl.store(out_ptr + offsets_m * (2 * N) + offsets_n2, z)


def test_interleave():
    M, N = 2, 3
    x = torch.zeros([M, N], dtype=torch.float32).npu()
    y = torch.zeros([M, N], dtype=torch.float32).npu()

    out = torch.empty((M, 2 * N), dtype=torch.float32, device="npu")
    interleave_2d_kernel[(1, )](out, x, y, M=M, N=N)

    assert out.shape == (M, 2 * N), f"Shape mismatch: expected {(M, 2*N)}, but got {out.shape}."
    print("Test passed.")


if __name__ == "__main__":
    test_interleave()
