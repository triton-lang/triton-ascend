import torch
import triton
import triton.language as tl


@triton.jit
def join_2d_kernel(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # (M, N) -> (2, M, N)
    offsets_m = tl.arange(0, M)[:, None]
    offsets_n = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + offsets_m * N + offsets_n)
    y = tl.load(y_ptr + offsets_m * N + offsets_n)
    z = tl.join(x, y)
    flat = tl.reshape(z, (2 * M * N, ))
    tl.store(out_ptr + tl.arange(0, 2 * M * N), flat)


def test_join():
    M, N = 2, 3
    x = torch.zeros([M, N], dtype=torch.float32).npu()
    y = torch.zeros([M, N], dtype=torch.float32).npu()

    out = torch.empty((2 * M * N, ), dtype=torch.float32, device="npu")
    join_2d_kernel[(1, )](out, x, y, M=M, N=N)

    assert out.shape == (2 * M * N, ), f"Shape mismatch: expected {(2*M*N,)} but got {out.shape}."
    print("Test passed.")


if __name__ == "__main__":
    test_join()
