import torch
import triton
import triton.language as tl


@triton.jit
def permute_3d_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # (M, N, K) -> (K, M, N)
    offsets_m = tl.arange(0, M)[:, None, None]
    offsets_n = tl.arange(0, N)[None, :, None]
    offsets_k = tl.arange(0, K)[None, None, :]
    x = tl.load(x_ptr + offsets_m * N * K + offsets_n * K + offsets_k)
    y = tl.permute(x, [2, 0, 1])
    flat = tl.reshape(y, (M * N * K, ))
    tl.store(out_ptr + tl.arange(0, M * N * K), flat)


def test_permute():
    M, N, K = 2, 3, 4
    x = torch.zeros([M, N, K], dtype=torch.float32).npu()
    out = torch.empty((M * N * K, ), dtype=torch.float32, device="npu")

    permute_3d_kernel[(1, )](out, x, M=M, N=N, K=K)

    assert out.shape == (M * N * K, ), "Shape mismatch"
    print("Test passed.")


if __name__ == "__main__":
    test_permute()
