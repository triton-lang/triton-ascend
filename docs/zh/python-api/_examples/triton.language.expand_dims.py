import torch
import triton
import triton.language as tl


@triton.jit
def expand_dims_kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # (M, N) -> (M, 1, N)
    offsets_m = tl.arange(0, M)[:, None]
    offsets_n = tl.arange(0, N)[None, :]
    x = tl.load(in_ptr + offsets_m * N + offsets_n)
    y = tl.expand_dims(x, axis=1)
    flat = tl.reshape(y, (M * N, ))
    tl.store(out_ptr + tl.arange(0, M * N), flat)


def test_expand_dims():
    M, N = 2, 3
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32).npu()
    out = torch.empty((M * N, ), dtype=torch.float32, device="npu")
    expand_dims_kernel[(1, )](out, x, M=M, N=N)
    ref = x.cpu().unsqueeze(1).flatten()
    torch.testing.assert_close(out.cpu(), ref)
    print("Test passed.")


if __name__ == "__main__":
    test_expand_dims()
