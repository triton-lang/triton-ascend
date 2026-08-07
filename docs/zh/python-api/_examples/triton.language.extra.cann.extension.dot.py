import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def dot_fractal_a_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    K1: tl.constexpr = K // 16
    M1: tl.constexpr = M // 16
    a0 = tl.arange(0, K1)[:, None, None, None]
    a1 = tl.arange(0, M1)[None, :, None, None]
    a2 = tl.arange(0, 16)[None, None, :, None]
    a3 = tl.arange(0, 16)[None, None, None, :]
    a = tl.load(a_ptr + (a0 * (M1 * 256) + a1 * 256 + a2 * 16 + a3))
    ob = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]
    b = tl.load(b_ptr + ob)
    d = al.dot(a, b, format_a="fractal", format_b="nd", format_c="nd")
    oo = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(out_ptr + oo, d)


def _to_zN(t: torch.Tensor) -> torch.Tensor:
    M, K = t.shape
    return t.reshape(M // 16, 16, K // 16, 16).permute(2, 0, 1, 3).contiguous()


def test_dot_a_fractal():
    M, K, N = 32, 64, 32
    a_nd = torch.randn(M, K, dtype=torch.float16)
    b_nd = torch.randn(K, N, dtype=torch.float16)

    a_zN = _to_zN(a_nd)
    a_npu = a_zN.npu()
    b_npu = b_nd.npu()
    out_npu = torch.empty(M, N, dtype=torch.float16).npu()

    dot_fractal_a_kernel[(1, )](a_npu, b_npu, out_npu, M, N, K)

    gold = torch.mm(a_nd.to(torch.float32), b_nd.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(out_npu.cpu(), gold, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_dot_a_fractal()
