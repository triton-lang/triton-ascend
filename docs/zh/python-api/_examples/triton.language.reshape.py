import torch
import triton
import triton.language as tl


@triton.jit
def reshape_kernel(x_ptr, out_ptr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, R1: tl.constexpr,
                   R2: tl.constexpr):
    # (D1, D2, D3) -> (R1, R2)
    d1 = tl.arange(0, D1)[:, None, None]
    d2 = tl.arange(0, D2)[None, :, None]
    d3 = tl.arange(0, D3)[None, None, :]
    x = tl.load(x_ptr + d1 * D2 * D3 + d2 * D3 + d3)
    y = tl.reshape(x, [R1, R2])
    flat = tl.reshape(y, (D1 * D2 * D3, ))
    tl.store(out_ptr + tl.arange(0, D1 * D2 * D3), flat)


def test_reshape():
    D1, D2, D3 = 2, 3, 4
    R1, R2 = 6, 4
    x = torch.arange(1, D1 * D2 * D3 + 1, dtype=torch.float32).reshape(D1, D2, D3).npu()
    out = torch.empty((D1 * D2 * D3, ), dtype=torch.float32, device="npu")
    reshape_kernel[(1, )](out, x, D1, D2, D3, R1, R2)

    assert out.shape == (D1 * D2 * D3, ), f"Shape mismatch: expected {(D1*D2*D3,)} but got {out.shape}."
    print("Test passed.")


if __name__ == "__main__":
    test_reshape()
