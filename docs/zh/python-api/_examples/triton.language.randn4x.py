import torch
import triton
import triton.language as tl


@triton.jit
def randn4x_kernel(output1_ptr, output2_ptr, output3_ptr, output4_ptr, seed, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    n1, n2, n3, n4 = tl.randn4x(seed, offsets)
    tl.store(output1_ptr + offsets, n1, mask=offsets < N)
    tl.store(output2_ptr + offsets, n2, mask=offsets < N)
    tl.store(output3_ptr + offsets, n3, mask=offsets < N)
    tl.store(output4_ptr + offsets, n4, mask=offsets < N)


def test_randn4x():
    N = 1024
    BLOCK = 128
    seed = 42
    grid = (triton.cdiv(N, BLOCK), )

    out1 = torch.empty(N, dtype=torch.float32, device="npu")
    out2 = torch.empty(N, dtype=torch.float32, device="npu")
    out3 = torch.empty(N, dtype=torch.float32, device="npu")
    out4 = torch.empty(N, dtype=torch.float32, device="npu")

    randn4x_kernel[grid](out1, out2, out3, out4, seed, N=N, BLOCK=BLOCK)

    # tl.randn4x is a pure function of (seed, offset): re-running with the same
    # seed reproduces exactly the same values.
    ref1 = torch.empty(N, dtype=torch.float32, device="npu")
    ref2 = torch.empty(N, dtype=torch.float32, device="npu")
    ref3 = torch.empty(N, dtype=torch.float32, device="npu")
    ref4 = torch.empty(N, dtype=torch.float32, device="npu")
    randn4x_kernel[grid](ref1, ref2, ref3, ref4, seed, N=N, BLOCK=BLOCK)
    for out, ref in [(out1, ref1), (out2, ref2), (out3, ref3), (out4, ref4)]:
        torch.testing.assert_close(out, ref)

    # all 4 outputs should be normally distributed (mean≈0, std≈1)
    cat = torch.cat([out1.cpu(), out2.cpu(), out3.cpu(), out4.cpu()])
    assert -0.2 < cat.mean() < 0.2, f"mean={cat.mean():.4f} not ~0"
    assert 0.8 < cat.std() < 1.2, f"std={cat.std():.4f} not ~1"


if __name__ == "__main__":
    test_randn4x()
    print("test_randn4x passed!")
