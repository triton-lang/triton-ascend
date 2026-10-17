import torch

import triton
import triton.language as tl


@triton.jit
def rand4x_kernel(output1_ptr, output2_ptr, output3_ptr, output4_ptr, seed, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    u1, u2, u3, u4 = tl.rand4x(seed, offsets)
    tl.store(output1_ptr + offsets, u1, mask=offsets < N)
    tl.store(output2_ptr + offsets, u2, mask=offsets < N)
    tl.store(output3_ptr + offsets, u3, mask=offsets < N)
    tl.store(output4_ptr + offsets, u4, mask=offsets < N)


def test_rand4x():
    N = 256
    BLOCK = 64
    seed = 42
    grid = (triton.cdiv(N, BLOCK), )

    out1 = torch.empty(N, dtype=torch.float32, device="npu")
    out2 = torch.empty(N, dtype=torch.float32, device="npu")
    out3 = torch.empty(N, dtype=torch.float32, device="npu")
    out4 = torch.empty(N, dtype=torch.float32, device="npu")

    rand4x_kernel[grid](out1, out2, out3, out4, seed, N=N, BLOCK=BLOCK)

    # all values should be in U(0, 1)
    for name, out in [("u1", out1), ("u2", out2), ("u3", out3), ("u4", out4)]:
        t = out.cpu()
        assert (t >= 0).all() and (t <= 1).all(), f"{name} values out of [0,1] range"


if __name__ == "__main__":
    test_rand4x()
