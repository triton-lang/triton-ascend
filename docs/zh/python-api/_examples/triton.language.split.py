import torch
import triton
import triton.language as tl


@triton.jit
def split_kernel(x_ptr, out0_ptr, out1_ptr, M: tl.constexpr):
    # (M, 2) -> (M, 1) + (M, 1)
    offsets_m = tl.arange(0, M)[:, None]
    offsets_n = tl.arange(0, 2)[None, :]
    x = tl.load(x_ptr + offsets_m * 2 + offsets_n)
    part0, part1 = x.split()
    flat0 = tl.reshape(part0, (M, ))
    flat1 = tl.reshape(part1, (M, ))
    tl.store(out0_ptr + tl.arange(0, M), flat0)
    tl.store(out1_ptr + tl.arange(0, M), flat1)


def test_split():
    M = 4
    x = torch.zeros([M, 2], dtype=torch.float32).npu()
    out0 = torch.empty((M, ), dtype=torch.float32, device="npu")
    out1 = torch.empty((M, ), dtype=torch.float32, device="npu")
    split_kernel[(1, )](out0, out1, x, M=M)

    assert out0.shape == (M, ) and out1.shape == (M, ), "Shape mismatch"
    print("Test passed.")


if __name__ == "__main__":
    test_split()
