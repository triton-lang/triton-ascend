import triton
import triton.language as tl
import torch
import torch_npu


@triton.jit
def swizzle2d_kernel(out_ptr, size_i: tl.constexpr, size_j: tl.constexpr, size_g: tl.constexpr):
    i = tl.arange(0, size_i)[:, None]
    j = tl.arange(0, size_j)[None, :]
    # Row-major linear index of each source position
    value = i * size_j + j
    # Swizzle the source position to its destination position
    new_i, new_j = tl.swizzle2d(i, j, size_i=size_i, size_j=size_j, size_g=size_g)
    tl.store(out_ptr + new_i * size_j + new_j, value)


def test_swizzle2d():
    size_i, size_j, size_g = 4, 4, 2
    out = torch.zeros((size_i, size_j), dtype=torch.int32, device="npu")
    swizzle2d_kernel[(1, )](out, size_i, size_j, size_g)

    # Documented transform for size_i = size_j = 4 and size_g = 2
    ref = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]], dtype=torch.int32, device="npu")
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    test_swizzle2d()
    print("test_swizzle2d passed!")
