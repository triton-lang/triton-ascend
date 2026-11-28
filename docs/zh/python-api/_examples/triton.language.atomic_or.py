import triton
import triton.language as tl
import torch


@triton.jit
def atomic_or(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_or(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)


def test_atomic_or():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    split_size = shape[0] // ncore

    val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()

    out = torch.randint(low=0, high=10, size=(split_size, shape[1]), dtype=eval(f'torch.{dtype}')).npu()
    out_old = torch.full_like(out, -10).npu()
    ref = out.clone()

    for i in range(ncore - 1):
        ref |= val[(i * split_size):((i + 1) * split_size)]

    ref |= val[((ncore - 1) * split_size):(ncore * split_size)]

    n_elements = shape[0] * shape[1]
    atomic_or[ncore, 1, 1](val, out, out_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    test_atomic_or()
    print("test_atomic_or passed!")
