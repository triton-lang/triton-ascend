import triton
import triton.language as tl
import torch


@triton.jit
def atomic_add(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    out = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (out), tmp0, xmask)
    tl.store(out_ptr1 + (out), tmp1, xmask)


def test_atomic_add():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval(f'torch.{dtype}')).npu()
    out = torch.full((split_size, shape[1]), 2, dtype=eval(f'torch.{dtype}')).npu()
    out_old = torch.full((split_size, shape[1]), -10, dtype=eval(f'torch.{dtype}')).npu()

    ref = out + ncore * x0_value

    n_elements = shape[0] * shape[1]
    atomic_add[ncore, 1, 1](x0, out, out_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    test_atomic_add()
    print("test_atomic_add passed!")
