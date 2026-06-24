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
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)


def test_atomic_add():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    block_size = shape[0] * shape[1] / ncore
    split_size = shape[0] // ncore
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval(f'torch.{dtype}')).npu()
    x1 = torch.full((split_size, shape[1]), 2, dtype=eval(f'torch.{dtype}')).npu()
    y = torch.full((split_size, shape[1]), -10, dtype=eval(f'torch.{dtype}')).npu()

    y_ref = x1 + 0
    x1_ref = x1 + ncore * x0_value

    n_elements = shape[0] * shape[1]
    atomic_add[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1])
    assert torch.equal(x1, x1_ref)


if __name__ == "__main__":
    test_atomic_add()
