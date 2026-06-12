import triton
import triton.language as tl
import torch


@triton.jit
def atomic_xor(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_xor(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)


def test_atomic_xor():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    split_size = shape[0] // ncore
    val_value = 3
    val = torch.full(shape, val_value, dtype=eval(f'torch.{dtype}')).npu()
    pointer_value = 5
    pointer = torch.full((split_size, shape[1]), pointer_value, dtype=eval(f'torch.{dtype}')).npu()
    pointer_old = torch.full_like(pointer, -10)
    pointer_result = pointer_value
    for _ in range(ncore):
        pointer_result ^= val_value

    pointer_ref = torch.full_like(pointer, pointer_result)

    n_elements = shape[0] * shape[1]
    atomic_xor[ncore, 1, 1](val, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    assert torch.equal(pointer, pointer_ref)


if __name__ == "__main__":
    test_atomic_xor()
