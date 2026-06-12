import triton
import triton.language as tl
import torch


@triton.jit
def atomic_cas(in_ptr0, in_ptr1, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    val = tl.load(in_ptr0 + (x0), xmask)
    cmp = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.atomic_cas(out_ptr0 + (x1), cmp, val)
    tl.store(out_ptr1 + (x1), tmp1, xmask)


def test_atomic_cas():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    block_size = shape[0] * shape[1] // ncore
    split_size = shape[0] // ncore

    import random
    cmp_val = [random.randint(0, 10) for _ in range(ncore)]

    cmp = torch.ones(split_size, shape[1], dtype=eval(f'torch.{dtype}')).to().npu() * cmp_val[0]
    for i in range(1, ncore):
        append = torch.ones(split_size, shape[1], dtype=eval(f'torch.{dtype}')).to().npu() * cmp_val[i]
        cmp = torch.cat([cmp, append], dim=0)

    val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()

    pointer = torch.randint(low=0, high=10, size=(split_size, shape[1]), dtype=eval(f'torch.{dtype}')).npu()
    pointer_old = torch.full_like(pointer, -10).npu()
    pointer_ref = pointer.clone()

    for i in range(ncore):
        val_subview = val[(i * split_size):((i + 1) * split_size)]
        pointer_ref = torch.where(pointer_ref == cmp_val[i], val_subview, pointer_ref)

    n_elements = shape[0] * shape[1]
    atomic_cas[ncore, 1, 1](val, cmp, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    assert torch.equal(pointer, pointer_ref)


if __name__ == "__main__":
    test_atomic_cas()
