import triton
import triton.language as tl
import torch


@triton.jit
def atomic_xchg(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_xchg(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp1, xmask)


def test_atomic_xchg():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    block_size = shape[0] * shape[1] // ncore
    split_size = shape[0] // ncore

    val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()

    pointer = torch.randint(low=0, high=10, size=(split_size, shape[1]), dtype=eval(f'torch.{dtype}')).npu()
    pointer_ref = pointer.clone()
    pointer_old = torch.full_like(val, -10).npu()
    pointer_old_ref = pointer_old.clone()

    pointer_ref = val[((ncore - 1) * split_size):(ncore * split_size)].clone()
    pointer_old_ref[0:split_size] = pointer
    pointer_old_ref[split_size:((ncore - 1) * split_size)] = val[0:(ncore - 2) * split_size]

    n_elements = shape[0] * shape[1]
    atomic_xchg[ncore, 1, 1](val, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    assert torch.equal(pointer, pointer_ref)


if __name__ == "__main__":
    test_atomic_xchg()
