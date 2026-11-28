import triton
import triton.language as tl
import torch


@triton.jit
def atomic_xchg(in_ptr0, out_ptr0, out_ptr1, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    tmp0 = tl.load(in_ptr0 + xindex)
    tmp1 = tl.atomic_xchg(out_ptr0 + xindex, tmp0)
    tl.store(out_ptr1 + xindex, tmp1)


def test_atomic_xchg():
    dtype, shape, ncore = ['int32', (32, 32), 2]

    val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()

    pointer = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()
    pointer_ref = pointer.clone()
    pointer_old = torch.full_like(val, -10).npu()

    n_elements = shape[0] * shape[1]
    atomic_xchg[ncore, 1, 1](val, pointer, pointer_old, BLOCK_SIZE=n_elements // ncore)

    assert torch.equal(pointer, val)
    assert torch.equal(pointer_old, pointer_ref)


if __name__ == "__main__":
    test_atomic_xchg()
    print("test_atomic_xchg passed!")
