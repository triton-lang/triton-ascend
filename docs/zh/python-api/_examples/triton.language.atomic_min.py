import triton
import triton.language as tl
import torch


@triton.jit
def triton_atomic_min(in_ptr0, out_ptr0, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_min(out_ptr0 + (x1), tmp0, xmask)


def test_atomic_min():
    dtype = 'int32'
    shape = (3, 1)

    x0 = torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype)).npu()

    x1_ref = torch.minimum(x0, x1)

    n_elements = shape[0] * shape[1]
    triton_atomic_min[shape[0], 1, 1](x0, x1, n_elements, BLOCK_SIZE=shape[1])
    assert torch.equal(x1, x1_ref)


if __name__ == "__main__":
    test_atomic_min()
