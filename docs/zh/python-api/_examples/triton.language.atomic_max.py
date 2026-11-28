import triton
import triton.language as tl
import torch


@triton.jit
def triton_atomic_max(in_ptr0, out_ptr0, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    out = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.atomic_max(out_ptr0 + (out), tmp0, xmask)


def test_atomic_max():
    dtype = 'int32'
    shape = (3, 1)

    x0 = torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype)).npu()
    out = torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype)).npu()

    ref = torch.maximum(x0, out)

    n_elements = shape[0] * shape[1]
    triton_atomic_max[shape[0], 1, 1](x0, out, n_elements, BLOCK_SIZE=shape[1])
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    test_atomic_max()
    print("test_atomic_max passed!")
