import torch

import triton
import triton.language as tl


@triton.jit
def triton_eq(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, mask=x_index < N)
        tmp1 = tl.load(in_ptr1 + x_index, mask=x_index < N)
        tmp2 = tmp0 == tmp1
        tl.store(out_ptr0 + x_index, tmp2, mask=x_index < N)


def test_eq():
    numel = 128
    block = 128
    sub_block = 32
    torch.manual_seed(0)
    x_cpu = torch.randint(-4, 5, (numel, ), dtype=torch.int32)
    y_cpu = torch.randint(-4, 5, (numel, ), dtype=torch.int32)

    out = torch.empty(numel, dtype=torch.bool, device="npu")
    triton_eq[(1, 1, 1)](x_cpu.npu(), y_cpu.npu(), out, N=numel, XBLOCK=block, XBLOCK_SUB=sub_block, debug=True)

    assert torch.equal(out.cpu(), torch.eq(x_cpu, y_cpu))


if __name__ == "__main__":
    test_eq()
