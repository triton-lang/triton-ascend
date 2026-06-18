import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def inline_asm_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = tl.inline_asm_elementwise(
        asm="ADD.s64 $0, $1, $2",
        constraints="=l,l,l",
        args=[x, y],
        dtype=tl.int64,
        is_pure=True,
        pack=1,
    )
    tl.store(out_ptr + offsets, out, mask=mask)


def test_inline_asm_elementwise():
    N = 128
    BLOCK_SIZE = 128
    x = torch.randint(0, 1000, (N, ), device="npu", dtype=torch.int64)
    y = torch.randint(0, 1000, (N, ), device="npu", dtype=torch.int64)
    out = torch.empty(N, device="npu", dtype=torch.int64)

    grid = (triton.cdiv(N, BLOCK_SIZE), )
    inline_asm_add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    assert_close(out, x + y)


if __name__ == "__main__":
    test_inline_asm_elementwise()
    print("test_inline_asm_elementwise PASSED!")
