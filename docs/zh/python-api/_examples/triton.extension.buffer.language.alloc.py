import torch
import torch_npu
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.backends.ascend.utils import is_compile_on_910_95
import pytest


@triton.jit
def add_kernel_func(A_ptr, B_ptr, Out_ptr, L: tl.constexpr = None, M: tl.constexpr = None):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    idx = lblk_idx[:, None] * M + mblk_idx[None, :]

    a_val = tl.load(A_ptr + idx)
    b_val = tl.load(B_ptr + idx)

    A_ub = bl.alloc(tl.float32, [L, M], al.ascend_address_space.UB)
    output = bl.to_tensor(A_ub)
    output = tl.add(a_val, b_val)

    bl.to_buffer(output, bind_buffer=A_ub)
    tl.store(Out_ptr + idx, output)


testlist = [
    # 2D
    (64, 64),
]


@pytest.mark.skipif(not is_compile_on_910_95(), reason="It's only support require Ascend 950 temporarily")
@pytest.mark.parametrize('shape', testlist)
def test_add(shape):

    A = torch.rand(size=shape, dtype=torch.float32).npu()
    B = torch.rand(size=shape, dtype=torch.float32).npu()
    triton_out_ub = torch.zeros(shape, dtype=torch.float32).npu()
    torch_out_ub = A + B

    add_kernel_func[(1, )](A, B, triton_out_ub, *shape)
    torch.testing.assert_close(triton_out_ub, torch_out_ub, atol=1e-5, rtol=1e-5)
