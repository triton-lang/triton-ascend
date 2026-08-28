import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit
def triton_ldexp(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = libdevice.ldexp(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


def test_ldexp():
    if is_compile_on_910_95():
        # TODO: re-enable once bisheng fixes __hmf_ldexpf on Ascend 950 — the
        # intrinsic currently returns wrong values (results overflow to inf).
        pytest.skip("ldexp is currently broken on Ascend 950")
    shape = (2, 256)
    ncore, xblock, xblock_sub = 2, 1024, 512
    x0 = torch.randn(size=shape, dtype=torch.float32).npu()
    x1 = torch.randint(-126, 128, size=shape, dtype=torch.int32, device='npu')

    torch_res = x0 * (2.0**x1.float())
    triton_res = torch.empty_like(x0)
    triton_ldexp[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub)

    torch.testing.assert_close(torch_res, triton_res, rtol=1e-03, atol=1e-03, equal_nan=True)


if __name__ == "__main__":
    test_ldexp()
