import os

os.environ.setdefault("TRITON_ENABLE_LIBDEVICE_SIMT", "1")

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
from triton.backends.ascend.utils import triton_enable_libdevice_simt

_SIMT_SKIP_MSG = ("SIMT libdevice ops are not supported on A3; "
                  "only runs on Ascend 950 with TRITON_ENABLE_LIBDEVICE_SIMT=1; skipping.")


def torch_ldexp_reference(x0, x1):
    assert x0.device.type == "cpu"
    assert x1.device.type == "cpu"
    assert x0.dtype == torch.float32
    assert x1.dtype == torch.int32
    assert x0.shape == x1.shape

    return torch.ldexp(x0, x1)


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


@pytest.mark.skipif(not triton_enable_libdevice_simt(), reason=_SIMT_SKIP_MSG)
def test_ldexp():
    shape = (2, 256)
    ncore, xblock, xblock_sub = 2, 1024, 512
    x0 = torch.randn(size=shape, dtype=torch.float32)
    x1 = torch.randint(-126, 128, size=shape, dtype=torch.int32)
    torch_res = torch_ldexp_reference(x0, x1)
    x0 = x0.npu()
    x1 = x1.npu()

    triton_res = torch.empty_like(x0)
    triton_ldexp[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub, compile_mode='simt_only')

    torch_res = torch_res.cpu()
    triton_res = triton_res.cpu()
    torch.testing.assert_close(torch_res, triton_res, rtol=1e-03, atol=1e-03, equal_nan=True)


if __name__ == "__main__":
    if not triton_enable_libdevice_simt():
        print(_SIMT_SKIP_MSG)
    else:
        test_ldexp()
