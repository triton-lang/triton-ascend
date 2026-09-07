import os

os.environ.setdefault("TRITON_ENABLE_LIBDEVICE_SIMT", "1")

import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.utils import triton_enable_libdevice_simt

_SIMT_SKIP_MSG = ("SIMT libdevice ops are not supported on A3; "
                  "only runs on Ascend 950 with TRITON_ENABLE_LIBDEVICE_SIMT=1; skipping.")


@triton.jit
def triton_kernel(input, input2, output, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input + (x0), mask=mask)
        tmp1 = tl.load(input2 + (x0), mask=mask)
        tmp2 = libdevice.nextafter(tmp0, tmp1)
        tl.store(output + (x0), tmp2, mask=mask)


@pytest.mark.skipif(not triton_enable_libdevice_simt(), reason=_SIMT_SKIP_MSG)
def test_nextafter():
    param_list = [(2, 256, 4), 2, 2048, 1024]
    shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=torch.float32).npu() * 0.5
    x1 = (torch.randn(size=shape, dtype=torch.float32) * 0.5 + 1).npu()

    torch_res = torch.nextafter(x0, x1)
    triton_res = torch.empty_like(x0)
    triton_kernel[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub, compile_mode='simt_only')

    torch_res = torch_res.cpu()
    triton_res = triton_res.cpu()
    torch.testing.assert_close(torch_res, triton_res, rtol=1e-03, atol=1e-03, equal_nan=True)


if __name__ == "__main__":
    if not triton_enable_libdevice_simt():
        print(_SIMT_SKIP_MSG)
    else:
        test_nextafter()
