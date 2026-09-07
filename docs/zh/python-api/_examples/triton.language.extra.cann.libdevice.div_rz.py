import os

os.environ.setdefault("TRITON_ENABLE_LIBDEVICE_SIMT", "1")

import pytest
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
        tmp2 = libdevice.div_rz(tmp0, tmp1)
        tl.store(output + (x0), tmp2, mask=mask)


@pytest.mark.skipif(not triton_enable_libdevice_simt(), reason=_SIMT_SKIP_MSG)
def test_div_rz():
    x0 = torch.tensor([10.0, -10.0, 7.5, -7.5], dtype=torch.float32, device='npu')
    x1 = torch.tensor([3.0, 3.0, 2.0, 2.0], dtype=torch.float32, device='npu')
    n = x0.numel()
    out = torch.empty(n, dtype=torch.float32, device='npu')

    triton_kernel[(1, )](x0, x1, out, n, XBLOCK=n, XBLOCK_SUB=n, compile_mode='simt_only')

    expected = x0 / x1
    out = out.cpu()
    expected = expected.cpu()
    torch.testing.assert_close(out, expected, rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    if not triton_enable_libdevice_simt():
        print(_SIMT_SKIP_MSG)
    else:
        test_div_rz()
