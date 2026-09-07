import os

os.environ.setdefault("TRITON_ENABLE_LIBDEVICE_SIMT", "1")

import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.utils import triton_enable_libdevice_simt

_SIMT_SKIP_MSG = ("SIMT libdevice ops require an Ascend 950 target "
                  "with TRITON_ENABLE_LIBDEVICE_SIMT=1; skipping.")


def torch_norm4d_reference(x0, x1, x2, x3):
    assert x0.device.type == "cpu"
    assert x1.device.type == "cpu"
    assert x2.device.type == "cpu"
    assert x3.device.type == "cpu"
    assert x0.dtype == torch.float32
    assert x1.dtype == torch.float32
    assert x2.dtype == torch.float32
    assert x3.dtype == torch.float32
    assert x0.shape == x1.shape
    assert x1.shape == x2.shape
    assert x2.shape == x3.shape
    return torch.sqrt(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3)


@triton.jit
def triton_kernel(input0, input1, input2, input3, output, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input0 + (x0), mask=mask)
        tmp1 = tl.load(input1 + (x0), mask=mask)
        tmp2 = tl.load(input2 + (x0), mask=mask)
        tmp3 = tl.load(input3 + (x0), mask=mask)
        tmp4 = libdevice.norm4d(tmp0, tmp1, tmp2, tmp3)
        tl.store(output + (x0), tmp4, mask=mask)


@pytest.mark.skipif(not triton_enable_libdevice_simt(), reason=_SIMT_SKIP_MSG)
def test_norm4d():
    x0 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x1 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x2 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x3 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    expected = (torch_norm4d_reference(x0, x1, x2, x3)).npu()
    x0 = x0.npu()
    x1 = x1.npu()
    x2 = x2.npu()
    x3 = x3.npu()
    output = torch.empty(8, dtype=torch.float32, device='npu')
    triton_kernel[(1, )](x0, x1, x2, x3, output, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simt_only')
    output = output.cpu()
    expected = expected.cpu()
    torch.testing.assert_close(output, expected, rtol=1e-03, atol=1e-03, equal_nan=True)


if __name__ == "__main__":
    if not triton_enable_libdevice_simt():
        print(_SIMT_SKIP_MSG)
    else:
        test_norm4d()
