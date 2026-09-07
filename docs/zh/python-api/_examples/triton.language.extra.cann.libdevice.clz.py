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


def torch_clz_reference(x0):
    assert x0.device.type == "cpu"
    assert x0.dtype == torch.int32

    vals = x0.view(-1).tolist()
    out = []
    for v in vals:
        u = v & 0xFFFFFFFF
        out.append(32 if u == 0 else 32 - u.bit_length())
    return torch.tensor(out, dtype=torch.int32).view(x0.shape)


@triton.jit
def triton_kernel(input0, output, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input0 + (x0), mask=mask)
        tmp1 = libdevice.clz(tmp0)
        tl.store(output + (x0), tmp1, mask=mask)


@pytest.mark.skipif(not triton_enable_libdevice_simt(), reason=_SIMT_SKIP_MSG)
def test_clz():
    x0 = (torch.randint(1, 16, (8, ))).to(torch.int32)
    expected = (torch_clz_reference(x0)).npu()
    x0 = x0.npu()
    output = torch.empty(8, dtype=torch.int32, device='npu')
    triton_kernel[(1, )](x0, output, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simt_only')
    output = output.cpu()
    expected = expected.cpu()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    if not triton_enable_libdevice_simt():
        print(_SIMT_SKIP_MSG)
    else:
        test_clz()
