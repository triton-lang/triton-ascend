import os

os.environ.setdefault("TRITON_ENABLE_LIBDEVICE_SIMT", "1")

import pytest
import numpy as np
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.utils import triton_enable_libdevice_simt

_SIMT_SKIP_MSG = ("SIMT libdevice ops require an Ascend 950 target "
                  "with TRITON_ENABLE_LIBDEVICE_SIMT=1; skipping.")


def torch_yn_reference(x0, x1):
    from scipy import special

    assert x0.device.type == "cpu"
    assert x1.device.type == "cpu"
    assert x0.dtype == torch.int32
    assert x1.dtype == torch.float32
    assert x0.shape == x1.shape

    n_np = x0.numpy()
    x_np = x1.numpy()

    # SciPy implements the mathematical yn, but libdevice returns NaN for
    # n < 0; substitute 0 and mask to NaN afterwards.
    n_for_scipy = np.where(n_np < 0, 0, n_np)

    with np.errstate(all="ignore"):
        y_np = special.yn(n_for_scipy, x_np)

    y_np = np.asarray(y_np)

    mask_n_neg = n_np < 0
    mask_x_nan = np.isnan(x_np)
    mask_x_neg = x_np < 0.0
    mask_x_zero = x_np == 0.0
    mask_x_posinf = np.isposinf(x_np)

    # n >= 0, 0 < x < +inf: yn(n, x)
    y_np = np.where(mask_x_zero & ~mask_n_neg, -np.inf, y_np)
    y_np = np.where(mask_x_posinf & ~mask_n_neg, 0.0, y_np)

    mask_nan = mask_n_neg | mask_x_neg | mask_x_nan
    y_np = np.where(mask_nan, np.nan, y_np)

    return torch.as_tensor(y_np, dtype=torch.float32)


@triton.jit
def triton_kernel(input0, input1, output, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input0 + (x0), mask=mask)
        tmp1 = tl.load(input1 + (x0), mask=mask)
        tmp2 = libdevice.yn(tmp0, tmp1)
        tl.store(output + (x0), tmp2, mask=mask)


if __name__ == "__main__":
    if not triton_enable_libdevice_simt():
        print(_SIMT_SKIP_MSG)
    else:
        x0 = (torch.randint(1, 16, (8, ))).to(torch.int32)
        x1 = (torch.rand((8, )) + 0.1).to(torch.float32)
        expected = (torch_yn_reference(x0, x1)).npu()
        x0 = x0.npu()
        x1 = x1.npu()
        output = torch.empty(8, dtype=torch.float32, device='npu')
        triton_kernel[(1, )](x0, x1, output, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simt_only')
        output = output.cpu()
        expected = expected.cpu()
        torch.testing.assert_close(output, expected, rtol=1e-03, atol=1e-03, equal_nan=True)
