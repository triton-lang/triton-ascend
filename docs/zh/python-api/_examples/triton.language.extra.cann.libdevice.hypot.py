import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.compiler import _is_a5_target_arch
from triton.backends.ascend.driver import NPUUtils

_SIMT_SKIP_MSG = ("SIMT libdevice ops are not supported on A3; "
                  "only runs on Ascend 950PR&950DT products; skipping.")


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
        tmp2 = libdevice.hypot(tmp0, tmp1)
        tl.store(output + (x0), tmp2, mask=mask)


@pytest.mark.skipif(not _is_a5_target_arch(NPUUtils().get_arch()), reason=_SIMT_SKIP_MSG)
def test_hypot():
    param_list = [(2, 256, 4), 2, 2048, 1024]
    shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=torch.float32).npu() * 0.5
    x1 = (torch.randn(size=shape, dtype=torch.float32) * 0.5 + 1).npu()

    torch_res = torch.hypot(x0, x1)
    triton_res = torch.empty_like(x0)
    triton_kernel[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub, compile_mode='simt_only')

    torch_res = torch_res.cpu()
    triton_res = triton_res.cpu()
    torch.testing.assert_close(torch_res, triton_res, rtol=1e-04, atol=1e-04, equal_nan=True)


def test_hypot_simd():
    param_list = [(2, 256, 4), 2, 2048, 1024]
    shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=torch.float32).npu() * 0.5
    x1 = (torch.randn(size=shape, dtype=torch.float32) * 0.5 + 1).npu()

    torch_res = torch.hypot(x0, x1)
    triton_res = torch.empty_like(x0)
    triton_kernel[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub, compile_mode='simd')

    torch_res = torch_res.cpu()
    triton_res = triton_res.cpu()
    torch.testing.assert_close(torch_res, triton_res, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == "__main__":
    test_hypot_simd()
    if not _is_a5_target_arch(NPUUtils().get_arch()):
        print(_SIMT_SKIP_MSG)
    else:
        test_hypot()
