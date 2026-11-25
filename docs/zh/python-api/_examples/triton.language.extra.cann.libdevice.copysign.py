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
def triton_kernel(input, input2, out, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input + (x0), mask=mask)
        tmp1 = tl.load(input2 + (x0), mask=mask)
        tmp2 = libdevice.copysign(tmp0, tmp1)
        tl.store(out + (x0), tmp2, mask=mask)


@pytest.mark.skipif(not _is_a5_target_arch(NPUUtils().get_arch()), reason=_SIMT_SKIP_MSG)
def test_copysign():
    param_list = [(2, 256, 4), 2, 2048, 1024]
    shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=torch.float32).npu() * 0.5
    x1 = (torch.randn(size=shape, dtype=torch.float32) * 0.5 + 1).npu()

    ref = torch.copysign(x0, x1)
    out = torch.empty_like(x0)
    triton_kernel[ncore, 1, 1](x0, x1, out, x0.numel(), xblock, xblock_sub, compile_mode='simt_only')

    ref = ref.cpu()
    out = out.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


def test_copysign_simd():
    param_list = [(2, 256, 4), 2, 2048, 1024]
    shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=torch.float32).npu() * 0.5
    x1 = (torch.randn(size=shape, dtype=torch.float32) * 0.5 + 1).npu()

    ref = torch.copysign(x0, x1)
    out = torch.empty_like(x0)
    triton_kernel[ncore, 1, 1](x0, x1, out, x0.numel(), xblock, xblock_sub, compile_mode='simd')

    ref = ref.cpu()
    out = out.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == "__main__":
    test_copysign_simd()
    print("test_copysign_simd passed!")
    if not _is_a5_target_arch(NPUUtils().get_arch()):
        print(_SIMT_SKIP_MSG)
    else:
        test_copysign()
        print("test_copysign passed!")
