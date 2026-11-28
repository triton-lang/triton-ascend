import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.compiler import _is_a5_target_arch
from triton.backends.ascend.driver import NPUUtils

_SIMT_SKIP_MSG = ("SIMT libdevice ops require an Ascend 950PR&950DT products target; skipping.")


def torch_rnorm4d_reference(x0, x1, x2, x3):
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
    return torch.rsqrt(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3)


@triton.jit
def triton_kernel(input0, input1, input2, input3, out, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
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
        tmp4 = libdevice.rnorm4d(tmp0, tmp1, tmp2, tmp3)
        tl.store(out + (x0), tmp4, mask=mask)


@pytest.mark.skipif(not _is_a5_target_arch(NPUUtils().get_arch()), reason=_SIMT_SKIP_MSG)
def test_rnorm4d():
    x0 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x1 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x2 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x3 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    ref = (torch_rnorm4d_reference(x0, x1, x2, x3)).npu()
    x0 = x0.npu()
    x1 = x1.npu()
    x2 = x2.npu()
    x3 = x3.npu()
    out = torch.empty(8, dtype=torch.float32, device='npu')
    triton_kernel[(1, )](x0, x1, x2, x3, out, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simt_only')
    out = out.cpu()
    ref = ref.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


def test_rnorm4d_simd():
    x0 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x1 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x2 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    x3 = (torch.rand((8, )) * 3.0 - 1.5).to(torch.float32)
    ref = (torch_rnorm4d_reference(x0, x1, x2, x3)).npu()
    x0 = x0.npu()
    x1 = x1.npu()
    x2 = x2.npu()
    x3 = x3.npu()
    out = torch.empty(8, dtype=torch.float32, device='npu')
    triton_kernel[(1, )](x0, x1, x2, x3, out, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simd')
    out = out.cpu()
    ref = ref.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == "__main__":
    test_rnorm4d_simd()
    print("test_rnorm4d_simd passed!")
    if not _is_a5_target_arch(NPUUtils().get_arch()):
        print(_SIMT_SKIP_MSG)
    else:
        test_rnorm4d()
        print("test_rnorm4d passed!")
