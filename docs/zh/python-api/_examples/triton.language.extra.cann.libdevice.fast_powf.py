import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.compiler import _is_a5_target_arch
from triton.backends.ascend.driver import NPUUtils

_SIMT_SKIP_MSG = ("SIMT libdevice ops require an Ascend 950PR&950DT products target; skipping.")


def torch_fast_powf_reference(x0, x1):
    assert x0.device.type == "cpu"
    assert x1.device.type == "cpu"
    assert x0.dtype == torch.float32
    assert x1.dtype == torch.float32
    assert x0.shape == x1.shape

    return torch.pow(x0, x1)


@triton.jit
def triton_kernel(input0, input1, out, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input0 + (x0), mask=mask)
        tmp1 = tl.load(input1 + (x0), mask=mask)
        tmp2 = libdevice.fast_powf(tmp0, tmp1)
        tl.store(out + (x0), tmp2, mask=mask)


@pytest.mark.skipif(not _is_a5_target_arch(NPUUtils().get_arch()), reason=_SIMT_SKIP_MSG)
def test_fast_powf():
    x0 = (torch.rand((8, )) + 0.1).to(torch.float32)
    x1 = (torch.rand((8, )) + 0.1).to(torch.float32)
    ref = (torch_fast_powf_reference(x0, x1)).npu()
    x0 = x0.npu()
    x1 = x1.npu()
    out = torch.empty(8, dtype=torch.float32, device='npu')
    triton_kernel[(1, )](x0, x1, out, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simt_only')
    out = out.cpu()
    ref = ref.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


def test_fast_powf_simd():
    x0 = (torch.rand((8, )) + 0.1).to(torch.float32)
    x1 = (torch.rand((8, )) + 0.1).to(torch.float32)
    ref = (torch_fast_powf_reference(x0, x1)).npu()
    x0 = x0.npu()
    x1 = x1.npu()
    out = torch.empty(8, dtype=torch.float32, device='npu')
    triton_kernel[(1, )](x0, x1, out, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simd')
    out = out.cpu()
    ref = ref.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == "__main__":
    test_fast_powf_simd()
    print("test_fast_powf_simd passed!")
    if not _is_a5_target_arch(NPUUtils().get_arch()):
        print(_SIMT_SKIP_MSG)
    else:
        test_fast_powf()
        print("test_fast_powf passed!")
