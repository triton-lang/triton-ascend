import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.compiler import _is_a5_target_arch
from triton.backends.ascend.driver import NPUUtils

_SIMT_SKIP_MSG = ("SIMT libdevice ops require an Ascend 950PR&950DT products target; skipping.")


def torch_int_as_float_reference(x0):
    assert x0.device.type == "cpu"
    assert x0.dtype == torch.int32

    return x0.view(torch.float32)


@triton.jit
def triton_kernel(input0, output, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input0 + (x0), mask=mask)
        tmp1 = libdevice.int_as_float(tmp0)
        tl.store(output + (x0), tmp1, mask=mask)


@pytest.mark.skipif(not _is_a5_target_arch(NPUUtils().get_arch()), reason=_SIMT_SKIP_MSG)
def test_int_as_float():
    x0 = (torch.randint(1, 16, (8, ))).to(torch.int32)
    expected = (torch_int_as_float_reference(x0)).npu()
    x0 = x0.npu()
    output = torch.empty(8, dtype=torch.float32, device='npu')
    triton_kernel[(1, )](x0, output, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simt_only')
    output = output.cpu()
    expected = expected.cpu()
    torch.testing.assert_close(output, expected, rtol=1e-04, atol=1e-04, equal_nan=True)
    assert torch.equal(output.cpu().view(torch.int32), expected.cpu().view(torch.int32))


def test_int_as_float_simd():
    x0 = (torch.randint(1, 16, (8, ))).to(torch.int32)
    expected = (torch_int_as_float_reference(x0)).npu()
    x0 = x0.npu()
    output = torch.empty(8, dtype=torch.float32, device='npu')
    triton_kernel[(1, )](x0, output, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simd')
    output = output.cpu()
    expected = expected.cpu()
    torch.testing.assert_close(output, expected, rtol=1e-04, atol=1e-04, equal_nan=True)
    assert torch.equal(output.cpu().view(torch.int32), expected.cpu().view(torch.int32))


if __name__ == "__main__":
    test_int_as_float_simd()
    if not _is_a5_target_arch(NPUUtils().get_arch()):
        print(_SIMT_SKIP_MSG)
    else:
        test_int_as_float()
