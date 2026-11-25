import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
from triton.backends.ascend.compiler import _is_a5_target_arch
from triton.backends.ascend.driver import NPUUtils

_SIMT_SKIP_MSG = ("SIMT libdevice ops are not supported on A3; "
                  "only runs on Ascend 950PR&950DT products; skipping.")


def torch_ldexp_reference(x0, x1):
    assert x0.device.type == "cpu"
    assert x1.device.type == "cpu"
    assert x0.dtype == torch.float32
    assert x1.dtype == torch.int32
    assert x0.shape == x1.shape

    return torch.ldexp(x0, x1)


@triton.jit
def triton_ldexp(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = libdevice.ldexp(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.skipif(not _is_a5_target_arch(NPUUtils().get_arch()), reason=_SIMT_SKIP_MSG)
def test_ldexp():
    shape = (2, 256)
    ncore, xblock, xblock_sub = 2, 1024, 512
    x0 = torch.randn(size=shape, dtype=torch.float32)
    x1 = torch.randint(-126, 128, size=shape, dtype=torch.int32)
    ref = torch_ldexp_reference(x0, x1)
    x0 = x0.npu()
    x1 = x1.npu()

    out = torch.empty_like(x0)
    triton_ldexp[ncore, 1, 1](x0, x1, out, x0.numel(), xblock, xblock_sub, compile_mode='simt_only')

    ref = ref.cpu()
    out = out.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


def test_ldexp_simd():
    shape = (2, 256)
    ncore, xblock, xblock_sub = 2, 1024, 512
    x0 = torch.randn(size=shape, dtype=torch.float32)
    x1 = torch.randint(-126, 128, size=shape, dtype=torch.int32)
    ref = torch_ldexp_reference(x0, x1)
    x0 = x0.npu()
    x1 = x1.npu()

    out = torch.empty_like(x0)
    triton_ldexp[ncore, 1, 1](x0, x1, out, x0.numel(), xblock, xblock_sub, compile_mode='simd')

    ref = ref.cpu()
    out = out.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == "__main__":
    test_ldexp_simd()
    print("test_ldexp_simd passed!")
    if not _is_a5_target_arch(NPUUtils().get_arch()):
        print(_SIMT_SKIP_MSG)
    else:
        test_ldexp()
        print("test_ldexp passed!")
