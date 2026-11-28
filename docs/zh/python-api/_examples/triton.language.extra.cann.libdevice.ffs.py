import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch
from triton.backends.ascend.compiler import _is_a5_target_arch
from triton.backends.ascend.driver import NPUUtils

_SIMT_SKIP_MSG = ("SIMT libdevice ops require an Ascend 950PR&950DT products target; skipping.")


def torch_ffs_reference(x0):
    assert x0.device.type == "cpu"
    assert x0.dtype == torch.int32

    x_vals = x0.view(-1).tolist()
    results = []
    for xv in x_vals:
        if xv == 0:
            results.append(0)
        else:
            for i in range(32):
                if (xv >> i) & 1:
                    results.append(i + 1)
                    break
    return torch.tensor(results, dtype=torch.int32).view(x0.shape)


@triton.jit
def triton_kernel(input0, out, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input0 + (x0), mask=mask)
        tmp1 = libdevice.ffs(tmp0)
        tl.store(out + (x0), tmp1, mask=mask)


@pytest.mark.skipif(not _is_a5_target_arch(NPUUtils().get_arch()), reason=_SIMT_SKIP_MSG)
def test_ffs():
    x0 = (torch.randint(1, 16, (8, ))).to(torch.int32)
    ref = (torch_ffs_reference(x0)).npu()
    x0 = x0.npu()
    out = torch.empty(8, dtype=torch.int32, device='npu')
    triton_kernel[(1, )](x0, out, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simt_only')
    out = out.cpu()
    ref = ref.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04)


def test_ffs_simd():
    x0 = (torch.randint(1, 16, (8, ))).to(torch.int32)
    ref = (torch_ffs_reference(x0)).npu()
    x0 = x0.npu()
    out = torch.empty(8, dtype=torch.int32, device='npu')
    triton_kernel[(1, )](x0, out, 8, XBLOCK=8, XBLOCK_SUB=8, compile_mode='simd')
    out = out.cpu()
    ref = ref.cpu()
    torch.testing.assert_close(out, ref, rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    test_ffs_simd()
    print("test_ffs_simd passed!")
    if not _is_a5_target_arch(NPUUtils().get_arch()):
        print(_SIMT_SKIP_MSG)
    else:
        test_ffs()
        print("test_ffs passed!")
