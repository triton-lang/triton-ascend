import re

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _dot_multidim_print_kernel(X, Y, Z, RANK: tl.constexpr, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
    x = tl.load(X + tl.arange(0, 256 << RANK)).reshape([2] * (RANK - 2) + [32, 32])
    y = tl.load(Y + tl.arange(0, 256 << RANK)).reshape([2] * (RANK - 2) + [32, 32])
    if TRANS_A:
        x = tl.trans(x)
    if TRANS_B:
        y = tl.trans(y)
    z = tl.dot(x, y)
    tl.device_print("x", x)
    tl.device_print("y", y)
    tl.device_print("z", z)
    tl.store(Z + tl.arange(0, 256 << RANK), z.reshape([256 << RANK]))


@pytest.mark.parametrize("rank", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_device_print_dot_multidim(rank, trans_a, trans_b, monkeypatch):
    monkeypatch.setenv("TRITON_DEVICE_PRINT", "1")
    shape = (2, ) * (rank - 2) + (32, 32)
    a = torch.randint(-4, 5, shape, dtype=torch.bfloat16, device="npu")
    b = torch.randint(-4, 5, shape, dtype=torch.bfloat16, device="npu")
    c = torch.empty(shape, dtype=torch.float32, device="npu")
    compiled = _dot_multidim_print_kernel[(1, )](a, b, c, rank, trans_a, trans_b)

    if trans_a:
        a = a.transpose(-1, -2)
    if trans_b:
        b = b.transpose(-1, -2)
    expected = a.float() @ b.float()
    torch.npu.synchronize()
    assert torch.equal(c, expected)

    # Do not rely on the device's bounded print buffer containing every value.
    # All three print calls must survive with a matching supported tensor rank.
    print_types = re.findall(r"call @triton_print\w*\([^\n]*\) : \(tensor<([^>]+)>\) -> \(\)",
                             compiled.asm["ttadapter"])
    batch = f"{2 ** (rank - 2)}x" if rank >= 3 else ""
    assert print_types == [f"{batch}32x32xbf16", f"{batch}32x32xbf16", f"{batch}32x32xf32"]
