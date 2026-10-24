import pytest
import torch

import triton as tr
import triton.language as tl


@tr.jit
def _scaled_load_snapshot(A, B, Snapshot, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    # A nonmonotonic N permutation exercises the indirect-load path.
    if MODE == "plain_direct" or MODE == "alias_store_direct" or MODE == "shared_direct":
        bn = n
    else:
        bn = (n * 17) % N
    bp = B + k[:, None] * N + bn[None, :]
    b = tl.load(bp, volatile=MODE == "volatile")
    a = tl.load(A + m[:, None] * K + k[None, :])
    if MODE == "alias_store" or MODE == "alias_store_direct":
        tl.store(bp, tl.full((K, N), 0, tl.float16))
    if MODE == "shared" or MODE == "shared_direct":
        tl.store(Snapshot + k[:, None] * N + n[None, :], b)
    c = tl.dot_scaled(a, None, "fp16", b, None, "fp16")
    tl.store(C + m[:, None] * N + n[None, :], c)


@pytest.mark.parametrize(
    "mode", ["plain", "alias_store", "shared", "volatile", "plain_direct", "alias_store_direct", "shared_direct"])
def test_dot_scaled_load_snapshot(mode):
    m, n, k = 16, 64, 64
    a = ((torch.arange(m * k).reshape(m, k) % 11 - 5) / 8).half()
    b = ((torch.arange(k * n).reshape(k, n) % 13 - 6) / 4).half()
    permutation = torch.arange(n) if mode.endswith("_direct") else torch.arange(n) * 17 % n
    expected = a.float() @ b[:, permutation].float()
    bn = b.to("npu")
    snapshot = torch.empty((k, n), dtype=torch.float16, device="npu")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_load_snapshot[(1, )](a.to("npu"), bn, snapshot, actual, m, n, k, mode)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
    if mode.startswith("alias_store"):
        torch.testing.assert_close(bn.cpu(), torch.zeros_like(b), atol=0, rtol=0)
    if mode.startswith("shared"):
        torch.testing.assert_close(snapshot.cpu(), b[:, permutation], atol=0, rtol=0)
