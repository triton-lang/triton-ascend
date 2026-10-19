"""FP16 N512 device acceptance; requires matching IR and template bitcode."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def f16_n512_step(S, M, L, P, MN, BS, LN, BM: tl.constexpr,
                  CAUSAL: tl.constexpr = False, LIMITS=None, MASK_REQUIRED=None):
    i = tl.arange(0, BM * 512)
    r = tl.arange(0, BM)
    s = tl.load(S + i).reshape(32, BM // 16, 16, 16)
    m = tl.load(M + r)
    if CAUSAL:
        limits = tl.load(LIMITS + r)
        p, mn, total = al.online_softmax_nz(s, m, causal_limits=limits, mask_required=MASK_REQUIRED)
    else:
        p, mn, total = al.online_softmax_nz(s, m)
    alpha = tl.exp(m.to(tl.float32) - mn.to(tl.float32))
    tl.store(P + i, p.reshape(BM * 512))
    tl.store(MN + r, mn)
    tl.store(BS + r, total)
    tl.store(LN + r, tl.load(L + r) * alpha + total)


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_f16_n512_joint_halves_multiround(rows):
    m = torch.full((rows,), -float("inf"), dtype=torch.float16)
    l = torch.ones(rows)
    for iteration, offset in enumerate([0., 4., -4., 8.]):
        # Half-exact values, different row maxima, alternating dominant halves.
        s = ((torch.arange(rows * 512).reshape(rows, 512) % 31) / 8
             + torch.arange(rows)[:, None] / 16 + offset).half()
        s[:, (iteration % 2) * 256:(iteration % 2 + 1) * 256] -= 8
        nz = s.reshape(rows // 16, 16, 32, 16).permute(2, 0, 1, 3).contiguous().npu()
        p = torch.empty((16, rows // 16, 16, 32), dtype=torch.float8_e4m3fn, device="npu")
        mn = torch.empty_like(m, device="npu")
        total, ln = [torch.empty_like(l, device="npu") for _ in range(2)]
        compiled = f16_n512_step[(1,)](nz, m.npu(), l.npu(), p, mn, total, ln, rows)
        torch.npu.synchronize()
        expected_m = torch.maximum(m, s.amax(1))
        e = torch.exp(s.float() - expected_m.float()[:, None])
        expected_sum = e.sum(1)
        expected_l = l * torch.exp(m.float() - expected_m.float()) + expected_sum
        decoded = p.cpu().float().permute(1, 2, 0, 3).reshape(rows, 512)
        torch.testing.assert_close(decoded, e.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
        torch.testing.assert_close(mn.cpu(), expected_m, rtol=0, atol=0)
        torch.testing.assert_close(total.cpu(), expected_sum, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ln.cpu(), expected_l, rtol=1e-5, atol=1e-5)
        assert "__builtin_online_softmax_nz" in compiled.asm["ttadapter"]
        m, l = mn.cpu(), ln.cpu()


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_f16_n512_causal_joint_halves_multiround(rows):
    # Finite old maxima also exercise fully masked rows: their block sum is
    # zero and their online state is preserved.
    m = torch.zeros(rows, dtype=torch.float16)
    l = torch.ones(rows)
    for iteration, base in enumerate([-32, 240, 496, 1024]):
        limits = torch.arange(rows, dtype=torch.int32) + base
        s = ((torch.arange(rows * 512).reshape(rows, 512) % 31) / 8
             + torch.arange(rows)[:, None] / 16 + iteration).half()
        # Hidden future columns dominate raw scores. A max-only or exp-only
        # mask, or a separate half-softmax, would fail this reference.
        visible = torch.arange(512)[None, :] <= limits[:, None]
        s = torch.where(visible, s, s + 16).half()
        nz = s.reshape(rows // 16, 16, 32, 16).permute(2, 0, 1, 3).contiguous().npu()
        p = torch.empty((16, rows // 16, 16, 32), dtype=torch.float8_e4m3fn, device="npu")
        mn = torch.empty_like(m, device="npu")
        total, ln = [torch.empty_like(l, device="npu") for _ in range(2)]
        compiled = f16_n512_step[(1,)](
            nz, m.npu(), l.npu(), p, mn, total, ln, rows,
            CAUSAL=True, LIMITS=limits.npu())
        torch.npu.synchronize()
        masked = s.masked_fill(~visible, -float("inf"))
        expected_m = torch.maximum(m, masked.amax(1))
        e = torch.exp(masked.float() - expected_m.float()[:, None])
        expected_sum = e.sum(1)
        expected_l = l * torch.exp(m.float() - expected_m.float()) + expected_sum
        decoded = p.cpu().float().permute(1, 2, 0, 3).reshape(rows, 512)
        torch.testing.assert_close(decoded, e.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
        torch.testing.assert_close(mn.cpu(), expected_m, rtol=0, atol=0)
        torch.testing.assert_close(total.cpu(), expected_sum, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ln.cpu(), expected_l, rtol=1e-5, atol=1e-5)
        assert not decoded[~visible].count_nonzero()
        assert "__builtin_online_softmax_nz" in compiled.asm["ttadapter"]
        m, l = mn.cpu(), ln.cpu()


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_f16_n512_causal_history_skip_then_boundary(rows):
    # Alternate fully visible history and masked boundary tiles through a
    # scalar flag; later and final rows must still receive their boundary mask.
    all_visible = torch.full((rows,), 511, dtype=torch.int32)
    nonmonotonic_visible = 511 + ((torch.arange(rows, dtype=torch.int32) * 37 + 17) % 193)
    later_masked = torch.full((rows,), 767, dtype=torch.int32)
    later_masked[1] = 191
    later_masked[rows // 2 + 1] = 319
    last_masked = torch.full((rows,), 1023, dtype=torch.int32)
    last_masked[-1] = 255
    limits_by_round = [all_visible, nonmonotonic_visible, later_masked, last_masked]
    m = torch.zeros(rows, dtype=torch.float16)
    l = torch.ones(rows)
    for limits, offset in zip(limits_by_round, [0., 4., -4., 8.]):
        visible = torch.arange(512)[None, :] <= limits[:, None]
        s = ((torch.arange(rows * 512).reshape(rows, 512) % 31) / 8
             + torch.arange(rows)[:, None] / 16 + offset).half()
        # Make any incorrectly skipped hidden position change both max and P.
        s = torch.where(visible, s, s + 16).half()
        nz = s.reshape(rows // 16, 16, 32, 16).permute(2, 0, 1, 3).contiguous().npu()
        p = torch.empty((16, rows // 16, 16, 32), dtype=torch.float8_e4m3fn, device="npu")
        mn = torch.empty_like(m, device="npu")
        total, ln = [torch.empty_like(l, device="npu") for _ in range(2)]
        f16_n512_step[(1,)](
            nz, m.npu(), l.npu(), p, mn, total, ln, rows,
            CAUSAL=True, LIMITS=limits.npu(), MASK_REQUIRED=bool((limits < 511).any()))
        torch.npu.synchronize()
        masked = s.masked_fill(~visible, -float("inf"))
        expected_m = torch.maximum(m, masked.amax(1))
        e = torch.exp(masked.float() - expected_m.float()[:, None])
        expected_sum = e.sum(1)
        expected_l = l * torch.exp(m.float() - expected_m.float()) + expected_sum
        decoded = p.cpu().float().permute(1, 2, 0, 3).reshape(rows, 512)
        torch.testing.assert_close(decoded, e.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
        torch.testing.assert_close(mn.cpu(), expected_m, rtol=0, atol=0)
        torch.testing.assert_close(total.cpu(), expected_sum, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ln.cpu(), expected_l, rtol=1e-5, atol=1e-5)
        assert not decoded[~visible].count_nonzero()
        m, l = mn.cpu(), ln.cpu()


@pytest.mark.parametrize("value", [-float("inf"), float("inf"), float("nan")])
def test_f16_n512_nonfinite(value):
    rows = 32
    s = torch.full((32, 2, 16, 16), value, dtype=torch.float16, device="npu")
    m = torch.full((rows,), -float("inf"), dtype=torch.float16, device="npu")
    l = torch.ones(rows, device="npu")
    p = torch.empty((16, 2, 16, 32), dtype=torch.float8_e4m3fn, device="npu")
    mn = torch.empty_like(m)
    total, ln = torch.empty_like(l), torch.empty_like(l)
    f16_n512_step[(1,)](s, m, l, p, mn, total, ln, rows)
    torch.npu.synchronize()
    assert torch.isnan(p.cpu().float()).all()  # NO_SAT, not CCE SAT zeroing
    assert torch.isnan(total.cpu()).all()
    assert torch.isnan(ln.cpu()).all()
    torch.testing.assert_close(mn.cpu(), torch.full((rows,), value, dtype=torch.float16),
                               rtol=0, atol=0, equal_nan=True)
