"""TLA 3-in-1 test: long-vector tail handling with runtime N (f32 and f16).

This is the consolidated tail/multi-chunk/multi-core case.  ``N`` is a runtime
kernel argument, so the predicate cannot be precomputed on the host: the kernel
broadcasts ``N`` into a register, builds the lane index with ``vci`` (f32) or
loads a padded ``i16`` index row (f16) and compares once with a signed
``vcmp <LT>``.  The compare operands are always signed lane indices, never
unsigned payload data.

Per chunk (``CHUNK`` lanes) the kernel appends 13 rows to ``O``:

  0..8   ALL-store arithmetic: vdiv / vmax / vmin / vmaxs / vexp / vdup /
         vdups / vdupv(LOWEST) / vdupv(HIGHEST)
  9..12  the same four shapes stored through an actual masked ``vsts``
         (vmax / vdiv / vdupv LOWEST / vdupv HIGHEST)

The input buffers are legally padded to a whole number of register chunks
(``CAP``) with an 8-lane guard on both sides, so the non-masked ``vlds`` only
ever reads within ``GIN + CAP + GOUT``.  That is why the whole-register load is
legal here: the allocation is padded, not masked.

Device semantics that must be checked on hardware (IR alone is not enough):

* ALL-store rows: every lane in ``[GIN+N, GIN+CAP)`` is bitwise ``+0``;
* masked-store rows: those lanes keep the non-zero sentinel, i.e. the masked
  store never touches them;
* the guard lanes outside ``[GIN, GIN+CAP)`` keep the sentinel in every row.

``N`` is parametrized over below/equal/just-above one chunk and multi-chunk
tails (``TAIL_F32_NS`` / ``TAIL_F16_NS``); each value gets its own row block in
one launch, and the pads cover zero-padding and non-divisible shapes.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SENTINEL = 17.0
GUARD = 8
CORES = 2
ROWS = 13
F32_CHUNK = 64
F32_CAP = 320
F32_NS = (1, 31, 32, 63, 64, 65, 127, 128, 129, 257)
F16_CHUNK = 128
F16_CAP = 512
F16_NS = (1, 63, 127, 128, 129, 255, 256, 257, 385)

F32_SIGNATURE = {"A": "*fp32", "B": "*fp32", "C": "*fp32", "O": "*fp32", "N": "i32", "ROW0": "i32"}
F16_SIGNATURE = {"A": "*fp16", "B": "*fp16", "C": "*fp16", "I": "*i16", "O": "*fp16", "N": "i32", "ROW0": "i32"}

SHAPE = dict(f32_cap=F32_CAP, f32_chunk=F32_CHUNK, f16_cap=F16_CAP, f16_chunk=F16_CHUNK, guard=GUARD, cores=CORES,
             rows=ROWS)
SIGNATURE = F32_SIGNATURE


def constexprs(cap=F32_CAP, chunk=F32_CHUNK):
    """Kernel constexprs for one dtype variant (CAP/CHUNK pick f32 or f16)."""
    return {"CAP": cap, "CHUNK": chunk, "CORES": CORES, "GIN": GUARD, "GOUT": GUARD}


@tla.jit
def simd_tail_f32(A, B, C, O, N, ROW0, CAP: tl.constexpr, CHUNK: tl.constexpr, CORES: tl.constexpr, GIN: tl.constexpr,
                  GOUT: tl.constexpr):
    a_ub = tla.alloc_buffer((CHUNK, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((CHUNK, ), tl.float32, space="ub")
    c_ub = tla.alloc_buffer((CHUNK, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((13, CHUNK), tl.float32, space="ub")
    NBLK: tl.constexpr = (CAP + CHUNK - 1) // CHUNK
    for p in range(tl.program_id(0), NBLK, CORES):
        off = p * CHUNK
        a = tla.view(A, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        b = tla.view(B, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        c = tla.view(C, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        o = tla.view(O, (13, CHUNK), strides=(CHUNK, 1), offsets=(ROW0, GIN + off))
        tla.copy(a, a_ub)
        tla.copy(b, b_ub)
        tla.copy(c, c_ub)
        # Seed the UB rows from GM: ALL rows are fully overwritten, masked rows
        # keep this non-zero sentinel in every lane >= N.
        tla.copy(o, o_ub)
        with tla.scope.simd():
            av = tla.simd.vlds(a_ub[0], dist="NORM")
            bv = tla.simd.vlds(b_ub[0], dist="NORM")
            cv = tla.simd.vlds(c_ub[0], dist="NORM")
            idx = tla.simd.vci(off)
            nv = tla.simd.vdup(N, tl.int32)
            # Signed lane index vs signed runtime N: a single vcmp <LT>.
            tail = tla.simd.vcmp(idx, nv, op=tla.simd.CMP.LT)
            tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[1, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[2, 0], tla.simd.vmin(av, bv, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[3, 0], tla.simd.vmaxs(av, 0.0, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[4, 0], tla.simd.vexp(av, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[5, 0], tla.simd.vdup(1.5, tl.float32, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[6, 0], tla.simd.vdups(1.5, tl.float32, mask=tail), dist="NORM_B32")
            tla.simd.vsts(o_ub[7, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST), dist="NORM_B32")
            tla.simd.vsts(o_ub[8, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST), dist="NORM_B32")
            tla.simd.vsts(o_ub[9, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[10, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B32", mask=tail)
            tla.simd.vsts(o_ub[11, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST), dist="NORM_B32",
                          mask=tail)
            tla.simd.vsts(o_ub[12, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST), dist="NORM_B32",
                          mask=tail)
        tla.copy(o_ub, o)


@tla.jit
def simd_tail_f16(A, B, C, I, O, N, ROW0, CAP: tl.constexpr, CHUNK: tl.constexpr, CORES: tl.constexpr,
                  GIN: tl.constexpr, GOUT: tl.constexpr):
    a_ub = tla.alloc_buffer((CHUNK, ), tl.float16, space="ub")
    b_ub = tla.alloc_buffer((CHUNK, ), tl.float16, space="ub")
    c_ub = tla.alloc_buffer((CHUNK, ), tl.float16, space="ub")
    i_ub = tla.alloc_buffer((CHUNK, ), tl.int16, space="ub")
    o_ub = tla.alloc_buffer((13, CHUNK), tl.float16, space="ub")
    NBLK: tl.constexpr = (CAP + CHUNK - 1) // CHUNK
    for p in range(tl.program_id(0), NBLK, CORES):
        off = p * CHUNK
        a = tla.view(A, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        b = tla.view(B, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        c = tla.view(C, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        i = tla.view(I, (CHUNK, ), strides=(1, ), offsets=(GIN + off, ))
        o = tla.view(O, (13, CHUNK), strides=(CHUNK, 1), offsets=(ROW0, GIN + off))
        tla.copy(a, a_ub)
        tla.copy(b, b_ub)
        tla.copy(c, c_ub)
        tla.copy(i, i_ub)
        tla.copy(o, o_ub)
        with tla.scope.simd():
            av = tla.simd.vlds(a_ub[0], dist="NORM")
            bv = tla.simd.vlds(b_ub[0], dist="NORM")
            cv = tla.simd.vlds(c_ub[0], dist="NORM")
            iv = tla.simd.vlds(i_ub[0], dist="NORM")
            # f16 has 128 lanes: the i16 lane index keeps the compare signed.
            nv = tla.simd.vdup(N, tl.int16)
            tail = tla.simd.vcmp(iv, nv, op=tla.simd.CMP.LT)
            tla.simd.vsts(o_ub[0, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[1, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[2, 0], tla.simd.vmin(av, bv, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[3, 0], tla.simd.vmaxs(av, 0.0, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[4, 0], tla.simd.vexp(av, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[5, 0], tla.simd.vdup(1.5, tl.float16, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[6, 0], tla.simd.vdups(1.5, tl.float16, mask=tail), dist="NORM_B16")
            tla.simd.vsts(o_ub[7, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST), dist="NORM_B16")
            tla.simd.vsts(o_ub[8, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST), dist="NORM_B16")
            tla.simd.vsts(o_ub[9, 0], tla.simd.vmax(av, bv, mask=tail), dist="NORM_B16", mask=tail)
            tla.simd.vsts(o_ub[10, 0], tla.simd.vdiv(av, cv, mask=tail), dist="NORM_B16", mask=tail)
            tla.simd.vsts(o_ub[11, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.LOWEST), dist="NORM_B16",
                          mask=tail)
            tla.simd.vsts(o_ub[12, 0], tla.simd.vdupv(av, mask=tail, pos=tla.simd.POS.HIGHEST), dist="NORM_B16",
                          mask=tail)
        tla.copy(o_ub, o)


def _strip_locs(text):
    text = re.sub(r" loc\(#loc\d+\)", "", text)
    return "\n".join(ln for ln in text.splitlines() if not re.match(r"\s*#loc\d* = loc\(", ln))


def _tail_ir(kernel, signature, consts, cap, chunk, label):
    module, ctx = compile_offline(kernel, signature, consts)
    text = _strip_locs(str(module))
    nblk = (cap + chunk - 1) // chunk
    assert text.count("ave.hir.vcmp <LT>") == 1, label
    assert "ave.hir.vci" in text or "ave.hir.vload" in text, label
    assert text.count("ave.hir.masked_store") == ROWS, label
    tail_ssa = next(ln for ln in text.splitlines() if "ave.hir.vcmp <LT>" in ln).split("=")[0].strip()
    all_ssas = {ln.split("=")[0].strip() for ln in text.splitlines() if "ave.hir.pge <ALL>" in ln}
    stores = [ln for ln in text.splitlines() if "ave.hir.masked_store" in ln]
    assert sum(f", {tail_ssa}," in ln for ln in stores) == 4, label
    assert sum(any(f", {ssa}," in ln for ssa in all_ssas) for ln in stores) == 9, label
    assert set(re.findall(r"ave\.hir\.pge <([^>]+)>", text)) == {"ALL"}, \
        f"{label}: pge must only be the ALL mask, never a tail prefix"
    # Four vdupv calls per chunk: rows 7/8 (ALL store) and rows 11/12 (masked
    # store) each emit LOWEST + HIGHEST; no CSE runs before lowering.
    assert text.count("ave.hir.vector_broadcast ") == 4, label
    assert re.search(r"scf\.for .* to %\w+ step %\w+\s*: i32", text), \
        f"{label}: expected a chunk loop"
    assert f"arith.constant {nblk} : i32" in text, f"{label}: expected {nblk} chunks"
    assert f"arith.constant {CORES} : i32" in text, f"{label}: expected step {CORES}"
    module = lower_offline(module, ctx)
    check_lowered_ir(str(module), loads=4 if label == "f32" else 5, stores=1, buffers=None)


def test_ir(output_dir=None):
    _tail_ir(simd_tail_f32, F32_SIGNATURE, constexprs(), F32_CAP, F32_CHUNK, "f32")
    _tail_ir(simd_tail_f16, F16_SIGNATURE, constexprs(F16_CAP, F16_CHUNK), F16_CAP, F16_CHUNK, "f16")
    print("IR PASS: runtime-N signed vcmp tail predicate feeds 4 masked stores; 9 ALL rows keep pge<ALL>")


def _inputs(cap, dtype):
    data = torch.arange(cap, dtype=torch.float64) / 64.0
    a = data.to(dtype)
    b = (data + 0.5).to(dtype)
    c = torch.full((cap, ), 1.0 / 16.0, dtype=dtype)
    g = torch.full((GUARD, ), 123.5, dtype=dtype)
    idx = torch.cat([
        torch.zeros(GUARD, dtype=torch.int16),
        torch.arange(cap, dtype=torch.int16),
        torch.zeros(GUARD, dtype=torch.int16)
    ])
    return torch.cat([g, a, g]), torch.cat([g, b, g]), torch.cat([g, c, g]), idx


def _values(a, b, c, dtype, chunk):
    af = a.float()
    div = (af / c.float()).to(dtype)
    expv = torch.exp(af).to(dtype)
    low = torch.empty_like(a)
    high = torch.empty_like(a)
    for start in range(0, a.numel(), chunk):
        low[start:start + chunk] = a[start]
        high[start:start + chunk] = a[start + chunk - 1]
    return {
        0: div,
        1: torch.maximum(a, b),
        2: torch.minimum(a, b),
        3: torch.maximum(a, torch.zeros_like(a)),
        4: expv,
        5: torch.full_like(a, 1.5),
        6: torch.full_like(a, 1.5),
        7: low,
        8: high,
        9: torch.maximum(a, b),
        10: div,
        11: low,
        12: high,
    }


def _reference(ns, cap, chunk, dtype):
    a, b, c, _ = _inputs(cap, dtype)
    a, b, c = a[GUARD:GUARD + cap], b[GUARD:GUARD + cap], c[GUARD:GUARD + cap]
    values = _values(a, b, c, dtype, chunk)
    width = GUARD + cap + GUARD
    out = []
    for n in ns:
        for row in range(ROWS):
            line = torch.full((width, ), SENTINEL, dtype=torch.float32).to(dtype)
            if row < 9:
                line[GUARD:GUARD + cap] = torch.zeros((), dtype=dtype)
            line[GUARD:GUARD + n] = values[row][:n]
            out.append(line)
    return torch.stack(out)


def _launch(ns, cap, chunk, dtype):
    a, b, c, idx = _inputs(cap, dtype)
    width = GUARD + cap + GUARD
    rows = len(ns) * ROWS
    out = torch.full((rows, width), SENTINEL, dtype=dtype, device="npu")
    consts = constexprs(cap, chunk)
    for k, n in enumerate(ns):
        args = [a.npu(), b.npu(), c.npu()] if dtype == torch.float32 else [a.npu(), b.npu(), c.npu(), idx.npu()]
        kernel = simd_tail_f32 if dtype == torch.float32 else simd_tail_f16
        kernel[(CORES, )](*args, out, n, k * ROWS, **consts)
    torch.npu.synchronize()
    return out.cpu(), _reference(ns, cap, chunk, dtype)


def _check(got, ref, ns, cap, dtype):
    div_rtol = 1e-6 if dtype == torch.float32 else 2e-3
    exp_rtol = 1e-5 if dtype == torch.float32 else 5e-3
    width = GUARD + cap + GUARD
    for k, n in enumerate(ns):
        base = k * ROWS
        for row in range(ROWS):
            g, r = got[base + row], ref[base + row]
            if row in (0, 10):
                torch.testing.assert_close(g, r, rtol=div_rtol, atol=0)
            elif row == 4:
                torch.testing.assert_close(g, r, rtol=exp_rtol, atol=0)
            else:
                torch.testing.assert_close(
                    g.view(torch.int32) if dtype == torch.float32 else g.view(torch.int16),
                    r.view(torch.int32) if dtype == torch.float32 else r.view(torch.int16), rtol=0, atol=0)
            # Guard lanes outside [GUARD, GUARD+CAP) always keep the sentinel.
            assert bool((g[:GUARD] == SENTINEL).all()) and bool((g[GUARD + cap:] == SENTINEL).all()), \
                f"n={n} row={row}: guard lane was overwritten"
            # ALL rows zero the inactive data lanes; masked rows keep the sentinel.
            beyond = g[GUARD + n:GUARD + cap] if n < cap else g[:0]
            if row < 9:
                assert bool((beyond == 0).all()), f"n={n} row={row}: ALL store must zero lanes >= N"
            else:
                assert bool((beyond == SENTINEL).all()), f"n={n} row={row}: masked store touched a lane >= N"
    assert got.shape[1] == width


def test_compile():
    _launch(F32_NS[:1], F32_CAP, F32_CHUNK, torch.float32)
    _launch(F16_NS[:1], F16_CAP, F16_CHUNK, torch.float16)
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: simd_tail generated artifacts", len(artifacts), "->", artifacts[0])


# N matrix: f32 covers 5 chunks of 64, f16 covers 4 chunks of 128; each value
# is below / equal to / just above one chunk and reaches multi-chunk tails.
TAIL_CASES = [(torch.float32, F32_CAP, F32_CHUNK, n) for n in F32_NS] + \
             [(torch.float16, F16_CAP, F16_CHUNK, n) for n in F16_NS]
TAIL_IDS = [f"f32-n{n}" for n in F32_NS] + [f"f16-n{n}" for n in F16_NS]


@pytest.mark.parametrize("dtype,cap,chunk,n", TAIL_CASES, ids=TAIL_IDS)
def test_run(dtype, cap, chunk, n):
    got, ref = _launch([n], cap, chunk, dtype)
    _check(got, ref, [n], cap, dtype)
    tag = "f32" if dtype == torch.float32 else "f16"
    print(f"SIMD_TAIL {tag} OK: N={n} chunks={(cap + chunk - 1) // chunk} core={CORES}")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        for dtype, cap, chunk, n in TAIL_CASES:
            test_run(dtype, cap, chunk, n)
    print("ALL PASS: simd_tail", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
