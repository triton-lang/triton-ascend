"""Tests for AVE stable-lane compaction."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, get_stages, lower_offline, check_compile_artifacts)


@tla.jit
def vsqz_kernel(X, M, O, D: tl.constexpr, I: tl.constexpr, N: tl.constexpr, DIST: tl.constexpr, MODE: tl.constexpr):
    x = tla.alloc((N, ), D, space="ub")
    m = tla.alloc((N, ), I, space="ub")
    o = tla.alloc((N, ), D, space="ub")
    tla.load_gm_to_ub(tla.view(X, (N, )), x)
    tla.load_gm_to_ub(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0], tla.simd.vsqz(tla.simd.vlds(x[0]), mask, mode=MODE), dist=DIST)
    tla.store_ub_to_gm(o, tla.view(O, (N, )))


CASES = [
    ("i32", torch.int32, torch.int32, tl.int32, tl.int32, "i32", "i32", 64, "NORM_B32", ("zero", "one", "alternating",
                                                                                         "sparse")),
    ("u8", torch.uint8, torch.uint8, tl.uint8, tl.uint8, "u8", "u8", 256, "NORM_B8", ("sparse", )),
    ("f16", torch.float16, torch.int16, tl.float16, tl.int16, "fp16", "i16", 128, "NORM_B16", ("sparse", )),
    ("f32", torch.float32, torch.int32, tl.float32, tl.int32, "fp32", "i32", 64, "NORM_B32", ("sparse", )),
]


def _masks(name, n):
    return {
        "zero": torch.zeros(n, dtype=torch.bool),
        "one": torch.ones(n, dtype=torch.bool),
        "alternating": torch.arange(n) % 2 == 0,
        "sparse": torch.arange(n) % 3 != 0,
    }[name]


def test_ir():
    for _, _, _, dt, mt, short, mshort, n, dist, _ in CASES:
        for mode in (tla.simd.SQUEEZE_STORE.STORED, tla.simd.SQUEEZE_STORE.NO_STORED):
            mod, ctx = compile_offline(vsqz_kernel, {"X": "*" + short, "M": "*" + mshort, "O": "*" + short},
                                       {"D": dt, "I": mt, "N": n, "DIST": dist, "MODE": mode})
            text = str(lower_offline(mod, ctx))
            assert "ave.hir.vsqz" in text
            assert "ave.hir.vscatter" not in text
            # squeeze_store is a hardware operand, not a write mode: no vsel is added.
            assert "ave.hir.vsel" not in text
            assert ("no_stored" in text) == (mode is tla.simd.SQUEEZE_STORE.NO_STORED)
    print("IR PASS: vsqz")


def _launch():
    _, td, md, dt, mt, _, _, n, dist, _ = CASES[0]
    base = torch.arange(n, dtype=td)
    mask = torch.ones(n, dtype=torch.bool)
    out = torch.empty(n, dtype=td, device="npu")
    vsqz_kernel[(1, )](base.npu(), mask.to(md).npu(), out, D=dt, I=mt, N=n, DIST=dist,
                       MODE=tla.simd.SQUEEZE_STORE.STORED)
    torch.npu.synchronize()
    return out


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vsqz generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    by_name = {case[0]: case for case in CASES}
    stored, no_stored = tla.simd.SQUEEZE_STORE.STORED, tla.simd.SQUEEZE_STORE.NO_STORED
    runs = [
        ("i32", "zero", stored),
        ("i32", "sparse", stored),
        ("i32", "sparse", no_stored),
        ("i32", "alternating", stored),
        ("i32", "alternating", no_stored),
        ("u8", "sparse", stored),
        ("u8", "sparse", no_stored),
        ("f16", "sparse", stored),
        ("f32", "sparse", no_stored),
    ]
    for name, mask_name, mode in runs:
        _, td, md, dt, mt, _, _, n, dist, _ = by_name[name]
        base = torch.arange(n, dtype=td)
        mask = _masks(mask_name, n)
        out = torch.empty(n, dtype=td, device="npu")
        vsqz_kernel[(1, )](base.npu(), mask.to(md).npu(), out, D=dt, I=mt, N=n, DIST=dist, MODE=mode)
        torch.npu.synchronize()
        # squeeze_store does not change the vector result: compacted lanes then zeros.
        ref = torch.zeros(n, dtype=td)
        ref[:int(mask.sum())] = base[mask]
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        print("RUN PASS: vsqz", name, n, mask_name, "no_stored" if mode is no_stored else "stored")
    print("RUN PASS: vsqz all")


if __name__ == "__main__":
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
