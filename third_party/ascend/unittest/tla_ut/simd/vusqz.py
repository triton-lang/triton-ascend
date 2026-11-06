"""Tests for AVE exclusive prefix mask count."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, lower_offline


@tla.jit
def vusqz_kernel(M, O, D: tl.constexpr, N: tl.constexpr, DIST: tl.constexpr):
    m = tla.alloc_buffer((N, ), D, space="ub")
    o = tla.alloc_buffer((N, ), D, space="ub")
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0], tla.simd.vusqz(mask, D), dist=DIST)
    tla.copy(o, tla.view(O, (N, )))


CASES = [
    (torch.uint8, tl.uint8, "u8", 256, "NORM_B8"),
    (torch.uint16, tl.uint16, "u16", 128, "NORM_B16"),
    (torch.uint32, tl.uint32, "u32", 64, "NORM_B32"),
]


def test_ir():
    for _, dtype, short, n, dist in CASES:
        mod, ctx = compile_offline(vusqz_kernel, {"M": "*" + short, "O": "*" + short},
                                   {"D": dtype, "N": n, "DIST": dist})
        text = str(lower_offline(mod, ctx))
        assert "ave.hir.vusqz" in text
        assert "ave.hir.vslide" not in text
    print("IR PASS: vusqz")


def test_compile():
    """Exercise backend instruction selection without initializing an NPU."""
    from triton.backends.ascend.utils import _get_npucompiler_path

    compiler, env = _get_npucompiler_path()
    for _, dtype, short, n, dist in CASES:
        mod, ctx = compile_offline(vusqz_kernel, {"M": "*" + short, "O": "*" + short},
                                   {"D": dtype, "N": n, "DIST": dist})
        with tempfile.TemporaryDirectory(prefix="vusqz-" + short + "-") as tmp:
            root = Path(tmp)
            source = root / "kernel.mlir"
            source.write_text(str(lower_offline(mod, ctx)))
            output = root / "kernel"
            cmd = [
                compiler,
                str(source), "--target=Ascend950", "--custom-aic-number=32", "--custom-aiv-number=64",
                "--enable-hfusion-compile=true", "--enable-triton-kernel-compile=true", "--enable-tla-mode=true",
                "--enable-optimized-metaop=true", "--save-temps=" + tmp, "-o",
                str(output)
            ]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            diagnostics = result.stdout + result.stderr
            assert result.returncode == 0, diagnostics
            binary = root / "kernel.o"
            assert binary.is_file() and binary.stat().st_size > 0, diagnostics
            llvm_ir = (root / "kernel.ll").read_text()
            assert f"@llvm.hivm.vbr.v{n}i{2048 // n}" in llvm_ir, llvm_ir
            assert f"@llvm.hivm.vusqz.m.v{n}i{2048 // n}" in llvm_ir, llvm_ir
            assert "zeroinitializer" not in llvm_ir, llvm_ir
        print("COMPILE PASS: vusqz", short, "VBR initialization and kernel.o")


def test_run():
    for td, dtype, _, n, dist in CASES:
        for name, mask in [("zero", torch.zeros(n)), ("one", torch.ones(n)), ("alternating", torch.arange(n) % 2 == 0),
                           ("sparse", torch.arange(n) % 3 != 0)]:
            out = torch.empty(n, dtype=td, device="npu")
            vusqz_kernel[(1, )](mask.to(td).npu(), out, D=dtype, N=n, DIST=dist)
            torch.npu.synchronize()
            ref = mask.to(torch.int64).cumsum(0) - mask.to(torch.int64)
            torch.testing.assert_close(out.cpu().to(torch.int64), ref, rtol=0, atol=0)
            print("RUN PASS: vusqz", td, n, name)
    print("RUN PASS: vusqz all")


if __name__ == "__main__":
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
