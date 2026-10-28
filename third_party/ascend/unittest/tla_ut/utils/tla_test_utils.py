"""Shared helpers for TLA 3-in-1 tests (ir / compile / run).

Stages are driven by env ``TLA_TEST_STAGE`` (default ``all``):
- ``ir``      : offline-compile the HIVM IR finally handed to npuir and
                assert key structure (hivm.hir.load/store, multi_buffer,
                pointer_cast, ...); no NPU needed.
- ``compile`` : on top of ``ir``, full compile and confirm .o/.npubin
                artifacts; needs Ascend driver (device or op simulator).
- ``run``     : on top of ``compile``, real-device launch + accuracy; needs NPU.

``all`` (default) auto-degrades: compile/run only when NPU is available,
otherwise ir only.
"""
import os
from pathlib import Path

from triton.language.extra.tla.compiler import ast_to_ttir, lower_ttir
from triton._C.libtriton import ir, buffer_ir, passes
from triton._C.libtriton.ascend import ir as ascend_ir


class Options:
    """Default offline compile options (same as legacy gen_*.py / pytest)."""
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    sanitize_overflow = True
    arch = "Ascend950"
    supported_fp8_dtypes = ("fp8e5", "fp8e4nv", "fp8e4b8", "fp8e4b15", "fp8e5b16")


def npu_available():
    try:
        import torch
        import torch_npu
        return torch.npu.is_available()
    except Exception:
        return False


def get_stages():
    """Parse TLA_TEST_STAGE, return (do_ir, do_compile, do_run)."""
    stage = os.environ.get("TLA_TEST_STAGE", "all").strip().lower()
    if stage == "ir":
        return True, False, False
    if stage == "compile":
        return True, True, False
    if stage == "run":
        return True, True, True
    # all (default): full run on NPU, else degrade to ir only.
    run = npu_available()
    return True, run, run


def make_context():
    ctx = ir.context()
    ir.load_dialects(ctx)
    buffer_ir.load_dialects(ctx)
    ascend_ir.load_dialects(ctx)
    return ctx


def compile_offline(fn, signature, constexprs, output_dir=None):
    """AST -> TTIR -> symbol_dce: front-end IR handed to npuir."""
    ctx = make_context()
    module = ast_to_ttir(fn, signature, constexprs, ctx, Options())
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "00_ast.mlir").write_text(str(module))
    pm = ir.pass_manager(ctx)
    passes.common.add_symbol_dce(pm)
    pm.run(module, "module")
    assert module.verify()
    if output_dir is not None:
        (output_dir / "01_symbol_dce.mlir").write_text(str(module))
    return module, ctx


def lower_offline(module, ctx, output_dir=None):
    """Run full triton-ascend chain to ttadapter boundary: no TTIR leaks to npuir."""
    return lower_ttir(module, ctx, output_dir=output_dir)


def check_lowered_ir(text, loads, stores, buffers, extra=()):
    """Unified assert: no TTIR entities + hivm load/store/multi_buffer counts + op-specific checks."""
    for forbidden in ("!tt.ptr", "tt.ptr_to_int", "bind_buffer", "bufferization.to_tensor", "tensor<"):
        assert forbidden not in text, forbidden
    assert text.count("hivm.hir.load") == loads, "unexpected hivm.hir.load count"
    assert text.count("hivm.hir.store") == stores, "unexpected hivm.hir.store count"
    assert text.count("hivm.multi_buffer") == buffers, "unexpected hivm.multi_buffer count"
    for probe in extra:
        assert probe in text, probe


def check_compile_artifacts():
    """After launch-compile, check .npubin/.o exists in TRITON_CACHE_DIR (or default)."""
    cache_candidates = []
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir:
        cache_candidates.append(Path(cache_dir))
    cache_candidates.append(Path.home() / ".triton" / "cache")
    for base in cache_candidates:
        if not base.is_dir():
            continue
        artifacts = sorted(base.rglob("*.npubin")) + sorted(base.rglob("*.o"))
        if artifacts:
            return [str(p) for p in artifacts[:5]]
    return []
