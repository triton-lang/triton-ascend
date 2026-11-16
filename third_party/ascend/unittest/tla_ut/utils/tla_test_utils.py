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
import re
from pathlib import Path

import pytest

from triton.language.extra.tla.compiler import ast_to_ttir, lower_ttir
from triton._C.libtriton import ir, buffer_ir, passes
from triton._C.libtriton.ascend import ir as ascend_ir

# Test glue only: ``tla.jit`` is a function, but the first decoration runs
# ``from .jit import TLAJITFunction`` inside it, which rebinds the package
# attribute ``tla.jit`` to the ``tla.jit`` *module*.  Without this pin only the
# very first ``@tla.jit`` in a process works, so every later kernel (and every
# later case file) would fail with "'module' object is not callable".  Pinning
# the canonical decorator keeps the documented ``@tla.jit`` spelling usable for
# every module-level kernel.  This does not change the product surface.
try:
    import triton.language.extra.tla as _tla
    from triton.language.extra.tla.jit import TLAJITFunction as _TLAJITFunction
except Exception:  # stale/native build without the TLA jit module
    _tla = _TLAJITFunction = None
if _TLAJITFunction is not None and getattr(_tla, "jit", None) is not _TLAJITFunction:
    _tla.jit = _TLAJITFunction

# Set TLA_IR_DUMP_DIR=<dir> to also dump the final IR under <dir>/<kernel>/.
_DUMP_DIR = os.environ.get("TLA_IR_DUMP_DIR")


class Options:
    """Default offline compile options (same as legacy gen_*.py / pytest)."""
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    sanitize_overflow = True
    arch = "Ascend910_9589"
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


def _dump_dir(name, output_dir):
    """Dump dir: explicit output_dir wins, else $TLA_IR_DUMP_DIR/<kernel>/."""
    if output_dir:
        return output_dir
    return Path(_DUMP_DIR) / name if _DUMP_DIR else None


def _entry_func_name(module):
    """TLA frontend entry func name (per-case subdir for the final dump)."""
    m = re.search(r"tt\.func\s+(?:public\s+)?@(\w+)", str(module))
    return m.group(1) if m else "lower"


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
    """Run full triton-ascend chain to ttadapter boundary: no TTIR leaks to npuir.

    output_dir defaults to $TLA_IR_DUMP_DIR/<kernel>/; None = no dump.  Only the
    final ttadapter-boundary IR (01_linalg.mlir) is dumped.
    """
    # Entry name must be read before lowering: the pass manager rewrites the
    # module in place (tt.func -> func.func at the ttadapter boundary).
    name = _entry_func_name(module)
    lowered = lower_ttir(module, ctx)
    dump = _dump_dir(name, output_dir)
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        (dump / "01_linalg.mlir").write_text(str(lowered))
    return lowered


def check_lowered_ir(text, loads, stores, buffers=None, extra=()):
    """Unified assert: no TTIR entities + hivm load/store/multi_buffer counts + op-specific checks."""
    for forbidden in ("!tt.ptr", "tt.ptr_to_int", "bind_buffer", "bufferization.to_tensor", "tensor<"):
        assert forbidden not in text, forbidden
    assert text.count("hivm.hir.load") == loads, "unexpected hivm.hir.load count"
    assert text.count("hivm.hir.store") == stores, "unexpected hivm.hir.store count"
    if buffers is not None:
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


def check_kernel_binary(kernel):
    """Verify the binary and artifact belonging to this exact specialization."""
    binary = kernel.asm.get("npubin")
    assert isinstance(binary, bytes) and binary, "kernel did not produce an NPU binary"
    paths = [Path(path) for name, path in kernel.metadata_group.items() if name.endswith((".npubin", ".o"))]
    assert paths, "kernel metadata contains no .npubin/.o artifact"
    assert any(path.is_file() and path.read_bytes() == binary
               for path in paths), "generated binary is missing or differs from this kernel's cache artifact"
    print("COMPILE PASS:", kernel.name, "->", ", ".join(str(path) for path in paths))


def matrix(cases):
    return pytest.mark.parametrize("case", cases, ids=[case["name"] for case in cases])
