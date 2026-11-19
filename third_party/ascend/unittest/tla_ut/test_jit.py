# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Exercise the TLA launch option through JIT and BiShengIR argv construction."""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import triton
import triton.language as tl
from triton.backends.ascend import compiler as ascend
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import CompiledKernel
from triton.language.extra import tla
from triton.language.extra.tla.jit import TLASTSource
from triton.runtime import _async_compile
from triton.runtime.driver import driver
from triton.runtime.jit import MockTensor


def _kernel(N: tl.constexpr):
    with tla.scope.vector():
        tla.sync.barrier(tla.sync.PIPE.ALL)


def _sync_kernel(LOCAL: tl.constexpr, CROSS: tl.constexpr, GLOBAL: tl.constexpr):
    with tla.scope.vector():
        if LOCAL == "pipe" or LOCAL == "signal":
            tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
            if LOCAL == "pipe":
                tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
        if LOCAL == "mutex":
            tla.sync.mutex_lock(0, tla.sync.PIPE.MTE2)
            tla.sync.mutex_unlock(0, tla.sync.PIPE.MTE2)
        if CROSS is not None:
            tla.sync.block_set_flag(CROSS, tla.sync.PIPE.MTE3, 0)
            tla.sync.block_wait_flag(CROSS, tla.sync.PIPE.MTE3, 0)
        if GLOBAL:
            tla.sync.sync_all(False)


def _multibuffer_kernel(MANUAL: tl.constexpr):
    with tla.scope.vector():
        scratch = tla.alloc((32, ), tl.float32)
        tla.annotate_multi_buffer(scratch, 2)
        if MANUAL:
            tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
        with tla.scope.simd():
            tla.simd.vlds(scratch[0], dist="NORM")


def _unit_flag_kernel(CONTROL: tl.constexpr):
    with tla.scope.cube():
        a = tla.alloc((1, 1, 16, 16), tl.float16, space="l1")
        b = tla.alloc((1, 1, 16, 16), tl.float16, space="l1")
        acc = tla.alloc((1, 1, 16, 16), tl.float32, space="l0c")
        tla.cube.mmad(a, b, acc, clear_accum=True, unit_flag_ctrl=CONTROL)


def _mark_gm(ptr):
    tla.annotate_multi_buffer(tla.view(ptr, (32, )), 2)


@pytest.fixture(params=["Ascend910B2", "Ascend910_9589"])
def compiler_argv(request, tmp_path, monkeypatch):
    target = GPUTarget("npu", request.param, 32)
    monkeypatch.setattr(
        driver, "_active",
        SimpleNamespace(get_current_target=lambda: target, get_current_device=lambda: 0,
                        get_current_stream=lambda _: None))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ascend, "NPUUtils",
                        lambda: SimpleNamespace(get_arch=lambda: target.arch, get_device_limit=lambda: (False, 24, 48)))
    monkeypatch.setattr(ascend, "get_cann_version_file_hash", lambda: "test-cann")
    monkeypatch.setattr(ascend, "is_cann_version_at_least", lambda *_: False)
    monkeypatch.setattr(ascend, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))
    monkeypatch.setattr(ascend, "_check_bishengir_api_change", lambda: True)
    monkeypatch.setattr(ascend, "_check_bishengir_is_regbased", lambda: False)
    monkeypatch.setattr(ascend, "get_libdevice", lambda _: "/fake/libdevice.bc")
    calls = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        output = Path(argv[argv.index("-o") + 1] + ".o")
        output.write_bytes(b"kernel")
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_extra_options_cache_and_argv(compiler_argv):
    flags = ["--mlir-print-ir-after-all", "--probe=two words"]
    first_jit = tla.jit(_kernel)
    plain = first_jit.warmup(32, grid=(1, ))
    assert first_jit.warmup(32, grid=(1, ), npuir_extra_options=[]) is plain

    first = first_jit.warmup(32, grid=(1, ), npuir_extra_options=flags)
    assert first_jit.warmup(32, grid=(1, ), npuir_extra_options=tuple(flags)) is first
    assert compiler_argv[1][-4:-2] == tuple(flags)
    assert first.hash != plain.hash

    second_jit = tla.jit(_kernel)
    cached = second_jit.warmup(32, grid=(1, ), npuir_extra_options=flags)
    assert cached.hash == first.hash
    assert len(compiler_argv) == 2

    flags.reverse()
    reordered = second_jit.warmup(32, grid=(1, ), npuir_extra_options=flags)
    assert reordered.hash != first.hash
    assert compiler_argv[2][-4:-2] == tuple(flags)

    with pytest.raises(KeyError, match="npuir_extra_options"):
        triton.jit(_kernel).warmup(32, grid=(1, ), npuir_extra_options=flags)
    assert len(compiler_argv) == 3


def test_extra_options_async(compiler_argv):
    jit = tla.jit(_kernel)
    with ThreadPoolExecutor(max_workers=2) as executor, _async_compile.AsyncCompileMode(executor):
        first = jit.warmup(32, grid=(1, ), npuir_extra_options=["--flag=1"])
        second = jit.warmup(32, grid=(1, ), npuir_extra_options=["--flag=2"])
    assert first.result().hash != second.result().hash
    assert {argv[-3] for argv in compiler_argv} == {"--flag=1", "--flag=2"}


def test_extra_options_launch(compiler_argv, monkeypatch):

    def unexpected_reuse(*args, **kwargs):
        pytest.fail("a TLA launch must compile its own TLA source")

    reuse_handler = SimpleNamespace(run=unexpected_reuse)
    monkeypatch.setattr(ascend.AscendBackend, "get_jit_reuse_handler", staticmethod(lambda: reuse_handler))
    monkeypatch.setattr(CompiledKernel, "run", lambda *args: None)
    monkeypatch.setattr(CompiledKernel, "launch_metadata", lambda *args: None)
    specialization_data = []

    def capture(**kwargs):
        specialization_data.append(kwargs["compile"]["specialization_data"])

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", capture)
    kernel = tla.jit(_kernel)[(1, )](32, npuir_extra_options=["--flag"])
    assert compiler_argv[0][-3] == "--flag"
    assert isinstance(kernel.src, TLASTSource)

    preloaded = tla.jit(_kernel).preload(specialization_data[0])
    assert preloaded.hash == kernel.hash
    assert len(compiler_argv) == 1


@pytest.mark.parametrize("value, exception", [("--flag", TypeError), ([1], TypeError), (["\0"], ValueError)])
def test_extra_options_validation(value, exception):
    with pytest.raises(exception, match="npuir_extra_options"):
        tla.jit(_kernel).warmup(32, grid=(1, ), npuir_extra_options=value)


def test_extra_option_shadowed_by_kernel_parameter(compiler_argv):

    @tla.jit
    def kernel(npuir_extra_options: tl.constexpr):
        with tla.scope.vector():
            tla.sync.barrier(tla.sync.PIPE.ALL)

    kernel.warmup(npuir_extra_options=17, grid=(1, ))
    assert len(compiler_argv) == 1


def test_inferred_sync_options(compiler_argv):
    intra = "--disable-hivm-auto-inject-sync"
    cross = "--disable-auto-inject-block-sync"
    block, counter = tla.sync.MODE.BLOCK, tla.sync.MODE.INTRA
    jit = tla.jit(_sync_kernel)

    for local, cross_mode, global_barrier, manual_intra in (
        ("", None, False, False),
        ("", None, True, False),
        ("signal", None, False, True),
        ("pipe", None, True, True),
        ("mutex", None, False, True),
        ("", block, False, False),
        ("pipe", counter, False, True),
    ):
        jit.warmup(local, cross_mode, global_barrier, grid=(1, ))
        argv = compiler_argv[-1]
        assert (f"{intra}=true" in argv) == manual_intra
        assert f"{cross}=true" in argv


def test_unit_flag_sync_ownership(compiler_argv):
    jit = tla.jit(_unit_flag_kernel)
    disable = "--disable-hivm-auto-inject-sync=true"
    jit.warmup(tla.UnitFlagMode.DISABLED, grid=(1, ))
    assert disable not in compiler_argv[-1]
    jit.warmup(tla.UnitFlagMode.ENABLED_WITH_UPDATE, grid=(1, ), unit_flag=False)
    assert "--enable-hivm-unit-flag-sync=False" in compiler_argv[-1]
    assert disable not in compiler_argv[-1]


@pytest.mark.parametrize("mode, enabled", [
    (tla.UnitFlagMode.DISABLED, False),
    (tla.UnitFlagMode.RESERVED, False),
    (tla.UnitFlagMode.ENABLED_WITH_UPDATE, True),
    (tla.UnitFlagMode.ENABLED_WITHOUT_UPDATE, True),
])
def test_kernel_features_unit_flag(mode, enabled):
    from triton._C.libtriton.ascend import ir as ascend_ir
    from utils.tla_test_utils import compile_offline

    module, _ = compile_offline(tla.jit(_unit_flag_kernel), {}, {"CONTROL": mode})
    features = ascend_ir.get_tla_kernel_features(module)
    assert features.has_unit_flag == enabled
    assert not features.has_intra_core_sync


def test_sync_ownership_rejects_conflicting_options(compiler_argv):
    jit = tla.jit(_sync_kernel)
    with pytest.raises(triton.TritonError, match="cannot combine handwritten and automatic intra-core"):
        jit.warmup("pipe", None, False, grid=(1, ), npuir_extra_options=["--disable-hivm-auto-inject-sync=false"])
    with pytest.raises(triton.TritonError, match="requires handwritten cross-core"):
        jit.warmup("", None, False, grid=(1, ), disable_auto_inject_block_sync=False)
    assert not compiler_argv
    with pytest.raises(triton.TritonError, match="cannot combine handwritten and automatic intra-core"):
        jit.warmup(
            "pipe", None, False, grid=(1, ),
            npuir_extra_options=["--disable-hivm-auto-inject-sync=true", "--disable-hivm-auto-inject-sync=false"])
    assert not compiler_argv


def test_multibuffering_requires_automatic_intra_core_sync(compiler_argv):
    kernel = tla.jit(_multibuffer_kernel)
    with pytest.raises(triton.TritonError, match="requires compiler-managed intra-core"):
        kernel.warmup(True, grid=(1, ), npuir_extra_options=["--disable-hivm-auto-inject-sync=false"])
    with pytest.raises(triton.TritonError, match="requires compiler-managed intra-core"):
        kernel.warmup(False, grid=(1, ), npuir_extra_options=["--disable-hivm-auto-inject-sync=true"])
    assert not compiler_argv
    kernel.warmup(False, grid=(1, ))
    assert "--disable-auto-inject-block-sync=true" in compiler_argv[-1]
    assert "--disable-hivm-auto-inject-sync=true" not in compiler_argv[-1]


def test_multibuffering_rejects_global_memory(compiler_argv):
    with pytest.raises(triton.TritonError, match="requires a core-local buffer"):
        tla.jit(_mark_gm).warmup(MockTensor(tl.float32), grid=(1, ))
    assert not compiler_argv


@pytest.mark.parametrize("body, argv", [
    ("", ()),
    ("hivm.hir.pipe_barrier[<PIPE_ALL>]", ()),
    ("", ("--disable-auto-inject-block-sync", )),
])
def test_automatic_pipeline_sync_ownership(tmp_path, body, argv):
    from triton._C.libtriton import ir
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.language.extra.tla._options import infer_sync_options
    from utils.tla_test_utils import make_context

    path = tmp_path / "pipeline.mlir"
    path.write_text(f"""module {{
        tt.func @kernel() {{
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %i = %c0 to %c1 step %c1 {{
                {body}
            }} {{hivm.cv_pipelined_loop}}
            tt.return
        }}
    }}""")
    context = make_context()
    module = ir.parse_mlir_module(str(path), context)
    features = ascend_ir.get_tla_kernel_features(module)
    assert features.uses_pipeline
    if body or argv:
        with pytest.raises(ValueError, match="tla.pipeline owns synchronization"):
            infer_sync_options(features, None, argv)
    else:
        assert infer_sync_options(features, None, argv) == ()


def test_sync_arguments_require_bishengir(compiler_argv, monkeypatch):
    monkeypatch.setattr(ascend, "_get_npucompiler_path", lambda: ("/fake/npuc", {}))
    with pytest.raises(triton.TritonError, match="require bishengir-compile"):
        tla.jit(_kernel).warmup(32, grid=(1, ))
    assert not compiler_argv


def test_debug_command_includes_effective_arguments(compiler_argv, capsys):
    tla.jit(_kernel).warmup(32, grid=(1, ), debug=True, npuir_extra_options=["--probe=two words"])
    import shlex
    command = next(
        line.split(": ", 1)[1] for line in capsys.readouterr().out.splitlines() if line.startswith("[DEBUG] cmd_list:"))
    printed = shlex.split(command)
    actual = list(compiler_argv[-1])
    # Debug logging substitutes stable dump paths for the temporary input and output.
    assert printed[1] != actual[1]
    for index in (1, actual.index("-o") + 1):
        printed[index] = actual[index]
    assert printed == actual
