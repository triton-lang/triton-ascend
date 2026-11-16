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
    assert first_jit.warmup(32, grid=(1, ), bishengir_extra_options=[]) is plain

    first = first_jit.warmup(32, grid=(1, ), bishengir_extra_options=flags)
    assert first_jit.warmup(32, grid=(1, ), bishengir_extra_options=tuple(flags)) is first
    assert compiler_argv[1][-2:] == tuple(flags)
    assert first.hash != plain.hash

    second_jit = tla.jit(_kernel)
    cached = second_jit.warmup(32, grid=(1, ), bishengir_extra_options=flags)
    assert cached.hash == first.hash
    assert len(compiler_argv) == 2

    flags.reverse()
    reordered = second_jit.warmup(32, grid=(1, ), bishengir_extra_options=flags)
    assert reordered.hash != first.hash
    assert compiler_argv[2][-2:] == tuple(flags)

    with pytest.raises(KeyError, match="bishengir_extra_options"):
        triton.jit(_kernel).warmup(32, grid=(1, ), bishengir_extra_options=flags)
    assert len(compiler_argv) == 3


def test_extra_options_async(compiler_argv):
    jit = tla.jit(_kernel)
    with ThreadPoolExecutor(max_workers=2) as executor, _async_compile.AsyncCompileMode(executor):
        first = jit.warmup(32, grid=(1, ), bishengir_extra_options=["--flag=1"])
        second = jit.warmup(32, grid=(1, ), bishengir_extra_options=["--flag=2"])
    assert first.result().hash != second.result().hash
    assert {argv[-1] for argv in compiler_argv} == {"--flag=1", "--flag=2"}


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
    kernel = tla.jit(_kernel)[(1, )](32, bishengir_extra_options=["--flag"])
    assert compiler_argv[0][-1] == "--flag"
    assert isinstance(kernel.src, TLASTSource)

    preloaded = tla.jit(_kernel).preload(specialization_data[0])
    assert preloaded.hash == kernel.hash
    assert len(compiler_argv) == 1


@pytest.mark.parametrize("value, exception", [("--flag", TypeError), ([1], TypeError), (["\0"], ValueError)])
def test_extra_options_validation(value, exception):
    with pytest.raises(exception, match="bishengir_extra_options"):
        tla.jit(_kernel).warmup(32, grid=(1, ), bishengir_extra_options=value)


def test_debug_command_includes_effective_arguments(compiler_argv, capsys):
    tla.jit(_kernel).warmup(32, grid=(1, ), debug=True, bishengir_extra_options=["--probe=two words"])
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
