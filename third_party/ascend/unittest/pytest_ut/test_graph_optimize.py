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

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource


pytestmark = pytest.mark.backend("none")


@triton.jit
def graph_optimize_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x_ptr + offsets)
    tl.store(y_ptr + offsets, value)


def make_ast_ttir(options):
    source = ASTSource(
        graph_optimize_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32"},
        {"BLOCK": 16},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return ast_to_ttir(graph_optimize_kernel, source, context, options, {}, {})


def assert_ttir_text_reparseable(ttir, tmp_path, name):
    path = tmp_path / f"{name}.ttir.mlir"
    path.write_text(ttir)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    reparsed = ir.parse_mlir_module(str(path), context)
    assert "tt.func" in str(reparsed)


def assert_reparseable(module, tmp_path, name):
    assert_ttir_text_reparseable(str(module), tmp_path, name)


def test_graph_optimize_pass_accepts_zero_rule_mask(tmp_path):
    module = make_ast_ttir(NPUOptions(arch="Ascend910_95"))
    pm = ir.pass_manager(module.context)
    ascend.passes.ttir.add_graph_optimize(pm, rule_mask=0)
    pm.run(module)

    assert "tt.func" in str(module)
    assert_reparseable(module, tmp_path, "zero-rule-mask")


def test_make_ttir_supports_graph_optimize_toggle(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    for enabled in (False, True):
        options = NPUOptions(
            arch="Ascend910_95",
            enable_graph_optimize=enabled,
        )
        for attempt in range(2):
            module = make_ast_ttir(options)
            result = make_ttir(module, {}, options)

            assert result is module
            assert "tt.func" in str(result)
            assert_reparseable(
                result,
                tmp_path,
                f"graph-optimize-{int(enabled)}-{attempt}",
            )


def _require_npu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    return torch


def test_graph_optimize_numerical_equivalence(monkeypatch, tmp_path):
    """Exercise the complete JIT pipeline on hardware with cache bypassed."""
    torch = _require_npu()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    source = torch.arange(16, dtype=torch.float32, device="npu")
    outputs = {}

    for enabled in (False, True):
        repeated_outputs = []
        for attempt in range(2):
            # JITFunction has its own in-memory cache, so clear it before each
            # launch to ensure TRITON_ALWAYS_COMPILE reaches triton.compile().
            graph_optimize_kernel.cache.clear()
            output = torch.empty_like(source)
            compiled = graph_optimize_kernel[(1,)](
                source,
                output,
                BLOCK=16,
                enable_graph_optimize=enabled,
            )
            torch.npu.synchronize()
            assert_ttir_text_reparseable(
                compiled.asm["ttir"],
                tmp_path,
                f"hardware-{int(enabled)}-{attempt}",
            )
            repeated_outputs.append(output.cpu())

        torch.testing.assert_close(repeated_outputs[0], source.cpu())
        torch.testing.assert_close(repeated_outputs[1], repeated_outputs[0])
        outputs[enabled] = repeated_outputs[-1]

    torch.testing.assert_close(outputs[True], outputs[False])


def test_graph_optimize_options_contribute_to_npu_hash():
    assert (
        NPUOptions(enable_graph_optimize=False).hash()
        != NPUOptions(enable_graph_optimize=True).hash()
    )
