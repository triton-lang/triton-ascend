# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import contextlib

import pytest

import triton
import triton.language as tl
from triton._C.libtriton import ir
import triton.backends.ascend.compiler as compiler
from triton.backends.compiler import GPUTarget

pytestmark = pytest.mark.backend("native")


@pytest.fixture(params=["Ascend910B1", "Ascend910_9581"])
def backend(request):
    return compiler.AscendBackend(GPUTarget("npu", request.param, 32))


@pytest.mark.parametrize("dtype, deprecated", [
    (tl.float8e5, False),
    (tl.float8e4nv, False),
    (tl.float8e4b8, False),
    (tl.float8e5b16, False),
    (tl.float8e4b15, False),
    (tl.float8e4b15, True),
    (tl.float8e4b8, True),
    (tl.float8e5b16, True),
])
def test_fp8_dot_frontend_options(backend, dtype, deprecated):

    @triton.jit
    def kernel(dtype: tl.constexpr):
        a = tl.full((64, 64), 0.0, dtype)
        tl.dot(a, a)

    raw = {"deprecated_fp8_dot_operand_dtypes": (str(dtype), )} if deprecated else {}
    options = backend.parse_options(raw)
    source = triton.compiler.ASTSource(kernel, {"dtype": "constexpr"}, {"dtype": dtype})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    warning_message = "the use of fp8e4b15 is deprecated" if dtype == tl.float8e4b15 else "upcasted to fp16"
    warning = pytest.warns(UserWarning, match=warning_message) if deprecated else contextlib.nullcontext()
    with warning:
        module = source.make_ir(backend.target, options, backend.get_codegen_implementation(options),
                                backend.get_module_map(), context)

    assert module.verify()
    dot = next(line for line in str(module).splitlines() if "tt.dot " in line)
    assert ("tensor<64x64xf16>" in dot) == (deprecated or dtype in (tl.float8e4b15, tl.float8e4b8, tl.float8e5b16))
