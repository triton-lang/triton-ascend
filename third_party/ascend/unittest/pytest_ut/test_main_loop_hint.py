# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend import _apply_ascend_patch
from triton.backends.ascend.compiler import NPUOptions, make_ttir, min_dot_size, ttir_to_linalg
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource


_apply_ascend_patch()


class Options:
    num_warps = 4
    num_stages = 1
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    sanitize_overflow = True


@triton.jit
def loop_hint_kernel(out, n):
    for i in tl.range(0, n, compile_hint="main_loop"):
        tl.store(out + i, i)


def test_range_compile_hint_is_emitted():
    src = ASTSource(loop_hint_kernel, {"out": "*i32", "n": "i32"}, {})
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(loop_hint_kernel, src, context, Options(), {}, {})

    ttir = str(module)
    assert ttir.count('tt.compile_hint = "main_loop"') == 1


def test_range_compile_hint_survives_ttadapter_lowering():
    src = ASTSource(loop_hint_kernel, {"out": "*i32", "n": "i32"}, {})
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = NPUOptions(
        arch="Ascend910_9589",
        enable_dynamic_cv_pipeline=False,
        enable_graph_optimize=False,
    )
    module = ast_to_ttir(loop_hint_kernel, src, context, options, {"min_dot_size": min_dot_size(None)}, {})
    metadata = {**options.__dict__}
    module = make_ttir(module, metadata, options)
    ttadapter = ttir_to_linalg(module, metadata, options)

    assert ttadapter.count('tt.compile_hint = "main_loop"') == 1
