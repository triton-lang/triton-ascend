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
"""JIT compilation through the TLA frontend."""
from hashlib import sha256

from triton.compiler.compiler import ASTSource, CompiledKernel, make_backend
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction, create_function_from_signature

from ._backend import _EXTRA_OPTIONS, _TLABackend, compile_tla, normalize_extra_options


class TLASTSource(ASTSource):

    def parse_options(self):
        return {"_frontend_kind": "tla"}

    def hash(self):
        return sha256(f"tla:{super().hash()}".encode()).hexdigest()

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from triton.language.extra.tla.compiler import ast_to_ttir

        constexprs = {self.fn.arg_names[i[0]]: value for i, value in self.constants.items()}
        return ast_to_ttir(self.fn, self.signature, constexprs, context=context, options=options)


class TLAJITFunction(JITFunction):
    """``@tla.jit``: ``triton.jit`` whose kernels lower via the TLA frontend."""

    ASTSource = TLASTSource
    CompiledKernel = CompiledKernel
    compile = staticmethod(compile_tla)

    def create_binder(self):
        target = driver.active.get_current_target()
        if target.backend == "npu":
            backend = _TLABackend(target, accept_extra_options=_EXTRA_OPTIONS not in self.signature.parameters)
        else:
            backend = make_backend(target)
        binder = create_function_from_signature(self.signature, self.params, backend)
        return {}, {}, target, backend, binder

    def run(self, *args, grid, warmup, **kwargs):
        if _EXTRA_OPTIONS in kwargs and _EXTRA_OPTIONS not in self.signature.parameters:
            extra_options = normalize_extra_options(kwargs.pop(_EXTRA_OPTIONS))
            if extra_options:
                kwargs[_EXTRA_OPTIONS] = extra_options
        return super().run(*args, grid=grid, warmup=warmup, **kwargs)
