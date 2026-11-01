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
"""Runtime JIT source that lowers ``@tla.jit`` kernels via the TLA frontend.

A plain ``triton.jit`` kernel runs the generic ``make_ttir`` frontend and
skips the TLA kernel-mode stamping.  Reusing ``ast_to_ttir`` keeps runtime
kernels consistent with the offline tla_ut flow and sets
``options.enable_tla_mode`` for the backend.

``@tla.jit`` additionally adapts kernel parameters annotated as
``tla.buffer_type`` (or a bare ``tla.buffer``) for launch: the host passes a
tensor, the binder validates it and encodes the buffer type into the
specialization signature, and the launcher maps that entry onto the pointer
argument the device ABI expects.  See :mod:`triton.language.extra.tla.host`.
"""
from triton.compiler.compiler import ASTSource
from triton.runtime.jit import JITFunction

from .host import buffer_spec, is_buffer_annotation, resolve_buffer_arg


class TLASTSource(ASTSource):
    """ASTSource that emits TLA TTIR instead of the community frontend."""

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from triton.language.extra.tla.compiler import ast_to_ttir

        # Enable the backend TLA lowering and --enable-tla-mode compile flag.
        object.__setattr__(options, "enable_tla_mode", True)
        constexprs = {self.fn.arg_names[i[0]]: value for i, value in self.constants.items()}
        return ast_to_ttir(self.fn, self.signature, constexprs, context=context, options=options)


class TLAJITFunction(JITFunction):
    """``@tla.jit``: ``triton.jit`` whose kernels lower via the TLA frontend."""

    def _buffer_annotations(self):
        return {
            param.name: param._param.annotation
            for param in self.params
            if is_buffer_annotation(param._param.annotation)
        }

    def _wrap_binder(self, binder, annotations):
        """Adapt host values of buffer parameters into TLA buffer signatures."""
        positions = {name: self.arg_names.index(name) for name in annotations}

        def wrapped(*args, **kwargs):
            bound_args, specialization, options = binder(*args, **kwargs)
            specialization = list(specialization)
            for name, annotation in annotations.items():
                ty = resolve_buffer_arg(bound_args[name], annotation, name)
                specialization[positions[name]] = (buffer_spec(ty), None)
            return bound_args, specialization, options

        return wrapped

    def create_binder(self):
        info = super().create_binder()
        self.ASTSource = TLASTSource
        kernel_cache, key_cache, target, backend, binder = info
        annotations = self._buffer_annotations()
        if annotations:
            binder = self._wrap_binder(binder, annotations)
        return kernel_cache, key_cache, target, backend, binder
