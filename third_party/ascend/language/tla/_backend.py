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
"""Ascend backend and compilation stages for TLA kernels."""
import os
from types import FunctionType

from triton.backends.ascend import compiler as ascend
from triton.compiler import compiler as triton_compiler

from ._schedule import sync_arguments

_EXTRA_OPTIONS = "bishengir_extra_options"


def normalize_extra_options(value):
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{_EXTRA_OPTIONS} must be a list or tuple of strings")
    options = tuple(value)
    if any(not isinstance(arg, str) for arg in options):
        raise TypeError(f"{_EXTRA_OPTIONS} must contain only strings")
    if any("\0" in arg for arg in options):
        raise ValueError(f"{_EXTRA_OPTIONS} cannot contain NUL characters")
    return options


def _with_globals(fn, **bindings):
    """Bind compiler dependencies without mutating shared globals."""
    bound = FunctionType(fn.__code__, {**fn.__globals__, **bindings}, fn.__name__, fn.__defaults__, fn.__closure__)
    bound.__kwdefaults__ = fn.__kwdefaults__
    return bound


def make_tla_ttir(mod, metadata, opt):
    from triton.language.extra.tla.compiler import add_ttir_pipeline

    if "hash" not in metadata:
        metadata["hash"] = ascend.hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
    pm = ascend.ir.pass_manager(mod.context)
    pm.enable_debug()
    add_ttir_pipeline(pm)
    pm.run(mod, 'make_tla_ttir')
    if opt.debug:
        ascend.get_dump_manager(metadata["hash"]).put(str(mod), "kernel.ttir.mlir", binary=False)
    return mod


def tla_to_ttadapter(mod, metadata, opt):
    from triton.language.extra.tla.compiler import lower_ttir

    metadata["has_auto_blockify_blacklist_op"] = False
    lower_ttir(mod, mod.context)
    ascend._export_profile_metadata(mod, metadata)
    ascend._export_program_grid_metadata(mod, metadata)
    text = ascend._serialize_module(mod, opt)
    if opt.debug:
        ascend.get_dump_manager(metadata["hash"]).put(text, "kernel.ttadapter.mlir", binary=False)
    return text


class _TLABackend(ascend.AscendBackend):

    # Dynamic reuse can replace the TLA source with a generic ASTSource.
    get_jit_reuse_handler = None

    def __init__(self, target, *, accept_extra_options=True):
        super().__init__(target)
        self.accept_extra_options = accept_extra_options

    def parse_options(self, opts):
        options = super().parse_options(opts)
        if self.accept_extra_options and _EXTRA_OPTIONS in opts:
            object.__setattr__(options, _EXTRA_OPTIONS, normalize_extra_options(opts[_EXTRA_OPTIONS]))
        return options

    def add_stages(self, stages, options, language):
        if not options.enable_tla_mode:
            return super().add_stages(stages, options, language)

        extra_options = getattr(options, _EXTRA_OPTIONS, ())
        sync_options = ()

        def compiler_arguments(executable):
            arguments = sync_options + extra_options
            if arguments and os.path.basename(executable) != "bishengir-compile":
                raise ValueError(f"TLA compiler arguments require bishengir-compile, got {executable!r}")
            return arguments

        compiler = (ascend.linalg_to_bin_enable_npu_compile_910_95
                    if options.compile_on_910_95 else ascend.linalg_to_bin_enable_npu_compile_A2_A3)

        def infer_then_lower(module, metadata):
            nonlocal sync_options
            # Look at the inlined kernel before sync_all expands to its
            # implementation's block flags and pipe barriers.
            sync_options = sync_arguments(module, options.disable_auto_inject_block_sync, extra_options)
            return tla_to_ttadapter(module, metadata, options)

        stages["ttir"] = lambda module, metadata: make_tla_ttir(module, metadata, options)
        stages["ttadapter"] = infer_then_lower

        def to_binary(module, metadata):
            return compiler(module, metadata, options, extra_args=compiler_arguments)

        stages["npubin"] = ascend._with_debug_line(to_binary, options)


def compile_tla(src, target=None, options=None, _env_vars=None):
    if target is None:
        target = triton_compiler.driver.active.get_current_target()
    if target.backend != "npu":
        return triton_compiler.compile(src, target=target, options=options, _env_vars=_env_vars)
    if options and options.get("block_batching_factor", 1) != 1:
        raise ValueError("TLA kernels do not support automatic block batching")
    compile_one = _with_globals(triton_compiler._compile, make_backend=_TLABackend)
    return compile_one(src, target=target, options=options, _env_vars=_env_vars)
