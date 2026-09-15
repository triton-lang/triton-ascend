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
__version__ = '3.6.0'

import logging
from triton._C.libtriton.ascend import ir as ascend_ir

from .testing import do_bench_npu

_ASCEND_UTILS_PACKAGE = "triton_ascend_op_utils"


def _load_ascend_op_utils():
    """Load the standalone ``third_party/ascend/utils`` config package by path.

    That folder is shipped next to the backend but is not installed onto
    ``sys.path`` (and must not collide with this backend's own ``utils.py``),
    so it is imported explicitly via importlib using the backend file location
    and the ``TRITON_PLUGIN_DIRS`` environment variable as fallbacks.
    """
    import importlib.util
    import os
    import sys
    from pathlib import Path

    if _ASCEND_UTILS_PACKAGE in sys.modules:
        return sys.modules[_ASCEND_UTILS_PACKAGE]

    candidates = [Path(__file__).resolve().parent.parent / "utils"]
    for plugin_dir in os.environ.get("TRITON_PLUGIN_DIRS", "").split(os.pathsep):
        if plugin_dir.strip():
            plugin_root = Path(plugin_dir.strip())
            candidates.append(plugin_root / "utils")
            # The plugin root may be either the ascend root (containing
            # ``backend/``) or ``third_party``; cover both layouts.
            candidates.append(plugin_root / "ascend" / "utils")

    for utils_dir in candidates:
        init_file = utils_dir / "__init__.py"
        if not init_file.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            _ASCEND_UTILS_PACKAGE,
            init_file,
            submodule_search_locations=[str(utils_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_ASCEND_UTILS_PACKAGE] = module
        spec.loader.exec_module(module)
        return module
    return None


def _apply_ascend_patch():
    from triton.compiler.code_generator import CodeGenerator

    if not getattr(CodeGenerator, "_ascend_patch_applied", False):
        _original_cg_init = CodeGenerator.__init__

        def _patched_cg_init(self, *args, **kwargs):
            """
            Monkey Patch for Ascend:
            Injects 'hacc.target' attribute into the module after initilization of module.
            """
            _original_cg_init(self, *args, **kwargs)
            options = self.builder.options
            context = self.context
            if hasattr(options, "arch") and options.arch:
                try:
                    builder = ascend_ir.ascendnpu_ir_builder(context, options.arch)

                    target_attr_str = f'#hacc.target<"{options.arch}">'
                    self.module.set_attr("hacc.target", builder.parse_attr(target_attr_str))
                except Exception as e:
                    logging.warning(f"[Ascend Patch] Failed to set hacc.target: {e}")

        CodeGenerator.__init__ = _patched_cg_init
        CodeGenerator._ascend_patch_applied = True

    # Patch compiler.parse to support Ascend-specific IR extensions
    # for ir_override and TRITON_KERNEL_OVERRIDE features.
    from triton.compiler import compiler as _compiler_module

    if not getattr(_compiler_module, "_ascend_parse_patch_applied", False):
        from pathlib import Path as _Path
        from triton._C.libtriton import ir as _ir

        # Keep the original community logic intact, adding Ascend extensions
        # to the existing text/binary branches.
        def _patched_parse(full_name, ext, context):
            if ext == "ttir" or ext == "ttgir":
                module = _ir.parse_mlir_module(full_name, context)
                module.context = context
                return module
            if ext in ("llir", "ptx", "amdgcn", "ttadapter", "bcmlir"):
                return _Path(full_name).read_text()
            if ext in ("cubin", "hsaco", "mlirbc", "npubin"):
                return _Path(full_name).read_bytes()

        _compiler_module.parse = _patched_parse
        _compiler_module._ascend_parse_patch_applied = True

    # Patch TritonSemantic.dot for Ascend-specific HF32 guard and
    # max_num_imprecise_acc warning.
    from triton.language.semantic import TritonSemantic

    if not getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
        _original_dot = TritonSemantic.dot

        def _patched_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype):
            """
            Monkey Patch for Ascend:
            - HF32 precision only works for fp32 x fp32.
              When either input is not fp32, silently fall back
              to default precision (ieee).
            - Warn when max_num_imprecise_acc is explicitly set, since
              Ascend NPU does not support imprecise accumulation.
            """
            # Ascend does not support tf32, tf32 is converted to hf32. Avoid tf32 verification interception
            if input_precision == "tf32":
                input_precision = "hf32"
            # Ascend NPU does not support imprecise accumulation.
            # Force max_num_imprecise_acc to None so the upstream None
            # branch handles it (via max_num_imprecise_acc_default = 0),
            # avoiding the fp8 ValueError path which is NVIDIA-only.
            if max_num_imprecise_acc is not None:
                print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. "
                      "Thus it is ignored.")
                max_num_imprecise_acc = None

            return _original_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype)

        TritonSemantic.dot = _patched_dot
        TritonSemantic._ascend_dot_patch_applied = True

    # Install the unified, non-intrusive dtype interception for operators
    # whose dtypes are unsupported on Ascend. The rules live in the standalone
    # package shipped at third_party/ascend/utils.
    if not getattr(CodeGenerator, "_ascend_dtype_guard_applied", False):
        op_utils = _load_ascend_op_utils()
        if op_utils is not None and hasattr(op_utils, "install_dtype_guard"):
            try:
                op_utils.install_dtype_guard()
                CodeGenerator._ascend_dtype_guard_applied = True
            except Exception as e:
                logging.warning(f"[Ascend Patch] Failed to install dtype guard: {e}")


__all__ = ["do_bench_npu"]
