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
"""Offline AST-to-TTIR entry point for explicit TLA memref signatures.

This does not define a device launcher ABI. Standard Triton ASTSource accepts
pointer strings; this entry point accepts tla.buffer_type without creating
pointer tensors or changing Triton's global type parser.

The TLA wiring (``TLA_WITH_DISPATCH`` merged into the ``WITH_DISPATCH``
registry, ``tla_setup_unified_builder`` mounted in ``CodeGenerator.__init__``)
is baked into the wheel by the build-time triton-ascend patch, so the plain
patched ``CodeGenerator`` already handles ``with tla.scope.simd():`` and the
TLA builder methods; no TLA-specific subclass or runtime dispatch is needed.
"""
import re
from collections import namedtuple
from triton.language import constexpr, str_to_ty
from triton.compiler.code_generator import (
    ASTFunction,
    CodeGenerator,
    get_jit_fn_file_line,
)
from .core import buffer_type


def lower_ttir(module, context, compile_on_910_95=True, compile_mode="simd_simt_template", output_dir=None,
               named_ops=True):
    """Run the default non-distributed lowering prefix on a TLA TTIR module.

    A standard inliner prefix (mirroring ``make_ttir``) first resolves JIT
    helpers (``tt.func``/``tt.call``) via the Triton inliner interface; TLA
    SIMD VFs are inline ``scope.scope`` ops at this stage, which the inliner
    does not touch (outline-scope in bishengir extracts them later).

    The TLA frontend already emits HIVM-HIR-level IR directly (``hivm.hir.*``
    ops, ``annotation.mark``, ``memref`` ops, ``scf.for``, SIMD VF bodies
    inside inline ``scope.scope`` regions with ``ave.hir.*`` ops), so the
    TTIR-oriented stages of the generic runtime
    chain are no-ops on TLA IR. Stage dumps confirm: ``triton_control_flow_opt``,
    ``triton_to_structure``, ``discrete_mask_access_conversion``,
    ``triton_to_annotation``, ``triton_to_unstructure``, ``triton_to_hivm``,
    ``triton_to_hfusion``, ``triton_to_llvm``, ``bubble_up_operation`` and
    ``merge_concat_load_buffer`` leave the module byte-identical (the first
    ``triton_to_structure`` run only performs CSE/cleanup that the final
    ``triton_to_linalg`` pass already reproduces via its own CSE+canonicalizer
    sub-pipeline). They are therefore removed; only ``triton_to_linalg`` runs,
    producing the ttadapter boundary (``func.func`` entry plus kernel ABI
    attributes), so no ``tt.func``/TTIR reaches npuir (bishengir). The only
    stage dumps written are ``00_inline.mlir`` and ``01_linalg.mlir``.
    Distributed and dynamic CV pipelines, later adapter/backend stages and
    launcher ABI are outside this offline entry point.
    """
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes
    from triton._C.libtriton import ascend as ascend_binding

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Inline JIT helpers before conversion. The MLIR inliner handles both
    # tt.call (Triton dialect interface) and func.call (registered interface).
    pm = ir.pass_manager(context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    pm.run(module, "module")
    if not module.verify():
        raise RuntimeError("inlining produced invalid IR")
    if output_dir is not None:
        (output_dir / "00_inline.mlir").write_text(str(module))

    # Calls that survive inlining (e.g. explicitly noinline helpers) cannot
    # cross the ttadapter boundary, so refuse them before conversion.
    calls = []
    module.walk(lambda op: calls.append(op.get_name()) if op.get_name() == "tt.call" else None)
    if calls:
        raise ValueError("unresolved tt.call after inlining; noinline JIT helpers cannot "
                         "cross the ttadapter boundary")
    # Only triton_to_linalg runs on TLA IR (see docstring): the TTIR-oriented
    # stages of the runtime chain are no-ops on TLA IR and are removed.
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend_binding.passes.ttir.add_triton_to_linalg(pm, False, named_ops, False, True, compile_on_910_95, compile_mode)
    pm.run(module, "module")
    if not module.verify():
        raise RuntimeError("triton_to_linalg produced invalid IR")
    if output_dir is not None:
        (output_dir / "01_linalg.mlir").write_text(str(module))
    check_ttadapter(module)
    return module


def check_ttadapter(module):
    """Validate the single-AIV-kernel boundary, not a runnable launcher ABI.

    Operation checks use the IR walk. The two integer ABI attributes are
    checked on the canonical printed entry header because this binding exposes
    only string/bool attributes on generic operations.
    """
    if not module.verify():
        raise ValueError("invalid ttadapter module")
    operations = []
    module.walk(operations.append)
    forbidden = [op.get_name() for op in operations if op.get_name().startswith("tt.")]
    if forbidden:
        raise ValueError(f"TTIR operations must not reach npuir: {forbidden}")
    entries = [
        op for op in operations if op.get_name() == "func.func" and op.get_str_attr("sym_visibility") != "private"
    ]
    if len(entries) != 1:
        raise ValueError("expected exactly one public func.func kernel entry")
    entry = entries[0]
    for key, value in (("global_kernel", "local"), ("mix_mode", "aiv"), ("parallel_mode", "simd")):
        if entry.get_str_attr(key) != value:
            raise ValueError(f"kernel entry must have {key} = {value!r}")
    name = entry.get_str_attr("sym_name")
    header = next((line for line in str(module).splitlines()
                   if re.match(r"\s*func\.func (?:public )?@" + re.escape(name) + r"\(", line)), "")
    for key, index in (("SyncBlockLockArgIdx", 0), ("WorkspaceArgIdx", 1)):
        if not re.search(r"\b" + key + r" = " + str(index) + r" : i64\b", header):
            raise ValueError(f"kernel entry must have {key} = {index} : i64")


def ast_to_ttir(fn, signature, constexprs, context, options, module=None):
    """Compile a kernel whose runtime arguments are TLA buffers or scalars.

    ``signature`` maps argument names to buffer_type or scalar Triton strings.
    Only top-level constexpr parameters are supported by this offline entry.
    """
    types, constants = [], {}
    for index, name in enumerate(fn.arg_names):
        if name in constexprs:
            types.append(constexpr(constexprs[name]).type)
            constants[(index, )] = constexprs[name]
        else:
            ty = signature[name]
            if not isinstance(ty, buffer_type):
                if isinstance(ty, str):
                    ty = str_to_ty(ty, None)
                if ty.is_block():
                    raise TypeError("TLA runtime memory arguments must be buffer_type "
                                    "or pointer; the entry should tla.view pointers "
                                    "into GM buffers")
            types.append(ty)
    prototype = ASTFunction([], types, constants, {})
    file_name, begin_line = get_jit_fn_file_line(fn)
    proxy = namedtuple("SpecializationProxy", "constants signature")(constexprs, signature)
    generator = CodeGenerator(context, prototype, gscope=fn.get_capture_scope(), function_name=fn.repr(proxy),
                              jit_fn=fn, is_kernel=True, file_name=file_name, begin_line=begin_line, options=options,
                              codegen_fns={}, module_map={}, module=module, is_gluon=False)
    generator.visit(fn.parse())
    module = generator.module
    module.context = context
    if not module.verify():
        raise RuntimeError("TLA module verification failed")
    return module
