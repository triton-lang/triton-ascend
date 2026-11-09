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

The TLA wiring is self-contained: ``setup_unified_builder`` (mounted in
``CodeGenerator.__init__`` by the patch) registers the ``with tla.scope.*``
handlers and attaches the TLA builder methods.
"""
import re
from collections import namedtuple
from triton.language import constexpr, str_to_ty
from triton.compiler.code_generator import (
    ASTFunction,
    CodeGenerator,
    get_jit_fn_file_line,
)


def _stamp_tla_mode(module):
    """Set ``hacc.mode = "tla"`` to select the TLA backend pipeline."""
    from triton._C.libtriton.ascend import ir as ascend_ir

    builder = ascend_ir.ascendnpu_ir_builder(module.context)
    module.set_attr("hacc.mode", builder.parse_attr('"tla"'))


def lower_ttir(module, context, output_dir=None):
    """Run the TLA-specific lowering prefix on a TLA TTIR module.

    A standard inliner prefix in the caller first resolves JIT
    helpers (``tt.func``/``tt.call``) via the Triton inliner interface; TLA
    SIMD VFs are inline ``scope.scope`` ops at this stage, which the inliner
    does not touch (outline-scope in bishengir extracts them later).
    Callers run ``add_ttir_pipeline`` (Inliner and SymbolDCE) before
    this entry point: ``make_tla_ttir`` at runtime and ``lower_offline`` in
    offline tests. The remaining kernel must be a single public,
    single-block, void ``tt.func``.

    The TLA frontend already emits HIVM-HIR-level IR directly (``hivm.hir.*``
    ops, ``annotation.mark``, ``memref`` ops, ``scf.for``, SIMD VF bodies
    inside inline ``scope.scope`` regions with ``ave.hir.*`` ops), so the
    TTIR-oriented stages of the generic runtime
    chain are skipped: ``triton_control_flow_opt``,
    ``triton_to_structure``, ``discrete_mask_access_conversion``,
    ``triton_to_annotation``, ``triton_to_unstructure``, ``triton_to_hivm``,
    ``triton_to_hfusion``, ``triton_to_llvm``, ``bubble_up_operation`` and
    ``merge_concat_load_buffer``. Only ``tla_to_linalg`` and local
    canonicalization run here, producing the ttadapter boundary
    (``func.func`` entry plus kernel ABI attributes). Scalar GM loads are
    temporarily lowered using the generic scalar-load conversion. This pass
    converts its owned operations rather than rejecting the entire Triton
    dialect; remaining operations still require downstream support. Core/VF
    attributes are then inferred from
    explicit scopes and the ttadapter boundary is checked.
    The only stage dumps written are ``00_inline.mlir`` and ``01_linalg.mlir``
    when ``output_dir`` is provided; the former records the caller-inlined
    input after TLA mode stamping.
    Distributed and dynamic CV pipelines, later backend stages and kernel
    launch are outside this entry point.
    """
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes
    from triton._C.libtriton import ascend as ascend_binding

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Mark as TLA for the backend pipeline; unconditional (TLA-only entry).
    _stamp_tla_mode(module)

    if output_dir is not None:
        (output_dir / "00_inline.mlir").write_text(str(module))

    # Calls that survive inlining (e.g. explicitly noinline helpers) cannot
    # cross the ttadapter boundary, so refuse them before conversion.
    calls = []
    module.walk(lambda op: calls.append(op.get_name()) if op.get_name() == "tt.call" else None)
    if calls:
        raise ValueError("unresolved tt.call after inlining; noinline JIT helpers cannot "
                         "cross the ttadapter boundary")
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend_binding.passes.ttir.add_tla_to_linalg(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module, "module")
    if not module.verify():
        raise RuntimeError("tla_to_linalg produced invalid IR")
    _infer_tla_kernel_mode(module, context)
    from triton._C.libtriton.ascend import ir as ascend_ir
    ascend_ir.lower_tla_sync_all(module)
    if output_dir is not None:
        (output_dir / "01_linalg.mlir").write_text(str(module))
    check_ttadapter(module)
    return module


def add_ttir_pipeline(pm):
    """Resolve JIT helpers without applying generic Triton optimization."""
    from triton._C.libtriton import passes

    passes.common.add_inliner(pm)
    passes.common.add_symbol_dce(pm)


def _kernel_entries(module):
    """All non-private func.func ops: the ttadapter kernel entry candidates."""
    operations = []
    module.walk(operations.append)
    return [op for op in operations if op.get_name() == "func.func" and op.get_str_attr("sym_visibility") != "private"]


def infer_func_core_type(module):
    """Infer AIC/AIV/MIX from scopes in a single-kernel module.

    The scope classes stamp ``#hivm.tcore_type<...>`` on the emitted
    ``scope.scope`` ops (see :meth:`tla.scope.mlir_attrs`); the binding
    exposes no generic attribute reader, so the printed module is parsed.
    VECTOR -> AIV, CUBE -> AIC.
    """
    text = str(module)
    core_types = set()
    if "#hivm.tcore_type<CUBE>" in text:
        core_types.add("AIC")
    if "#hivm.tcore_type<VECTOR>" in text:
        core_types.add("AIV")
    if "AIC" in core_types and "AIV" in core_types:
        return "MIX"
    if "AIC" in core_types:
        return "AIC"
    if "AIV" in core_types:
        return "AIV"
    return None


def _infer_tla_kernel_mode(module, context):
    """Stamp the entry core type, launch mode and VF mode on the kernel entry."""
    from triton._C.libtriton.ascend import ir as ascend_ir

    entries = _kernel_entries(module)
    if len(entries) != 1:
        raise ValueError("expected exactly one public func.func kernel entry")
    entry = entries[0]
    core_type = infer_func_core_type(module)
    if core_type is None:
        return
    builder = ascend_ir.tla_builder(context, "")
    builder.set_op_attr(entry, "hivm.func_core_type", builder.parse_attr(f"#hivm.func_core_type<{core_type}>"))
    builder.set_op_attr(entry, "mix_mode", builder.get_string_attr("aiv" if core_type == "AIV" else "mix"))
    module.set_attr("hivm.module_core_type", builder.parse_attr(f"#hivm.module_core_type<{core_type}>"))
    # MIX only when a SIMT scope exists; the binding can't read dialect-typed
    # vf_mode, so match it on the printed scope too.
    scopes = []
    module.walk(scopes.append)
    has_simt = any(op.get_name() == "scope.scope" and (
        op.get_str_attr("vector_mode") == "simt" or "#hivm.vf_mode<SIMT>" in str(op)) for op in scopes)
    if has_simt:
        builder.set_op_attr(entry, "hivm.vf_mode", builder.parse_attr("#hivm.vf_mode<MIX>"))


def check_ttadapter(module):
    """Validate the single-kernel boundary, not a runnable launcher ABI.

    Operation checks use the IR walk. The two integer ABI attributes are
    checked on the canonical printed entry header because this binding exposes
    only string/bool attributes on generic operations.
    """
    if not module.verify():
        raise ValueError("invalid ttadapter module")
    operations = []
    module.walk(operations.append)
    # Check the entry ABI and operations owned by the adapter, not the whole
    # Triton dialect. This does not enable SIMT scopes or a Tile lowering path.
    converted_ops = {
        "tt.func", "tt.call", "tt.return", "tt.addptr", "tt.ptr_to_int", "tt.get_program_id", "tt.get_num_programs",
        "tt.assert", "tt.load", "tt.clampf"
    }
    forbidden = [op.get_name() for op in operations if op.get_name() in converted_ops]
    if forbidden:
        raise ValueError(f"unconverted TLA adapter operations: {forbidden}")
    if any(op.get_name() == "hivm.hir.sync_block" for op in operations):
        raise ValueError("TLA global barriers must be expanded before npuir")
    entries = _kernel_entries(module)
    if len(entries) != 1:
        raise ValueError("expected exactly one public func.func kernel entry")
    entry = entries[0]
    for key, value in (("global_kernel", "local"), ("parallel_mode", "simd")):
        if entry.get_str_attr(key) != value:
            raise ValueError(f"kernel entry must have {key} = {value!r}")
    mix = entry.get_str_attr("mix_mode")
    if mix not in ("aiv", "mix"):
        raise ValueError(f"kernel entry mix_mode must be 'aiv' or 'mix', got {mix!r}")
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
    from triton._C.libtriton.ascend import ir as ascend_ir
    types, constants = [], {}
    for index, name in enumerate(fn.arg_names):
        if name in constexprs:
            types.append(constexpr(constexprs[name]).type)
            constants[(index, )] = constexprs[name]
        else:
            ty = signature[name]
            if isinstance(ty, str):
                ty = str_to_ty(ty, None)
            types.append(ty)
    prototype = ASTFunction([], types, constants, {})
    file_name, begin_line = get_jit_fn_file_line(fn)
    proxy = namedtuple("SpecializationProxy", "constants signature")(constexprs, signature)
    generator = CodeGenerator(context, prototype, gscope=fn.get_capture_scope(), function_name=fn.repr(proxy),
                              jit_fn=fn, is_kernel=True, file_name=file_name, begin_line=begin_line, options=options,
                              codegen_fns={}, module_map={}, module=module, is_gluon=False)
    arch = getattr(options, "arch", None)
    # Mount the TLA builder on the code generator before visiting the AST.
    from triton.language.extra.tla.builder import setup_unified_builder

    tla_builder = ascend_ir.tla_builder(context, arch or "")
    tla_builder.set_loc(file_name, begin_line, 0)
    setup_unified_builder(generator.builder, tla_builder)
    generator.visit(fn.parse())
    module = generator.module
    module.context = context
    # Mirror the runtime Ascend patch (backend._apply_ascend_patch) and inject
    # #hacc.target<arch> so target-aware verifiers (e.g. FixpipeOp::verify,
    # which rejects dst=UB / NZ2DN fixpipe unless isAscend950(moduleOp))
    # behave like the on-device path. The arch string passes through as-is and
    # must be a TargetDevice enum symbol (e.g. Ascend910_9589): its parse
    # matches exactly, so family names such as "Ascend950" resolve nowhere.
    if arch:
        builder = ascend_ir.ascendnpu_ir_builder(context, arch)
        module.set_attr("hacc.target", builder.parse_attr(f'#hacc.target<"{arch}">'))
    if not module.verify():
        raise RuntimeError("TLA module verification failed")
    return module
