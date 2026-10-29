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
"""
Ascend-specific code generation handlers for 'with' statement context managers.
"""

__all__ = ["handle_scope_with"]


def _vf_scope_attrs(builder, effect_strs, core_mode="vector"):
    """Scope attributes that outline-scope turns back into the old SIMD VF shape.

    The bishengir HFusion pipeline unconditionally runs outline-scope with
    outlineMarkedScopesOnly=true, which clones the scope attrs onto the
    extracted func.func and keeps an inline func.call at the original site.
    ``hivm.vector_function`` + ``no_inline`` and ``hivm.func_core_type<AIV>``
    reproduce the attrs the previously frontend-outlined VF carried, so the
    HIVM stage behaves identically; no ``hivm.vf_mode`` is set, keeping the
    SIMT path away. ``vector_mode``/``tcore_type`` mark this as a manual VF
    scope for the HFusion pass pipeline (HFusionVectorizeOps, AutoVectorize).
    ``tla.memory_effect`` carries the per-argument buffer effects in the
    outline input order; outline-scope lowers it to per-argument
    ``hivm.memory_effect`` attrs.
    """
    tcore_type_str = ("#hivm.tcore_type<CUBE>" if core_mode == "cube" else "#hivm.tcore_type<VECTOR>")
    return {
        "no_inline": builder.get_unit_attr(),
        "outline": builder.get_unit_attr(),
        "vector_mode": builder.get_string_attr("vec"),
        "tcore_type": builder.parse_attr(tcore_type_str),
        "hivm.func_core_type": builder.parse_attr("#hivm.func_core_type<AIV>"),
        "hivm.vector_function": builder.get_unit_attr(),
        "tla.memory_effect": builder.get_array_attr([builder.get_string_attr(eff) for eff in effect_strs]),
    }


def handle_scope_with(generator, node):
    """
    Handle 'with tla.scope.simd()/tla.scope.cube():' by emitting an
    inline scope.scope carrying the VF attributes; outline-scope in the
    bishengir HFusion pipeline extracts the marked scope into the func.func +
    func.call that the HIVM stage expects.

    The SIMD register ops stay inline (sharing the enclosing function's scope,
    so buffer handles are used directly, without function-argument remapping).
    The memory effect is inferred from the simd.vlds/vsts calls inside the
    block: load -> read, store -> write (read_write when both).
    """
    from triton.compiler.code_generator import enter_sub_region
    from . import simd as simd_mod

    context_expr = node.items[0].context_expr
    withitem_class = generator.visit(context_expr.func)
    core_mode = getattr(withitem_class, "core_mode", "vector")

    with enter_sub_region(generator):
        ip, last_loc = generator._get_insertion_point_and_loc()

        # Dummy visit to infer the (buffer, effect) pairs via simd._VF_EFFECTS.
        simd_mod._VF_EFFECTS = []
        dummy = generator.builder.create_block()
        generator.builder.set_insertion_point_to_start(dummy)
        generator.visit_compound_statement(node.body)
        dummy.erase()
        effects = simd_mod._VF_EFFECTS
        simd_mod._VF_EFFECTS = None

        # Deduplicate buffers; merge read + write into read_write.
        ordered = []
        index_by_id = {}
        for buf, eff in effects:
            if id(buf) in index_by_id:
                prev = ordered[index_by_id[id(buf)]][1]
                if prev != eff:
                    ordered[index_by_id[id(buf)]][1] = "read_write"
            else:
                index_by_id[id(buf)] = len(ordered)
                ordered.append([buf, eff])

        effect_strs = [eff for _, eff in ordered]

        generator._set_insertion_point_and_loc(ip, last_loc)
        scope_op, body = generator.builder.create_scope_op(_vf_scope_attrs(generator.builder, effect_strs, core_mode),
                                                           [])
        generator.builder.set_insertion_point_to_start(body)
        generator.visit_compound_statement(node.body)
        generator.builder.set_insertion_point_to_end(body)
        if not body.has_terminator():
            generator.builder.scope_return([])
    return None
