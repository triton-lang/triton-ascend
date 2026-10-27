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
Ascend-specific builder utilities for code generation.
"""

# The wrapper helpers are shared verbatim with the cann extension; only the
# method list below is TLA-specific.
from triton.language.extra.cann.extension.builder import (
    create_builder_method_wrapper,
    attach_builder_methods,
)

__all__ = [
    "create_builder_method_wrapper",
    "attach_builder_methods",
    "setup_unified_builder",
]


def setup_unified_builder(main_builder, ascend_builder):
    """Set up a unified builder interface by attaching methods from specialized builders."""
    main_builder._ascend_builder = ascend_builder
    ascend_methods = [
        'create_tla_buffer_type',
        'create_tla_vector_type',
        'create_tla_alloc',
        'create_tla_bind',
        'create_tla_subview',
        'create_tla_reinterpret_cast',
        'create_tla_collapse_shape',
        'create_tla_expand_shape',
        'create_hivm_load',
        'create_hivm_store',
        'create_hivm_nd2nz',
        'create_hivm_fixpipe',
        'create_hivm_mmadl1',
        'create_annotation_mark',
        'get_array_attr',
        'parse_attr',
        'is_simt_mode',
        'get_i64_array_attr',
        'get_buffer_ty',
        'get_buffer_ty_with_strides',
        'create_simd_vlds',
        'create_simd_vsts',
        'create_simd_binary',
        'create_simd_unary',
        'create_simd_vfma',
        'create_simd_vreduce',
        'create_simd_vci',
        'create_simd_vmembar',
        'create_simd_vcmp',
        'create_simd_vbroadcast',
        'create_simd_scalar',
        'create_simd_vsplat',
        'create_simd_vcvt',
        'create_simd_pge',
        'create_simd_vsel',
        'create_simd_vintlv',
        'create_simd_vdintlv',
        'create_simd_bitcast',
        'create_scope_op',
        'get_scope_entry_block',
        'scope_return',
    ]
    attach_builder_methods(main_builder, ascend_builder, ascend_methods)
