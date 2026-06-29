# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

__all__ = ["scope"]

from triton.language.core import _constexpr_to_value


def _py_value_to_mlir_attr(builder, value):
    """Convert Python value to MLIR attribute."""
    attr_creators = {
        str: lambda v: builder.get_str_attr(v),
        bool: lambda v: builder.get_bool_attr(v),
        int: lambda v: builder.get_int32_attr(v),
        list: lambda v: builder.get_i64_array_attr(v),
    }
    creator = attr_creators.get(type(value))
    return creator(value) if creator else value


def _handle_core_mode_attr(builder, core_mode):
    """Handle core_mode attribute conversion."""
    if core_mode not in ("cube", "vector"):
        return {}
    return {
        builder.get_t_core_type_attr_name():
        (builder.get_t_core_type_cube_attr() if core_mode == "cube" else builder.get_t_core_type_vector_attr())
    }


def _build_mlir_attrs_from_scope_attrs(builder, scope_attrs):
    """Convert Python scope attributes to MLIR attributes."""
    mlir_attrs = {"noinline": builder.get_unit_attr()}
    for k, v in scope_attrs.items():
        if k == "core_mode":
            mlir_attrs.update(_handle_core_mode_attr(builder, v))
        elif k == "noinline":
            if not v:
                mlir_attrs.pop("noinline")
        elif k == "disable_auto_sync":
            if v:
                mlir_attrs["hivm.disable_auto_sync"] = _py_value_to_mlir_attr(builder, v)
        else:
            mlir_attrs[k] = _py_value_to_mlir_attr(builder, v)
    return mlir_attrs


def _verify_loop_carried_variable(_is_triton_value, _is_triton_tensor, name, loop_val, live_val):
    """Verify that loop-carried variable types are consistent."""
    assert _is_triton_value(loop_val), f'cannot reassign constxpr {name} in the loop'
    assert _is_triton_value(live_val), f'cannot reasign constexpr {name} in the loop'
    assert type(loop_val) == type(live_val), f'Loop carried variable {name} changed type'
    assert not _is_triton_tensor(loop_val) or loop_val.type == live_val.type, \
        f'Loop-carried variable {name} has initial type {live_val.type} '\
        f'but is re-assigned to {loop_val.type} in loop! '\
        f'Please make sure that the type stays consistent.'


def _reconstruct_value_from_ir(language, entry_block_arg, ret_type):
    """Reconstruct a tensor value from IR."""
    return language.core.tensor(entry_block_arg, ret_type)


class scope:
    """
    Context manager for entering and exiting a scope, where operations within a scope shares some common characteristics.

    Example:
    ```python
        import triton.language.extra.cann.extension as extension

        @triton.jit
        def kernel(x_ptr, y_ptr, N):
            # specify annotation
            with extension.scope(feature_a=True):
                a = tl.load(x_ptr)
                b = tl.load(y_ptr)
                result = tl.dot(a, b)
    ```

    Reserved keywords:
        - `core_mode`: Allows explicitly specify which core type should be used for operations within a code block, helping the compiler generate appropriate code for cube or vector cores.
    """

    def __init__(self, core_mode: str, _builder=None, _semantic=None, **kwargs):
        """
        :param core_mode: Either "cube" or "vector" to specify the core type
        :param _builder: Internal builder object (set by code_generator)
        :param _semantic: Internal semantic object (set by code_generator)
        :param kwargs: Additional internal parameters
        """
        self.core_mode = _constexpr_to_value(core_mode)
        self._builder = _builder
        self._semantic = _semantic
        self._generator = getattr(_semantic, "generator", None)
        self._scope_attrs = {
            "core_mode": self.core_mode,
            **{key: _constexpr_to_value(value)
               for key, value in kwargs.items()},
        }
        self._scope_region = None

        # Validate core_mode
        if self.core_mode not in ("cube", "vector"):
            raise ValueError(f'core_mode must be "cube" or "vector", got {self.core_mode}')

    def __enter__(self):
        if self._builder is None and self._semantic is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        if self._generator is None:
            return self

        from triton.compiler.code_generator import enter_sub_region

        generator = self._generator
        sub_region = enter_sub_region(generator)
        liveins, _ = sub_region.__enter__()
        ip, last_loc = generator._get_insertion_point_and_loc()
        body_block = generator.builder.create_block()
        generator.builder.set_insertion_point_to_start(body_block)
        self._scope_region = (sub_region, liveins, ip, last_loc, body_block)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._scope_region is None:
            return False

        from triton import language
        from triton.compiler.code_generator import _is_triton_tensor, _is_triton_value

        generator = self._generator
        sub_region, liveins, ip, last_loc, body_block = self._scope_region
        scope_defs = generator.local_defs

        names = []
        ret_types = []
        for name in scope_defs:
            scope_val = scope_defs[name]
            ret_types.append(scope_val.type)
            names.append(name)
            if name in liveins:
                live_val = liveins[name]
                _verify_loop_carried_variable(_is_triton_value, _is_triton_tensor, name, scope_val, live_val)

        mlir_attrs = _build_mlir_attrs_from_scope_attrs(generator.builder, self._scope_attrs)
        generator._set_insertion_point_and_loc(ip, last_loc)
        scope_op = generator.builder.create_scope_op(
            mlir_attrs,
            [ty.to_ir(generator.builder) for ty in ret_types],
        )

        entry_block = generator.builder.create_block_with_parent(scope_op.get_region(0), [])
        body_block.merge_block_before(entry_block)
        generator.builder.set_insertion_point_to_end(entry_block)
        generator.builder.scope_return([generator.lscope[name].handle for name in names])

        sub_region.__exit__(exc_type, exc_val, exc_tb)
        for i, name in enumerate(names):
            generator.set_value(name, _reconstruct_value_from_ir(language, scope_op.get_result(i), ret_types[i]))
        self._scope_region = None
        return False
