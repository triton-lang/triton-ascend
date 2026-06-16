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

from triton.language.core import _unwrap_if_constexpr

_VALID_CORE_MODES = ("cube", "vector")
_VALID_VEC_MODES = ("simd", "simt")


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
        - `core_mode`: Allows explicitly specify which core type should be used for operations
            within a code block, helping the compiler generate appropriate code for cube or vector cores.
            Valid values: "cube", "vector".
        - `vec_mode`: Within a mixed compile mode (compile_mode="simd_simt" or "simt_template"),
            explicitly select the SIMD or SIMT path for ops in this scope. This is a per-scope
            override of the default routing within mix-compile modes.
            Valid values:"simd", "simt"
            Note: vec_mode targets the vector core compile path. It cannot be combined with
            core_mode="cube". It only takes effect when compile_mode is a mix-compile mode;
            in pure simd / simt_only mode it is ignored.
    """

    def __init__(self, core_mode: str = None, _builder=None, _semantic=None, vec_mode: str = None, **kwargs):
        """
        :param core_mode: Either "cube" or "vector" to specify the core type (optional)
        :param vec_mode: Vector core path selector within mix-compile modes:
                         "simd" forces SIMD path, "simt" forces SIMT path. (optional)
        :param _builder: Internal builder object (set by code_generator)
        :param _semantic: Internal semantic object (set by code_generator)
        :param kwargs: Additional internal parameters
        """
        # Convert constexpr to value if not being called from code generator
        self.core_mode = _unwrap_if_constexpr(core_mode) if _builder is None else core_mode
        self.vec_mode = _unwrap_if_constexpr(vec_mode) if _builder is None else vec_mode
        self._builder = _builder
        self._semantic = _semantic

        # Validate core_mode
        if self.core_mode is not None and self.core_mode not in _VALID_CORE_MODES:
            raise ValueError(f'core_mode must be one of {_VALID_CORE_MODES}, got {self.core_mode!r}')

        # Validate vec_mode
        if self.vec_mode is not None and self.vec_mode not in _VALID_VEC_MODES:
            raise ValueError(f'vec_mode must be one of {_VALID_VEC_MODES}, got {self.vec_mode!r}')

        # vec_mode is a vector-core directive; reject when core_mode explicitly targets cube
        if self.core_mode == "cube" and self.vec_mode is not None:
            raise ValueError('vec_mode cannot be set when core_mode="cube"; '
                             'vec_mode targets the vector core compile path')

        # At least one of core_mode or vec_mode (or other kwargs) must be provided
        if self.core_mode is None and self.vec_mode is None and not kwargs:
            raise ValueError('scope requires at least one argument (core_mode, vec_mode, or '
                             'custom attributes)')

    def __enter__(self):
        if self._builder is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
