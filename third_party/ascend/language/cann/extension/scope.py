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
        - `core_mode`: Selects cube or vector core operations.
        - `vector_mode`: Selects the SIMD or SIMT vector path inside a mixed
            compile mode (`compile_mode="simd_simt"` or
            `"simd_simt_template"`).
    """

    def __init__(self, core_mode: str = None, _builder=None, _semantic=None, vector_mode: str = None, **kwargs):
        """
        :param core_mode: Either "cube" or "vector" to specify the core type (optional)
        :param vector_mode: Either "simd" or "simt" to select the vector path (optional)
        :param _builder: Internal builder object (set by code_generator)
        :param _semantic: Internal semantic object (set by code_generator)
        :param kwargs: Additional internal parameters
        """
        # Convert constexpr to value if not being called from code generator
        self.core_mode = _unwrap_if_constexpr(core_mode) if _builder is None else core_mode
        self.vector_mode = _unwrap_if_constexpr(vector_mode) if _builder is None else vector_mode
        self._builder = _builder
        self._semantic = _semantic

        # Validate core_mode
        if self.core_mode is not None and self.core_mode not in _VALID_CORE_MODES:
            raise ValueError(f'core_mode must be one of {_VALID_CORE_MODES}, got {self.core_mode!r}')
        if self.vector_mode is not None and self.vector_mode not in _VALID_VEC_MODES:
            raise ValueError(f'vector_mode must be one of {_VALID_VEC_MODES}, got {self.vector_mode!r}')
        if self.core_mode == "cube" and self.vector_mode is not None:
            raise ValueError('vector_mode cannot be set when core_mode="cube"')
        if self.core_mode is None and self.vector_mode is None and not kwargs:
            raise ValueError("scope requires at least one annotation")

    def __enter__(self):
        if self._builder is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
