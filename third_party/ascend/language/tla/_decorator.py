# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""Decorators for TLA builtins."""

import warnings
from functools import wraps
from inspect import Parameter, signature


def experimental(fn):
    """Emit a UserWarning at the kernel call site when compiling a call to this TLA builtin."""
    sig = signature(fn)
    takes_generator = "_generator" in sig.parameters

    @wraps(fn)
    def wrapper(*args, **kwargs):
        generator = kwargs.get("_generator") if takes_generator else kwargs.pop("_generator", None)
        if generator is not None:
            warnings.warn_explicit(
                f"TLA API {fn.__name__} is experimental and may change without notice.",
                category=UserWarning,
                filename=generator.file_name,
                lineno=generator.begin_line + generator.cur_node.lineno,
                module=generator.jit_fn.__module__,
                registry=generator.jit_fn.__globals__.setdefault("__warningregistry__", {}),
            )
        return fn(*args, **kwargs)

    if not takes_generator:
        # Request Triton's call-site context while preserving the builtin's other parameters.
        params = list(sig.parameters.values())
        index = len(params)
        if params and params[-1].kind == Parameter.VAR_KEYWORD:
            index -= 1
        params.insert(index, Parameter("_generator", Parameter.KEYWORD_ONLY, default=None))
        wrapper.__signature__ = sig.replace(parameters=params)
    return wrapper
