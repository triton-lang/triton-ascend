# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Vendored from upstream fla/utils/_decorators.py, keeping only the decorators
# required by fla.ops.wall_attn and its vendored helpers:
#   tensor_cache / input_guard / contiguous.

import contextlib
import functools
import inspect
import sys
from collections import deque
from collections.abc import Callable
from typing import Any

import torch

from ._config import FLA_DISABLE_TENSOR_CACHE, FLA_TENSOR_CACHE_SIZE
from ._device import custom_device_ctx


def tensor_cache(fn: Callable[..., torch.Tensor], ) -> Callable[..., torch.Tensor]:
    """
    A decorator that memoizes the most recent results of a function call by argument identity.

    The decorator keeps a bounded queue of up to ``FLA_TENSOR_CACHE_SIZE`` (default 4)
    recent ``(args, kwargs, result)`` triples. On each call, every cached entry is checked
    in order; an entry is considered a hit when the positional arg count and kwarg key set
    match and every argument is the *same object* (``is`` identity) as the cached one. On a
    hit the cached result is returned and ``fn`` is skipped; on a miss ``fn`` is invoked and
    the new triple is appended (evicting the oldest when the queue is full).

    Caching is fully bypassed when the ``FLA_DISABLE_TENSOR_CACHE`` environment variable is
    set to ``'1'``.
    """
    cached: deque = deque(maxlen=FLA_TENSOR_CACHE_SIZE)

    def cache_disabled() -> bool:
        utils_module = sys.modules.get('fla.utils')
        return getattr(utils_module, 'FLA_DISABLE_TENSOR_CACHE', FLA_DISABLE_TENSOR_CACHE)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if cache_disabled():
            return fn(*args, **kwargs)

        for cached_args, cached_kwargs, cached_result in cached:
            if len(args) != len(cached_args) or len(kwargs) != len(cached_kwargs):
                continue
            if all(a is b for a, b in zip(args, cached_args, strict=False)) and \
                    all(k in cached_kwargs and v is cached_kwargs[k] for k, v in kwargs.items()):
                return cached_result

        result = fn(*args, **kwargs)
        cached.append((args, kwargs, result))
        return result

    return wrapper


def _skip_contiguous(
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str],
    param_name: str,
    skip_params: set[str],
) -> bool:
    return no_guard_contiguous is True or param_name in skip_params


def _contiguous_if_needed(arg: Any, skip: bool) -> Any:
    if isinstance(arg, torch.Tensor) and not skip:
        return arg.contiguous()
    return arg


def input_guard(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str] = False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]] | Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.

    Args:
        no_guard_contiguous (bool | list[str] | tuple[str, ...] | set[str]):
            If True, skip all contiguous checks. If a list/tuple/set of parameter names,
            skip contiguous check for those parameters.
    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        skip_params = set(no_guard_contiguous) if isinstance(no_guard_contiguous, (list, tuple, set)) else set()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    param_name = f"__arg_{i}"

                processed_args.append(
                    _contiguous_if_needed(arg, _skip_contiguous(no_guard_contiguous, param_name, skip_params)))

            processed_kwargs = {}
            for k, v in kwargs.items():
                processed_kwargs[k] = _contiguous_if_needed(v, _skip_contiguous(no_guard_contiguous, k, skip_params))

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    # Handle direct usage without parentheses: @input_guard
    if fn is not None:
        return decorator(fn)

    return decorator


def contiguous(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Alias for input_guard() without parameters."""
    return input_guard(fn)
