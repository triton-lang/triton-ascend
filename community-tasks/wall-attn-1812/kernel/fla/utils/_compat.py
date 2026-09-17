# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Vendored from upstream fla/utils/_compat.py, trimmed to autotune-cache and
# the Ascend compile-kwargs helper (which upstream added exactly for NPU).

import inspect
import logging

import triton

from ._config import FLA_CACHE_RESULTS
from ._device import IS_NPU

logger = logging.getLogger(__name__)

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


def ascend_compile_kwargs(*, blacklist_auto_blockify: bool = False) -> dict:
    """Return NPU Triton launch kwargs that disable auto-multi-buffer.

    Empty on non-NPU devices. Disabling auto-multi-buffer bounds UB usage of the
    attention tiles (correctness-first on A2/A3/950; performance tuning may
    revisit). When ``blacklist_auto_blockify`` is set, also disable AutoBlockify
    if the installed compiler exposes that option.
    """
    if not IS_NPU:
        return {}
    kwargs = {'multibuffer': False}
    if blacklist_auto_blockify:
        try:
            from triton.backends.ascend.compiler import NPUOptions
        except ImportError:
            return kwargs
        if 'has_auto_blockify_blacklist_op' in getattr(NPUOptions, '__dataclass_fields__', {}):
            kwargs['has_auto_blockify_blacklist_op'] = True
    return kwargs
