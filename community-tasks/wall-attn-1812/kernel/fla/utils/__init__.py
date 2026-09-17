# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Minimal vendored subset of upstream fla.utils for Ascend NPU (task #1812).

from ._compat import SUPPORTS_AUTOTUNE_CACHE, ascend_compile_kwargs, autotune_cache_kwargs  # noqa: F401
from ._config import (  # noqa: F401
    FLA_CACHE_RESULTS, FLA_CI_ENV, FLA_DISABLE_TENSOR_CACHE, FLA_TENSOR_CACHE_SIZE,
)
from ._decorators import contiguous, input_guard, tensor_cache  # noqa: F401
from ._device import (  # noqa: F401
    IS_NPU, IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, custom_device_ctx, device,
    device_name, device_platform, device_torch_lib, get_available_device, get_multiprocessor_count,
)
from ._testing import assert_close, get_abs_err, get_err_ratio  # noqa: F401
