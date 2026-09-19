# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Vendored from upstream fla/utils/_device.py, trimmed to the pieces required
# by fla.ops.wall_attn on Ascend NPU:
#   - torch_npu is imported up-front so that `torch.npu` is registered;
#   - check_shared_mem keeps the upstream name/signature (the upstream test
#     monkeypatches it inside fla.ops.wall_attn.parallel) but always returns
#     False on NPU: UB budget is far below GPU shared-memory tiers, so callers
#     take the conservative tile path (BV <= 64, BS <= 32).

import contextlib
import functools
import logging
import warnings
from enum import Enum
from functools import cache

import torch
import triton
from packaging import version as package_version

logger = logging.getLogger(__name__)

# `torch.npu` is registered by the torch_npu package; import it before any use.
try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False


@cache
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return package_version.parse(torch.__version__) >= package_version.parse(version_s)


@cache
def get_available_device() -> str:
    """Name of the active Triton backend: 'npu' on Ascend, 'cuda'/'hip' on GPU."""
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        if _HAS_TORCH_NPU:
            # triton-ascend is installed but the driver failed to report a
            # target (e.g. device busy); torch_npu still gives a usable device.
            return 'npu'
        warnings.warn('Triton is not supported on current platform, roll back to CPU.', stacklevel=2)
        return 'cpu'


@cache
def get_multiprocessor_count(tensor_idx: int = 0, *, use_aicore: bool = False) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']
    except Exception:
        # Maybe we use a NPU device.
        try:
            if triton.runtime.driver.active.get_current_target().backend == 'npu':
                props = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)
                return props['num_aicore'] if use_aicore else props['num_vectorcore']
        except Exception:
            logger.debug('Failed to get NPU multiprocessor count, falling back to 1.', exc_info=True)
        return 1


device_platform = get_available_device()
# For AMD GPUs, the triton backend is 'hip' while the torch backend is 'cuda'.
device = 'cuda' if device_platform == 'hip' else device_platform
device_torch_lib = getattr(torch, device)
device_name = device

IS_NPU = (device_platform == 'npu')
IS_NVIDIA = (device_platform == 'cuda')


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except Exception:
        warnings.warn('Triton is not supported on current platform, roll back to CPU.', stacklevel=2)
        return [-1]


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """GPU shared-memory tier check, kept with the upstream name and signature.

    On Ascend NPU this always returns False: the UB budget is far below the GPU
    shared-memory tiers, so callers deliberately take the conservative tile path
    (BV <= 64, BS <= 32, small num_warps). The upstream test-suite monkeypatches
    this function by name inside ``fla.ops.wall_attn.parallel`` when it needs a
    specific tile path, so the symbol must stay importable and patchable.
    """
    if IS_NPU:
        return False
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


if not check_pytorch_version('2.4'):
    raise RuntimeError('fla (Ascend vendor) requires PyTorch >= 2.4 (target env: torch 2.9.0 + torch_npu).')

autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)


def custom_device_ctx(index: int):
    if index is None:
        return contextlib.nullcontext()
    try:
        return device_torch_lib.device(index)
    except (AttributeError, AssertionError, RuntimeError):
        return contextlib.nullcontext()
