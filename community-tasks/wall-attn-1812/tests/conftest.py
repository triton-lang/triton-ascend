# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Pytest bootstrap for the Wall Attention Ascend task workspace:
#   1. put <task>/kernel on sys.path so that `import fla...` in the (unmodified)
#      upstream test-suite resolves to the vendored Ascend implementation;
#   2. import torch_npu early so `torch.npu` is registered before any test
#      touches the NPU (silently skipped where torch_npu is unavailable).

import os
import sys

_KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'kernel'))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
