# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Minimal vendored subset of upstream fla (fla-org/flash-linear-attention),
# adapted for Ascend NPU with triton-ascend, for community task #1812
# (Wall Attention Ascend 适配). See community-tasks/wall-attn-1812/docs/design.md.
# Only the modules required by `fla.ops.wall_attn` and its upstream test-suite
# `tests/ops/test_wall_attn.py` are provided, keeping upstream import paths so
# the test file stays byte-identical.

__version__ = '0.1.0+ascend'
