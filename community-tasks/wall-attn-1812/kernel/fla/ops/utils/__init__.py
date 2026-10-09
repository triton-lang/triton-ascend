# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Vendored from upstream fla/ops/utils/__init__.py, reduced to the exports
# required by fla.ops.wall_attn.

from .cumsum import chunk_global_cumsum
from .index import prepare_chunk_indices

__all__ = [
    'chunk_global_cumsum',
    'prepare_chunk_indices',
]
