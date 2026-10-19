# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the MIT License. See LICENSE for details.
"""Online softmax with native NZ score/probability layouts."""
import triton
import triton.language as tl
from .custom_op import custom


@triton.jit
def online_softmax_nz(scores, m, causal_limits=None, mask_required=None):
    """Return (FP8 NZ32 P, FP16 m_new, FP32 block_sum) for scaled FP16 NZ16 [512/16,M/16,16,16].
    M=32/64/128; optional int32[M] causal_limits are inclusive local column limits.
    mask_required=False requires all limits>=511; callers widen m/m_new before alpha and own l updates."""
    i = tl.arange(0, 256)
    indices = ((i // 32) * 32 + (i % 16) * 2 + (i % 32) // 16).to(tl.uint8)
    rows: tl.constexpr = scores.shape[1] * 16
    outputs = (tl.full((scores.shape[0] // 2, rows // 16, 16, 32), 0, tl.float8e4nv),
               tl.full((rows,), 0, scores.dtype),
               tl.full((rows,), 0, tl.float32))
    return custom("__builtin_online_softmax_nz", scores, m, indices, causal_limits, mask_required, out=outputs)
