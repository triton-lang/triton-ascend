# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""Shared fixtures and helpers for score_mode x dynamic-shape tests.

Reuses the dense SDPA reference and NPU monkey-patch pattern from
``dynamic-fix/test_flex_attention_dynamic_shape.py`` but kept self-contained
so the score_mode directory can run independently.

Precision: atol=5e-3, rtol=5e-3 per project requirement (fp32 baseline;
fp16/bf16 may be looser per-dtype but capped at 5e-3 here unless the per-test
override is necessary).
"""
import math

import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    noop_mask,
)

# NPU device validation monkey-patch: bypass flex_attention's CUDA/CPU-only
# check so eager reference calls work on NPU. torch_npu doesn't patch this
# upstream across all versions; this also covers Dynamo tracing paths because
# the patch happens before any torch.compile call in the test files.
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass

# Project-wide tolerance for score_mode x dynamic-shape tests.
SD_ATOL = 5e-3
SD_RTOL = 5e-3


@pytest.fixture(scope="session")
def npu_device():
    assert torch.npu.is_available(), "NPU is required for the Flex Attention tests"
    return "npu"


@pytest.fixture(autouse=True)
def _reset_dynamo_cache():
    """Clear Dynamo global cache before each test to prevent cross-test
    cache pollution (which can starve CompileCounterWithBackend.frame_count
    and produce false ``frame_count == 0``).
    """
    import torch._dynamo
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ----------------------------------------------------------------------------
# Static dimensions (kept fixed across runtime shapes; per A/B/C/D convention
# B>=2 to avoid Dynamo 0/1 specialization, H is a structural static dim).
# ----------------------------------------------------------------------------
DEFAULT_B = 2
DEFAULT_H = 8
DEFAULT_D = 64


# ----------------------------------------------------------------------------
# mask_mod helpers (same signatures as dynamic-fix shared module)
# ----------------------------------------------------------------------------
def causal_mask_mod(b, h, m, n):
    return m >= n


def windowed_mask_mod_factory(offset):

    def _mask(b, h, m, n):
        return (m - n).abs() <= offset

    return _mask


def head_dependent_mask_mod(b, h, m, n):
    # Odd heads use causal, even heads use full attention.
    return (h % 2 == 0) | (m >= n)


def full_masked_half_mod(b, h, m, n):
    # First half of Q is unmasked, second half fully masked -> NaN/zero check.
    M = None  # patched per-shape at call site via closure
    return m < M


def noop_mask_mod(b, h, m, n):
    return noop_mask(b, h, m, n)


# ----------------------------------------------------------------------------
# score_mod helpers (same signatures as dynamic-fix + advanced file)
# ----------------------------------------------------------------------------
def identity_score_mod(score, b, h, m, n):
    return score


def causal_score_mod(score, b, h, m, n):
    return torch.where(m >= n, score, float("-inf"))


def rel_bias_score_mod(score, b, h, m, n):
    return score + (m - n)


def alibi_bias_score_mod(score, b, h, m, n):
    scale = torch.exp2(-((h + 1) * 8.0 / 16))
    return score + (n - m) * scale


def composed_rel_causal_score_mod(score, b, h, m, n):
    # rel_bias then causal mask.
    score = score + (m - n)
    return torch.where(m >= n, score, float("-inf"))


def make_captured_buffer_score_mod(head_offset):
    """Return score_mod closure capturing an external head_offset tensor."""

    def _score_mod(score, b, h, m, n):
        return score + head_offset[h]

    return _score_mod


# ----------------------------------------------------------------------------
# Dense SDPA reference (mirrors dynamic-fix/_dense_reference but extends
# score_fn signature so head-dependent scores can be expressed).
#
# score_fn(scores, q_idx, kv_idx) -> scores
#   - scores: [B, H, Q, KV] float32
#   - q_idx: [Q, 1] long tensor
#   - kv_idx: [1, KV] long tensor
# mask_fn(b, h, q_idx, kv_idx) -> bool [Q, KV]  (sampled at b=0, h per-head)
# ----------------------------------------------------------------------------
def dense_reference(q, k, v, *, causal=False, score_fn=None, mask_fn=None):
    """Dense SDPA reference. mask_fn must NOT be head-dependent (b/h unused);
    for head-dependent masks use ``dense_reference_head_dependent_mask``.
    """
    q_len = q.shape[-2]
    kv_len = k.shape[-2]
    q_idx = torch.arange(q_len, device=q.device, dtype=torch.float32)[:, None]
    kv_idx = torch.arange(kv_len, device=k.device, dtype=torch.float32)[None, :]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        scores = scores.masked_fill(q_idx < kv_idx, float("-inf"))
    if score_fn is not None:
        scores = score_fn(scores, q_idx, kv_idx)
    if mask_fn is not None:
        b_idx = torch.zeros(1, dtype=torch.long, device=q.device)
        h_idx = torch.zeros(1, dtype=torch.long, device=q.device)
        keep = mask_fn(b_idx, h_idx, q_idx.long(), kv_idx.long())  # [Q, KV]
        keep_all = keep.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, KV] -> broadcast
        scores = scores.masked_fill(~keep_all, float("-inf"))
    # Rows that are fully masked (all -inf) produce NaN from softmax.
    # Flex attention outputs 0 for such rows, so match that behavior.
    fully_masked = (scores == float("-inf")).all(dim=-1, keepdim=True)
    attn_out = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    # Manually zero-out rows that had no valid KV positions.
    attn_out = attn_out.masked_fill(fully_masked, 0.0)
    return attn_out.to(q.dtype)


def dense_reference_head_dependent_mask(q, k, v, mask_fn, *, causal=False, score_fn=None):
    """Dense ref that honors head-dependent mask_fn(b, h, q, kv) by iterating heads."""
    q_len = q.shape[-2]
    kv_len = k.shape[-2]
    B, H = q.shape[0], q.shape[1]
    q_idx = torch.arange(q_len, device=q.device, dtype=torch.float32)[:, None]
    kv_idx = torch.arange(kv_len, device=k.device, dtype=torch.float32)[None, :]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        scores = scores.masked_fill(q_idx < kv_idx, float("-inf"))
    if score_fn is not None:
        scores = score_fn(scores, q_idx, kv_idx)
    b_idx = torch.zeros(1, dtype=torch.long, device=q.device)
    keep_per_head = []
    for h in range(H):
        h_idx = torch.tensor([h], dtype=torch.long, device=q.device)
        keep = mask_fn(b_idx, h_idx, q_idx.long(), kv_idx.long())  # [Q, KV]
        keep_per_head.append(keep)
    keep_all = torch.stack(keep_per_head, dim=0).unsqueeze(0)  # [1, H, Q, KV]
    scores = scores.masked_fill(~keep_all, float("-inf"))
    fully_masked = (scores == float("-inf")).all(dim=-1, keepdim=True)
    attn_out = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    attn_out = attn_out.masked_fill(fully_masked, 0.0)
    return attn_out.to(q.dtype)


# ----------------------------------------------------------------------------
# Flex attention eager reference (uses monkey-patched _validate_device).
# Used as ground truth when dense SDPA cannot express head-dependent score_mod
# (e.g., alibi bias can be expressed in dense ref; captured_buffer is also
# expressible). Kept for fallback only.
# ----------------------------------------------------------------------------
def eager_flex_reference(q, k, v, *, score_mod=None, block_mask=None):
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)


# ----------------------------------------------------------------------------
# Compile-counter call helper for frame_count assertions.
# ----------------------------------------------------------------------------
def make_compiled(counter, *, score_mod=None, block_mask=None, enable_gqa=False, return_lse=False, scale=None):
    """Build a compiled flex_attention callable sharing one backend."""
    kwargs = {}
    if score_mod is not None:
        kwargs["score_mod"] = score_mod
    if block_mask is not None:
        kwargs["block_mask"] = block_mask
    if enable_gqa:
        kwargs["enable_gqa"] = True
    if return_lse:
        kwargs["return_lse"] = True
    if scale is not None:
        kwargs["scale"] = scale
    fn = functools_partial(flex_attention, **kwargs)
    return torch.compile(fn, backend=counter, dynamic=True)


def functools_partial(fn, **kwargs):
    import functools
    return functools.partial(fn, **kwargs)
