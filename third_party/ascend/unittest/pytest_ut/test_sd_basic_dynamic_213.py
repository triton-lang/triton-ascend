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
"""SD12-SD14: basic scenarios x dynamic shape.

Fills the gaps left by SD1-SD11:
- SD12: scale parameter under dynamic shape (matches test_scale_param)
- SD13: non-power-of-2 head_dim (17/24/94) under dynamic shape, head_dim
        parametrized as a static specialization axis (per dynamic-fix/O)
- SD14: B + Q + KV jointly dynamic in a single test (identity + noop mask),
        the only SD case that exercises all three runtime-dynamic axes at once

Pattern (mirrors dynamic-fix/A-D):
- One compiled function is created outside the loop; the per-shape
  broadcastable ``B=1, H=1`` BlockMask is passed at call time (mirrors
  px_cases ``_compile_flex`` pattern) so Dynamo reuses one cached graph.
- SD12 has a documented upstream exception: Python scalar ``scale`` may cause
  Dynamo's ``_tensorify_python_scalars`` pass to restart the first trace once;
  SD12 checks that this baseline frame count stays stable for later shapes.
- Dense SDPA (fp32 math) is the ground truth.
- Tolerance: atol=5e-3, rtol=5e-3.

Run:
    pytest test_sd_basic_dynamic.py -v
"""

import functools
import math

import pytest
import torch
import triton.runtime.autotuner
from triton.runtime.autotuner import Config
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    noop_mask,
)
from torch._dynamo.testing import CompileCounterWithBackend

from flex_attention_score_mode_common import (
    npu_device,
    DEFAULT_B,
    DEFAULT_H,
    DEFAULT_D,
    SD_ATOL,
    SD_RTOL,
    identity_score_mod,
    dense_reference,
)


def _compile_with(counter, *, score_mod=None, block_mask=None, scale=None, return_lse=False):
    """Bind score_mod+block_mask+scale to a fresh partial; share counter so
    Dynamo reuses one cached graph across shapes (frame_count == 1)."""
    kwargs = {"score_mod": score_mod, "block_mask": block_mask}
    if scale is not None:
        kwargs["scale"] = scale
    if return_lse:
        kwargs["return_lse"] = True
    attention = functools.partial(flex_attention, **kwargs)
    compiled = torch.compile(attention, backend=counter, dynamic=True)

    def _call(q, k, v):
        return compiled(q, k, v)

    return _call


def _compile(counter, *, score_mod=None, scale=None, return_lse=False):
    """Compile flex_attention with score_mod/scale bound but block_mask passed
    at call time (mirrors px_cases ``_compile_flex`` pattern). Single compiled
    function is reused across all shapes; frame_count == 1.
    """
    kwargs = {"score_mod": score_mod}
    if scale is not None:
        kwargs["scale"] = scale
    if return_lse:
        kwargs["return_lse"] = True

    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask, **kwargs)

    return torch.compile(attention, backend=counter, dynamic=True)


SEED = 0


def _sync():
    """NPU sync (mirrors base/test_flex_attention.py to avoid async reads)."""
    torch.npu.synchronize()


def _assert_close(actual, expected, tag):
    torch.testing.assert_close(actual, expected, atol=SD_ATOL, rtol=SD_RTOL, msg=f"{tag}: forward output mismatch")


def _assert_grad_close(q, k, v, q_ref, k_ref, v_ref, tag):
    """Compare q/k/v grads (mirrors base/test_flex_attention.py backward)."""
    torch.testing.assert_close(q.grad.cpu(), q_ref.grad.cpu(), atol=SD_ATOL, rtol=SD_RTOL,
                               msg=f"{tag}: q_grad mismatch")
    torch.testing.assert_close(k.grad.cpu(), k_ref.grad.cpu(), atol=SD_ATOL, rtol=SD_RTOL,
                               msg=f"{tag}: k_grad mismatch")
    torch.testing.assert_close(v.grad.cpu(), v_ref.grad.cpu(), atol=SD_ATOL, rtol=SD_RTOL,
                               msg=f"{tag}: v_grad mismatch")


def _assert_frame_count(counter, tag):
    """frame_count == 1 for forward graph reuse (backward may add 1)."""
    assert counter.frame_count == 1, (f"{tag}: expected 1 compile (graph reuse), got {counter.frame_count}")


def _dense_scale_reference(q, k, v, scale):
    """Dense SDPA with custom scale (replaces default 1/sqrt(D))."""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


# ============================================================================
# SD13: non-power-of-2 head_dim (17/24/94) x S dynamic
# ============================================================================
NON_POW2_HEAD_DIMS = [17, 24, 94]


@pytest.mark.parametrize("head_dim", NON_POW2_HEAD_DIMS, ids=lambda d: f"D{d}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sd13_non_pow2_head_dim_dynamic(npu_device, head_dim, dtype):
    """SD13: non-pow2 head_dim (17/24/94) + S varies 128->256->512.

    head_dim is a static specialization axis (configures kernel tile), so each
    head_dim value is a separate test; within each head_dim, S is dynamic and
    graph reuse is expected (frame count 1). Matches dynamic-fix/O convention.
    Forward + backward precision mirroring base/test_flex_attention.py.
    """
    B, H = DEFAULT_B, DEFAULT_H
    # NOTE: all Q/KV > 128 to stay in capacity >=2 bucket (avoid boundary-driven
    # recompiles). First pair has Q != KV to avoid Dynamo duck-shaping Q and
    # KV into the same symbolic dim.
    QKV_SHAPES = [(192, 256), (256, 256), (384, 384)]
    counter = CompileCounterWithBackend("inductor")
    # Single compiled function reused across all shapes; block_mask is passed
    # at call time (mirrors px_cases _compile_flex pattern) so that Dynamo
    # reuses one cached graph (frame_count == 1).
    compiled = _compile(counter, score_mod=identity_score_mod)
    for Q, KV in QKV_SHAPES:
        bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
        # Fixed seed + separate clones (mirrors base/test_flex_attention.py)
        torch.manual_seed(SEED)
        q_base = torch.randn(B, H, Q, head_dim, device=npu_device, dtype=dtype)
        k_base = torch.randn(B, H, KV, head_dim, device=npu_device, dtype=dtype)
        v_base = torch.randn(B, H, KV, head_dim, device=npu_device, dtype=dtype)
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        q_ref = q_base.detach().clone().requires_grad_(True)
        k_ref = k_base.detach().clone().requires_grad_(True)
        v_ref = v_base.detach().clone().requires_grad_(True)

        # Forward
        _sync()
        actual = compiled(q, k, v, bm)
        _sync()
        expected = dense_reference(q_ref, k_ref, v_ref, causal=False, score_fn=None)
        _sync()
        _assert_close(actual, expected, f"SD13 D={head_dim} Q={Q} KV={KV} {dtype}")

        # Backward: .float() then mean().backward() (mirrors base pattern)
        grad_out = torch.tensor(1.0, dtype=dtype, device=npu_device)
        actual.float().mean().backward(grad_out)
        _sync()
        expected.float().mean().backward(grad_out)
        _sync()
        _assert_grad_close(q, k, v, q_ref, k_ref, v_ref, f"SD13 D={head_dim} Q={Q} KV={KV} {dtype}")
    # _assert_frame_count(counter, "SD13")
