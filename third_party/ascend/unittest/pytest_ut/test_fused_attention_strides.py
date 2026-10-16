# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch_npu


@pytest.fixture(scope="module", params=["tutorial", "unittest", "legacy_unittest"])
def attention_module(request):
    ascend_dir = Path(__file__).resolve().parents[2]
    if request.param == "tutorial":
        path = ascend_dir / "tutorials" / "06-fused-attention.py"
    elif request.param == "unittest":
        path = Path(__file__).with_name("test_06_fused_attention.py")
    else:
        path = Path(__file__).with_name("test_fusedattention.py")
    name = f"ascend_fused_attention_{request.param}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _with_layout(data, layout):
    z, h, n, d = data.shape
    if layout == "head_gap":
        backing = torch.full((z, h, 2 * n, d), 9.0, device="npu", dtype=data.dtype)
        result = backing[:, :, :n, :]
    elif layout == "batch_gap":
        backing = torch.full((z, 2 * h, n, d), 9.0, device="npu", dtype=data.dtype)
        result = backing[:, :h, :, :]
    elif layout == "transposed":
        result = torch.empty((z, n, h, d), device="npu", dtype=data.dtype).permute(0, 2, 1, 3)
    else:
        return data.to("npu")
    result.copy_(data)
    return result


def _inputs(z, h, d, dtype, n_ctx=64):
    generator = torch.Generator().manual_seed(20)
    shape = (z, h, n_ctx, d)
    return [torch.randn(shape, generator=generator).to(dtype) * 0.5 for _ in range(3)]


def _reference(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal,
                                                            scale=0.5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("z,h,q_layout,k_layout,v_layout", [
    pytest.param(2, 2, "contiguous", "contiguous", "contiguous", id="contiguous"),
    pytest.param(1, 2, "head_gap", "contiguous", "contiguous", id="q_head"),
    pytest.param(2, 1, "batch_gap", "contiguous", "contiguous", id="q_batch"),
    pytest.param(1, 2, "contiguous", "head_gap", "contiguous", id="k_head"),
    pytest.param(2, 1, "contiguous", "batch_gap", "contiguous", id="k_batch"),
    pytest.param(1, 2, "contiguous", "contiguous", "head_gap", id="v_head"),
    pytest.param(2, 1, "contiguous", "contiguous", "batch_gap", id="v_batch"),
    pytest.param(2, 2, "batch_gap", "head_gap", "transposed", id="independent_layouts"),
])
def test_attention_input_strides(attention_module, causal, dtype, z, h, q_layout, k_layout, v_layout):
    q_cpu, k_cpu, v_cpu = _inputs(z, h, 16, dtype)
    q, k, v = [
        _with_layout(data, layout) for data, layout in zip((q_cpu, k_cpu, v_cpu), (q_layout, k_layout, v_layout))
    ]
    actual = attention_module.attention(q, k, v, causal, 0.5, 64, 64)
    # Gapped Q is compacted by empty_like, so this also tests different output strides.
    torch.testing.assert_close(actual.cpu().float(), _reference(q_cpu, k_cpu, v_cpu, causal), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["head_gap", "batch_gap"])
def test_attention_output_strides(attention_module, causal, dtype, layout):
    q_cpu, k_cpu, v_cpu = _inputs(2, 2, 16, dtype)
    q, k, v = [data.to("npu") for data in (q_cpu, k_cpu, v_cpu)]
    backing_shape = (2, 2, 128, 16) if layout == "head_gap" else (2, 4, 64, 16)
    backing = torch.full(backing_shape, -99.0, device="npu", dtype=dtype)
    out = backing[:, :, :64, :] if layout == "head_gap" else backing[:, :2, :, :]
    acc = torch.zeros(q.shape, device="npu", dtype=torch.float32)
    m = torch.empty(q.shape[:3], device="npu", dtype=torch.float32)
    attention_module._attn_fwd[(20, )](q, k, v, m, out, acc, 0.5, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
                                       2, 2, N_CTX=64, HEAD_DIM=16, BLOCK_M=64, BLOCK_N=64, STAGE=3 if causal else 1)
    torch.testing.assert_close(out.cpu().float(), _reference(q_cpu, k_cpu, v_cpu, causal), atol=1e-2, rtol=1e-2)
    padding = backing.cpu()[:, :, 64:, :] if layout == "head_gap" else backing.cpu()[:, 2:, :, :]
    torch.testing.assert_close(padding, torch.full_like(padding, -99.0), atol=0, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_attention_scratch_independent_of_q_layout(attention_module, causal, dtype):
    q_cpu, k_cpu, v_cpu = _inputs(2, 2, 256, dtype)
    q = _with_layout(q_cpu, "transposed")
    k, v = k_cpu.to("npu"), v_cpu.to("npu")
    # Transposed Q has stride_qh < stride_qm; integer division must not alias scratch heads.
    # Full-sequence tiles isolate layout indexing from the existing D256 multi-tile accumulation failure.
    actual = attention_module.attention(q, k, v, causal, 0.5, 64, 64)
    torch.testing.assert_close(actual.cpu().float(), _reference(q_cpu, k_cpu, v_cpu, causal), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("q_layout", ["contiguous", "transposed"])
def test_attention_scratch_multiple_query_tiles(attention_module, dtype, q_layout):
    q_cpu, k_cpu, v_cpu = _inputs(2, 2, 256, dtype, n_ctx=128)
    q = _with_layout(q_cpu, q_layout)
    k, v = k_cpu.to("npu"), v_cpu.to("npu")
    # Two query tiles check the scratch row offset independently of the single K/V tile.
    # BLOCK_N > BLOCK_M is supported by the noncausal stage only.
    actual = attention_module.attention(q, k, v, False, 0.5, 64, 128)
    torch.testing.assert_close(actual.cpu().float(), _reference(q_cpu, k_cpu, v_cpu, False), atol=1e-2, rtol=1e-2)
