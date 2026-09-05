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

import json
import os

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver
from triton.backends.ascend.utils import is_compile_on_910_95

simd_simt_910_95_only = pytest.mark.xfail(
    not is_compile_on_910_95(),
    reason="SIMD/SIMT cost model only supports 910_95",
    run=False,
)

_BLOCK_X = 64
# The same shape family used by the original scalar-dominated Megablocks tests.
_SL, _HS, _NE, _TOP_K = 1024, 1536, 128, 4
_EXPERT_CAPACITY = (_SL * _TOP_K) // _NE


def _vector_core_count():
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(properties["num_vectorcore"])


def _load_route_report(path, expected):
    report = json.loads(path.read_text())
    assert report["stage_model"]["applied"]
    assert report["effective_decision_kind"] == expected
    return report


def _launch_options(report_path, logical_programs):
    options = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "enable_auto_blockify": True,
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": _vector_core_count(),
    }
    if os.getenv("TRITON_TEST_DISABLE_TTIR_LAYOUT_MERGE") == "1":
        options["enable_ttir_layout_merge"] = False
    return options


# ---------------------------------------------------------------------------
# Kernels copied from the scalar-dominated Megablocks cases.  The autotune
# layer is intentionally removed so a cost-model UT launches one fixed config.
# ---------------------------------------------------------------------------
@triton.jit
def _padded_copy_gather(
    a,
    b,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    index_a = tl.load(indices + tl.program_id(0))
    bin_idx = tl.load(bin_ids + tl.program_id(0))
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    scale = tl.load(weights + index_a) if SCALE else 1
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)
        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)
        offsets += BLOCK_X


@triton.jit
def _padded_copy_scatter(
    a,
    b,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    index_a = tl.load(indices + tl.program_id(0))
    bin_idx = tl.load(bin_ids + tl.program_id(0))
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    scale = tl.load(weights + index_a) if SCALE else 1
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)
        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)
        offsets += BLOCK_X


@triton.jit
def _padded_copy_wgrad(
    x,
    grad,
    wgrad,
    indices,
    bin_ids,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    index_out = tl.load(indices + tl.program_id(0))
    bin_idx = tl.load(bin_ids + tl.program_id(0))
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)
    index_x = offset_in_bin
    if bin_idx > 0:
        index_x += tl.load(padded_bins + bin_idx - 1)
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    acc = tl.zeros((BLOCK_X, ), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


@triton.jit
def _binned_copy_gather(
    a,
    b,
    num_experts,
    expert_capacity,
    indices,
    weights,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)
    index_b = expert_idx * expert_capacity + entry_idx
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx)
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    scale = tl.load(weights + index_a) if SCALE else 1
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)
        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)
        offsets += BLOCK_X


@triton.jit
def _binned_copy_scatter(
    a,
    b,
    num_experts,
    expert_capacity,
    indices,
    weights,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)
    index_b = expert_idx * expert_capacity + entry_idx
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx)
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    scale = tl.load(weights + index_a) if SCALE else 1
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)
        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)
        offsets += BLOCK_X


@triton.jit
def _binned_copy_wgrad(
    x,
    grad,
    wgrad,
    num_experts,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)
    index_x = expert_idx * expert_capacity + entry_idx
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start
    if entry_idx >= num_tokens:
        return
    index_out = tl.load(indices + start + entry_idx)
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    acc = tl.zeros((BLOCK_X, ), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


# ---------------------------------------------------------------------------
# Deterministic CPU data generation and references.
# ---------------------------------------------------------------------------
def _make_expert_data(seed):
    torch.manual_seed(seed)
    x_cpu = torch.randn((_SL, _HS), dtype=torch.float16)
    top_expert = torch.randint(0, _NE, (_SL * _TOP_K, ), dtype=torch.int32)
    bin_ids_cpu, indices_cpu = torch.sort(top_expert)
    tokens_per_expert = torch.bincount(top_expert, minlength=_NE)[:_NE].to(torch.int32)
    bins_cpu = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    padded_tokens_per_expert = torch.div(tokens_per_expert + 127, 128, rounding_mode="trunc") * 128
    padded_bins_cpu = torch.cumsum(padded_tokens_per_expert, dim=0).to(torch.int32)
    weights_cpu = torch.rand((_SL * _TOP_K, ), dtype=torch.float16)
    grads_cpu = torch.randn((_SL, _HS), dtype=torch.float16)
    return (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu)


def _to_npu(tensors):
    return tuple(tensor.contiguous().to("npu") for tensor in tensors)


def _padded_gather_reference(x, indices, bins, padded_bins, top_k):
    out = torch.zeros((padded_bins[-1].item(), x.shape[1]), dtype=x.dtype)
    for i in range(bins.numel()):
        start = 0 if i == 0 else bins[i - 1].item()
        end = bins[i].item()
        if start == end:
            continue
        out_start = 0 if i == 0 else padded_bins[i - 1].item()
        rows = indices[start:end] // top_k
        out[out_start:out_start + (end - start)] = x[rows]
    return out


def _padded_scatter_reference(gathered, indices, weights, bins, padded_bins, top_k):
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, gathered.shape[1]), dtype=torch.float32)
    for i in range(bins.numel()):
        start = 0 if i == 0 else bins[i - 1].item()
        end = bins[i].item()
        if start == end:
            continue
        in_idx = 0 if i == 0 else padded_bins[i - 1].item()
        n = end - start
        src = gathered[in_idx:in_idx + n].float()
        dst_pos = indices[start:end]
        dst = dst_pos // top_k
        scale = weights[dst_pos].float()
        out.index_add_(0, dst, src * scale[:, None])
    return out


def _padded_wgrad_reference(gathered, grads, indices, bins, padded_bins, top_k):
    out = torch.zeros((indices.shape[0], ), dtype=torch.float32)
    for i in range(bins.numel()):
        start = 0 if i == 0 else bins[i - 1].item()
        end = bins[i].item()
        if start == end:
            continue
        in_idx = 0 if i == 0 else padded_bins[i - 1].item()
        n = end - start
        dst_pos = indices[start:end]
        dst = dst_pos // top_k
        src = gathered[in_idx:in_idx + n].float()
        vals = (src * grads[dst].float()).sum(dim=1)
        out[dst_pos] = vals
    return out


def _binned_gather_reference(x, indices, bins, expert_capacity, top_k):
    ne = bins.numel()
    out = torch.zeros((ne, expert_capacity, x.shape[1]), dtype=x.dtype)
    for i in range(ne):
        start = 0 if i == 0 else bins[i - 1].item()
        end = bins[i].item()
        n = min(expert_capacity, end - start)
        if n == 0:
            continue
        rows = indices[start:start + n] // top_k
        out[i, :n, :] = x[rows]
    return out


def _binned_scatter_reference(gathered, indices, weights, bins, top_k):
    tokens = indices.shape[0] // top_k
    expert_capacity = gathered.shape[1]
    out = torch.zeros((tokens, gathered.shape[2]), dtype=torch.float32)
    for i in range(bins.numel()):
        start = 0 if i == 0 else bins[i - 1].item()
        end = bins[i].item()
        n = min(expert_capacity, end - start)
        if n == 0:
            continue
        dst_pos = indices[start:start + n]
        dst = dst_pos // top_k
        scale = weights[dst_pos].float()
        src = gathered[i, :n, :].float()
        out.index_add_(0, dst, src * scale[:, None])
    return out


def _binned_wgrad_reference(gathered, grads, indices, bins, top_k):
    out = torch.zeros((indices.shape[0], ), dtype=torch.float32)
    expert_capacity = gathered.shape[1]
    for i in range(bins.numel()):
        start = 0 if i == 0 else bins[i - 1].item()
        end = bins[i].item()
        n = min(expert_capacity, end - start)
        if n == 0:
            continue
        dst_pos = indices[start:start + n]
        dst = dst_pos // top_k
        src = gathered[i, :n, :].float()
        vals = (src * grads[dst].float()).sum(dim=1)
        out[dst_pos] = vals
    return out


# ---------------------------------------------------------------------------
# Cost-model regression cases: all six scalar-dominated Megablocks kernels
# must be selected as all_simt_only when AutoBlockify V1 is enabled.
#
# These tests intentionally focus on route selection plus numerical correctness.
# Wall-clock performance assertions are not encoded yet because this source tree
# does not contain per-kernel median baselines for these six operators.
# ---------------------------------------------------------------------------
@simd_simt_910_95_only
def test_costmodel_padded_copy_gather(tmp_path):
    (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu) = _make_expert_data(1)
    x, indices, bin_ids, bins, padded_bins, _, _ = _to_npu(
        (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu))
    del grads_cpu
    expected = _padded_gather_reference(x_cpu, indices_cpu, bins_cpu, padded_bins_cpu, _TOP_K)
    report_path = tmp_path / "padded_copy_gather_route.json"
    output = torch.zeros((padded_bins_cpu[-1].item(), x.shape[1]), dtype=x.dtype, device=x.device)
    logical_programs = indices.shape[0]

    def launch():
        _padded_copy_gather[(logical_programs, )](
            x,
            output,
            indices,
            bin_ids,
            None,
            bins,
            padded_bins,
            NUM_COLUMNS=_HS,
            TOP_K=_TOP_K,
            BLOCK_X=_BLOCK_X,
            A_TO_B=True,
            SCALE=False,
            **_launch_options(report_path, logical_programs),
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), expected.float(), rtol=1e-2, atol=1e-2)
    _load_route_report(report_path, "all_simt_only")


@simd_simt_910_95_only
def test_costmodel_padded_copy_scatter(tmp_path):
    (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu) = _make_expert_data(2)
    gathered_cpu = _padded_gather_reference(x_cpu, indices_cpu, bins_cpu, padded_bins_cpu, _TOP_K)
    gathered, indices, bin_ids, bins, padded_bins, weights, _ = _to_npu(
        (gathered_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu))
    del x_cpu, grads_cpu
    expected = _padded_scatter_reference(gathered_cpu, indices_cpu, weights_cpu, bins_cpu, padded_bins_cpu, _TOP_K)
    report_path = tmp_path / "padded_copy_scatter_route.json"
    output = torch.empty((_SL, _TOP_K, _HS), dtype=gathered.dtype, device=gathered.device)
    logical_programs = indices.shape[0]

    def launch():
        _padded_copy_scatter[(logical_programs, )](
            output,
            gathered,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            NUM_COLUMNS=_HS,
            TOP_K=_TOP_K,
            BLOCK_X=_BLOCK_X,
            A_TO_B=False,
            SCALE=True,
            **_launch_options(report_path, logical_programs),
        )

    launch()
    torch.npu.synchronize()
    result = output.sum(dim=1)
    torch.testing.assert_close(result.cpu().float(), expected, rtol=1e-2, atol=1e-2)
    _load_route_report(report_path, "all_simt_only")


@simd_simt_910_95_only
def test_costmodel_padded_copy_wgrad(tmp_path):
    (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu) = _make_expert_data(3)
    gathered_cpu = _padded_gather_reference(x_cpu, indices_cpu, bins_cpu, padded_bins_cpu, _TOP_K)
    gathered, indices, bin_ids, bins, padded_bins, _, grads = _to_npu(
        (gathered_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu))
    del x_cpu, weights_cpu
    expected = _padded_wgrad_reference(gathered_cpu, grads_cpu, indices_cpu, bins_cpu, padded_bins_cpu, _TOP_K)
    report_path = tmp_path / "padded_copy_wgrad_route.json"
    output = torch.empty((indices.shape[0], ), dtype=gathered.dtype, device=gathered.device)
    logical_programs = indices.shape[0]

    def launch():
        _padded_copy_wgrad[(logical_programs, )](
            gathered,
            grads,
            output,
            indices,
            bin_ids,
            bins,
            padded_bins,
            NUM_COLUMNS=_HS,
            TOP_K=_TOP_K,
            BLOCK_X=_BLOCK_X,
            **_launch_options(report_path, logical_programs),
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), expected, rtol=1e-2, atol=1e-2)
    _load_route_report(report_path, "all_simt_only")


@simd_simt_910_95_only
def test_costmodel_binned_copy_gather(tmp_path):
    (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu) = _make_expert_data(4)
    x, indices, _, bins, _, _, _ = _to_npu(
        (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu))
    del bin_ids_cpu, padded_bins_cpu, weights_cpu, grads_cpu
    expected = _binned_gather_reference(x_cpu, indices_cpu, bins_cpu, _EXPERT_CAPACITY, _TOP_K)
    report_path = tmp_path / "binned_copy_gather_route.json"
    output = torch.zeros((_NE, _EXPERT_CAPACITY, x.shape[1]), dtype=x.dtype, device=x.device)
    logical_programs = _NE * _EXPERT_CAPACITY

    def launch():
        _binned_copy_gather[(_NE, _EXPERT_CAPACITY)](
            x,
            output,
            _NE,
            _EXPERT_CAPACITY,
            indices,
            None,
            bins,
            NUM_COLUMNS=_HS,
            TOP_K=_TOP_K,
            BLOCK_X=_BLOCK_X,
            A_TO_B=True,
            SCALE=False,
            **_launch_options(report_path, logical_programs),
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), expected.float(), rtol=1e-2, atol=1e-2)
    _load_route_report(report_path, "all_simt_only")


@simd_simt_910_95_only
def test_costmodel_binned_copy_scatter(tmp_path):
    (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu) = _make_expert_data(5)
    gathered_cpu = _binned_gather_reference(x_cpu, indices_cpu, bins_cpu, _EXPERT_CAPACITY, _TOP_K)
    gathered, indices, _, bins, _, weights, _ = _to_npu(
        (gathered_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu))
    del x_cpu, bin_ids_cpu, padded_bins_cpu, grads_cpu
    expected = _binned_scatter_reference(gathered_cpu, indices_cpu, weights_cpu, bins_cpu, _TOP_K)
    report_path = tmp_path / "binned_copy_scatter_route.json"
    output = torch.zeros((_SL, _TOP_K, _HS), dtype=gathered.dtype, device=gathered.device)
    logical_programs = _NE * _EXPERT_CAPACITY

    def launch():
        _binned_copy_scatter[(_NE, _EXPERT_CAPACITY)](
            output,
            gathered,
            _NE,
            _EXPERT_CAPACITY,
            indices,
            weights,
            bins,
            NUM_COLUMNS=_HS,
            TOP_K=_TOP_K,
            BLOCK_X=_BLOCK_X,
            A_TO_B=False,
            SCALE=True,
            **_launch_options(report_path, logical_programs),
        )

    launch()
    torch.npu.synchronize()
    result = output.sum(dim=1)
    torch.testing.assert_close(result.cpu().float(), expected, rtol=1e-2, atol=1e-2)
    _load_route_report(report_path, "all_simt_only")


@simd_simt_910_95_only
def test_costmodel_binned_copy_wgrad(tmp_path):
    (x_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu) = _make_expert_data(6)
    gathered_cpu = _binned_gather_reference(x_cpu, indices_cpu, bins_cpu, _EXPERT_CAPACITY, _TOP_K)
    gathered, indices, _, bins, _, _, grads = _to_npu(
        (gathered_cpu, indices_cpu, bin_ids_cpu, bins_cpu, padded_bins_cpu, weights_cpu, grads_cpu))
    del x_cpu, bin_ids_cpu, padded_bins_cpu, weights_cpu
    expected = _binned_wgrad_reference(gathered_cpu, grads_cpu, indices_cpu, bins_cpu, _TOP_K)
    report_path = tmp_path / "binned_copy_wgrad_route.json"
    output = torch.empty((indices.shape[0], ), dtype=gathered.dtype, device=gathered.device)
    logical_programs = _NE * _EXPERT_CAPACITY

    def launch():
        _binned_copy_wgrad[(_NE, _EXPERT_CAPACITY)](
            gathered,
            grads,
            output,
            _NE,
            _EXPERT_CAPACITY,
            indices,
            bins,
            NUM_COLUMNS=_HS,
            TOP_K=_TOP_K,
            BLOCK_X=_BLOCK_X,
            **_launch_options(report_path, logical_programs),
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), expected, rtol=1e-2, atol=1e-2)
    _load_route_report(report_path, "all_simt_only")
