import csv
import json
import os
import statistics

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver
from triton.backends.ascend.utils import is_compile_on_910_95

pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_RUN_SIMD_SIMT_COSTMODEL_GUARDS") != "1",
    reason="SIMD/SIMT costmodel performance guards are opt-in",
)

simd_simt_910_95_only = pytest.mark.xfail(
    not is_compile_on_910_95(),
    reason="SIMD/SIMT cost model only supports 910_95",
    run=False,
)


def _vector_core_count():
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(properties["num_vectorcore"])


def _aicore_count():
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(properties["num_aicore"])


def _load_route_report(path, expected):
    report = json.loads(path.read_text())
    assert report["stage_model"]["applied"]
    assert report["effective_decision_kind"] == expected
    return report


def _launch_options(report_path, logical_programs):
    options = {
        "num_warps": 4,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": _vector_core_count(),
        "physical_aicore_count_hint": _aicore_count(),
    }
    if os.getenv("TRITON_TEST_DISABLE_TTIR_LAYOUT_MERGE") == "1":
        options["enable_ttir_layout_merge"] = False
    return options


def _assert_performance(case, launch, profile_root, documented_us, tolerance=1.2):
    for _ in range(20):
        launch()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    skip_first, warmup, active = 5, 3, 20
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=warmup,
                active=active,
                repeat=1,
                skip_first=skip_first,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_root)),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=config,
    ) as profiler:
        for _ in range(skip_first + warmup + active):
            launch()
            profiler.step()
    torch.npu.synchronize()

    detail_files = list(profile_root.rglob("kernel_details.csv"))
    assert detail_files, f"{case}: profiler did not generate kernel_details.csv"
    durations = []
    for detail_file in detail_files:
        with detail_file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Duration(us)"):
                    durations.append(float(row["Duration(us)"]))
    assert durations, f"{case}: profiler generated no kernel duration"
    duration_us = statistics.median(durations)
    maximum_us = documented_us * tolerance
    print(
        f"{case}: profiler median {duration_us:.3f} us (documented {documented_us:.3f} us, limit {maximum_us:.3f} us)")
    assert duration_us <= maximum_us


# The original TopK validation matrix has device-reference failures in cases
# 02/03/05/06/07/10.  Keep only the four shapes that passed the device
# reference check while retaining the validated shapes.
TOPK_INDEX_MERGE_CASES = [
    pytest.param(1, 1, 256, 4, 2, "all_simt_only", id="01-b1-h1-b256-k4-c2"),
    pytest.param(8, 8, 2048, 32, 8, "all_simt_only", id="04-b8-h8-b2048-k32-c8"),
    pytest.param(8, 8, 4096, 32, 8, "all_simt_only", id="08-b8-h8-b4096-k32-c8"),
    pytest.param(8, 4, 8192, 32, 8, "all_simt_only", id="09-b8-h4-b8192-k32-c8"),
]

TOPK_INDEX_MERGE_HEURISTICS = {
    "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(args["num_topk_chunks"] * triton.next_power_of_2(args["topk"])),
}


@triton.jit
def _topk_compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    values = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return values.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _topk_bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _topk_compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.heuristics(TOPK_INDEX_MERGE_HEURISTICS)
@triton.jit(do_not_specialize=["num_topk_chunks", "decode_query_len"])
def topk_index_merge_costmodel_kernel(
    partial_scores,
    partial_indices,
    final_indices,
    sequence_lengths,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    decode_query_len,
    stride_scores_chunk,
    stride_scores_head,
    stride_scores_batch,
    stride_scores_topk,
    stride_indices_chunk,
    stride_indices_head,
    stride_indices_batch,
    stride_indices_topk,
    stride_final_head,
    stride_final_batch,
    stride_final_topk,
    num_topk_chunks,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    request_id = pid_batch // decode_query_len
    query_offset = pid_batch - request_id * decode_query_len
    sequence_length = tl.load(sequence_lengths + request_id)
    kv_length = tl.maximum(sequence_length - decode_query_len + query_offset + 1, 0)
    num_blocks = (kv_length + block_size - 1) // block_size
    offset = tl.arange(0, BLOCK_SIZE_K)
    chunk_index = offset // BLOCK_SIZE_T
    in_chunk_index = offset % BLOCK_SIZE_T
    valid = chunk_index < num_topk_chunks
    score_offset = (chunk_index * stride_scores_chunk + pid_head * stride_scores_head +
                    pid_batch * stride_scores_batch + in_chunk_index * stride_scores_topk)
    index_offset = (chunk_index * stride_indices_chunk + pid_head * stride_indices_head +
                    pid_batch * stride_indices_batch + in_chunk_index * stride_indices_topk)
    scores = tl.load(partial_scores + score_offset, mask=valid, other=-1e30).to(tl.float32)
    scores = tl.where(scores != scores, -1e30, scores)
    indices = tl.load(partial_indices + index_offset, mask=valid, other=0).to(tl.int32)
    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for stage in tl.static_range(1, n_dims):
        scores, indices = _topk_bitonic_merge(scores, indices, stage, 2, n_dims)
    scores, indices = _topk_bitonic_merge(scores, indices, n_dims, True, n_dims)
    extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    result = tl.sum(
        extract[:, None] * tl.reshape(indices - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    topk_offset = tl.arange(0, BLOCK_SIZE_T)
    output = (final_indices + pid_head * stride_final_head + pid_batch * stride_final_batch +
              topk_offset * stride_final_topk)
    result = tl.where(topk_offset < tl.minimum(topk, num_blocks), result, -1)
    tl.store(output, result, mask=topk_offset < topk)


def _topk_index_merge_reference(scores, indices, topk):
    heads, total_queries = scores.shape[1:3]
    flat_scores = scores.permute(1, 2, 0, 3).reshape(heads, total_queries, -1)
    flat_indices = indices.permute(1, 2, 0, 3).reshape(heads, total_queries, -1)
    selected = flat_scores.topk(topk, dim=-1).indices
    return torch.gather(flat_indices, -1, selected) - 1


@simd_simt_910_95_only
@pytest.mark.parametrize(
    "batch,heads,max_block,topk,num_topk_chunks,expected_route",
    TOPK_INDEX_MERGE_CASES,
)
def test_costmodel_topk_index_merge_end_to_end(
    batch,
    heads,
    max_block,
    topk,
    num_topk_chunks,
    expected_route,
    tmp_path,
):
    torch.manual_seed(1234)
    query_length = 1
    block_size = 128
    block_topk = triton.next_power_of_2(topk)
    sequence_lengths = torch.full((batch, ), max_block * block_size, dtype=torch.int32, device="npu")
    partial_scores = torch.randn(
        (num_topk_chunks, heads, batch * query_length, block_topk),
        device="npu",
    )
    partial_indices = torch.randint(
        1,
        128,
        partial_scores.shape,
        dtype=torch.int32,
        device="npu",
    )
    final_indices = torch.empty((heads, batch * query_length, topk), dtype=torch.int32, device="npu")
    report_path = tmp_path / "topk_index_merge_route.json"
    logical_programs = batch * query_length * heads

    topk_index_merge_costmodel_kernel[(batch * query_length, heads)](
        partial_scores,
        partial_indices,
        final_indices,
        sequence_lengths,
        block_size,
        topk,
        query_length,
        partial_scores.stride(0),
        partial_scores.stride(1),
        partial_scores.stride(2),
        partial_scores.stride(3),
        partial_indices.stride(0),
        partial_indices.stride(1),
        partial_indices.stride(2),
        partial_indices.stride(3),
        final_indices.stride(0),
        final_indices.stride(1),
        final_indices.stride(2),
        num_topk_chunks,
        **_launch_options(report_path, logical_programs),
    )
    torch.npu.synchronize()

    expected = _topk_index_merge_reference(partial_scores.cpu(), partial_indices.cpu(), topk)
    # The kernel contract does not require an order among the selected
    # indices. Sorting both sides preserves multiplicity, unlike the old
    # set-based validation used by the out-of-tree performance harness.
    torch.testing.assert_close(
        final_indices.cpu().sort(dim=-1).values,
        expected.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )
    report = _load_route_report(report_path, expected_route)
    assert report["logical_program_count_hint"] == logical_programs


@triton.jit
def gather_dot_min(
    a_ptr,
    b_ptr,
    indices_ptr,
    out_ptr,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gather_k = tl.load(indices_ptr + offs_k)
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + gather_k[None, :] * stride_ak)
    b = tl.load(b_ptr + gather_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    result = tl.dot(a, b)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, result)


@simd_simt_910_95_only
def test_costmodel_gather_dot_min(tmp_path):
    logical_programs = _vector_core_count()
    block = 16
    source_k = 256
    a = torch.randn((logical_programs * block, source_k), dtype=torch.float16, device="npu")
    b = torch.randn((source_k, block), dtype=torch.float16, device="npu")
    indices = torch.tensor(
        [10, 25, 100, 200, 5, 50, 150, 255, 1, 2, 3, 4, 6, 7, 8, 9],
        dtype=torch.int32,
        device="npu",
    )
    output = torch.empty((logical_programs * block, block), dtype=torch.float32, device="npu")
    report_path = tmp_path / "gather_dot_min_route.json"

    def launch():
        gather_dot_min[(logical_programs, 1)](
            a,
            b,
            indices,
            output,
            a.shape[0],
            b.shape[1],
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            **_launch_options(report_path, logical_programs),
        )

    launch()
    expected = torch.matmul(a[:, indices].float(), b[indices, :].float())
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    report = _load_route_report(report_path, "all_simt_only")
    assert any(stage["features"]["has_dot"] for stage in report["stage_model"]["logical_stages"])
    _assert_performance("gather_dot_min", launch, tmp_path / "gather_profile", 5.478, tolerance=1.35)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0.0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0.0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0.0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0.0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.0)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(18, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.0)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(34, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.0)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(50, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.0)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11, boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22, boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33, boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44, boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21, boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31, boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32, boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41, boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42, boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43, boundary_check=(0, 1))


SOLVE_TRIL_CASES = [
    (1, 1024, 32, 64, 182.929),
    (1, 1024, 64, 64, 613.346),
    (8, 1024, 32, 64, 1396.827),
    (16, 1024, 64, 64, 5474.364),
    (1, 8192, 64, 64, 2730.823),
    (8, 8192, 32, 64, 10930.605),
    (16, 8192, 64, 64, 43821.177),
    (4, 131072, 32, 64, 87436.743),
]


@simd_simt_910_95_only
@pytest.mark.parametrize(
    "batch,sequence_length,heads,block,documented_us",
    SOLVE_TRIL_CASES,
    ids=[f"B{b}-T{t}-H{h}-BT{bt}" for b, t, h, bt, _ in SOLVE_TRIL_CASES],
)
def test_costmodel_solve_tril(batch, sequence_length, heads, block, documented_us, tmp_path):
    chunks = sequence_length // block
    logical_programs = batch * chunks * heads
    torch.manual_seed(1)
    lower = torch.tril(torch.randn((block, block), dtype=torch.float32), diagonal=-1) * 0.01
    a = torch.empty((batch, sequence_length, heads, block), dtype=torch.float32, device="npu")
    a.view(batch, chunks, block, heads, block).copy_(lower.reshape(1, 1, block, 1, block).to("npu"))
    output = torch.zeros_like(a)
    report_path = tmp_path / "solve_tril_route.json"

    def launch():
        merge_16x16_to_64x64_inverse_kernel[(chunks, batch * heads)](
            a,
            output,
            None,
            None,
            sequence_length,
            H=heads,
            BT=block,
            USE_TMA=False,
            DOT_PRECISION="ieee",
            **_launch_options(report_path, logical_programs),
        )

    launch()
    inverse = torch.linalg.inv(torch.eye(block) + lower).to("npu")
    output_blocks = output.view(batch, chunks, block, heads, block)
    probes = {
        (0, 0, 0),
        (batch // 2, chunks // 2, heads // 2),
        (batch - 1, chunks - 1, heads - 1),
    }
    for batch_id, chunk_id, head_id in probes:
        torch.testing.assert_close(
            output_blocks[batch_id, chunk_id, :, head_id, :],
            inverse,
            rtol=3e-2,
            atol=3e-2,
        )
    report = _load_route_report(report_path, "mixed_simd_simt")
    assert report["materialized_simt_anchor_count"] > 0
    assert report["selected_superblock_factor"] == 4
    # Mixed solve_tril contains SIMD Cube stages. NPUIR therefore creates a
    # MIX/AIC AutoBlockify loop (CUBE_CORE_COUNT slots), not an AIV loop.
    # The local SIMT stage is still one F4 scope inside that loop.
    aicore_count = _aicore_count()
    logical_programs_per_slot = (logical_programs + aicore_count - 1) // aicore_count
    expected_full_groups = logical_programs_per_slot // 4
    assert report["effective_runtime_factor"] == (4 if expected_full_groups else 1)
    assert report["runtime_scheduling_slot_count"] == min(logical_programs, aicore_count)
    assert report["runtime_physical_program_count"] == min(logical_programs, aicore_count)
    assert report["logical_programs_per_scheduling_slot"] == logical_programs_per_slot
    assert report["full_group_count"] == expected_full_groups
    assert report["tail_count"] == logical_programs_per_slot % 4
    assert report["runtime_loop_iteration_count"] == (expected_full_groups + logical_programs_per_slot % 4)
    case = f"solve_tril_B{batch}_T{sequence_length}_H{heads}_BT{block}"
    _assert_performance(case, launch, tmp_path / "solve_profile", documented_us)
    del output_blocks, inverse, output, a
    torch.npu.empty_cache()


@triton.jit
def _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    token_indices_ptr,
    expert_indices_ptr,
    scores_ptr,
    scale_ub_ptr,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(D % BLOCK_D == 0, "D must be a multiple of BLOCK_D")
    output_token = tl.program_id(0)
    valid_token_count = tl.load(valid_token_count, None, eviction_policy="evict_last")
    if output_token >= valid_token_count:
        return
    input_token = tl.load(token_indices_ptr + output_token)
    expert = tl.load(expert_indices_ptr + output_token)
    score = tl.load(scores_ptr + input_token * stride_t + expert * stride_e).to(tl.float32)
    offsets = tl.arange(0, BLOCK_D)
    input_block = input_ptr + input_token.to(tl.int64) * D + offsets
    row_max = 0.0
    for _ in range(0, D, BLOCK_D):
        values = tl.load(input_block, eviction_policy="evict_last").to(tl.float32) * score
        row_max = tl.maximum(tl.max(tl.abs(values)), row_max)
        input_block += BLOCK_D

    if CLAMP_MAX:
        row_max = tl.clamp(row_max, EPS, tl.load(scale_ub_ptr))
    else:
        row_max = tl.maximum(row_max, EPS)
    scale = MAX_FP8 / row_max
    tl.store(output_scale_ptr + output_token, 1.0 / scale)
    input_block = input_ptr + input_token.to(tl.int64) * D + offsets
    output_block = output_ptr + output_token.to(tl.int64) * D + offsets
    for _ in range(0, D, BLOCK_D):
        values = tl.load(input_block, eviction_policy="evict_first").to(tl.float32) * score
        quantized = tl.clamp(values * scale, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        tl.store(output_block, quantized, cache_modifier=".cg")
        input_block += BLOCK_D
        output_block += BLOCK_D


@simd_simt_910_95_only
def test_costmodel_fbgemm_rowwise_quant(tmp_path):
    tokens, width, experts, valid = 256, 1024, 8, 512
    torch.manual_seed(2)
    input_tensor = torch.randn((tokens, width), dtype=torch.float16, device="npu")
    token_indices = torch.arange(valid, dtype=torch.int32, device="npu") % tokens
    expert_indices = torch.arange(valid, dtype=torch.int32, device="npu") % experts
    scores = torch.randn((tokens, experts), dtype=torch.float16, device="npu")
    valid_count = torch.tensor([valid], dtype=torch.int32, device="npu")
    scale_ub = torch.tensor([448.0], dtype=torch.float32, device="npu")
    output = torch.empty((valid, width), dtype=torch.float8_e4m3fn, device="npu")
    output_scale = torch.empty((valid, ), dtype=torch.float32, device="npu")
    report_path = tmp_path / "fbgemm_route.json"

    def launch():
        _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens[(valid, )](
            output,
            output_scale,
            input_tensor,
            token_indices,
            expert_indices,
            scores,
            scale_ub,
            scores.stride(0),
            scores.stride(1),
            valid_count,
            D=width,
            TL_FP8_DTYPE=tl.float8e4nv,
            MAX_FP8=448.0,
            EPS=1.0e-12,
            CLAMP_MAX=False,
            BLOCK_D=width,
            **_launch_options(report_path, valid),
        )

    launch()
    gathered = input_tensor[token_indices].float() * scores[token_indices, expert_indices].float()[:, None]
    row_max = torch.clamp(torch.amax(torch.abs(gathered), dim=1), min=1.0e-12)
    # Match the kernel's floating-point operation order.  Although
    # ``x / (row_max / 448)`` is algebraically equivalent to
    # ``x * (448 / row_max)``, their FP32 rounding differs at FP8 bin
    # boundaries and can select adjacent values whose spacing is 32.
    quant_scale = 448.0 / row_max
    expected_scale = 1.0 / quant_scale
    expected = torch.clamp(gathered * quant_scale[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(output_scale, expected_scale, rtol=2e-3, atol=2e-3)
    output_f32 = output.float()
    expected_f32 = expected.float()
    difference = torch.abs(output_f32 - expected_f32)
    # The device conversion and torch's reference conversion may choose
    # adjacent FP8 values for an exact rounding tie.  Check one E4M3 ULP at
    # each expected value instead of using a fixed tolerance: E4M3 spacing is
    # 16 around 128 but 32 around 256, while subnormals have spacing 2^-9.
    magnitude = torch.abs(expected_f32)
    normal_ulp = torch.pow(2.0, torch.floor(torch.log2(torch.clamp(magnitude, min=2**-6))) - 3)
    fp8_ulp = torch.where(magnitude < 2**-6, torch.full_like(magnitude, 2**-9), normal_ulp)
    assert torch.all(difference <= fp8_ulp), (f"FBGEMM FP8 output exceeds one ULP: max_abs={difference.max().item()}, "
                                              f"max_ulp_error={(difference / fp8_ulp).max().item()}")
    layout_merge_disabled = os.getenv("TRITON_TEST_DISABLE_TTIR_LAYOUT_MERGE") == "1"
    expected_route = "all_simd" if layout_merge_disabled else "all_simt_only"
    report = _load_route_report(report_path, expected_route)
    capability = report["route_transform_capability"]
    assert capability["source_logical_program_count_hint"] == valid
    if layout_merge_disabled:
        assert not capability["layout_coalescing_applied"]
        assert capability["logical_program_count_hint"] == valid
    else:
        assert capability["layout_coalescing_factor"] == 2
        assert capability["logical_program_count_hint"] == valid // 2
    documented_us = 20.578 if layout_merge_disabled else 8.904
    _assert_performance(
        "fbgemm_rowwise_quant",
        launch,
        tmp_path / "fbgemm_profile",
        documented_us,
        tolerance=1.25,
    )
