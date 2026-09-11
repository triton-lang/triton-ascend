"""Unrolled K/V page loaders must execute on CUBE when block merging is enabled."""

import re

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir, min_dot_size, ttir_to_linalg
from triton.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir


@triton.jit
def _paged_attention(Q, K, V, Metadata, Counts, Out, Lse, TOPK: tl.constexpr, HEADS: tl.constexpr,
                     DIM: tl.constexpr, PAGE: tl.constexpr, BLOCK_R: tl.constexpr):
    query = tl.program_id(0)
    count = tl.load(Counts + query)
    heads = tl.arange(0, HEADS)
    dims = tl.arange(0, DIM)
    tokens = tl.arange(0, PAGE)
    q = tl.load(Q + query * HEADS * DIM + heads[:, None] * DIM + dims[None, :])
    maximum = tl.full((HEADS,), float("-inf"), tl.float32)
    denominator = tl.zeros((HEADS,), tl.float32)
    accumulator = tl.zeros((HEADS, DIM), tl.float32)
    for base in tl.range(0, TOPK, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        meta = tl.load(Metadata + query * TOPK + slots, slots < count, other=-1)
        valid = meta >> 24
        live = (slots < count) & (meta >= 0) & (valid > 0)
        token_ok = tl.reshape(live[:, None] & (tokens[None, :] < valid[:, None]), (BLOCK_R * PAGE,))
        keys = tl.zeros((BLOCK_R * PAGE, DIM), K.dtype.element_ty)
        for page_iter in tl.static_range(0, BLOCK_R):
            slot = base + page_iter
            slot_ok = (slot < count) & (slot < TOPK)
            page_meta = tl.load(Metadata + query * TOPK + slot, slot_ok, other=-1)
            physical = page_meta & 0xFFFFFF
            page_valid = page_meta >> 24
            page_live = slot_ok & (page_meta >= 0) & (page_valid > 0)
            offsets = (physical * PAGE + tokens)[:, None] * DIM + dims[None, :]
            key = tl.load(K + offsets, (tokens < page_valid * page_live)[:, None], other=0.0)
            keys = tl.extra.cann.extension.insert_slice(keys, key, (page_iter * PAGE, 0), (PAGE, DIM), (1, 1))
        scores = tl.dot(q, tl.trans(keys)) * (DIM**-0.5)
        scores = tl.where(token_ok[None, :], scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        empty = new_maximum == float("-inf")
        shift = tl.where(empty, 0.0, new_maximum)
        rescale = tl.where(empty, 1.0, tl.exp(maximum - shift))
        probabilities = tl.exp(scores - shift[:, None])
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        values = tl.zeros((BLOCK_R * PAGE, DIM), V.dtype.element_ty)
        for page_iter in tl.static_range(0, BLOCK_R):
            slot = base + page_iter
            slot_ok = (slot < count) & (slot < TOPK)
            page_meta = tl.load(Metadata + query * TOPK + slot, slot_ok, other=-1)
            physical = page_meta & 0xFFFFFF
            page_valid = page_meta >> 24
            page_live = slot_ok & (page_meta >= 0) & (page_valid > 0)
            offsets = (physical * PAGE + tokens)[:, None] * DIM + dims[None, :]
            value = tl.load(V + offsets, (tokens < page_valid * page_live)[:, None], other=0.0)
            values = tl.extra.cann.extension.insert_slice(values, value, (page_iter * PAGE, 0), (PAGE, DIM), (1, 1))
        accumulator = accumulator * rescale[:, None] + tl.dot(probabilities.to(values.dtype), values)
        maximum = new_maximum
    selected = denominator > 0
    result = tl.where(selected[:, None], accumulator / denominator[:, None], 0.0)
    tl.store(Out + query * HEADS * DIM + heads[:, None] * DIM + dims[None, :], result)
    tl.store(Lse + query * HEADS + heads, tl.where(selected, maximum + tl.log(denominator), float("-inf")))


def _check_page_scopes(adapter, block_r, merge):
    scopes = re.findall(r'scope\.scope[^\{]*\{(.*?)\}\s*\{[^\}]*hivm\.tcore_type = #hivm\.tcore_type<(CUBE|VECTOR)>',
                        adapter, re.S)
    assert {core for _, core in scopes} == {"CUBE", "VECTOR"}, "DynamicCVPipeline unexpectedly fell back"
    bodies = {core: "\n".join(body for body, kind in scopes if kind == core) for core in ("CUBE", "VECTOR")}
    assert "tensor.insert_slice" not in bodies["CUBE"]
    # Both K and V need masked scalar metadata loads and field decoding.
    if merge:
        assert bodies["CUBE"].count("hivm.hir.nd2nz") == 2 * block_r
        assert "tensor.insert_slice" not in bodies["VECTOR"]
        assert len(re.findall(r"memref.load .* : memref<1xi32", bodies["CUBE"])) >= 2 * block_r
        assert bodies["CUBE"].count("arith.shrsi") >= 2 * block_r
    else:
        assert bodies["VECTOR"].count("tensor.insert_slice") == 2 * block_r
    assert "math.exp" in bodies["VECTOR"]
    assert "math.exp" not in bodies["CUBE"]


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("block_r,dim,dtype", [(2, 64, "fp16"), (4, 128, "bf16"), (8, 128, "bf16")])
def test_paged_attention_core_assignment(block_r, dim, dtype, merge):
    constants = dict(TOPK=2 * block_r - 1, HEADS=16, DIM=dim, PAGE=8, BLOCK_R=block_r)
    signature = {name: f"*{dtype}" for name in ("Q", "K", "V", "Out")}
    signature.update(Metadata="*i32", Counts="*i32", Lse="*fp32")
    source = ASTSource(_paged_attention, signature, constexprs=constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = NPUOptions(arch="Ascend910_9589", enable_dynamic_cv_pipeline=True, enable_cube_block_merge=merge)
    metadata = dict(options.__dict__)
    module = ast_to_ttir(_paged_attention, source, context, options, {"min_dot_size": min_dot_size(None)}, {})
    module = make_ttir(module, metadata, options)
    module = ttir_to_linalg(module, metadata, options, named_ops=True)
    assert metadata["enable_dynamic_cv_pipeline"]
    _check_page_scopes(str(module), block_r, merge)


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("block_r", [2, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_attention_numerics(block_r, dtype, merge):
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch.startswith(("Ascend910_95", "Ascend950")):
        pytest.skip("DynamicCVPipeline requires Ascend 950")

    torch.manual_seed(42)
    queries, heads, dim, page, topk = 5, 16, 128, 8, 2 * block_r - 1
    q = torch.randn((queries, heads, dim), dtype=dtype)
    k = torch.randn((topk * page, dim), dtype=dtype)
    v = torch.randn_like(k)
    metadata = torch.full((queries, topk), -1, dtype=torch.int32)
    counts = torch.tensor([0, 1, topk, topk, topk], dtype=torch.int32)
    for row in range(1, queries):
        for slot in range(int(counts[row])):
            physical = topk - slot - 1
            valid = page if slot % 2 == 0 else page // 2
            metadata[row, slot] = physical | (valid << 24)
    metadata[3, :] = -1  # Active slots containing sentinels are still empty.
    metadata[4, :block_r] &= 0xFFFFFF  # Empty first tile followed by live pages.

    expected = torch.zeros_like(q)
    expected_lse = torch.full((queries, heads), float("-inf"))
    for row in range(queries):
        selected = []
        for packed in metadata[row, :counts[row]].tolist():
            if packed >= 0:
                first = (packed & 0xFFFFFF) * page
                selected.extend(range(first, first + (packed >> 24)))
        if selected:
            scores = q[row].float() @ k[selected].float().T / dim**0.5
            expected[row] = (scores.softmax(-1) @ v[selected].float()).to(dtype)
            expected_lse[row] = scores.logsumexp(-1)

    q, k, v, metadata, counts = [tensor.npu() for tensor in (q, k, v, metadata, counts)]
    output = torch.empty_like(q)
    lse = torch.empty((queries, heads), device="npu", dtype=torch.float32)
    compiled = _paged_attention[(queries,)](q, k, v, metadata, counts, output, lse, topk, heads, dim, page, block_r,
                                           enable_dynamic_cv_pipeline=True, enable_cube_block_merge=merge)
    _check_page_scopes(compiled.asm["ttadapter"], block_r, merge)
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-2, rtol=2e-2)
