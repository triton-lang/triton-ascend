"""Example 2 Stage 2 baseline indexer-logits operator.

Extracted from ``example2/kernel.py``.  One program computes one
``(query tile, key tile, provider group)``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_logits_kernel(
    index_q_ptr,
    index_k_ptr,
    weights_ptr,
    out_ptr,
    seq_q,
    seq_k,
    stride_q_token,
    stride_q_group,
    stride_q_head,
    stride_k_token,
    stride_w_token,
    stride_w_group,
    stride_out_row,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Compute weighted-ReLU scores for one Q/K tile and one group."""
    q_block = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    queries = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    query_ok = queries < seq_q
    key_ok = keys < seq_k

    key_tile = tl.load(
        index_k_ptr + dims[:, None] + keys[None, :] * stride_k_token,
        mask=key_ok[None, :],
        other=0.0,
    )

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr + queries[:, None] * stride_q_token + group * stride_q_group + head * stride_q_head +
            dims[None, :],
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr + queries * stride_w_token + group * stride_w_group + head,
            mask=query_ok,
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight[:, None]

    rows = group * seq_q + queries
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + keys[None, :],
        accumulator,
        mask=query_ok[:, None] & key_ok[None, :],
    )


def round_activations_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Round to E4M3 and losslessly widen the result to BF16."""
    return tensor.to(torch.float8_e4m3fn).to(torch.bfloat16)


def indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_q: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """Launch the baseline three-dimensional-grid implementation."""
    if index_k.shape[1] != 1:
        raise ValueError("indexer keys are MQA; expected one head, "
                         f"got {index_k.shape[1]}")
    seq_q, groups, heads_per_group, proxy_dim = index_q.shape
    seq_k = index_k.shape[0]
    if index_k.shape[2] != proxy_dim:
        raise ValueError(f"proxy_dim mismatch: q={proxy_dim} k={index_k.shape[2]}")

    quantized_q = round_activations_e4m3(index_q)
    quantized_k = round_activations_e4m3(index_k)
    out = torch.empty(
        (groups * seq_q, seq_k),
        device=index_q.device,
        dtype=torch.float32,
    )

    _indexer_logits_kernel[(
        triton.cdiv(seq_q, block_q),
        triton.cdiv(seq_k, block_k),
        groups,
    )](
        quantized_q,
        quantized_k,
        weights,
        out,
        seq_q,
        seq_k,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        quantized_k.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
    )
    return out
