"""Regression reproducer for flex-attention UB-overflow compile failure on Ascend.

See https://github.com/triton-lang/triton-ascend/issues/3029.

The kernel is the exact Triton program that torch_npu inductor generated for
``triton_flex_attention_fwd_workspace_mask_compact`` (UB overflow:
``requires 1839104 bits while 1572864 bits available``) on Ascend 910B2.

It is marked ``xfail(strict=False)``: it fails while the compiler issue is open
and simply xpasses once it is fixed, so it never turns CI red.

Note: the ``triton_flex_attention_fwd_mask_out`` SIGABRT from the same issue is
not reproduced by a bare kernel launch (it only triggers under the full
torch.compile path), so no xfail entry is added for it here.
"""
import pytest
import torch

import triton
import triton.language as tl


@pytest.mark.xfail(strict=False,
                   reason="triton-ascend#3029: ub overflow on 910B2 (needs 1839104 bits, 1572864 available)")
def test_flex_attention_workspace_mask_compact_ub_overflow(device):

    @triton.jit
    def triton_flex_attention_fwd_workspace_mask_compact(arg_SPARSE_MASK, arg_FLAT_TO_ROW, arg_FLAT_TO_BLK, arg_Q,
                                                         arg_K, arg_KV_IDX, arg_Q_OFFSETS, arg_KV_NUM_BLKS, in_ptr8,
                                                         in_ptr9):
        NUM_VECTOR_CORE: tl.constexpr = 48  # noqa: F841  (inductor codegen artifact)
        SPARSE_Q_BLOCK_SIZE: tl.constexpr = 128
        SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128
        SPARSE_MASK_STRIDE_BLK: tl.constexpr = 16384
        SPARSE_MASK_STRIDE_M: tl.constexpr = 128
        MASK_BLOCK_M: tl.constexpr = 128
        MASK_BLOCK_N: tl.constexpr = 128
        NUM_Q_SUB_BLOCKS: tl.constexpr = 1
        NUM_KV_SUB_BLOCKS: tl.constexpr = 1
        ALLOW_TF32: tl.constexpr = False  # noqa: F841  (inductor codegen artifact)
        INDEX_DTYPE: tl.constexpr = tl.int32  # noqa: F841  (inductor codegen artifact)
        SPARSE_MASK = arg_SPARSE_MASK
        FLAT_TO_ROW = arg_FLAT_TO_ROW
        FLAT_TO_BLK = arg_FLAT_TO_BLK
        Q = arg_Q  # noqa: F841  (inductor codegen artifact)
        K = arg_K  # noqa: F841  (inductor codegen artifact)
        KV_IDX = arg_KV_IDX
        Q_OFFSETS = arg_Q_OFFSETS
        KV_NUM_BLKS = arg_KV_NUM_BLKS

        stride_kv_idx_z = 4096
        stride_kv_idx_h = 4096
        stride_kv_idx_m = 64
        stride_kv_idx_blk = 1

        sparse_h_count = 1
        sparse_q_count = 64
        q_len = 8192
        kv_len = 8192
        sparse_z_count = 1
        count_z, count_h, count_q = 64, 64, 1
        last_row = sparse_z_count * sparse_h_count * sparse_q_count - 1
        last_count = tl.load(KV_NUM_BLKS + (sparse_z_count - 1) * count_z + (sparse_h_count - 1) * count_h +
                             (sparse_q_count - 1) * count_q)
        actual_blocks = tl.load(Q_OFFSETS + last_row) + last_count
        actual_entries = actual_blocks * NUM_Q_SUB_BLOCKS * NUM_KV_SUB_BLOCKS

        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)

        for entry_idx in range(pid, actual_entries, num_programs):
            kv_sub = entry_idx % NUM_KV_SUB_BLOCKS
            tmp = entry_idx // NUM_KV_SUB_BLOCKS
            q_sub = tmp % NUM_Q_SUB_BLOCKS
            flat_blk = tmp // NUM_Q_SUB_BLOCKS

            flat_row = tl.load(FLAT_TO_ROW + flat_blk)
            blk_pos = tl.load(FLAT_TO_BLK + flat_blk)
            sq_idx = flat_row % sparse_q_count
            tmp_row = flat_row // sparse_q_count
            sparse_h = tmp_row % sparse_h_count
            sparse_z = tmp_row // sparse_h_count

            idx_offset = (sparse_z * stride_kv_idx_z + sparse_h * stride_kv_idx_h + sq_idx * stride_kv_idx_m +
                          blk_pos * stride_kv_idx_blk)
            kv_block = tl.load(KV_IDX + idx_offset)

            q_start = sq_idx * SPARSE_Q_BLOCK_SIZE
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE

            offs_m = q_start + q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
            offs_m_local = q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
            offs_n = kv_start + kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)
            offs_n_local = kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)

            m = tl.minimum(offs_m[:, None], q_len - 1)
            n = tl.minimum(offs_n[None, :], kv_len - 1)
            off_z = sparse_z  # noqa: F841  (inductor codegen artifact)
            off_h = sparse_h  # noqa: F841  (inductor codegen artifact)

            tmp0 = tl.full([1], True, tl.int1)
            tmp1 = (m)
            tmp2 = (n)
            tmp3 = tmp1 >= tmp2
            tmp4 = tmp0 & tmp3
            tmp5 = tl.load(in_ptr8 + tmp1)
            tmp6 = tl.load(in_ptr9 + tmp5)
            tmp7 = tmp2 >= tmp6
            tmp8 = tmp4 & tmp7
            mask_mod_output = tmp8

            store_mask = (offs_m[:, None] < q_len) & (offs_n[None, :] < kv_len)
            mask_mod_output = mask_mod_output & store_mask
            mask_base = SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK
            mask_offsets = offs_m_local[:, None] * SPARSE_MASK_STRIDE_M + offs_n_local[None, :]
            tl.store(mask_base + mask_offsets, mask_mod_output.to(tl.int8))

    # Synthetic inputs sized for the max offsets baked into the kernel
    # (SPARSE_MASK_STRIDE_BLK=16384, sparse_q_count=64).
    SPARSE_MASK = torch.zeros(64 * 16384, dtype=torch.int8, device=device)
    FLAT_TO_ROW = torch.zeros(64, dtype=torch.int32, device=device)
    FLAT_TO_BLK = torch.zeros(64, dtype=torch.int32, device=device)
    Q = torch.zeros(1, dtype=torch.bfloat16, device=device)
    K = torch.zeros(1, dtype=torch.bfloat16, device=device)
    KV_IDX = torch.zeros(4096, dtype=torch.int32, device=device)
    Q_OFFSETS = torch.zeros(64, dtype=torch.int32, device=device)
    KV_NUM_BLKS = torch.ones(64, dtype=torch.int32, device=device)
    in_ptr8 = torch.zeros(8192, dtype=torch.int32, device=device)
    in_ptr9 = torch.zeros(8192, dtype=torch.int32, device=device)

    triton_flex_attention_fwd_workspace_mask_compact[(48, )](
        SPARSE_MASK,
        FLAT_TO_ROW,
        FLAT_TO_BLK,
        Q,
        K,
        KV_IDX,
        Q_OFFSETS,
        KV_NUM_BLKS,
        in_ptr8,
        in_ptr9,
    )
