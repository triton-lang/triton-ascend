"""Regression reproducers for flex-attention kernel compilation failures on Ascend.

See https://github.com/triton-lang/triton-ascend/issues/3029.

Both kernels are the exact Triton programs that torch_npu inductor generated for
``triton_flex_attention_fwd_workspace_mask_compact`` (UB overflow:
``requires 1839104 bits while 1572864 bits available``) and
``triton_flex_attention_fwd_mask_out`` (``bishengir-compile`` SIGABRT).

They are marked ``xfail(strict=False)``: they fail while the compiler issue is
open and simply xpass once it is fixed, so they never turn CI red.
"""
import pytest
import torch

import triton
import triton.language as tl


@pytest.mark.xfail(strict=False,
                   reason="triton-ascend#3029: ub overflow on 910B2 (needs 1839104 bits, 1572864 available)")
def test_flex_attention_workspace_mask_compact_ub_overflow(device):
    @triton.jit
    def triton_flex_attention_fwd_workspace_mask_compact(arg_SPARSE_MASK, arg_FLAT_TO_ROW, arg_FLAT_TO_BLK, arg_Q, arg_K, arg_KV_IDX, arg_Q_OFFSETS, arg_KV_NUM_BLKS, in_ptr8, in_ptr9):
        NUM_VECTOR_CORE : tl.constexpr = 48
        SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
        SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
        SPARSE_MASK_STRIDE_BLK : tl.constexpr = 16384
        SPARSE_MASK_STRIDE_M : tl.constexpr = 128
        MASK_BLOCK_M : tl.constexpr = 128
        MASK_BLOCK_N : tl.constexpr = 128
        NUM_Q_SUB_BLOCKS : tl.constexpr = 1
        NUM_KV_SUB_BLOCKS : tl.constexpr = 1
        ALLOW_TF32 : tl.constexpr = False
        INDEX_DTYPE : tl.constexpr = tl.int32
        SPARSE_MASK = arg_SPARSE_MASK
        FLAT_TO_ROW = arg_FLAT_TO_ROW
        FLAT_TO_BLK = arg_FLAT_TO_BLK
        Q = arg_Q
        K = arg_K
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
        last_count = tl.load(
            KV_NUM_BLKS + (sparse_z_count - 1) * count_z
            + (sparse_h_count - 1) * count_h + (sparse_q_count - 1) * count_q
        )
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
    
            idx_offset = (
                sparse_z * stride_kv_idx_z
                + sparse_h * stride_kv_idx_h
                + sq_idx * stride_kv_idx_m
                + blk_pos * stride_kv_idx_blk
            )
            kv_block = tl.load(KV_IDX + idx_offset)
    
            q_start = sq_idx * SPARSE_Q_BLOCK_SIZE
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE
    
            offs_m = q_start + q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
            offs_m_local = q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
            offs_n = kv_start + kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)
            offs_n_local = kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)
    
            m = tl.minimum(offs_m[:, None], q_len - 1)
            n = tl.minimum(offs_n[None, :], kv_len - 1)
            off_z = sparse_z
            off_h = sparse_h
    
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

    triton_flex_attention_fwd_workspace_mask_compact[(48,)](
        SPARSE_MASK, FLAT_TO_ROW, FLAT_TO_BLK, Q, K, KV_IDX, Q_OFFSETS,
        KV_NUM_BLKS, in_ptr8, in_ptr9,
    )


@pytest.mark.xfail(strict=False,
                   reason="triton-ascend#3029: bishengir-compile SIGABRT on 910B2")
def test_flex_attention_mask_out_sigabrt(device):
    @triton.jit
    def triton_flex_attention_fwd_mask_out(arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0):
        PRESCALE_QK : tl.constexpr = False
        ROWS_GUARANTEED_SAFE : tl.constexpr = False
        BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
        WRITE_DQ : tl.constexpr = True
        OUTPUT_LOGSUMEXP : tl.constexpr = True
        OUTPUT_MAX : tl.constexpr = False
        FLOAT32_PRECISION : tl.constexpr = 'ieee'
        TORCHINDUCTOR_FLEXATTENTION_MASKOUT : tl.constexpr = True
        IS_DIVISIBLE : tl.constexpr = True
        SM_SCALE : tl.constexpr = 0.25
        GQA_SHARED_HEADS : tl.constexpr = 1
        HAS_FULL_BLOCKS : tl.constexpr = True
        QK_HEAD_DIM : tl.constexpr = 16
        QK_HEAD_DIM_ROUNDED : tl.constexpr = 16
        V_HEAD_DIM : tl.constexpr = 16
        V_HEAD_DIM_ROUNDED : tl.constexpr = 16
        SAFE_HEAD_DIM : tl.constexpr = True
        SPARSE_MASK_STRIDE_BLK : tl.constexpr = 16384
        SPARSE_MASK_STRIDE_M : tl.constexpr = 128
        BLOCK_M : tl.constexpr = 64
        BLOCK_N : tl.constexpr = 64
        SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
        SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
        NUM_CUBE_CORE : tl.constexpr = 24
        ALLOW_TF32 : tl.constexpr = False
        INDEX_DTYPE : tl.constexpr = tl.int32
        Q = arg_Q
        K = arg_K
        V = arg_V
        SPARSE_MASK = arg_SPARSE_MASK
        Q_OFFSETS = arg_Q_OFFSETS
        KV_NUM_BLKS = arg_KV_NUM_BLKS
        KV_IDX = arg_KV_IDX
        LSE = arg_LSE
        MAX_SCORES = arg_MAX_SCORES
        FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
        FULL_KV_IDX = arg_FULL_KV_IDX
     # noqa: B950
        tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
        tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)
    
        stride_qz, stride_qh, stride_qm, stride_qk = 256, 16, 256, 1
        stride_kz, stride_kh, stride_kn, stride_kk = 256, 16, 256, 1
        stride_vz, stride_vh, stride_vn, stride_vk = 256, 16, 256, 1
    
        ZQ = 1
        HQ = 16
        Q_LEN = 8192
        ZKV = 1
        KV_LEN = 8192
    
        MATMUL_PRECISION = Q.dtype.element_ty
    
        SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
        NUM_Q_TILES = tl.cdiv(Q_LEN, BLOCK_M)
    
        for tile_id in range(tl.program_id(0), NUM_Q_TILES * ZQ * HQ, tl.num_programs(0)):
            tile_id_index = tile_id.to(INDEX_DTYPE)
            q_start = tile_id_index % NUM_Q_TILES
            off_zh = tile_id_index // NUM_Q_TILES
            off_zq = off_zh // HQ
            off_hq = off_zh % HQ
            off_zkv = off_zq % ZKV
            off_hkv = off_hq // GQA_SHARED_HEADS
    
            Q_tile = Q + off_zq * stride_qz + off_hq * stride_qh
            K_tile = K + off_zkv * stride_kz + off_hkv * stride_kh
            V_tile = V + off_zkv * stride_vz + off_hkv * stride_vh
    
            SPARSE_Z = 1
            SPARSE_HQ = 1
            sparse_idx_z = off_zq % SPARSE_Z
            sparse_idx_hq = off_hq % SPARSE_HQ
    
            SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
            FULL128_SUBTILES: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    
            stride_kv_num_blks_h = 64
            stride_kv_idx_h = 4096
            stride_kv_idx_m = 64
    
            m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)
    
            offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_k = tl.arange(0, QK_HEAD_DIM)
            offs_v = tl.arange(0, V_HEAD_DIM)
            q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
    
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
            sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_sparse_idx
            sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + q_sparse_idx * stride_kv_idx_m
    
            q = tl.load(
                Q_tile + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                mask=offs_m[:, None] < Q_LEN,
                other=0.0,
            )
    
            kv_indices = KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(
                kv_num_blocks * SPARSE_KV_MULTIPLE,
                tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
    
            acc, l_i, m_i = forward_inner_sparse_mask_direct_index(
                arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                q, K_tile, V_tile, Q_LEN, KV_LEN,
                stride_kk, stride_kn, stride_vn, stride_vk,
                acc, l_i, m_i,
                off_zq, off_hq, offs_m[:, None],
                kv_indices, kv_num_blocks,
                0, block_n_end,
                MATMUL_PRECISION,
                q_start,
                IS_FULL_BLOCKS=False,
            )
    
    
            FULL_SPARSE_Z = 1
            FULL_SPARSE_HQ = 1
            full_sparse_idx_z = off_zq % FULL_SPARSE_Z
            full_sparse_idx_hq = off_hq % FULL_SPARSE_HQ
    
            stride_full_kv_num_blks_h = 64
            stride_full_kv_idx_h = 4096
            stride_full_kv_idx_m = 64
    
            full_hz_offset = full_sparse_idx_z * FULL_SPARSE_HQ + full_sparse_idx_hq
            full_kv_num_blks_offset = full_hz_offset * stride_full_kv_num_blks_h + q_sparse_idx
            full_kv_idx_offset = full_hz_offset * stride_full_kv_idx_h + q_sparse_idx * stride_full_kv_idx_m
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + full_kv_num_blks_offset)
    
            if kv_num_blocks > 0:
                kv_indices = FULL_KV_IDX + full_kv_idx_offset
    
                for start_n in range(0, kv_num_blocks):
                    kv_block_start = tl.load(kv_indices + start_n) * SPARSE_KV_BLOCK_SIZE
                    for sub_idx in range(0, FULL128_SUBTILES):
                        kv_start = kv_block_start + sub_idx * BLOCK_N
                        offs_n = kv_start + tl.arange(0, BLOCK_N)
                        k = tl.load(
                            K_tile + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                            mask=offs_n[:, None] < KV_LEN,
                            other=0.0,
                        )
                        v = tl.load(
                            V_tile + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                            mask=offs_n[:, None] < KV_LEN,
                            other=0.0,
                        )
    
                        if IS_DIVISIBLE:
                            acc, l_i, m_i = forward_block_mn_full(
                                arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                                q, k, v, Q_LEN, KV_LEN,
                                acc, l_i, m_i,
                                off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                                MATMUL_PRECISION,
                            )
                        else:
                            acc, l_i, m_i = forward_block_mn_full(
                                arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                                q, k, v, Q_LEN, KV_LEN,
                                acc, l_i, m_i,
                                off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                                MATMUL_PRECISION,
                                CHECK_BLOCK_BOUNDARY=True,
                            )
    
    
            l_i = tl.where(l_i == 0.0, 1, l_i)
            acc = acc / l_i[:, None]
            idx_zq = off_zq
            idx_hq = off_hq
            idx_m = offs_m[:, None]
            idx_d = offs_v[None, :]
            mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)
    
            xindex = idx_d + 16*idx_m + 131072*idx_hq + 2097152*idx_zq
            tl.store(out_ptr0 + (tl.broadcast_to(idx_d + 16*idx_hq + 256*idx_m, acc.shape)), acc, mask)
    
            off_hz = off_zq * HQ + off_hq
            if OUTPUT_LOGSUMEXP:
                l_ptrs = LSE + off_hz * Q_LEN + offs_m
                lse = m_i + tl.math.log(l_i)
                if IS_DIVISIBLE:
                    tl.store(l_ptrs, lse)
                else:
                    tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)
    
            if OUTPUT_MAX:
                m_ptrs = MAX_SCORES + off_hz * Q_LEN + offs_m
                if IS_DIVISIBLE:
                    tl.store(m_ptrs, m_i)
                else:
                    tl.store(m_ptrs, m_i, mask=offs_m < Q_LEN)
    
    @triton.jit
    def forward_inner_sparse_mask_direct_index(
        arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
        q, K, V, Q_LEN, KV_LEN,
        stride_kk, stride_kn, stride_vn, stride_vk,
        # accumulated values
        acc, l_i, m_i,
        # Offsets used as inputs to score_mod & mask_mod
        off_z, off_h, offs_m,
        # blocksparse data
        kv_indices, kv_num_blocks,
        # start kv and end kv block
        block_n_start, block_n_end,
        MATMUL_PRECISION,
        q_start,
        IS_FULL_BLOCKS,
    ):
        PRESCALE_QK : tl.constexpr = False
        ROWS_GUARANTEED_SAFE : tl.constexpr = False
        BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
        WRITE_DQ : tl.constexpr = True
        OUTPUT_LOGSUMEXP : tl.constexpr = True
        OUTPUT_MAX : tl.constexpr = False
        FLOAT32_PRECISION : tl.constexpr = 'ieee'
        TORCHINDUCTOR_FLEXATTENTION_MASKOUT : tl.constexpr = True
        IS_DIVISIBLE : tl.constexpr = True
        SM_SCALE : tl.constexpr = 0.25
        GQA_SHARED_HEADS : tl.constexpr = 1
        HAS_FULL_BLOCKS : tl.constexpr = True
        QK_HEAD_DIM : tl.constexpr = 16
        QK_HEAD_DIM_ROUNDED : tl.constexpr = 16
        V_HEAD_DIM : tl.constexpr = 16
        V_HEAD_DIM_ROUNDED : tl.constexpr = 16
        SAFE_HEAD_DIM : tl.constexpr = True
        SPARSE_MASK_STRIDE_BLK : tl.constexpr = 16384
        SPARSE_MASK_STRIDE_M : tl.constexpr = 128
        BLOCK_M : tl.constexpr = 64
        BLOCK_N : tl.constexpr = 64
        SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
        SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
        NUM_CUBE_CORE : tl.constexpr = 24
        ALLOW_TF32 : tl.constexpr = False
        INDEX_DTYPE : tl.constexpr = tl.int32
    
    
        SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)
    
        if PRESCALE_QK:
            q = (q * SM_SCALE).to(MATMUL_PRECISION)
    
        for start_n in range(block_n_start, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N
            offs_n = kv_start + tl.arange(0, BLOCK_N)
            k = tl.load(
                K + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=offs_n[:, None] < KV_LEN,
                other=0.0,
            )
            v = tl.load(
                V + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=offs_n[:, None] < KV_LEN,
                other=0.0,
            )
    
            if IS_DIVISIBLE:
                acc, l_i, m_i = forward_block_mn_sparse_mask(
                    arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                    q, k, v, Q_LEN, KV_LEN,
                    acc, l_i, m_i,
                    off_z, off_h, offs_m, offs_n[None, :],
                    MATMUL_PRECISION,
                    q_start,
                    blk_idx_in_list,
                    IS_FULL_BLOCKS,
                )
            else:
                acc, l_i, m_i = forward_block_mn_sparse_mask(
                    arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                    q, k, v, Q_LEN, KV_LEN,
                    acc, l_i, m_i,
                    off_z, off_h, offs_m, offs_n[None, :],
                    MATMUL_PRECISION,
                    q_start,
                    blk_idx_in_list,
                    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
                )
    
        return acc, l_i, m_i
    
    
    @triton.jit
    def forward_block_mn_sparse_mask(
        arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
        q, k, v, Q_LEN, KV_LEN,
        # accumulated values
        acc, l_i, m_i,
        # Offsets
        off_z, off_h, offs_m, offs_n,
        MATMUL_PRECISION,
        q_start,
        blk_idx_in_list,
        IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
    
    ):
        # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
        PRESCALE_QK : tl.constexpr = False
        ROWS_GUARANTEED_SAFE : tl.constexpr = False
        BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
        WRITE_DQ : tl.constexpr = True
        OUTPUT_LOGSUMEXP : tl.constexpr = True
        OUTPUT_MAX : tl.constexpr = False
        FLOAT32_PRECISION : tl.constexpr = 'ieee'
        TORCHINDUCTOR_FLEXATTENTION_MASKOUT : tl.constexpr = True
        IS_DIVISIBLE : tl.constexpr = True
        SM_SCALE : tl.constexpr = 0.25
        GQA_SHARED_HEADS : tl.constexpr = 1
        HAS_FULL_BLOCKS : tl.constexpr = True
        QK_HEAD_DIM : tl.constexpr = 16
        QK_HEAD_DIM_ROUNDED : tl.constexpr = 16
        V_HEAD_DIM : tl.constexpr = 16
        V_HEAD_DIM_ROUNDED : tl.constexpr = 16
        SAFE_HEAD_DIM : tl.constexpr = True
        SPARSE_MASK_STRIDE_BLK : tl.constexpr = 16384
        SPARSE_MASK_STRIDE_M : tl.constexpr = 128
        BLOCK_M : tl.constexpr = 64
        BLOCK_N : tl.constexpr = 64
        SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
        SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
        NUM_CUBE_CORE : tl.constexpr = 24
        ALLOW_TF32 : tl.constexpr = False
        INDEX_DTYPE : tl.constexpr = tl.int32
    
        # -- compute qk ---
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")
        if not PRESCALE_QK:
            qk *= SM_SCALE
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
        n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)
    
        tmp0 = (qk)
        post_mod_scores = tmp0
    
    
        if not IS_FULL_BLOCKS:
    
            SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
            sparse_h_count = 1
            q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
            q_sparse_start = q_sparse_idx * SPARSE_Q_BLOCK_SIZE
            sparse_h = off_h % sparse_h_count
            # Broadcast a B=1 block-mask across B>1 QKV (mirror the other sparse paths).
            sparse_z_count = 1
            sparse_idx_z = off_z % sparse_z_count
    
            stride_kv_idx_z = 4096
            stride_kv_idx_h = 4096
            stride_kv_idx_m = 64
            stride_kv_idx_blk = 1
            kv_block = tl.load(
                arg_KV_IDX
                + sparse_idx_z * stride_kv_idx_z
                + sparse_h * stride_kv_idx_h
                + q_sparse_idx * stride_kv_idx_m
                + blk_idx_in_list * stride_kv_idx_blk
            )
    
            offs_m_local = offs_m - q_sparse_start
            offs_n_local = offs_n - kv_block * SPARSE_KV_BLOCK_SIZE
            stride_q_offsets_z = 64
            stride_q_offsets_h = 64
            stride_q_offsets_q = 1
            q_offsets_idx = (
                sparse_idx_z * stride_q_offsets_z
                + sparse_h * stride_q_offsets_h
                + q_sparse_idx * stride_q_offsets_q
            )
            flat_blk = tl.load(arg_Q_OFFSETS + q_offsets_idx) + blk_idx_in_list
            mask_base = arg_SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK
            mask_offsets = offs_m_local * SPARSE_MASK_STRIDE_M + offs_n_local
            mask_mod_output = tl.load(mask_base + mask_offsets) != 0
    
            # apply mask for partially unmasked blocks
            post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
        elif CHECK_BLOCK_BOUNDARY:
            post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))
    
        # -- compute scaling constant ---
        m_ij = tl.maximum(
            m_i,
            tl.max(post_mod_scores, 1, propagate_nan=True),
            propagate_nan=tl.PropagateNan.ALL,
        )
        if not ROWS_GUARANTEED_SAFE:
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
        else:
            m_ij_masked = m_ij
    
        alpha = tl.math.exp(m_i - m_ij_masked)
        p = tl.math.exp(post_mod_scores - m_ij_masked[:, None])
    
        # NB: l_i update is pulled up here since it's a bit faster
        # NB: For headdim=256, it's faster to move it back down to after m_i =
        # m_ij
        l_i = l_i * alpha + tl.sum(p, 1)
        # # -- scale and update acc --
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision="ieee")
        # -- update m_i
        m_i = m_ij
    
        return acc, l_i, m_i
    
    
    @triton.jit
    def forward_block_mn_full(
        arg_Q, arg_K, arg_V, arg_SPARSE_MASK, arg_Q_OFFSETS, arg_KV_NUM_BLKS, arg_KV_IDX, arg_LSE, arg_MAX_SCORES, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
        q, k, v, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_z, off_h, offs_m, offs_n,
        MATMUL_PRECISION,
        CHECK_BLOCK_BOUNDARY=False,
    ):
        PRESCALE_QK : tl.constexpr = False
        ROWS_GUARANTEED_SAFE : tl.constexpr = False
        BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
        WRITE_DQ : tl.constexpr = True
        OUTPUT_LOGSUMEXP : tl.constexpr = True
        OUTPUT_MAX : tl.constexpr = False
        FLOAT32_PRECISION : tl.constexpr = 'ieee'
        TORCHINDUCTOR_FLEXATTENTION_MASKOUT : tl.constexpr = True
        IS_DIVISIBLE : tl.constexpr = True
        SM_SCALE : tl.constexpr = 0.25
        GQA_SHARED_HEADS : tl.constexpr = 1
        HAS_FULL_BLOCKS : tl.constexpr = True
        QK_HEAD_DIM : tl.constexpr = 16
        QK_HEAD_DIM_ROUNDED : tl.constexpr = 16
        V_HEAD_DIM : tl.constexpr = 16
        V_HEAD_DIM_ROUNDED : tl.constexpr = 16
        SAFE_HEAD_DIM : tl.constexpr = True
        SPARSE_MASK_STRIDE_BLK : tl.constexpr = 16384
        SPARSE_MASK_STRIDE_M : tl.constexpr = 128
        BLOCK_M : tl.constexpr = 64
        BLOCK_N : tl.constexpr = 64
        SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
        SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
        NUM_CUBE_CORE : tl.constexpr = 24
        ALLOW_TF32 : tl.constexpr = False
        INDEX_DTYPE : tl.constexpr = tl.int32
    
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")
        if not PRESCALE_QK:
            qk *= SM_SCALE
    
        m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
        n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)
    
        tmp1 = (qk)
        post_mod_scores = tmp1
    
    
    
    
        m_ij = tl.maximum(
            m_i,
            tl.max(post_mod_scores, 1, propagate_nan=True),
            propagate_nan=tl.PropagateNan.ALL,
        )
        if not ROWS_GUARANTEED_SAFE:
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
        else:
            m_ij_masked = m_ij
    
        alpha = tl.math.exp(m_i - m_ij_masked)
        p = tl.math.exp(post_mod_scores - m_ij_masked[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision="ieee")
        m_i = m_ij
        return acc, l_i, m_i
    
    
    @triton.jit
    def get_bounded_indices(indices, max_len=None):
        return indices % max_len if max_len is not None else indices

    # Synthetic inputs sized for the max offsets baked into the kernel
    # (strides qz/qh 256/16, Q_LEN=KV_LEN=8192, HQ=16).
    Q = torch.zeros(2097152, dtype=torch.bfloat16, device=device)
    K = torch.zeros(2097152, dtype=torch.bfloat16, device=device)
    V = torch.zeros(2097152, dtype=torch.bfloat16, device=device)
    SPARSE_MASK = torch.zeros(64 * 16384, dtype=torch.int8, device=device)
    Q_OFFSETS = torch.zeros(64, dtype=torch.int32, device=device)
    KV_NUM_BLKS = torch.ones(64, dtype=torch.int32, device=device)
    KV_IDX = torch.zeros(4096, dtype=torch.int32, device=device)
    LSE = torch.zeros(131072, dtype=torch.float32, device=device)
    MAX_SCORES = torch.zeros(131072, dtype=torch.float32, device=device)
    FULL_KV_NUM_BLKS = torch.zeros(64, dtype=torch.int32, device=device)
    FULL_KV_IDX = torch.zeros(4096, dtype=torch.int32, device=device)
    out = torch.zeros(2097152, dtype=torch.bfloat16, device=device)

    triton_flex_attention_fwd_mask_out[(24,)](
        Q, K, V, SPARSE_MASK, Q_OFFSETS, KV_NUM_BLKS, KV_IDX, LSE, MAX_SCORES,
        FULL_KV_NUM_BLKS, FULL_KV_IDX, out,
    )
