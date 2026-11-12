"""BF16 paged FlashMLA, with FP32 online state and LSE."""

import math
import triton
import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def attention(
    Q,
    KV,
    BLOCK_TABLE,
    KV_LENGTHS,
    NUMERATOR,
    MAXIMUM,
    DENOMINATOR,
    BATCH: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    SPLIT_PAGES: tl.constexpr,
    SPLITS: tl.constexpr,
    CUBE_CORES: tl.constexpr,
    SCALE: tl.constexpr,
    Q_TILE_ROWS: tl.constexpr,
    KV_L2: tl.constexpr,
    OUT,
    LSE,
    FUSE_MERGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):

    # Hardware tiles and NZ layouts. QK spans latent 512 plus RoPE 64.
    PAGE: tl.constexpr = 128
    VALUE_DIM: tl.constexpr = 512
    ROPE_DIM: tl.constexpr = 64
    QK_DIM: tl.constexpr = VALUE_DIM + ROPE_DIM
    STREAM_MERGE: tl.constexpr = FUSE_MERGE
    MERGE_ARENA: tl.constexpr = FUSE_MERGE
    KEY_ROWS: tl.constexpr = 112
    SLOT_ELEMS: tl.constexpr = (VALUE_DIM + (Q_TILE_ROWS if Q_TILE_ROWS > ROPE_DIM else ROPE_DIM)) * KEY_ROWS
    QK_K: tl.constexpr = 96
    QK_CHUNKS: tl.constexpr = (96, 112, 128, 112, 128)
    QK_OFFSETS: tl.constexpr = (0, 6, 13, 21, 28)
    PV_K: tl.constexpr = 64
    PV_N: tl.constexpr = 256
    PV_CHUNKS: tl.constexpr = (PV_K, KEY_ROWS - PV_K)
    Q_L0_SHAPE: tl.constexpr = (QK_K // 16, Q_TILE_ROWS // 16, 16, 16)
    P_L0_SHAPE: tl.constexpr = (KEY_ROWS // 16, Q_TILE_ROWS // 16, 16, 16)
    L0B_BANK_ELEMENTS: tl.constexpr = PV_K * PV_N
    L0C_BANK_ELEMENTS: tl.constexpr = 32768
    L0C_EVENTS: tl.constexpr = 2
    KV_L1_SHAPE: tl.constexpr = (QK_DIM // 16, KEY_ROWS // 16, 16, 16)
    V_L1_SHAPE: tl.constexpr = (VALUE_DIM // 16, KEY_ROWS // 16, 16, 16)
    tl.static_assert(Q_TILE_ROWS <= 96 and QK_DIM % QK_K == 0)
    tl.static_assert(SPLIT_PAGES * PAGE % KEY_ROWS == 0)
    VECTOR_ROWS: tl.constexpr = Q_TILE_ROWS // 2
    MERGE_ROWS: tl.constexpr = 16 if MERGE_ARENA else 4
    QUERY_TILES: tl.constexpr = QUERY_LENGTH * QUERY_HEADS // Q_TILE_ROWS
    REQUEST_K_TILES: tl.constexpr = (MAX_PAGES * PAGE + KEY_ROWS - 1) // KEY_ROWS
    GLOBAL_K_TILES: tl.constexpr = BATCH * QUERY_TILES * REQUEST_K_TILES
    CORE_K_TILES: tl.constexpr = (GLOBAL_K_TILES + CUBE_CORES - 1) // CUBE_CORES
    STREAM_K_SPLITS: tl.constexpr = min(CUBE_CORES, (REQUEST_K_TILES + 2 * CORE_K_TILES - 2) // CORE_K_TILES)
    STREAM_K: tl.constexpr = FUSE_MERGE and SPLITS >= STREAM_K_SPLITS
    JOBS: tl.constexpr = BATCH * QUERY_TILES * SPLITS
    ALL_ROWS: tl.constexpr = BATCH * QUERY_LENGTH * QUERY_HEADS

    # Cross-core credits protect score, probability slots, and PV bands.
    SCORE_READY: tl.constexpr = 0
    SCORE_FREE: tl.constexpr = 1
    PROB_FLAG_BASE: tl.constexpr = 10
    PRODUCT_READY: tl.constexpr = 4
    PRODUCT_FREE: tl.constexpr = 5
    BAND_READY: tl.constexpr = (PRODUCT_READY, 8)
    BAND_FREE: tl.constexpr = (PRODUCT_FREE, 9)
    JOB_DONE: tl.constexpr = 6
    JOB_ACK: tl.constexpr = 7

    # Three L1 slots share K/V/P. P overwrites RoPE after QK consumes it.
    # The Q prefix remains in L0A while scratch regions alternate Q/P fragments.
    query = tla.alloc((QK_DIM // 16, Q_TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l1')
    slots = tla.alloc((3 * SLOT_ELEMS,), tl.bfloat16, space='l1')
    q_first = tla.alloc(Q_L0_SHAPE, tl.bfloat16, space='l0a')
    q_even = tla.alloc((7, Q_TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l0a')
    q_odd = tla.alloc((8, Q_TILE_ROWS // 16, 16, 16), tl.bfloat16, space='l0a')
    b_banks = tla.alloc((2 * L0B_BANK_ELEMENTS,), tl.bfloat16, space='l0b')
    c_banks = tla.alloc((2 * L0C_BANK_ELEMENTS,), tl.float32, space='l0c')
    P_PACK_SHAPE: tl.constexpr = (PAGE // 16, (VECTOR_ROWS + 1) * 16)
    PARTIAL_SHAPE: tl.constexpr = (2 * VECTOR_ROWS, PV_N)

    # Attention and FlashDecode reuse UB storage across the AIV fence.
    if MERGE_ARENA:
        SCORE_WORDS: tl.constexpr = VECTOR_ROWS * PAGE
        VALUE_WORDS: tl.constexpr = VECTOR_ROWS * VALUE_DIM
        STATE_OFFSET: tl.constexpr = SCORE_WORDS + 2 * VALUE_WORDS
        MERGE_NUM_WORDS: tl.constexpr = 3 * MERGE_ROWS * VALUE_DIM
        MERGE_STAT_WORDS: tl.constexpr = SPLITS * MERGE_ROWS
        MERGE_READ_WORDS: tl.constexpr = MERGE_STAT_WORDS + (64 - MERGE_ROWS)
        WEIGHT_OFFSET: tl.constexpr = MERGE_NUM_WORDS + 2 * MERGE_READ_WORDS
        LSE_OFFSET: tl.constexpr = WEIGHT_OFFSET + MERGE_STAT_WORDS
        INVERSE_OFFSET: tl.constexpr = LSE_OFFSET + MERGE_ROWS
        F32_WORDS: tl.constexpr = max(STATE_OFFSET + 256, INVERSE_OFFSET + MERGE_ROWS)
        BF16_WORDS: tl.constexpr = max(PAGE * (VECTOR_ROWS + 1), MERGE_ROWS * VALUE_DIM)
        float_arena = tla.alloc((F32_WORDS,), tl.float32, space='ub')
        bf16_arena = tla.alloc((BF16_WORDS,), tl.bfloat16, space='ub')
        score = tla.reshape(tla.view(float_arena, (SCORE_WORDS,)), (VECTOR_ROWS, PAGE))
        partial = tla.reshape(tla.view(float_arena, (VALUE_WORDS,), offsets=(SCORE_WORDS,)), PARTIAL_SHAPE)
        accumulator = tla.reshape(
            tla.view(float_arena, (VALUE_WORDS,), offsets=(SCORE_WORDS + VALUE_WORDS,)),
            (VECTOR_ROWS, VALUE_DIM),
        )
        maximum = tla.view(float_arena, (64,), offsets=(STATE_OFFSET,))
        denominator = tla.view(float_arena, (64,), offsets=(STATE_OFFSET + 64,))
        corrections = tla.view(float_arena, (128,), offsets=(STATE_OFFSET + 128,))
        packed_p = tla.reshape(tla.view(bf16_arena, (PAGE * (VECTOR_ROWS + 1),)), P_PACK_SHAPE)
    else:
        score = tla.alloc((VECTOR_ROWS, PAGE), tl.float32, space='ub')
        packed_p = tla.alloc(P_PACK_SHAPE, tl.bfloat16, space='ub')
        partial = tla.alloc(PARTIAL_SHAPE, tl.float32, space='ub')
        maximum = tla.alloc((64,), tl.float32, space='ub')
        denominator = tla.alloc((64,), tl.float32, space='ub')
        corrections = tla.alloc((128,), tl.float32, space='ub')
        accumulator = tla.alloc((VECTOR_ROWS, VALUE_DIM), tl.float32, space='ub')
    tile_sums = tla.alloc((128,), tl.float32, space='ub')
    sub = tla.sub_block_id()

    # Distribute the global KV tile stream over the runtime Cube grid.
    core = tl.program_id(0)
    if STREAM_K:
        core_first = tl.minimum(core * CORE_K_TILES, GLOBAL_K_TILES)
        core_end = tl.minimum(core_first + CORE_K_TILES, GLOBAL_K_TILES)
        job_start = core_first // REQUEST_K_TILES
        job_end = tl.where(core_first < core_end, tl.cdiv(core_end, REQUEST_K_TILES), job_start)
        job_stride: tl.constexpr = 1
    else:
        job_start = core
        job_end = JOBS
        job_stride: tl.constexpr = CUBE_CORES
    for job in tl.range(job_start, job_end, job_stride):
        if STREAM_K:
            batch = job // QUERY_TILES
            tile = job % QUERY_TILES
            request_first = job * REQUEST_K_TILES
            split = core - request_first // CORE_K_TILES
        else:
            batch = job // (QUERY_TILES * SPLITS)
            split = job // QUERY_TILES % SPLITS
            tile = job % QUERY_TILES
        request_row = tile * Q_TILE_ROWS
        q_row = batch * QUERY_LENGTH * QUERY_HEADS + request_row
        first_page = split * SPLIT_PAGES
        length = tl.load(KV_LENGTHS + batch)
        last_query = (request_row + Q_TILE_ROWS - 1).to(tl.int32) // QUERY_HEADS
        last_page = (length - QUERY_LENGTH + last_query) // PAGE
        num_pages = tl.maximum(0, tl.minimum(SPLIT_PAGES, last_page + 1 - first_page))
        if STREAM_K:
            visible_tiles = tl.cdiv(length - QUERY_LENGTH + last_query + 1, KEY_ROWS)
            intersection_first = tl.maximum(request_first, core_first)
            intersection_end = tl.minimum(request_first + visible_tiles, core_end)
            first_token = (intersection_first - request_first) * KEY_ROWS
            num_tiles = tl.maximum(0, intersection_end - intersection_first)
        else:
            first_token = first_page * PAGE
            num_tiles = tl.cdiv(num_pages * PAGE, KEY_ROWS)
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + q_row * QK_DIM, (Q_TILE_ROWS, QK_DIM)), query)
            tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 3)
            tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 3)
            for event in tl.static_range(3):
                tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, event)
            for event in tl.static_range(L0C_EVENTS):
                tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, event)
            for event in tl.static_range(2):
                tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, event)
            tla.copy_l1_to_l0(
                query,
                q_first,
                tla.CopyL1ToL0Params(
                    k_part_idx=0,
                    k_part=QK_K,
                    k_part_ceil=QK_K,
                    k_part_loop=QK_DIM // QK_K,
                    k_part_actual=QK_K,
                    m=Q_TILE_ROWS,
                ),
            )
        with tla.scope.vector():
            tla.sync.set_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
            tla.sync.set_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 1)
            tla.sync.wait_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 1)
            with tla.scope.simd():
                tla.simd.vsts(maximum[0], tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64))
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                tla.simd.vsts(denominator[0], zero)
            if num_tiles == 0:
                with tla.scope.simd():
                    zero = tla.simd.vfull(0.0, tl.float32, 64)
                    for row in tl.range(VECTOR_ROWS):
                        for col in tl.static_range(0, VALUE_DIM, 64):
                            tla.simd.vsts(accumulator[row, col], zero)

        # Tick t prefetches t+1, computes QK(t), and consumes PV(t-1).
        # The -1 and final ticks fill and drain these dependencies.
        for step in tl.range(-1, num_tiles + 1):
            has_qk = (step >= 0) & (step < num_tiles)
            has_pv = step > 0
            slot = step % 3
            correction_slot = step % 2
            with tla.scope.cube():

                # Reclaim L1 only after the last PV operand load from that slot.
                if step + 1 < num_tiles:
                    key_start = first_token + (step + 1) * KEY_ROWS
                    page = key_start // PAGE
                    in_page = key_start % PAGE
                    first_count = tl.minimum(KEY_ROWS, PAGE - in_page)
                    next_id = (step + 1) % 3
                    next_slot = next_id * SLOT_ELEMS
                    next_key = tla.reshape(
                        tla.view(slots, (QK_DIM * KEY_ROWS,), offsets=(next_slot,)), KV_L1_SHAPE
                    )
                    first = tl.load(BLOCK_TABLE + batch * MAX_PAGES + tl.minimum(page, last_page)).to(
                        tl.int64
                    )
                    tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, next_id)
                    tla.load_gm_to_l1(
                        tla.view(KV + first * PAGE * QK_DIM, (PAGE, QK_DIM)),
                        next_key,
                        tla.LoadGM2L1Params(l2_cache_ctrl=KV_L2, row_count=first_count, source_row=in_page),
                    )
                    if first_count < KEY_ROWS:
                        second = tl.load(
                            BLOCK_TABLE + batch * MAX_PAGES + tl.minimum(page + 1, last_page)
                        ).to(tl.int64)
                        tla.load_gm_to_l1(
                            tla.view(KV + second * PAGE * QK_DIM, (PAGE, QK_DIM)),
                            next_key,
                            tla.LoadGM2L1Params(
                                l2_cache_ctrl=KV_L2,
                                row_count=KEY_ROWS - first_count,
                                destination_row=first_count,
                            ),
                        )
                    tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, next_id)

                # Five QK fragments reuse the resident Q prefix.
                if has_qk:
                    key = tla.reshape(
                        tla.view(slots, (QK_DIM * KEY_ROWS,), offsets=(slot * SLOT_ELEMS,)), KV_L1_SHAPE
                    )
                    qk_event = step % 2
                    tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, slot)
                    tla.sync.wait_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, qk_event)
                    qk = tla.reshape(
                        tla.view(c_banks, (Q_TILE_ROWS * KEY_ROWS,), offsets=(qk_event * L0C_BANK_ELEMENTS,)),
                        P_L0_SHAPE,
                    )
                    for chunk in tl.static_range(5):
                        qk_bank = (step + chunk) % 2
                        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, qk_bank)
                        if chunk == 0:
                            q_operand = q_first
                        else:
                            q_fragment = query
                            q_operand = q_even if chunk % 2 else q_odd
                            q_fragment = tla.view(
                                query,
                                (QK_CHUNKS[chunk] // 16, Q_TILE_ROWS // 16, 16, 16),
                                offsets=(QK_OFFSETS[chunk], 0, 0, 0),
                            )
                            tla.copy_l1_to_l0(
                                q_fragment,
                                q_operand,
                                tla.CopyL1ToL0Params(
                                    k_part_idx=0,
                                    k_part=QK_CHUNKS[chunk],
                                    k_part_ceil=QK_CHUNKS[chunk],
                                    k_part_loop=1,
                                    k_part_actual=QK_CHUNKS[chunk],
                                    m=Q_TILE_ROWS,
                                ),
                            )
                        k_fragment = key
                        bank = tla.view(b_banks, (L0B_BANK_ELEMENTS,), offsets=(qk_bank * L0B_BANK_ELEMENTS,))
                        k_operand = tla.reshape(
                            tla.view(bank, (QK_CHUNKS[chunk] * KEY_ROWS,)),
                            (QK_CHUNKS[chunk] // 16, KEY_ROWS // 16, 16, 16),
                        )
                        k_fragment = tla.view(
                            key,
                            (QK_CHUNKS[chunk] // 16, KEY_ROWS // 16, 16, 16),
                            offsets=(QK_OFFSETS[chunk], 0, 0, 0),
                        )
                        tla.copy_l1_to_l0(
                            k_fragment,
                            k_operand,
                            tla.CopyL1ToL0Params(
                                is_transpose=True,
                                k_part_idx=0,
                                k_part=QK_CHUNKS[chunk],
                                k_part_ceil=QK_CHUNKS[chunk],
                                k_part_loop=1,
                                k_part_actual=QK_CHUNKS[chunk],
                                m=KEY_ROWS,
                            ),
                        )
                        tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, qk_bank)
                        tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, qk_bank)
                        tla.cube.mmad_l0(
                            q_operand,
                            k_operand,
                            qk,
                            m=Q_TILE_ROWS,
                            n=KEY_ROWS,
                            k=QK_CHUNKS[chunk],
                            clear_accum=chunk == 0,
                        )
                        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, qk_bank)
                    tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, qk_event)
                    tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, qk_event)
                    if has_pv:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, SCORE_FREE)
                    score_tile = tla.view(score, (VECTOR_ROWS, KEY_ROWS))
                    tla.copy_l0c_to_ub(
                        qk,
                        score_tile,
                        tla.FixpipeL0C2UBParams(
                            dma_mode=tla.FixpipeDMAMode.NZ2ND,
                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                            sub_block_idx=None,
                        ),
                    )
                    tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, qk_event)
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, SCORE_READY)

                # The two 256-column PV bands reuse the same L0A probability.
                if has_pv:
                    previous = (step + 2) % 3 * SLOT_ELEMS
                    value = tla.reshape(
                        tla.view(slots, (VALUE_DIM * KEY_ROWS,), offsets=(previous,)), V_L1_SHAPE
                    )
                    probability_tile = tla.reshape(
                        tla.view(
                            slots, (Q_TILE_ROWS * KEY_ROWS,), offsets=(previous + VALUE_DIM * KEY_ROWS,)
                        ),
                        P_L0_SHAPE,
                    )
                    tla.sync.block_wait_flag(
                        tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, PROB_FLAG_BASE + (step + 2) % 3
                    )
                    pv_phase = (step + ((step > 1) | has_qk).to(tl.int32)) % 2
                    for band in tl.static_range(VALUE_DIM // PV_N):
                        pv_event = (step + band + has_qk.to(tl.int32)) % 2
                        tla.sync.wait_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, pv_event)
                        v_band = tla.view(
                            value,
                            (PV_N // 16, KEY_ROWS // 16, 16, 16),
                            offsets=(band * (PV_N // 16), 0, 0, 0),
                        )
                        product = tla.reshape(
                            tla.view(c_banks, (Q_TILE_ROWS * PV_N,), offsets=(pv_event * L0C_BANK_ELEMENTS,)),
                            (PV_N // 16, Q_TILE_ROWS // 16, 16, 16),
                        )
                        for chunk in tl.static_range(2):
                            pv_bank = (pv_phase + chunk) % 2
                            tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, pv_bank)
                            if band == 0 and chunk == 0:
                                tla.copy_l1_to_l0(
                                    probability_tile,
                                    q_even,
                                    tla.CopyL1ToL0Params(
                                        k_part_idx=0,
                                        k_part=KEY_ROWS,
                                        k_part_ceil=KEY_ROWS,
                                        k_part_loop=1,
                                        k_part_actual=KEY_ROWS,
                                        m=Q_TILE_ROWS,
                                    ),
                                )
                            p_operand = tla.view(
                                q_even,
                                (PV_CHUNKS[chunk] // 16, Q_TILE_ROWS // 16, 16, 16),
                                offsets=(chunk * (PV_K // 16), 0, 0, 0),
                            )
                            bank = tla.view(
                                b_banks, (L0B_BANK_ELEMENTS,), offsets=(pv_bank * L0B_BANK_ELEMENTS,)
                            )
                            v_operand = tla.reshape(
                                tla.view(bank, (PV_CHUNKS[chunk] * PV_N,)),
                                (PV_CHUNKS[chunk] // 16, PV_N // 16, 16, 16),
                            )
                            tla.copy_l1_to_l0(
                                v_band,
                                v_operand,
                                tla.CopyL1ToL0Params(
                                    k_part_idx=chunk,
                                    k_part=PV_K,
                                    k_part_ceil=PV_CHUNKS[chunk],
                                    k_part_loop=2,
                                    k_part_actual=PV_CHUNKS[chunk],
                                    m=PV_N,
                                ),
                            )
                            if band == 1 and chunk == 1:
                                tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, (step + 2) % 3)
                            tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, pv_bank)
                            tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, pv_bank)
                            tla.cube.mmad_l0(
                                p_operand,
                                v_operand,
                                product,
                                m=Q_TILE_ROWS,
                                n=PV_N,
                                k=PV_CHUNKS[chunk],
                                clear_accum=chunk == 0,
                            )
                            tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, pv_bank)
                        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, pv_event)
                        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, pv_event)
                        if step > 1:
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, BAND_FREE[band])
                        band_partial = tla.view(partial, (VECTOR_ROWS, PV_N), offsets=(band * VECTOR_ROWS, 0))
                        tla.copy_l0c_to_ub(
                            product,
                            band_partial,
                            tla.FixpipeL0C2UBParams(
                                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                sub_block_idx=None,
                            ),
                        )
                        tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, pv_event)
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, BAND_READY[band])
            with tla.scope.vector():
                if has_qk:
                    first_row = request_row + sub * VECTOR_ROWS
                    remaining = (
                        length - QUERY_LENGTH + 1 - first_token - step * KEY_ROWS + first_row // QUERY_HEADS
                    ).to(tl.int32)
                    correction = tla.view(corrections, (64,), offsets=(correction_slot * 64,))
                    tile_sum = tla.view(tile_sums, (64,), offsets=(correction_slot * 64,))
                    tla.sync.wait_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, SCORE_READY)
                    for MASKED in tl.static_range(0, 2):
                        if (remaining < KEY_ROWS) == MASKED:
                            CAUSAL_ROWS: tl.constexpr = (
                                QUERY_HEADS if MASKED and QUERY_HEADS % Q_TILE_ROWS != 0 else VECTOR_ROWS
                            )
                            with tla.scope.simd():
                                neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                                for query in tl.static_range(VECTOR_ROWS // CAUSAL_ROWS):
                                    if MASKED:
                                        end = tla.simd.vdup(remaining + query, tl.int32)
                                        valid0 = tla.simd.vcmp(tla.simd.vci(0), end, op=tla.simd.CMP.LT)
                                    end1 = tl.minimum(remaining + query, KEY_ROWS) if MASKED else KEY_ROWS
                                    tail = tla.simd.vdup(end1, tl.int32)
                                    valid1 = tla.simd.vcmp(tla.simd.vci(64), tail, op=tla.simd.CMP.LT)
                                    for i in tl.range(CAUSAL_ROWS):
                                        row = query * CAUSAL_ROWS + i
                                        x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                                        x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                                        if MASKED:
                                            x0 = tla.simd.vsel(valid0, x0, neg)
                                        x1 = tla.simd.vsel(valid1, x1, neg)
                                        old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                                        new_max = tla.simd.vmax(
                                            tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max
                                        )
                                        alpha = tla.simd.vexpdif(old_max, new_max)
                                        e0 = tla.simd.vexpdif(x0, new_max)
                                        e1 = tla.simd.vexpdif(x1, new_max)
                                        total = tla.simd.vbroadcast(
                                            tla.simd.vreduce_sum(tla.simd.vadd(e0, e1))
                                        )
                                        tla.simd.vsts(maximum[row], new_max, dist='ONEPT_B32')
                                        tla.simd.vsts(tile_sum[row], total, dist='ONEPT_B32')
                                        tla.simd.vsts(correction[row], alpha, dist='ONEPT_B32')
                                        p0 = tla.simd.vtruncf(e0, tl.bfloat16)
                                        p1 = tla.simd.vtruncf(e1, tl.bfloat16)
                                        interleaved, _ = tla.simd.vdintlv(p0, p1)
                                        tla.simd.vsstb(
                                            interleaved, packed_p[0, row * 16], (VECTOR_ROWS + 1) * 16
                                        )
                                tla.simd.membar('VST_VLD')
                                row_mask = tla.simd.vcmp(
                                    tla.simd.vci(0),
                                    tla.simd.vfull(VECTOR_ROWS, tl.int32, 64),
                                    op=tla.simd.CMP.LT,
                                )
                                running = tla.simd.vfma(
                                    tla.simd.vlds(tile_sum[0]),
                                    tla.simd.vlds(denominator[0]),
                                    tla.simd.vlds(correction[0]),
                                )
                                tla.simd.vsts(denominator[0], running, mask=row_mask)
                    if step + 1 < num_tiles:
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, SCORE_FREE)
                    destination = tla.reshape(
                        tla.view(
                            slots,
                            (Q_TILE_ROWS * KEY_ROWS,),
                            offsets=(slot * SLOT_ELEMS + VALUE_DIM * KEY_ROWS,),
                        ),
                        (KEY_ROWS // 16, Q_TILE_ROWS * 16),
                    )
                    tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
                    tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
                    tla.copy_ub_to_l1(
                        tla.view(packed_p, (KEY_ROWS // 16, VECTOR_ROWS * 16)),
                        tla.view(
                            destination,
                            (KEY_ROWS // 16, VECTOR_ROWS * 16),
                            offsets=(0, sub * VECTOR_ROWS * 16),
                        ),
                    )
                    tla.sync.set_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, PROB_FLAG_BASE + slot)
                if has_pv:
                    correction = tla.view(corrections, (64,), offsets=((1 - correction_slot) * 64,))
                    for band in tl.static_range(2):
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, BAND_READY[band])
                        result = tla.view(partial, (VECTOR_ROWS, PV_N), offsets=(band * VECTOR_ROWS, 0))
                        if step == 1:
                            with tla.scope.simd():
                                for row in tl.range(VECTOR_ROWS):
                                    for col in tl.static_range(0, PV_N, 64):
                                        tla.simd.vsts(
                                            accumulator[row, band * PV_N + col],
                                            tla.simd.vlds(result[row, col]),
                                        )
                        else:
                            with tla.scope.simd():
                                for row in tl.range(VECTOR_ROWS):
                                    alpha = tla.simd.vlds(correction[row], dist='BRC_B32')
                                    for col in tl.static_range(0, PV_N, 64):
                                        update = tla.simd.vfma(
                                            tla.simd.vlds(result[row, col]),
                                            tla.simd.vlds(accumulator[row, band * PV_N + col]),
                                            alpha,
                                        )
                                        tla.simd.vsts(accumulator[row, band * PV_N + col], update)
                        if has_qk:
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, BAND_FREE[band])
        with tla.scope.vector():
            tla.sync.wait_flag(tla.sync.PIPE.MTE3, tla.sync.PIPE.V, 0)
            tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
            tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 0)
            output_row = q_row + sub * VECTOR_ROWS
            if FUSE_MERGE:
                tl.static_assert(VECTOR_ROWS % MERGE_ROWS == 0)
                workspace_row = (output_row // MERGE_ROWS * SPLITS + split) * MERGE_ROWS
                values: tl.constexpr = (VECTOR_ROWS // MERGE_ROWS, MERGE_ROWS, VALUE_DIM)
                stats: tl.constexpr = (VECTOR_ROWS // MERGE_ROWS, MERGE_ROWS)
                stats_stride: tl.constexpr = (SPLITS * MERGE_ROWS, 1)
                tla.store_ub_to_gm(
                    tla.reshape(accumulator, values),
                    tla.view(
                        NUMERATOR + workspace_row * VALUE_DIM,
                        values,
                        strides=(SPLITS * MERGE_ROWS * VALUE_DIM, VALUE_DIM, 1),
                    ),
                )
                tla.store_ub_to_gm(
                    tla.reshape(tla.view(maximum, (VECTOR_ROWS,)), stats),
                    tla.view(MAXIMUM + workspace_row, stats, strides=stats_stride),
                )
                tla.store_ub_to_gm(
                    tla.reshape(tla.view(denominator, (VECTOR_ROWS,)), stats),
                    tla.view(DENOMINATOR + workspace_row, stats, strides=stats_stride),
                )
            else:
                workspace_row = split * ALL_ROWS + output_row
                tla.store_ub_to_gm(
                    accumulator, tla.view(NUMERATOR + workspace_row * VALUE_DIM, (VECTOR_ROWS, VALUE_DIM))
                )
                tla.store_ub_to_gm(
                    tla.view(maximum, (VECTOR_ROWS,)), tla.view(MAXIMUM + workspace_row, (VECTOR_ROWS,))
                )
                tla.store_ub_to_gm(
                    tla.view(denominator, (VECTOR_ROWS,)),
                    tla.view(DENOMINATOR + workspace_row, (VECTOR_ROWS,)),
                )
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, JOB_DONE)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, JOB_ACK)
        with tla.scope.cube():
            for event in tl.static_range(3):
                tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.MTE2, event)
            for event in tl.static_range(L0C_EVENTS):
                tla.sync.wait_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, event)
            for event in tl.static_range(2):
                tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, event)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, JOB_DONE)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, JOB_ACK)

    # All AIV writers complete before any AIV starts FlashDecode.
    if FUSE_MERGE:
        with tla.scope.vector():
            tla.sync.barrier(tla.sync.PIPE.MTE3)
            tla.sync.sync_all(False)
            worker = tl.program_id(0) * 2 + sub.to(tl.int32)
            ROW_TILE: tl.constexpr = MERGE_ROWS
            STAT_VALUES: tl.constexpr = SPLITS * ROW_TILE
            STAT_ELEMENTS: tl.constexpr = STAT_VALUES + (64 - ROW_TILE)
            tl.static_assert(ALL_ROWS % ROW_TILE == 0)
            numerators = tla.reshape(
                tla.view(float_arena, (2 * ROW_TILE * VALUE_DIM,)), (2, ROW_TILE, VALUE_DIM)
            )
            reduced = tla.reshape(
                tla.view(float_arena, (ROW_TILE * VALUE_DIM,), offsets=(2 * ROW_TILE * VALUE_DIM,)),
                (ROW_TILE, VALUE_DIM),
            )
            maxima = tla.view(float_arena, (STAT_ELEMENTS,), offsets=(MERGE_NUM_WORDS,))
            sums = tla.view(float_arena, (STAT_ELEMENTS,), offsets=(MERGE_NUM_WORDS + MERGE_READ_WORDS,))
            weights = tla.reshape(
                tla.view(float_arena, (STAT_VALUES,), offsets=(WEIGHT_OFFSET,)), (SPLITS, ROW_TILE)
            )
            merged_output = tla.reshape(tla.view(bf16_arena, (ROW_TILE * VALUE_DIM,)), (ROW_TILE, VALUE_DIM))
            inverse_sums = tla.view(float_arena, (ROW_TILE,), offsets=(INVERSE_OFFSET,))
            with tla.scope.simd():
                zeros = tla.simd.vfull(0.0, tl.float32, 64)
                tla.simd.vsts(maxima[(SPLITS - 1) * ROW_TILE], zeros)
                tla.simd.vsts(sums[(SPLITS - 1) * ROW_TILE], zeros)
            tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, 2)
            tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, 2)
            if RETURN_LSE:
                result_lse = tla.view(float_arena, (ROW_TILE,), offsets=(LSE_OFFSET,))
            for merge_tile in tl.range(worker, ALL_ROWS // ROW_TILE, 2 * CUBE_CORES):
                row_start = merge_tile * ROW_TILE
                if STREAM_K:
                    request_first_tile = row_start // Q_TILE_ROWS * REQUEST_K_TILES
                    request_last_tile = request_first_tile + REQUEST_K_TILES - 1
                    merge_splits = request_last_tile // CORE_K_TILES - request_first_tile // CORE_K_TILES + 1
                else:
                    merge_splits = SPLITS
                start = merge_tile * SPLITS * ROW_TILE
                tla.load_gm_to_ub(tla.view(MAXIMUM + start, (STAT_VALUES,)), tla.view(maxima, (STAT_VALUES,)))
                tla.load_gm_to_ub(
                    tla.view(DENOMINATOR + start, (STAT_VALUES,)), tla.view(sums, (STAT_VALUES,))
                )
                tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 2)
                for preload_bank in tl.static_range(min(2, SPLITS)):
                    if not STREAM_K or preload_bank < merge_splits:
                        preload_source = NUMERATOR + (start + preload_bank * ROW_TILE) * VALUE_DIM
                        tla.load_gm_to_ub(
                            tla.view(preload_source, (1, ROW_TILE, VALUE_DIM)),
                            tla.view(numerators, (1, ROW_TILE, VALUE_DIM), offsets=(preload_bank, 0, 0)),
                        )
                        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, preload_bank + 3)
                tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 2)
                with tla.scope.simd():
                    zero = tla.simd.vfull(0.0, tl.float32, 64)
                    one = tla.simd.vfull(1.0, tl.float32, 64)
                    merge_mask = tla.simd.pge(ROW_TILE, 64)
                    row_maxima = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
                    for merge_split in tl.range(merge_splits):
                        stat = merge_split * ROW_TILE
                        row_maxima = tla.simd.vmax(row_maxima, tla.simd.vlds(maxima[stat]), mask=merge_mask)
                    row_sums = zero
                    for merge_split in tl.range(merge_splits):
                        stat = merge_split * ROW_TILE
                        weight = tla.simd.vexpdif(tla.simd.vlds(maxima[stat]), row_maxima, mask=merge_mask)
                        row_sums = tla.simd.vfma(row_sums, tla.simd.vlds(sums[stat]), weight)
                        tla.simd.vsts(weights[merge_split, 0], weight, mask=merge_mask)
                    tla.simd.vsts(
                        inverse_sums[0], tla.simd.vdiv(one, row_sums, mask=merge_mask), mask=merge_mask
                    )
                    if RETURN_LSE:
                        lse = tla.simd.vadd(
                            row_maxima, tla.simd.vln(row_sums, mask=merge_mask), mask=merge_mask
                        )
                        tla.simd.vsts(result_lse[0], lse, mask=merge_mask)
                    tla.simd.membar('VST_VLD')
                with tla.scope.simd():
                    zero = tla.simd.vfull(0.0, tl.float32, 64)
                    for row in tl.range(ROW_TILE):
                        for col in tl.static_range(0, VALUE_DIM, 64):
                            tla.simd.vsts(reduced[row, col], zero)
                for merge_split in tl.range(merge_splits):
                    merge_bank = merge_split % 2
                    event = merge_bank + 3
                    tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, event)
                    with tla.scope.simd():
                        for row in tl.range(ROW_TILE):
                            weight = tla.simd.vlds(weights[merge_split, row], dist='BRC_B32')
                            for col in tl.static_range(0, VALUE_DIM, 64):
                                value = tla.simd.vlds(numerators[merge_bank, row, col])
                                previous = tla.simd.vlds(reduced[row, col])
                                tla.simd.vsts(reduced[row, col], tla.simd.vfma(previous, value, weight))
                    if merge_split + 2 < merge_splits:
                        tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, event)
                        tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE2, event)
                        refill_source = NUMERATOR + (start + (merge_split + 2) * ROW_TILE) * VALUE_DIM
                        tla.load_gm_to_ub(
                            tla.view(refill_source, (1, ROW_TILE, VALUE_DIM)),
                            tla.view(numerators, (1, ROW_TILE, VALUE_DIM), offsets=(merge_bank, 0, 0)),
                        )
                        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, event)
                with tla.scope.simd():
                    for row in tl.range(ROW_TILE):
                        inv_denominator = tla.simd.vlds(inverse_sums[row], dist='BRC_B32')
                        for col in tl.static_range(0, VALUE_DIM, 128):
                            lo = tla.simd.vmul(tla.simd.vlds(reduced[row, col]), inv_denominator)
                            hi = tla.simd.vmul(tla.simd.vlds(reduced[row, col + 64]), inv_denominator)
                            packed, _ = tla.simd.vdintlv(
                                tla.simd.vtruncf(lo, tl.bfloat16), tla.simd.vtruncf(hi, tl.bfloat16)
                            )
                            tla.simd.vsts(merged_output[row, col], packed, dist='NORM_B16')
                tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 2)
                tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 2)
                tla.store_ub_to_gm(
                    merged_output, tla.view(OUT + row_start * VALUE_DIM, (ROW_TILE, VALUE_DIM))
                )
                if RETURN_LSE:
                    tla.store_ub_to_gm(result_lse, tla.view(LSE + row_start, (ROW_TILE,)))
                tla.sync.barrier(tla.sync.PIPE.ALL)


@triton.jit
def merge_splits(
    NUMERATOR,
    MAXIMUM,
    DENOMINATOR,
    OUT,
    LSE,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT_TILE: tl.constexpr,
    COL_TILE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    LATENT: tl.constexpr = 512
    col_tiles: tl.constexpr = LATENT // COL_TILE
    job = tl.program_id(0)
    row = job // col_tiles
    col = job % col_tiles * COL_TILE + tl.arange(0, COL_TILE)
    split = tl.arange(0, SPLIT_TILE)
    split_row = split * ROWS + row
    maxima = tl.load(MAXIMUM + split_row, split < SPLITS, -float('inf'))
    sums = tl.load(DENOMINATOR + split_row, split < SPLITS, 0.0)
    maximum = tl.max(maxima, 0)
    weight = tl.exp(maxima - maximum)
    denominator = tl.sum(sums * weight, 0)
    numerators = tl.load(NUMERATOR + split_row[:, None] * LATENT + col[None, :], split[:, None] < SPLITS, 0.0)
    result = tl.sum(numerators * weight[:, None], 0) / denominator
    tl.store(OUT + row * LATENT + col, result)
    if RETURN_LSE:
        tl.store(LSE + row, maximum + tl.log(denominator), job % col_tiles == 0)


def device_cores(device):
    """Read the available hardware grid through the active Triton driver."""
    import torch
    from triton.runtime import driver

    with torch.npu.device(device):
        properties = driver.active.utils.get_device_properties(device)
    return (properties['num_aicore'], properties['num_vectorcore'])


def prepare_call(
    q, kv, block_table, kv_lengths, *, scale=0.04, return_lse=True, head_tile=None, split_pages=None, kv_l2=4
):
    """Prepare paged MLA for packed BF16 Q and a single latent KV head.

    Q is [batch, query_length, query_heads, 576], KV is [pages, 128, 576].
    The first 512 KV columns supply both latent K and V, and the final
    64 columns supply RoPE K. Causality is aligned to the end of each KV
    sequence. Every actual KV length must be in [query_length, table capacity].
    The table must address valid physical pages below that actual length.
    A prepared call captures the tensor objects, and reads their current
    contents on every invocation. It allocates fresh output and workspace.
    """
    import torch

    if q.ndim != 4 or kv.ndim != 3:
        raise ValueError('Expected Q [B,Sq,H,576] and KV [pages,128,576]')
    batch, query_length, heads, width = q.shape
    if (
        min(batch, query_length, heads) <= 0
        or width != 576
        or kv.shape[0] <= 0
        or (kv.shape[1:] != (128, 576))
    ):
        raise ValueError('Expected nonempty packed Q/KV with 576 columns and 128-token pages')
    if block_table.ndim != 2 or block_table.shape[0] != batch or kv_lengths.shape != (batch,):
        raise ValueError('Expected block table [B,max_pages] and KV lengths [B]')
    if block_table.shape[1] * 128 < query_length:
        raise ValueError('The page table capacity must cover the query length')
    if q.dtype != torch.bfloat16 or kv.dtype != q.dtype:
        raise ValueError('Q and KV must both be BF16')
    if block_table.dtype != torch.int32 or kv_lengths.dtype != torch.int32:
        raise ValueError('Page metadata must be int32')
    tensors = (q, kv, block_table, kv_lengths)
    if q.device.type != 'npu' or any((t.device != q.device or not t.is_contiguous() for t in tensors)):
        raise ValueError('Inputs must be contiguous and on the same Ascend NPU')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Softmax scale must be finite and positive')
    if head_tile is None:
        head_tile = 96 if heads % 32 and query_length * heads % 96 == 0 else 64 if heads % 64 == 0 else 32
    if head_tile not in (32, 64, 96) or query_length * heads % head_tile:
        raise ValueError('Query rows must form complete 32-, 64- or 96-row tiles')
    if heads % head_tile and head_tile // 2 % heads:
        raise ValueError('Each Vector sub-block must cover complete query head groups')
    if split_pages is not None and (not isinstance(split_pages, int) or split_pages <= 0):
        raise ValueError('split_pages must be a positive integer')
    cube_cores, vector_cores = device_cores(q.device)
    if cube_cores <= 0 or vector_cores != 2 * cube_cores:
        raise ValueError(f'Expected two Vector cores per Cube core, got {cube_cores}:{vector_cores}')
    rows = batch * query_length * heads
    query_tiles = rows // head_tile
    max_pages = block_table.shape[1]
    if split_pages is None:
        groups = cube_cores // math.gcd(query_tiles, cube_cores)
        split_pages = triton.cdiv(max_pages, groups) if head_tile == 96 else 64
    split_pages = triton.cdiv(split_pages, 7) * 7
    splits = triton.cdiv(max_pages, split_pages)
    request_tiles = triton.cdiv(max_pages * 128, 112)
    tiles_per_core = triton.cdiv(query_tiles * request_tiles, cube_cores)
    stream_splits = min(cube_cores, triton.cdiv(request_tiles + tiles_per_core - 1, tiles_per_core))
    if stream_splits <= 7:
        splits = stream_splits
    fuse_merge = splits <= 7
    launch_cores = cube_cores if fuse_merge else min(cube_cores, query_tiles * splits)
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False,
                   enable_optimized_metaop=True)
    options.update(unit_flag=False, sync_solver=True, disable_auto_inject_sync=True)

    def call():
        numerator = torch.empty((splits, rows, 512), dtype=torch.float32, device=q.device)
        maximum = torch.empty((splits, rows), dtype=torch.float32, device=q.device)
        denominator = torch.empty_like(maximum)
        output = torch.empty((batch, query_length, heads, 512), dtype=q.dtype, device=q.device)
        lse = (
            torch.empty((batch, query_length, heads), dtype=torch.float32, device=q.device)
            if return_lse
            else maximum
        )
        args = (
            q,
            kv,
            block_table,
            kv_lengths,
            numerator,
            maximum,
            denominator,
            batch,
            query_length,
            heads,
            max_pages,
            split_pages,
            splits,
            launch_cores,
            scale,
            head_tile,
            kv_l2,
        )
        with torch.npu.device(q.device):
            attention[launch_cores,](*args, output, lse, fuse_merge, return_lse, **options)
            if not fuse_merge:
                merge_splits[rows,](
                    numerator,
                    maximum,
                    denominator,
                    output,
                    lse,
                    rows,
                    splits,
                    triton.next_power_of_2(splits),
                    512,
                    return_lse,
                    sanitize_overflow=False,
                )
        return (output, lse) if return_lse else output

    call.metadata = dict(
        cube_grid=launch_cores,
        vector_cores=vector_cores,
        head_tile=head_tile,
        split_pages=split_pages,
        splits=splits,
        kv_tile=112,
        fuse_merge=fuse_merge,
        return_lse=return_lse,
        scale=scale,
        kv_l2=kv_l2,
    )
    return call
