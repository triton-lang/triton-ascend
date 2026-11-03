"""BF16 sparse MLA prefill with a three-slot pipeline.

Supported contract: B=1, SQ=KV=8192, TopK=2048, H in {16,32,64,128},
latent/value dimension 512 plus a separate 64-wide Q/K RoPE, contiguous BSND
inputs, int32 indices. Indices outside [0,8192) are masked; each query needs at
least one valid index; duplicate valid indices are retained. The operator does
not compute an indexer or TopK.

Three latent key slots rotate through L1, so the gather of tile t+1 is not
blocked by BMM2 of tile t-1; the vector can gather while the cube runs BMM1 of
the current tile. Splitting the key tile into its 512-wide latent part and
64-wide RoPE part is what makes three key slots plus Q plus P fit in the
512 KiB L1; BMM1 is two accumulating MMADs into one L0C tile.

Per tile:
  Vector: gather(t+1), softmax(t), FP32 accumulator update(t-1)
  Cube  : BMM1(t) -> Fixpipe score(t), BMM2(t-1) -> Fixpipe PV(t-1)
The order matters: the vector releases the score tile (FLAG_SCORE_FREE) before
the next fixpipe may overwrite it; the next gather is ordered after the
accumulator update of tile t-1 (which itself waits for that tile's PV fixpipe);
and the next P store waits for BMM2 to release the previous P tile.
"""
from dataclasses import dataclass

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from triton.runtime import driver


@dataclass(frozen=True)
class Config:
    """Static tile sizes. Dispatch depends only on the public tensor shape."""

    heads: int
    gather_rows: int


def default_config(heads):
    if heads not in (16, 32, 64, 128):
        raise ValueError('Supported head counts are 16, 32, 64, 128')
    return Config(min(heads, 64), 16)


def device_cores(device):
    """Query runtime core limits, rather than assuming a physical core count."""
    with torch.npu.device(device):
        properties = driver.active.utils.get_device_properties(device)
    cube, vector = properties['num_aicore'], properties['num_vectorcore']
    if cube <= 0 or vector != 2 * cube:
        raise ValueError(f'This A5 kernel requires a 1:2 Cube/Vector ratio; got {cube}:{vector}')
    return cube, vector


@triton.jit
def _pack_kv(KV, KR, PACKED):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = tl.arange(0, 1024)
    latent = tl.load(KV + row[:, None] * 512 + col[None, :], col[None, :] < 512, 0)
    rope = tl.load(KR + row[:, None] * 64 + col[None, :] - 512, (col[None, :] >= 512) & (col[None, :] < 576), 0)
    tl.store(
        PACKED + row[:, None] * 576 + col[None, :],
        tl.where(col[None, :] < 512, latent, rope),
        col[None, :] < 576,
    )


# Cross-core event ids (0-15). Single ids are semaphores: producers and
# consumers walk the tiles in order, so one id carries the whole sequence.
FLAG_KV_READY = tl.constexpr(0)  # a gathered key tile is in L1
FLAG_SCORE_TILE = tl.constexpr(1)  # qk fixpipe produced the score tile
FLAG_SCORE_FREE = tl.constexpr(13)  # vector finished reading the score tile
FLAG_P_TILE = tl.constexpr(2)  # BF16 P tile landed in L1
FLAG_P_FREE = tl.constexpr(6)  # BMM2 finished reading the P tile
FLAG_PV_TILE = tl.constexpr(3)  # pv fixpipe produced the accumulator tile
FLAG_PV_FREE = tl.constexpr(4)  # vector folded the previous pv tile into acc
FLAG_JOB_DONE = tl.constexpr(10)  # vector stores finished
FLAG_JOB_ACK = tl.constexpr(11)  # cube observed them


@tla.jit
def _sparse_mla(
    Q,
    QR,
    IDS,
    IDS64,
    OUT,
    MAX,
    SUM,
    PACKED_KV,
    SQ: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    GR: tl.constexpr,
    CORES: tl.constexpr,
):
    BN: tl.constexpr = 128
    VM: tl.constexpr = BM // 2
    HEAD_TILES: tl.constexpr = H // BM
    TILES: tl.constexpr = TOPK // BN
    OUT_ROWS: tl.constexpr = 8
    GRP: tl.constexpr = GR + 1  # padded row stride used by the NZ packers
    # Cyclic jobs cover tails for any runtime core count; no H/core divisibility is required.
    for job in tl.range(tl.program_id(0), SQ * HEAD_TILES, CORES):
        token = job // HEAD_TILES
        head = job % HEAD_TILES * BM
        # L1: three 512-wide latent slots, two 64-wide RoPE slots, Q, one P tile.
        q_lat = tla.alloc((32, BM // 16, 16, 16), tl.bfloat16, space='l1')
        q_rope = tla.alloc((4, BM // 16, 16, 16), tl.bfloat16, space='l1')
        kv_lat = tla.alloc((3 * 32, BN // 16, 16, 16), tl.bfloat16, space='l1')
        kv_lat_flat = tla.reshape(kv_lat, (3 * 32, BN * 16))
        kv_rope = tla.alloc((2 * 4, BN // 16, 16, 16), tl.bfloat16, space='l1')
        kv_rope_flat = tla.reshape(kv_rope, (2 * 4, BN * 16))
        p = tla.alloc((BN // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
        pflat = tla.reshape(p, (BN // 16, BM * 16))
        qk_l0c = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        pv_l0c = tla.alloc((512 // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        score = tla.alloc((VM, BN), tl.float32, space='ub')
        pv = tla.alloc((VM, 512), tl.float32, space='ub')
        acc = tla.alloc((VM, 512), tl.float32, space='ub')
        output = tla.alloc((OUT_ROWS, 512), tl.bfloat16, space='ub')
        prob = tla.alloc((BN // 16, VM * 16 + 16), tl.bfloat16, space='ub')
        ids = tla.alloc((2 * BN, ), tl.int32, space='ub')
        maximum = tla.alloc((64, ), tl.float32, space='ub')
        denominator = tla.alloc((64, ), tl.float32, space='ub')
        corrections = tla.alloc((128, ), tl.float32, space='ub')
        # Pad each gathered row to a full 128-BF16 SIMD load for its RoPE tail.
        key0 = tla.alloc((GR, 640), tl.bfloat16, space='ub')
        key1 = tla.alloc((GR, 640), tl.bfloat16, space='ub')
        rope0 = tla.view(key0, (GR, 128), offsets=(0, 512))
        rope1 = tla.view(key1, (GR, 128), offsets=(0, 512))
        packed0 = tla.alloc((36, GRP * 16), tl.bfloat16, space='ub')
        packed1 = tla.alloc((36, GRP * 16), tl.bfloat16, space='ub')
        ids_view0 = tla.view(ids, (BN, ), offsets=(0, ))

        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + (token * H + head) * 512, (BM, 512)), q_lat)
            tla.load_gm_to_l1(tla.view(QR + (token * H + head) * 64, (BM, 64)), q_rope)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                neg = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
                tla.simd.vsts(maximum[0], neg)
                tla.simd.vsts(denominator[0], zero)
                for row in tl.range(VM):
                    for col in tl.static_range(8):
                        tla.simd.vsts(acc[row, col * 64], zero)
        # Prologue: gather tile 0; the loop's first iteration gathers tile 1
        # while the cube runs BMM1(0). The job-end handshake already drained
        # the previous job's key reads, so the first gather needs no free wait.
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.load_gm_to_ub(tla.view(IDS + token * TOPK, (BN, )), ids_view0)
            gather_start = 0
            for chunk in tl.range(BN // (4 * GR)):
                for stage in tl.static_range(2):
                    if stage == 0:
                        keys, ropes, packed = key0, rope0, packed0
                    else:
                        keys, ropes, packed = key1, rope1, packed1
                    position = sub * (BN // 2) + (chunk * 2 + stage) * GR
                    for pair in tl.static_range(GR // 2):
                        idx2 = tl.load(IDS64 + (token * TOPK + gather_start + position) // 2 + pair)
                        # Wrap addresses safely; softmax masks invalid indices using the original int32 values.
                        safe0 = (idx2 & 0xFFFFFFFF).to(tl.int32) & (L - 1)
                        safe1 = (idx2 >> 32).to(tl.int32) & (L - 1)
                        row = pair * 2
                        tla.load_gm_to_ub(
                            tla.view(PACKED_KV + safe0 * 576, (1, 576)),
                            tla.view(keys, (1, 576), offsets=(row, 0)),
                        )
                        tla.load_gm_to_ub(
                            tla.view(PACKED_KV + safe1 * 576, (1, 576)),
                            tla.view(keys, (1, 576), offsets=(row + 1, 0)),
                        )
                    with tla.scope.simd():
                        # Only the first 64 BF16 lanes of the RoPE tail are valid.
                        half = tla.simd.pge(64, 128)
                        for local_row in tl.range(GR):
                            for col in tl.static_range(4):
                                value = tla.simd.vlds(keys[local_row, col * 128])
                                tla.simd.vsstb(value, packed[col * 8, local_row * 16], GRP * 16)
                            rope = tla.simd.vlds(ropes[local_row, 0])
                            tla.simd.vsstb(rope, packed[32, local_row * 16], GRP * 16, mask=half)
                    tla.copy_ub_to_l1(
                        tla.view(packed, (32, GR * 16)),
                        tla.view(
                            kv_lat_flat,
                            (32, GR * 16),
                            offsets=(0, position * 16),
                        ),
                    )
                    tla.copy_ub_to_l1(
                        tla.view(packed, (4, GR * 16), offsets=(32, 0)),
                        tla.view(
                            kv_rope_flat,
                            (4, GR * 16),
                            offsets=(0, position * 16),
                        ),
                    )
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, FLAG_KV_READY)

        for t in tl.range(TILES):
            lat_cur = tla.view(kv_lat, (32, BN // 16, 16, 16), offsets=((t % 3) * 32, 0, 0, 0))
            rope_cur = tla.view(kv_rope, (4, BN // 16, 16, 16), offsets=((t % 2) * 4, 0, 0, 0))
            with tla.scope.cube():
                # BMM1(t): 512-wide latent part plus the 64-wide RoPE part.
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_KV_READY)
                tla.cube.mmad(q_lat, lat_cur, qk_l0c, transpose_b=True, clear_accum=True)
                tla.cube.mmad(q_rope, rope_cur, qk_l0c, transpose_b=True, clear_accum=False)
                if t >= 1:
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_SCORE_FREE)
                tla.copy_l0c_to_ub(
                    qk_l0c,
                    score,
                    tla.FixpipeL0C2UBParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND,
                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                        sub_block_idx=None,
                    ),
                )
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG_SCORE_TILE)
                if t >= 1:
                    # BMM2(t-1): P(t-1) x V(t-1), one 512-wide fixpipe.
                    lat_prev = tla.view(kv_lat, (32, BN // 16, 16, 16), offsets=(((t - 1) % 3) * 32, 0, 0, 0))
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_P_TILE)
                    tla.cube.mmad(p, lat_prev, pv_l0c, clear_accum=True)
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.M, FLAG_P_FREE)
                    if t >= 2:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PV_FREE)
                    tla.copy_l0c_to_ub(
                        pv_l0c,
                        pv,
                        tla.FixpipeL0C2UBParams(
                            dma_mode=tla.FixpipeDMAMode.NZ2ND,
                            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                            sub_block_idx=None,
                        ),
                    )
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG_PV_TILE)
            with tla.scope.vector():
                sub = tla.sub_block_id()
                if t + 1 < TILES:
                    # Slot (t+1)%3 was last read by BMM2(t-2); the RoPE slot by BMM1(t-1).
                    gather_start = (t + 1) * BN
                    tla.load_gm_to_ub(
                        tla.view(IDS + token * TOPK + gather_start, (BN, )),
                        tla.view(ids, (BN, ), offsets=(((t + 1) % 2) * BN, )),
                    )
                    # Rolling two-stage gather: keep one group of row DMAs in flight while
                    # the previous group is packed into nZ and copied to L1.
                    # Compile-time stages preserve the rolling two-buffer schedule:
                    # load 0, load 1, pack 0/load 2, pack 1/load 3, pack 2, pack 3.
                    for step in tl.static_range(6):
                        if step % 2 == 0:
                            keys, ropes, packed = key0, rope0, packed0
                        else:
                            keys, ropes, packed = key1, rope1, packed1
                        if step >= 2:
                            position = sub * (BN // 2) + (step - 2) * GR
                            with tla.scope.simd():
                                half = tla.simd.pge(64, 128)
                                for local_row in tl.range(GR):
                                    for col in tl.static_range(4):
                                        value = tla.simd.vlds(keys[local_row, col * 128])
                                        tla.simd.vsstb(value, packed[col * 8, local_row * 16], GRP * 16)
                                    rope = tla.simd.vlds(ropes[local_row, 0])
                                    tla.simd.vsstb(rope, packed[32, local_row * 16], GRP * 16, mask=half)
                            tla.copy_ub_to_l1(
                                tla.view(packed, (32, GR * 16)),
                                tla.view(kv_lat_flat, (32, GR * 16), offsets=(((t + 1) % 3) * 32, position * 16)),
                            )
                            tla.copy_ub_to_l1(
                                tla.view(packed, (4, GR * 16), offsets=(32, 0)),
                                tla.view(kv_rope_flat, (4, GR * 16), offsets=(((t + 1) % 2) * 4, position * 16)),
                            )
                        if step < 4:
                            position = sub * (BN // 2) + step * GR
                            for pair in tl.static_range(GR // 2):
                                idx2 = tl.load(IDS64 + (token * TOPK + gather_start + position) // 2 + pair)
                                safe0 = (idx2 & 0xFFFFFFFF).to(tl.int32) & (L - 1)
                                safe1 = (idx2 >> 32).to(tl.int32) & (L - 1)
                                row = pair * 2
                                tla.load_gm_to_ub(
                                    tla.view(PACKED_KV + safe0 * 576, (1, 576)),
                                    tla.view(keys, (1, 576), offsets=(row, 0)),
                                )
                                tla.load_gm_to_ub(
                                    tla.view(PACKED_KV + safe1 * 576, (1, 576)),
                                    tla.view(keys, (1, 576), offsets=(row + 1, 0)),
                                )
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, FLAG_KV_READY)
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_SCORE_TILE)
                ids_cur = tla.view(ids, (BN, ), offsets=((t % 2) * BN, ))
                corr_view = tla.view(corrections, (64, ), offsets=((t % 2) * 64, ))
                with tla.scope.simd():
                    neg = tla.simd.vfull(float('-inf'), tl.float32, 64)
                    scale = tla.simd.vsplat(1.0 / 24.0, tl.float32)
                    limit = tla.simd.vsplat(L, tl.int32)
                    i0 = tla.simd.vlds(ids_cur, (0, ))
                    i1 = tla.simd.vlds(ids_cur, (64, ))
                    # One unsigned compare rejects both index < 0 and index >= L.
                    valid0 = tla.simd.vcmp(i0, limit, op=tla.simd.CMP.ULT)
                    valid1 = tla.simd.vcmp(i1, limit, op=tla.simd.CMP.ULT)
                    for row in tl.range(VM):
                        x0 = tla.simd.vmul(tla.simd.vlds(score[row, 0]), scale)
                        x1 = tla.simd.vmul(tla.simd.vlds(score[row, 64]), scale)
                        x0 = tla.simd.vsel(valid0, x0, neg)
                        x1 = tla.simd.vsel(valid1, x1, neg)
                        tile_max = tla.simd.vmax(x0, x1)
                        old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                        new_max = tla.simd.vmax(tla.simd.vreduce_max(tile_max), old_max)
                        alpha = tla.simd.vexp(tla.simd.vsub(old_max, new_max))
                        p0 = tla.simd.vexp(tla.simd.vsub(x0, new_max))
                        p1 = tla.simd.vexp(tla.simd.vsub(x1, new_max))
                        partial_sum = tla.simd.vadd(p0, p1)
                        total = tla.simd.vbroadcast(tla.simd.vreduce_sum(partial_sum))
                        total = tla.simd.vadd(total,
                                              tla.simd.vmul(tla.simd.vlds(denominator[row], dist='BRC_B32'), alpha))
                        tla.simd.vsts(maximum[row], new_max, dist='ONEPT_B32')
                        tla.simd.vsts(denominator[row], total, dist='ONEPT_B32')
                        tla.simd.vsts(corr_view[row], alpha, dist='ONEPT_B32')
                        packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p0, tl.bfloat16),
                                                          tla.simd.vtruncf(p1, tl.bfloat16))
                        tla.simd.vsstb(packed, prob[0, row * 16], (VM + 1) * 16)
                if t < TILES - 1:
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, FLAG_SCORE_FREE)
                if t >= 1:
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_P_FREE)
                tla.copy_ub_to_l1(
                    tla.view(prob, (BN // 16, VM * 16)),
                    tla.view(pflat, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)),
                )
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, FLAG_P_TILE)
                if t >= 1:
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PV_TILE)
                    corr_prev = tla.view(corrections, (64, ), offsets=(((t - 1) % 2) * 64, ))
                    with tla.scope.simd():
                        # In-place FP32 accumulator: acc = pv + alpha * acc.
                        for row in tl.range(VM):
                            alpha = tla.simd.vlds(corr_prev[row], dist='BRC_B32')
                            for col in tl.static_range(8):
                                value = tla.simd.vfma(
                                    tla.simd.vlds(pv[row, col * 64]),
                                    tla.simd.vlds(acc[row, col * 64]),
                                    alpha,
                                )
                                tla.simd.vsts(acc[row, col * 64], value)
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, FLAG_PV_FREE)
        # Drain: BMM2(TILES-1) and the last accumulator update.
        with tla.scope.cube():
            lat_last = tla.view(kv_lat, (32, BN // 16, 16, 16), offsets=((TILES - 1) % 3 * 32, 0, 0, 0))
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_P_TILE)
            tla.cube.mmad(p, lat_last, pv_l0c, clear_accum=True)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PV_FREE)
            tla.copy_l0c_to_ub(
                pv_l0c,
                pv,
                tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                    sub_block_idx=None,
                ),
            )
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG_PV_TILE)
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PV_TILE)
            with tla.scope.simd():
                for row in tl.range(VM):
                    alpha = tla.simd.vlds(tla.view(corrections, (64, ), offsets=(64, ))[row], dist='BRC_B32')
                    for col in tl.static_range(8):
                        value = tla.simd.vfma(
                            tla.simd.vlds(pv[row, col * 64]),
                            tla.simd.vlds(acc[row, col * 64]),
                            alpha,
                        )
                        tla.simd.vsts(acc[row, col * 64], value)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, FLAG_PV_FREE)
            for row_block in tl.range(VM // OUT_ROWS):
                with tla.scope.simd():
                    for local_row in tl.range(OUT_ROWS):
                        row = row_block * OUT_ROWS + local_row
                        den = tla.simd.vlds(denominator[row], dist='BRC_B32')
                        for col in tl.static_range(4):
                            b0 = tla.simd.vtruncf(
                                tla.simd.vdiv(tla.simd.vlds(acc[row, col * 128]), den),
                                tl.bfloat16,
                            )
                            b1 = tla.simd.vtruncf(
                                tla.simd.vdiv(tla.simd.vlds(acc[row, col * 128 + 64]), den),
                                tl.bfloat16,
                            )
                            packed, unused = tla.simd.vdintlv(b0, b1)
                            tla.simd.vsts(output[local_row, col * 128], packed)
                tla.store_ub_to_gm(
                    output,
                    tla.view(
                        OUT + (token * H + head + sub * VM + row_block * OUT_ROWS) * 512,
                        (OUT_ROWS, 512),
                        strides=(512, 1),
                    ),
                )
            tla.store_ub_to_gm(tla.view(maximum, (VM, )), tla.view(MAX + token * H + head + sub * VM, (VM, )))
            tla.store_ub_to_gm(tla.view(denominator, (VM, )), tla.view(SUM + token * H + head + sub * VM, (VM, )))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, FLAG_JOB_DONE)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_JOB_ACK)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_JOB_DONE)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_JOB_ACK)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PV_FREE)


def forward(q, kv, qr, kr, indices, config=None, *, core_limit=None):
    """Return BF16 output and FP32 online-softmax maximum and sum.

    Outputs have shapes [1,8192,H,512], [1,1,8192,H], [1,1,8192,H].
    core_limit is a diagnostic grid cap, not a fabricated device property.
    Production launches use the runtime core count. Jobs cover every query and
    head tile cyclically, without assuming that heads divide the core count.
    """
    if q.ndim != 4:
        raise ValueError('q must have shape [1,8192,H,512]')
    h = q.shape[2]
    tensors = q, kv, qr, kr, indices
    expected = [(1, 8192, h, 512), (1, 8192, 1, 512), (1, 8192, h, 64), (1, 8192, 1, 64), (1, 8192, 1, 2048)]
    if h not in (16, 32, 64, 128) or any(tuple(t.shape) != s for t, s in zip(tensors, expected)):
        raise ValueError('Only the four documented B1/SQ8192/KV8192/TopK2048 specifications are supported')
    if q.device.type != 'npu' or any(t.device != q.device or not t.is_contiguous() for t in tensors):
        raise ValueError('Inputs must be contiguous tensors on the same NPU')
    if any(t.dtype != torch.bfloat16 for t in tensors[:4]) or indices.dtype != torch.int32:
        raise ValueError('Expected BF16 data and int32 indices')
    config = config or default_config(h)
    if config != default_config(h):
        raise ValueError('Only the validated shape-specific tile configuration is exposed')
    bm, gr = config.heads, config.gather_rows
    cores, _ = device_cores(q.device)
    if core_limit is not None:
        if not 1 <= core_limit <= cores:
            raise ValueError('core_limit must be between 1 and the runtime Cube count')
        cores = core_limit
    cores = min(cores, 8192 * (h // bm))
    packed_kv = torch.empty((8192, 576), dtype=q.dtype, device=q.device)
    output = torch.empty_like(q)
    maximum = torch.empty((1, 1, 8192, h), dtype=torch.float32, device=q.device)
    denominator = torch.empty_like(maximum)
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
    indices64 = indices.view(torch.int64)
    args = (q, qr, indices, indices64, output, maximum, denominator, packed_kv, 8192, h, 8192, 2048, bm, gr, cores)
    with torch.npu.device(q.device):
        _pack_kv[(1024, )](kv, kr, packed_kv, **options)
        _sparse_mla[(cores, )](*args, **options)
    return output, maximum, denominator
