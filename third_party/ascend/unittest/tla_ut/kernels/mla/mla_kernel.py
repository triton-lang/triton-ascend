"""Paged BF16 MLA decode using public Ascend TLA interfaces.

Q is BSND [1, 4, H, 576], KV is PA_BBND [pages, 128, 1, 576],
and block_table is int32 [1, 1024]. H is 96, 48, 24 or 12. The first
512 KV channels are also V; the last 64 participate only in QK.
All 131072 logical KV positions are present. Causal attention is bottom-right
aligned: query i sees positions below 131072 - 4 + i + 1.

Stage 1 partitions pages and flattened query/head rows between Cube blocks.
Each block uses two AIVs, FP32 online softmax and full-width 512-column PV.
Stage 2 rescales and merges FP32 split numerators/denominators, then casts once.
No input-dependent dispatch, cached input data or compiler modifications.
"""

import math

import torch
import torch_npu  # noqa: F401
import triton.language as tl
import triton.language.extra.tla as tla
from triton.runtime import driver


def device_cores(device):
    """Return runtime Cube/Vector counts; this kernel requires A5's 1:2 ratio."""
    with torch.npu.device(device):
        props = driver.active.utils.get_device_properties(device)
    cube, vector = props['num_aicore'], props['num_vectorcore']
    if cube < 1 or vector != 2 * cube:
        raise ValueError(f'Expected positive 1:2 Cube/Vector counts, got {cube}:{vector}')
    return cube, vector


def launch_config(heads, cube_cores, vector_cores):
    """Balance page splits without assuming a fixed number of physical cores."""
    if heads not in (96, 48, 24, 12):
        raise ValueError('Supported head counts: 96, 48, 24, 12')
    block_m = 96 if heads >= 24 else 48
    row_tiles = 4 * heads // block_m
    splits = min(1024, max(1, cube_cores // row_tiles))
    return dict(block_m=block_m, splits=splits, cube_grid=min(cube_cores, row_tiles * splits),
                vector_grid=min(vector_cores, 4 * heads))


@tla.jit
def _mla_split(Q, KV, TABLE, PART, MAXIMUM, DENOMINATOR, H: tl.constexpr, BM: tl.constexpr, SPLITS: tl.constexpr,
               CORES: tl.constexpr, SCALE: tl.constexpr, CAUSAL: tl.constexpr):
    BN: tl.constexpr = 128
    VM: tl.constexpr = BM // 2
    ROWS: tl.constexpr = 4 * H
    core = tl.program_id(0)
    for job in tl.range(core, (ROWS // BM) * SPLITS, CORES):
        row_start = (job // SPLITS) * BM
        split = job % SPLITS
        first_page = split * 1024 // SPLITS
        end_page = (split + 1) * 1024 // SPLITS

        # L1 NZ layout: [channels/16, rows/16, 16, 16].
        q = tla.alloc((36, BM // 16, 16, 16), tl.bfloat16, space='l1')
        kv0 = tla.alloc((36, 8, 16, 16), tl.bfloat16, space='l1')
        kv1 = tla.alloc((36, 8, 16, 16), tl.bfloat16, space='l1')
        p = tla.alloc((8, BM // 16, 16, 16), tl.bfloat16, space='l1')
        pflat = tla.reshape(p, (8, BM * 16))
        qk = tla.alloc((8, BM // 16, 16, 16), tl.float32, space='l0c')
        pv = tla.alloc((32, BM // 16, 16, 16), tl.float32, space='l0c')
        # QK and PV exchange buffers have disjoint lifetimes. A contiguous
        # prefix holds scores; flag 4 protects reuse after the preceding PV.
        exchange = tla.alloc((VM * 512, ), tl.float32, space='ub')
        score = tla.reshape(tla.view(exchange, (VM * BN, )), (VM, BN))
        product = tla.reshape(exchange, (VM, 512))
        accumulator = tla.alloc((VM, 512), tl.float32, space='ub')
        probability = tla.alloc((8, (VM + 1) * 16), tl.bfloat16, space='ub')
        maximum = tla.alloc((64, ), tl.float32, space='ub')
        denominator = tla.alloc((64, ), tl.float32, space='ub')
        correction = tla.alloc((64, ), tl.float32, space='ub')

        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + row_start * 576, (BM, 576)), q)
            page_id = tl.load(TABLE + first_page).to(tl.int64)
            tla.load_gm_to_l1(tla.view(KV + page_id * BN * 576, (BN, 576)), kv0)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                negative = tla.simd.vfull(float('-inf'), tl.float32, 64)
                tla.simd.vsts(maximum[0], negative)
                tla.simd.vsts(denominator[0], zero)
                for row in tl.range(VM):
                    for col in tl.static_range(8):
                        tla.simd.vsts(accumulator[row, col * 64], zero)
            # Flag 4 grants Cube permission to overwrite the PV destination UB.
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)

        # Each tile is one complete page. A scalar table lookup precedes ND2NZ;
        # no contiguous-page assumption, full-cache packing or GM P workspace.
        for page_pair in tl.range(first_page, end_page, 2):
            for phase in tl.static_range(2):
                page = page_pair + phase
                if page < end_page:
                    if phase == 0:
                        current, following = kv0, kv1
                    else:
                        current, following = kv1, kv0
                    with tla.scope.cube():
                        if page + 1 < end_page:
                            next_id = tl.load(TABLE + page + 1).to(tl.int64)
                            tla.load_gm_to_l1(tla.view(KV + next_id * BN * 576, (BN, 576)), following)
                        tla.cube.mmad(q, current, qk, transpose_b=True, clear_accum=True)
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                        tla.copy_l0c_to_ub(
                            qk, score,
                            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 1)
                    with tla.scope.vector():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                        sub = tla.sub_block_id()
                        # Only the last physical tile needs a causal mask:
                        # every earlier logical page is visible to all four queries.
                        if CAUSAL and page == 1023:
                            with tla.scope.simd():
                                negative = tla.simd.vfull(float('-inf'), tl.float32, 64)
                                positions0 = tla.simd.vci(page * BN)
                                positions1 = tla.simd.vci(page * BN + 64)
                                for row in tl.range(VM):
                                    logical_row = (row_start + sub * VM + row).to(tl.int32)
                                    bound = 131072 - 4 + logical_row // H + 1
                                    x0 = tla.simd.vlds(score[row, 0])
                                    x1 = tla.simd.vlds(score[row, 64])
                                    x0 = tla.simd.vsel(tla.simd.vcmps(positions0, bound, op=tla.simd.CMP.LT), x0,
                                                       negative)
                                    x1 = tla.simd.vsel(tla.simd.vcmps(positions1, bound, op=tla.simd.CMP.LT), x1,
                                                       negative)
                                    tla.simd.vsts(score[row, 0], x0)
                                    tla.simd.vsts(score[row, 64], x1)
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                x0 = tla.simd.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                                x1 = tla.simd.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                                old_max = tla.simd.vlds(maximum[row], dist='BRC_B32')
                                new_max = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_max)
                                alpha = tla.simd.vexpdif(old_max, new_max)
                                p0 = tla.simd.vexpdif(x0, new_max)
                                p1 = tla.simd.vexpdif(x1, new_max)
                                total = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(p0, p1)))
                                total = tla.simd.vadd(
                                    total, tla.simd.vmul(tla.simd.vlds(denominator[row], dist='BRC_B32'), alpha))
                                tla.simd.vsts(maximum[row], new_max, dist='ONEPT_B32')
                                tla.simd.vsts(denominator[row], total, dist='ONEPT_B32')
                                tla.simd.vsts(correction[row], alpha, dist='ONEPT_B32')
                                packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p0, tl.bfloat16),
                                                                  tla.simd.vtruncf(p1, tl.bfloat16))
                                tla.simd.vsstb(packed, probability[0, row * 16], (VM + 1) * 16)
                        tla.copy_ub_to_l1(tla.view(probability, (8, VM * 16)),
                                          tla.view(pflat, (8, VM * 16), offsets=(0, sub * VM * 16)))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 2)
                    with tla.scope.cube():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                        tla.cube.mmad(p, tla.view(current, (32, 8, 16, 16)), pv, clear_accum=True)
                        tla.copy_l0c_to_ub(
                            pv, product,
                            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 3)
                    with tla.scope.vector():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 3)
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                alpha = tla.simd.vlds(correction[row], dist='BRC_B32')
                                for col in tl.static_range(8):
                                    value = tla.simd.vfma(tla.simd.vlds(product[row, col * 64]),
                                                          tla.simd.vlds(accumulator[row, col * 64]), alpha)
                                    tla.simd.vsts(accumulator[row, col * 64], value)
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)

        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        with tla.scope.vector():
            sub = tla.sub_block_id()
            offset = split * ROWS + row_start + sub * VM
            tla.store_ub_to_gm(accumulator, tla.view(PART + offset * 512, (VM, 512)))
            tla.store_ub_to_gm(tla.view(maximum, (VM, )), tla.view(MAXIMUM + offset, (VM, )))
            tla.store_ub_to_gm(tla.view(denominator, (VM, )), tla.view(DENOMINATOR + offset, (VM, )))
            # Finish both AIV stores before this Cube block reuses job buffers.
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 7)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 7)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)


@tla.jit
def _mla_merge(PART, MAXIMUM, DENOMINATOR, OUT, LSE, ROWS: tl.constexpr, SPLITS: tl.constexpr, CORES: tl.constexpr):
    """Rescale splits to one maximum and reduce each output chunk in registers.

    A split stores unnormalized FP32 numerator, maximum and denominator.
    Weighting each numerator/denominator by exp(split_max - global_max)
    gives the same online-softmax result without renormalizing it per split.
    """
    if SPLITS <= 64:
        for row in tl.range(tl.program_id(0), ROWS, CORES):
            with tla.scope.vector():
                values = tla.alloc((SPLITS, 512), tl.float32, space='ub')
                split_maximum = tla.alloc((SPLITS, 8), tl.float32, space='ub')
                split_denominator = tla.alloc((SPLITS, 8), tl.float32, space='ub')
                weights = tla.alloc((64, ), tl.float32, space='ub')
                output = tla.alloc((512, ), tl.bfloat16, space='ub')
                logsum = tla.alloc((64, ), tl.float32, space='ub')
                tla.load_gm_to_ub(tla.view(MAXIMUM + row, (SPLITS, 1), strides=(ROWS, 1)),
                                  tla.view(split_maximum, (SPLITS, 1)))
                tla.load_gm_to_ub(tla.view(DENOMINATOR + row, (SPLITS, 1), strides=(ROWS, 1)),
                                  tla.view(split_denominator, (SPLITS, 1)))
                tla.load_gm_to_ub(tla.view(PART + row * 512, (SPLITS, 512), strides=(ROWS * 512, 1)), values)
                with tla.scope.simd():
                    maximum = tla.simd.vfull(float('-inf'), tl.float32, 64)
                    for split in tl.range(SPLITS):
                        maximum = tla.simd.vmax(maximum, tla.simd.vlds(split_maximum[split, 0], dist='BRC_B32'))
                    total = tla.simd.vfull(0.0, tl.float32, 64)
                    for split in tl.range(SPLITS):
                        weight = tla.simd.vexpdif(tla.simd.vlds(split_maximum[split, 0], dist='BRC_B32'), maximum)
                        total = tla.simd.vfma(total, tla.simd.vlds(split_denominator[split, 0], dist='BRC_B32'), weight)
                        tla.simd.vsts(weights[split], weight, dist='ONEPT_B32')
                    for col in tl.static_range(4):
                        left = tla.simd.vfull(0.0, tl.float32, 64)
                        right = tla.simd.vfull(0.0, tl.float32, 64)
                        for split in tl.range(SPLITS):
                            weight = tla.simd.vlds(weights[split], dist='BRC_B32')
                            left = tla.simd.vfma(left, tla.simd.vlds(values[split, col * 128]), weight)
                            right = tla.simd.vfma(right, tla.simd.vlds(values[split, col * 128 + 64]), weight)
                        packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(tla.simd.vdiv(left, total), tl.bfloat16),
                                                          tla.simd.vtruncf(tla.simd.vdiv(right, total), tl.bfloat16))
                        tla.simd.vsts(output[col * 128], packed)
                    tla.simd.vsts(logsum[0], tla.simd.vadd(maximum, tla.simd.vln(total)))
                tla.store_ub_to_gm(output, tla.view(OUT + row * 512, (512, )))
                tla.store_ub_to_gm(tla.view(logsum, (1, )), tla.view(LSE + row, (1, )))
    else:
        # At most 128 KiB of UB for split numerators. Larger core configurations
        # retain the scalar split traversal instead of exceeding UB capacity.
        BATCH: tl.constexpr = SPLITS if SPLITS <= 64 else 1
        for row in tl.range(tl.program_id(0), ROWS, CORES):
            with tla.scope.vector():
                values = tla.alloc((BATCH, 512), tl.float32, space='ub')
                accumulator = tla.alloc((512, ), tl.float32, space='ub')
                maximum = tla.alloc((64, ), tl.float32, space='ub')
                denominator = tla.alloc((64, ), tl.float32, space='ub')
                output = tla.alloc((512, ), tl.bfloat16, space='ub')
                logsum = tla.alloc((64, ), tl.float32, space='ub')
                split_maximum = tla.alloc((BATCH, 8), tl.float32, space='ub')
                split_denominator = tla.alloc((BATCH, 8), tl.float32, space='ub')
                with tla.scope.simd():
                    zero = tla.simd.vfull(0.0, tl.float32, 64)
                    tla.simd.vsts(maximum[0], tla.simd.vfull(float('-inf'), tl.float32, 64))
                    tla.simd.vsts(denominator[0], zero)
                    for col in tl.static_range(8):
                        tla.simd.vsts(accumulator[col * 64], zero)
                # Batch strided split reads: one DMA per tensor and chunk,
                # then merge all loaded splits in order inside one SIMD scope.
                # Eight-float statistics rows preserve 32-byte DMA alignment.
                for group in tl.range(SPLITS // BATCH):
                    offset = group * BATCH * ROWS + row
                    tla.load_gm_to_ub(tla.view(MAXIMUM + offset, (BATCH, 1), strides=(ROWS, 1)),
                                      tla.view(split_maximum, (BATCH, 1)))
                    tla.load_gm_to_ub(tla.view(DENOMINATOR + offset, (BATCH, 1), strides=(ROWS, 1)),
                                      tla.view(split_denominator, (BATCH, 1)))
                    tla.load_gm_to_ub(tla.view(PART + offset * 512, (BATCH, 512), strides=(ROWS * 512, 1)), values)
                    with tla.scope.simd():
                        for split in tl.range(BATCH):
                            part_max = tla.simd.vlds(split_maximum[split, 0], dist='BRC_B32')
                            old_max = tla.simd.vlds(maximum[0])
                            new_max = tla.simd.vmax(old_max, part_max)
                            alpha = tla.simd.vexpdif(old_max, new_max)
                            beta = tla.simd.vexpdif(part_max, new_max)
                            total = tla.simd.vadd(
                                tla.simd.vmul(tla.simd.vlds(denominator[0]), alpha),
                                tla.simd.vmul(beta, tla.simd.vlds(split_denominator[split, 0], dist='BRC_B32')))
                            tla.simd.vsts(maximum[0], new_max)
                            tla.simd.vsts(denominator[0], total)
                            for col in tl.static_range(8):
                                value = tla.simd.vadd(tla.simd.vmul(tla.simd.vlds(accumulator[col * 64]), alpha),
                                                      tla.simd.vmul(tla.simd.vlds(values[split, col * 64]), beta))
                                tla.simd.vsts(accumulator[col * 64], value)
                with tla.scope.simd():
                    den = tla.simd.vlds(denominator[0])
                    for col in tl.static_range(4):
                        left = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(accumulator[col * 128]), den), tl.bfloat16)
                        right = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(accumulator[col * 128 + 64]), den),
                                                 tl.bfloat16)
                        packed, unused = tla.simd.vdintlv(left, right)
                        tla.simd.vsts(output[col * 128], packed)
                    tla.simd.vsts(logsum[0], tla.simd.vadd(tla.simd.vlds(maximum[0]), tla.simd.vln(den)))
                tla.store_ub_to_gm(output, tla.view(OUT + row * 512, (512, )))
                tla.store_ub_to_gm(tla.view(logsum, (1, )), tla.view(LSE + row, (1, )))


def forward(q, kv_cache, block_table, *, softmax_scale=0.04, causal=True, core_limit=None, compile_only=False):
    """Return BF16 [1,4,H,512] output and FP32 [1,4,H] logsumexp.

    Inputs must be contiguous on the same NPU. block_table contains valid physical
    page indices; repeated pages are allowed. Exactly 1024 logical pages of size
    128 are used. core_limit is only a diagnostic cap on the queried Cube count.
    Every execution allocates fresh workspace/output tensors and launches both
    stages. compile_only returns the two compiled kernels without execution.
    """
    if q.ndim != 4 or q.shape[:2] != (1, 4) or q.shape[-1] != 576:
        raise ValueError('q must be BF16 [1,4,H,576]')
    if kv_cache.ndim != 4 or kv_cache.shape[1:] != (128, 1, 576):
        raise ValueError('kv_cache must be BF16 [physical_pages,128,1,576]')
    if tuple(block_table.shape) != (1, 1024) or block_table.dtype != torch.int32:
        raise ValueError('block_table must be int32 [1,1024] with valid physical page indices')
    if q.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        raise ValueError('q and kv_cache must be BF16')
    if q.device.type != 'npu' or any(x.device != q.device or not x.is_contiguous() for x in (q, kv_cache, block_table)):
        raise ValueError('Inputs must be contiguous on the same NPU')
    if not isinstance(causal, bool) or not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError('Expected a Boolean causal and finite positive softmax_scale')
    cube, vector = device_cores(q.device)
    if core_limit is not None:
        if not isinstance(core_limit, int) or not 1 <= core_limit <= cube:
            raise ValueError('core_limit must be in [1, runtime Cube count]')
        cube, vector = core_limit, 2 * core_limit
    h = q.shape[2]
    config = launch_config(h, cube, vector)
    splits, rows = config['splits'], 4 * h
    part = torch.empty((splits, rows, 512), dtype=torch.float32, device=q.device)
    maximum = torch.empty((splits, rows), dtype=torch.float32, device=q.device)
    denominator = torch.empty_like(maximum)
    output = torch.empty((1, 4, h, 512), dtype=q.dtype, device=q.device)
    lse = torch.empty((1, 4, h), dtype=torch.float32, device=q.device)
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
    args = (q, kv_cache, block_table, part, maximum, denominator, h, config['block_m'], splits, config['cube_grid'],
            softmax_scale, causal)
    merge_args = (part, maximum, denominator, output, lse, rows, splits, config['vector_grid'])
    with torch.npu.device(q.device):
        if compile_only:
            first = _mla_split.warmup(*args, grid=(config['cube_grid'], ), **options)
            second = _mla_merge.warmup(*merge_args, grid=(config['vector_grid'], ), **options)
            return first, second
        else:
            _mla_split[(config['cube_grid'], )](*args, **options)
            _mla_merge[(config['vector_grid'], )](*merge_args, **options)
    return output, lse
