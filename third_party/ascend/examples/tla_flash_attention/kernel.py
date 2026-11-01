"""Readable source templates for the eight-case Ascend950 FlashAttention study.

This file is parsed, not imported. ``inline_*`` calls expand syntactically before
TLA JIT, preserving expression order and avoiding runtime helper calls.
``pipeline_phases(6)`` expands the 2-score/3-probability ring's six phases.
See expand_kernel.py and OPTIMIZATION_GUIDE.md for the expansion contract.
"""


# Online softmax: each FP32 register lane owns a query row. Four independent
# accumulators shorten the max/sum dependency chains. WIDTH is BN or the exact
# causal half-tail BM; PROB has the matching packed row pitch.
def inline_softmax(SCORE, ALPHA, WIDTH, PROB, RECURRING, MASKED):
    with tla.scope.simd():
        scale = tla.simd.vfull(SCALE, tl.float32, 64)
        neg_inf = tla.simd.vfull(float('-inf'), tl.float32, 64)
        if MASKED:
            if CAUSAL:
                queries = tla.simd.vci((sm_row + sub * VM).to(tl.int32))
        mx0 = neg_inf
        mx1 = neg_inf
        mx2 = neg_inf
        mx3 = neg_inf
        for i in tl.range(0, WIDTH // 4):
            x0 = tla.simd.vlds(SCORE[i * 4 + 0, 0])
            x1 = tla.simd.vlds(SCORE[i * 4 + 1, 0])
            x2 = tla.simd.vlds(SCORE[i * 4 + 2, 0])
            x3 = tla.simd.vlds(SCORE[i * 4 + 3, 0])
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + (i * 4 + 0), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x0 = tla.simd.vsel(valid, x0, neg_inf)
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + (i * 4 + 1), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x1 = tla.simd.vsel(valid, x1, neg_inf)
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + (i * 4 + 2), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x2 = tla.simd.vsel(valid, x2, neg_inf)
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + (i * 4 + 3), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x3 = tla.simd.vsel(valid, x3, neg_inf)
            mx0 = tla.simd.vmax(mx0, x0)
            mx1 = tla.simd.vmax(mx1, x1)
            mx2 = tla.simd.vmax(mx2, x2)
            mx3 = tla.simd.vmax(mx3, x3)
        new_m = ext.vmuls(tla.simd.vmax(tla.simd.vmax(mx0, mx1), tla.simd.vmax(mx2, mx3)), SCALE)
        alpha_v = tla.simd.vfull(1.0, tl.float32, 64)
        if RECURRING:
            old_m = tla.simd.vlds(m_ub[sm_state, 0])
            new_m = tla.simd.vmax(new_m, old_m)
            alpha_v = ext.vexpdif(old_m, new_m)
            tla.simd.vsts(ALPHA[0, 0], alpha_v)
        tla.simd.vsts(m_ub[sm_state, 0], new_m)
        sum0 = tla.simd.vfull(0.0, tl.float32, 64)
        sum1 = tla.simd.vfull(0.0, tl.float32, 64)
        sum2 = tla.simd.vfull(0.0, tl.float32, 64)
        sum3 = tla.simd.vfull(0.0, tl.float32, 64)
        for i in tl.range(0, WIDTH // 4):
            x0 = tla.simd.vlds(SCORE[i + 0 * (WIDTH // 4), 0])
            x1 = tla.simd.vlds(SCORE[i + 1 * (WIDTH // 4), 0])
            x2 = tla.simd.vlds(SCORE[i + 2 * (WIDTH // 4), 0])
            x3 = tla.simd.vlds(SCORE[i + 3 * (WIDTH // 4), 0])
            x0 = ext.vmuls(x0, SCALE)
            x1 = ext.vmuls(x1, SCALE)
            x2 = ext.vmuls(x2, SCALE)
            x3 = ext.vmuls(x3, SCALE)
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + i + 0 * (WIDTH // 4), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x0 = tla.simd.vsel(valid, x0, neg_inf)
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + i + 1 * (WIDTH // 4), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x1 = tla.simd.vsel(valid, x1, neg_inf)
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + i + 2 * (WIDTH // 4), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x2 = tla.simd.vsel(valid, x2, neg_inf)
            if MASKED:
                if CAUSAL:
                    keys = tla.simd.vdup(sm_key + i + 3 * (WIDTH // 4), tl.int32)
                    valid = tla.simd.vcmp(keys, queries, op=tla.simd.CMP.LE)
                    x3 = tla.simd.vsel(valid, x3, neg_inf)
            e0 = ext.vexpdif(x0, new_m)
            e1 = ext.vexpdif(x1, new_m)
            e2 = ext.vexpdif(x2, new_m)
            e3 = ext.vexpdif(x3, new_m)
            sum0 = tla.simd.vadd(sum0, e0)
            sum1 = tla.simd.vadd(sum1, e1)
            sum2 = tla.simd.vadd(sum2, e2)
            sum3 = tla.simd.vadd(sum3, e3)
            pc0 = tla.simd.vtruncf(e0, tl.bfloat16)
            pc1 = tla.simd.vtruncf(e1, tl.bfloat16)
            pc2 = tla.simd.vtruncf(e2, tl.bfloat16)
            pc3 = tla.simd.vtruncf(e3, tl.bfloat16)
            (pa, unused_a) = tla.simd.vdintlv(pc0, pc2)
            (pb, unused_b) = tla.simd.vdintlv(pc1, pc3)
            tla.simd.vsstb(pa, PROB[0, i * 16], (WIDTH // 2 + PAD) * 16)
            tla.simd.vsstb(pb, PROB[0, (i + WIDTH // 4) * 16], (WIDTH // 2 + PAD) * 16)
        new_l = tla.simd.vadd(tla.simd.vadd(sum0, sum1), tla.simd.vadd(sum2, sum3))
        if RECURRING:
            old_l = tla.simd.vlds(l_ub[sm_state, 0])
            new_l = tla.simd.vadd(tla.simd.vmul(old_l, alpha_v), new_l)
        tla.simd.vsts(l_ub[sm_state, 0], new_l)


# Cube stage QK(t): K @ Q.T, then column-split NZ2ND into a DN score tile.
def inline_qk_full_tile(KEY):
    with tla.scope.cube():
        if c_key == 0:
            tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
        tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), KEY)
        score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
        ext.matmul(KEY, q_l1, score_l0c, transpose_b=True, clear_accum=True)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.copy_l0c_to_ub(
            score_l0c, s0,
            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                    dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 0)


# Cube stage PV(t-2): consume the published P slot using MMAD transpose-A.
def inline_pv_full_tile(VALUE, PV):
    with tla.scope.cube():
        tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), VALUE)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 5)
        pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        ext.matmul(p1, VALUE, pv_l0c, clear_accum=True, transpose_a=True)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.copy_l0c_to_ub(
            pv_l0c, PV,
            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                    sub_block_idx=None))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 2)


# Vector stage softmax(t-1): publish score-free only after the last score
# read, then publish P-ready after UB-to-L1 copy completion.
def inline_softmax_full_tile():
    with tla.scope.vector():
        sub = tla.sub_block_id()
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        if sm_key == 0:
            if CAUSAL and sm_key + BN > sm_row:
                inline_softmax(s1, alpha2, BN, p_ub, False, True)
            else:
                inline_softmax(s1, alpha2, BN, p_ub, False, False)
        elif CAUSAL and sm_key + BN > sm_row:
            inline_softmax(s1, alpha2, BN, p_ub, True, True)
        else:
            inline_softmax(s1, alpha2, BN, p_ub, True, False)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                          tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
        tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                          tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 13)
        if sm_key + BN >= sm_end:
            with tla.scope.simd():
                bits = tla.simd.vbitcast(tla.simd.vlds(m_ub[sm_state, 0]), 'i32')
                (a0, a1) = tla.simd.vintlv(bits, bits)
                (b0, b1) = tla.simd.vintlv(a0, a0)
                (c0, c1) = tla.simd.vintlv(b0, b0)
                tla.simd.vsts(stat_m[0, 0], tla.simd.vbitcast(c0, 'f32'))
                tla.simd.vsts(stat_m[8, 0], tla.simd.vbitcast(c1, 'f32'))
                if VM >= 32:
                    (c2, c3) = tla.simd.vintlv(b1, b1)
                    tla.simd.vsts(stat_m[16, 0], tla.simd.vbitcast(c2, 'f32'))
                    tla.simd.vsts(stat_m[24, 0], tla.simd.vbitcast(c3, 'f32'))
                if VM == 64:
                    (b2, b3) = tla.simd.vintlv(a1, a1)
                    (c4, c5) = tla.simd.vintlv(b2, b2)
                    (c6, c7) = tla.simd.vintlv(b3, b3)
                    tla.simd.vsts(stat_m[32, 0], tla.simd.vbitcast(c4, 'f32'))
                    tla.simd.vsts(stat_m[40, 0], tla.simd.vbitcast(c5, 'f32'))
                    tla.simd.vsts(stat_m[48, 0], tla.simd.vbitcast(c6, 'f32'))
                    tla.simd.vsts(stat_m[56, 0], tla.simd.vbitcast(c7, 'f32'))
                bits = tla.simd.vbitcast(tla.simd.vlds(l_ub[sm_state, 0]), 'i32')
                (a0, a1) = tla.simd.vintlv(bits, bits)
                (b0, b1) = tla.simd.vintlv(a0, a0)
                (c0, c1) = tla.simd.vintlv(b0, b0)
                tla.simd.vsts(stat_l[0, 0], tla.simd.vbitcast(c0, 'f32'))
                tla.simd.vsts(stat_l[8, 0], tla.simd.vbitcast(c1, 'f32'))
                if VM >= 32:
                    (c2, c3) = tla.simd.vintlv(b1, b1)
                    tla.simd.vsts(stat_l[16, 0], tla.simd.vbitcast(c2, 'f32'))
                    tla.simd.vsts(stat_l[24, 0], tla.simd.vbitcast(c3, 'f32'))
                if VM == 64:
                    (b2, b3) = tla.simd.vintlv(a1, a1)
                    (c4, c5) = tla.simd.vintlv(b2, b2)
                    (c6, c7) = tla.simd.vintlv(b3, b3)
                    tla.simd.vsts(stat_l[32, 0], tla.simd.vbitcast(c4, 'f32'))
                    tla.simd.vsts(stat_l[40, 0], tla.simd.vbitcast(c5, 'f32'))
                    tla.simd.vsts(stat_l[48, 0], tla.simd.vbitcast(c6, 'f32'))
                    tla.simd.vsts(stat_l[56, 0], tla.simd.vbitcast(c7, 'f32'))
            tla.store_ub_to_gm(stat_m,
                               tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            tla.store_ub_to_gm(stat_l,
                               tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))


# Vector stage update(t-3): acc = old_acc * alpha + PV, followed by FP32
# division only on the last key tile. Release the PV slot after its last read.
def inline_output_update(PV, PV_READY, PV_FREE):
    with tla.scope.vector():
        sub = tla.sub_block_id()
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, PV_READY)
        if u_key + BN < u_end:
            if u_key == 0:
                with tla.scope.simd():
                    for r in tl.range(0, VM):
                        for col in tl.static_range(D // 64):
                            acc = tla.simd.vlds(PV[r, col * 64])
                            tla.simd.vsts(acc_ub[r, col * 64], acc)
            else:
                with tla.scope.simd():
                    for r in tl.range(0, VM):
                        alpha = tla.simd.vlds(alpha0[r, 0], dist='BRC_B32')
                        for col in tl.static_range(D // 64):
                            acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                            acc = tla.simd.vadd(acc, tla.simd.vlds(PV[r, col * 64]))
                            tla.simd.vsts(acc_ub[r, col * 64], acc)
        else:
            if u_key == 0:
                with tla.scope.simd():
                    for r in tl.range(0, VM):
                        for col in tl.static_range(D // 64):
                            acc = tla.simd.vlds(PV[r, col * 64])
                            den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                            value = tla.simd.vdiv(acc, den)
                            packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                            tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
            else:
                with tla.scope.simd():
                    for r in tl.range(0, VM):
                        alpha = tla.simd.vlds(alpha0[r, 0], dist='BRC_B32')
                        for col in tl.static_range(D // 64):
                            acc = tla.simd.vmul(tla.simd.vlds(acc_ub[r, col * 64]), alpha)
                            acc = tla.simd.vadd(acc, tla.simd.vlds(PV[r, col * 64]))
                            den = tla.simd.vlds(l_ub[u_state + r, 0], dist='BRC_B32')
                            value = tla.simd.vdiv(acc, den)
                            packed = tla.simd.vbitcast(tla.simd.vtruncf(value, tl.bfloat16), 'i32')
                            tla.simd.vsts(out_ub[r, col * 64], packed, dist='PK_B32')
            tla.store_ub_to_gm(out_ub, tla.view(O + u_base, (VM, D), strides=(D, 1), offsets=(u_row + sub * VM, 0)))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, PV_FREE)


# Tail specialization: L0C shape must match real M, not the full BN buffer.
def inline_qk_compact_tail(KEY):
    with tla.scope.cube():
        if c_key == 0:
            tla.load_gm_to_l1(tla.view(Q + c_base, (BM, D), strides=(D, 1), offsets=(c_row, 0)), q_l1)
        tla.load_gm_to_l1(tla.view(K + c_base, (BN, D), strides=(D, 1), offsets=(c_key, 0)), KEY)
        if CAUSAL and c_end - c_key < BN:
            score_tail_l0c = tla.alloc((BM // 16, BM // 16, 16, 16), tl.float32, space='l0c')
            ext.matmul(KEY, q_l1, score_tail_l0c, transpose_b=True, clear_accum=True, real_m=BM)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
            tla.copy_l0c_to_ub(
                score_tail_l0c, tla.view(s0, (BM, VM)),
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
        else:
            score_l0c = tla.alloc((BM // 16, BN // 16, 16, 16), tl.float32, space='l0c')
            ext.matmul(KEY, q_l1, score_l0c, transpose_b=True, clear_accum=True)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
            tla.copy_l0c_to_ub(
                score_l0c, s0,
                tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                        dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT, sub_block_idx=None))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 0)


# The masked causal half-tail has exactly BM valid keys, so real K is BM.
def inline_pv_compact_tail(VALUE, PV):
    with tla.scope.cube():
        tla.load_gm_to_l1(tla.view(V + p_base, (BN, D), strides=(D, 1), offsets=(p_key, 0)), VALUE)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 5)
        pv_l0c = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        if CAUSAL and p_end - p_key < BN:
            ext.matmul(p1, VALUE, pv_l0c, clear_accum=True, transpose_a=True, real_k=BM)
        else:
            ext.matmul(p1, VALUE, pv_l0c, clear_accum=True, transpose_a=True)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.copy_l0c_to_ub(
            pv_l0c, PV,
            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                    sub_block_idx=None))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 2)


# Trim vector work and P-copy pitch along with the two matrix operations.
def inline_softmax_compact_tail():
    with tla.scope.vector():
        sub = tla.sub_block_id()
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        if sm_key == 0:
            if CAUSAL and sm_key + BN > sm_row:
                if CAUSAL and sm_end - sm_key < BN:
                    inline_softmax(s1, alpha2, BM, p_tail, False, True)
                else:
                    inline_softmax(s1, alpha2, BN, p_ub, False, True)
            elif CAUSAL and sm_end - sm_key < BN:
                inline_softmax(s1, alpha2, BM, p_tail, False, False)
            else:
                inline_softmax(s1, alpha2, BN, p_ub, False, False)
        elif CAUSAL and sm_key + BN > sm_row:
            if CAUSAL and sm_end - sm_key < BN:
                inline_softmax(s1, alpha2, BM, p_tail, True, True)
            else:
                inline_softmax(s1, alpha2, BN, p_ub, True, True)
        elif CAUSAL and sm_end - sm_key < BN:
            inline_softmax(s1, alpha2, BM, p_tail, True, False)
        else:
            inline_softmax(s1, alpha2, BN, p_ub, True, False)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        if CAUSAL and sm_end - sm_key < BN:
            tla.copy_ub_to_l1(tla.view(p_tail, (VM // 16, BM // 2 * 16), offsets=(0, 0)),
                              tla.view(pf2, (VM // 16, BM // 2 * 16), offsets=(sub * (VM // 16), 0)))
        else:
            tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(0, 0)),
                              tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), 0)))
        if CAUSAL and sm_end - sm_key < BN:
            tla.copy_ub_to_l1(tla.view(p_tail, (VM // 16, BM // 2 * 16), offsets=(VM // 16, 0)),
                              tla.view(pf2, (VM // 16, BM // 2 * 16), offsets=(sub * (VM // 16), BM // 2 * 16)))
        else:
            tla.copy_ub_to_l1(tla.view(p_ub, (VM // 16, BN // 2 * 16), offsets=(VM // 16, 0)),
                              tla.view(pf2, (VM // 16, BN // 2 * 16), offsets=(sub * (VM // 16), BN // 2 * 16)))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 13)
        if sm_key + BN >= sm_end:
            with tla.scope.simd():
                for stat_row in tl.range(0, VM, 8):
                    max8 = tla.simd.vlds(m_ub[sm_state + stat_row, 0], dist='E2B_B32')
                    sum8 = tla.simd.vlds(l_ub[sm_state + stat_row, 0], dist='E2B_B32')
                    tla.simd.vsts(stat_m[stat_row, 0], max8)
                    tla.simd.vsts(stat_l[stat_row, 0], sum8)
            tla.store_ub_to_gm(stat_m,
                               tla.view(SM + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))
            tla.store_ub_to_gm(stat_l,
                               tla.view(SL + sm_bh * S * 8, (VM, 8), strides=(8, 1), offsets=(sm_row + sub * VM, 0)))


# Fixed B4/H8/S1024 schedule: every query tile appears exactly once across
# 28 cores. Each uint64 stores eight public query IDs, not input-dependent data.
# The cost model uses valid key tiles plus a fixed per-query cost of two.
def inline_balanced_query_schedule():
    tl.static_assert(CAUSAL and S == 1024 and (BM == 128) and (TILES == 256) and (CORES == 28))
    schedule_lo = tl.full((), 0, tl.uint64)
    schedule_hi = tl.full((), 0, tl.uint64)
    query_count = 0
    if tl.program_id(0) == 0:
        schedule_lo = tl.full((), 2378432372355180038, tl.uint64)
        schedule_hi = tl.full((), 57344, tl.uint64)
        query_count = 10
    if tl.program_id(0) == 1:
        schedule_lo = tl.full((), 2957153755059793422, tl.uint64)
        schedule_hi = tl.full((), 59400, tl.uint64)
        query_count = 10
    if tl.program_id(0) == 2:
        schedule_lo = tl.full((), 3535875137764406806, tl.uint64)
        schedule_hi = tl.full((), 61456, tl.uint64)
        query_count = 10
    if tl.program_id(0) == 3:
        schedule_lo = tl.full((), 4114596520469020190, tl.uint64)
        schedule_hi = tl.full((), 63512, tl.uint64)
        query_count = 10
    if tl.program_id(0) == 4:
        schedule_lo = tl.full((), 4702324002918893351, tl.uint64)
        schedule_hi = tl.full((), 32, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 5:
        schedule_lo = tl.full((), 5281045385623506735, tl.uint64)
        schedule_hi = tl.full((), 40, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 6:
        schedule_lo = tl.full((), 5859766768328120119, tl.uint64)
        schedule_hi = tl.full((), 48, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 7:
        schedule_lo = tl.full((), 6438488151032733503, tl.uint64)
        schedule_hi = tl.full((), 56, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 8:
        schedule_lo = tl.full((), 7017209533737396039, tl.uint64)
        schedule_hi = tl.full((), 64, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 9:
        schedule_lo = tl.full((), 7595930916442009423, tl.uint64)
        schedule_hi = tl.full((), 72, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 10:
        schedule_lo = tl.full((), 8174652299146622807, tl.uint64)
        schedule_hi = tl.full((), 80, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 11:
        schedule_lo = tl.full((), 8753373681851236191, tl.uint64)
        schedule_hi = tl.full((), 88, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 12:
        schedule_lo = tl.full((), 9332130245169727079, tl.uint64)
        schedule_hi = tl.full((), 96, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 13:
        schedule_lo = tl.full((), 9910851627874340463, tl.uint64)
        schedule_hi = tl.full((), 104, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 14:
        schedule_lo = tl.full((), 10489573010578953847, tl.uint64)
        schedule_hi = tl.full((), 112, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 15:
        schedule_lo = tl.full((), 11068294393283567231, tl.uint64)
        schedule_hi = tl.full((), 120, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 16:
        schedule_lo = tl.full((), 11647015775988180615, tl.uint64)
        schedule_hi = tl.full((), 128, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 17:
        schedule_lo = tl.full((), 12225737158692793999, tl.uint64)
        schedule_hi = tl.full((), 136, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 18:
        schedule_lo = tl.full((), 12804458541397407383, tl.uint64)
        schedule_hi = tl.full((), 144, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 19:
        schedule_lo = tl.full((), 13383179924102020767, tl.uint64)
        schedule_hi = tl.full((), 152, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 20:
        schedule_lo = tl.full((), 13961901306806634151, tl.uint64)
        schedule_hi = tl.full((), 160, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 21:
        schedule_lo = tl.full((), 14540622689511247535, tl.uint64)
        schedule_hi = tl.full((), 168, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 22:
        schedule_lo = tl.full((), 15119344072215860919, tl.uint64)
        schedule_hi = tl.full((), 176, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 23:
        schedule_lo = tl.full((), 15698065454920474303, tl.uint64)
        schedule_hi = tl.full((), 184, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 24:
        schedule_lo = tl.full((), 16276786975047263943, tl.uint64)
        schedule_hi = tl.full((), 192, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 25:
        schedule_lo = tl.full((), 16855508357751877327, tl.uint64)
        schedule_hi = tl.full((), 200, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 26:
        schedule_lo = tl.full((), 17434229740456490711, tl.uint64)
        schedule_hi = tl.full((), 208, tl.uint64)
        query_count = 9
    if tl.program_id(0) == 27:
        schedule_lo = tl.full((), 18012951123161104095, tl.uint64)
        schedule_hi = tl.full((), 216, tl.uint64)
        query_count = 9


# FA1/FA2 and replayable v4 baseline: dual K/V L1, one PV UB slot.
def fa_forward_phase4_kqt_dualkv_native(Q, K, V, O, SM, SL, S: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                                        BM: tl.constexpr, CAUSAL: tl.constexpr, BN: tl.constexpr, TILES: tl.constexpr,
                                        CORES: tl.constexpr, FIRST_OPT: tl.constexpr, MASK_OPT: tl.constexpr,
                                        QUERY_DRAIN: tl.constexpr, ROTATION: tl.constexpr):
    PAD: tl.constexpr = 1
    VM: tl.constexpr = BM // 2
    SCALE: tl.constexpr = D**(-0.5)
    tl.static_assert(VM == 64 and BN >= 128 and (BN <= 512) and FIRST_OPT)
    p_ub = tla.alloc((VM // 16 * 2, (BN // 2 + PAD) * 16), tl.bfloat16, space='ub')
    acc_ub = tla.alloc((VM, D), tl.float32, space='ub')
    out_ub = tla.alloc((VM, D), tl.bfloat16, space='ub')
    m_ub = tla.alloc((192, 1), tl.float32, space='ub')
    l_ub = tla.alloc((192, 1), tl.float32, space='ub')
    m_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    l_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    stat_m = tla.alloc((VM, 8), tl.float32, space='ub')
    stat_l = tla.alloc((VM, 8), tl.float32, space='ub')
    pv_ub = tla.alloc((VM, D), tl.float32, space='ub')
    q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1_0 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1_1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1_0 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1_1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    s0 = tla.alloc((BN, VM), tl.float32, space='ub')
    p0 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf0 = tla.reshape(p0, (BM // 16, BN * 16))
    alpha0 = tla.alloc((64, 1), tl.float32, space='ub')
    s1 = tla.alloc((BN, VM), tl.float32, space='ub')
    p1 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf1 = tla.reshape(p1, (BM // 16, BN * 16))
    alpha1 = tla.alloc((64, 1), tl.float32, space='ub')
    p2 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf2 = tla.reshape(p2, (BM // 16, BN * 16))
    alpha2 = tla.alloc((64, 1), tl.float32, space='ub')
    with tla.scope.vector():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        with tla.scope.simd():
            zeros = tla.simd.vfull(0.0, tl.float32, 64)
            for state in tl.static_range(3):
                tla.simd.vsts(m_ub[state * 64, 0], zeros)
                tla.simd.vsts(l_ub[state * 64, 0], zeros)
            tla.simd.vsts(m_tmp[0, 0], zeros)
            tla.simd.vsts(l_tmp[0, 0], zeros)
    query_count = (TILES - 1 - tl.program_id(0)) // CORES + 1
    task_count = 0
    for q_index in tl.range(query_count):
        count_pid = tl.program_id(0) + q_index * CORES
        count_bh = count_pid // (S // BM)
        count_row = (count_pid % (S // BM) + count_bh * ROTATION) % (S // BM) * BM
        count_end = count_row + BM if CAUSAL else S
        task_count += (count_end + BN - 1) // BN
    c_query = 0
    c_key = 0
    p_query = 0
    p_key = 0
    sm_query = 0
    sm_key = 0
    u_query = 0
    u_key = 0
    for step in tl.range(0, task_count + 3, 6):
        for PHASE in pipeline_phases(6):
            # Physical ring period lcm(2 score slots, 3 P/alpha slots) = 6.
            # Expand all phases before JIT; no runtime buffer selection.
            t = step + 0
            if t < task_count:
                c_pid = tl.program_id(0) + c_query * CORES
                c_bh = c_pid // (S // BM)
                c_row = (c_pid % (S // BM) + c_bh * ROTATION) % (S // BM) * BM
                c_base = c_bh * S * D
                c_end = c_row + BM if CAUSAL else S
                inline_qk_full_tile(k_l1_0)
                c_key += BN
                if c_key >= c_end:
                    c_query += 1
                    c_key = 0
            if t >= 2 and t < task_count + 2:
                p_pid = tl.program_id(0) + p_query * CORES
                p_bh = p_pid // (S // BM)
                p_row = (p_pid % (S // BM) + p_bh * ROTATION) % (S // BM) * BM
                p_base = p_bh * S * D
                p_end = p_row + BM if CAUSAL else S
                inline_pv_full_tile(v_l1_0, pv_ub)
                p_key += BN
                if p_key >= p_end:
                    p_query += 1
                    p_key = 0
            if t >= 1 and t < task_count + 1:
                sm_pid = tl.program_id(0) + sm_query * CORES
                sm_bh = sm_pid // (S // BM)
                sm_row = (sm_pid % (S // BM) + sm_bh * ROTATION) % (S // BM) * BM
                sm_base = sm_bh * S * D
                sm_end = sm_row + BM if CAUSAL else S
                sm_state = sm_query % 3 * 64
                inline_softmax_full_tile()
                sm_key += BN
                if sm_key >= sm_end:
                    sm_query += 1
                    sm_key = 0
            if t >= 3 and t < task_count + 3:
                u_pid = tl.program_id(0) + u_query * CORES
                u_bh = u_pid // (S // BM)
                u_row = (u_pid % (S // BM) + u_bh * ROTATION) % (S // BM) * BM
                u_base = u_bh * S * D
                u_end = u_row + BM if CAUSAL else S
                u_state = u_query % 3 * 64
                inline_output_update(pv_ub, 2, 9)
                u_key += BN
                if u_key >= u_end:
                    u_query += 1
                    u_key = 0
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)


# FA3/FA4: dual K/V L1 and two PV UB slots (D64 capacity).
def fa_forward_phase4_pv2_dualkv(Q, K, V, O, SM, SL, S: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                                 BM: tl.constexpr, CAUSAL: tl.constexpr, BN: tl.constexpr, TILES: tl.constexpr,
                                 CORES: tl.constexpr, FIRST_OPT: tl.constexpr, MASK_OPT: tl.constexpr,
                                 QUERY_DRAIN: tl.constexpr, ROTATION: tl.constexpr):
    PAD: tl.constexpr = 1
    VM: tl.constexpr = BM // 2
    SCALE: tl.constexpr = D**(-0.5)
    tl.static_assert(VM == 64 and BN >= 128 and (BN <= 512) and FIRST_OPT)
    p_ub = tla.alloc((VM // 16 * 2, (BN // 2 + PAD) * 16), tl.bfloat16, space='ub')
    acc_ub = tla.alloc((VM, D), tl.float32, space='ub')
    out_ub = tla.alloc((VM, D), tl.bfloat16, space='ub')
    m_ub = tla.alloc((192, 1), tl.float32, space='ub')
    l_ub = tla.alloc((192, 1), tl.float32, space='ub')
    m_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    l_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    stat_m = tla.alloc((VM, 8), tl.float32, space='ub')
    stat_l = tla.alloc((VM, 8), tl.float32, space='ub')
    tl.static_assert(D == 64)
    pv_ub0 = tla.alloc((VM, D), tl.float32, space='ub')
    pv_ub1 = tla.alloc((VM, D), tl.float32, space='ub')
    q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1_0 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1_1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1_0 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1_1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    s0 = tla.alloc((BN, VM), tl.float32, space='ub')
    p0 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf0 = tla.reshape(p0, (BM // 16, BN * 16))
    alpha0 = tla.alloc((64, 1), tl.float32, space='ub')
    s1 = tla.alloc((BN, VM), tl.float32, space='ub')
    p1 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf1 = tla.reshape(p1, (BM // 16, BN * 16))
    alpha1 = tla.alloc((64, 1), tl.float32, space='ub')
    p2 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf2 = tla.reshape(p2, (BM // 16, BN * 16))
    alpha2 = tla.alloc((64, 1), tl.float32, space='ub')
    with tla.scope.vector():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 12)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        with tla.scope.simd():
            zeros = tla.simd.vfull(0.0, tl.float32, 64)
            for state in tl.static_range(3):
                tla.simd.vsts(m_ub[state * 64, 0], zeros)
                tla.simd.vsts(l_ub[state * 64, 0], zeros)
            tla.simd.vsts(m_tmp[0, 0], zeros)
            tla.simd.vsts(l_tmp[0, 0], zeros)
    query_count = (TILES - 1 - tl.program_id(0)) // CORES + 1
    task_count = 0
    for q_index in tl.range(query_count):
        count_pid = tl.program_id(0) + q_index * CORES
        count_bh = count_pid // (S // BM)
        count_row = (count_pid % (S // BM) + count_bh * ROTATION) % (S // BM) * BM
        count_end = count_row + BM if CAUSAL else S
        task_count += (count_end + BN - 1) // BN
    c_query = 0
    c_key = 0
    p_query = 0
    p_key = 0
    sm_query = 0
    sm_key = 0
    u_query = 0
    u_key = 0
    for step in tl.range(0, task_count + 3, 6):
        for PHASE in pipeline_phases(6):
            # Physical ring period lcm(2 score slots, 3 P/alpha slots) = 6.
            # Expand all phases before JIT; no runtime buffer selection.
            t = step + 0
            if t < task_count:
                c_pid = tl.program_id(0) + c_query * CORES
                c_bh = c_pid // (S // BM)
                c_row = (c_pid % (S // BM) + c_bh * ROTATION) % (S // BM) * BM
                c_base = c_bh * S * D
                c_end = c_row + BM if CAUSAL else S
                inline_qk_full_tile(k_l1_0)
                c_key += BN
                if c_key >= c_end:
                    c_query += 1
                    c_key = 0
            if t >= 2 and t < task_count + 2:
                p_pid = tl.program_id(0) + p_query * CORES
                p_bh = p_pid // (S // BM)
                p_row = (p_pid % (S // BM) + p_bh * ROTATION) % (S // BM) * BM
                p_base = p_bh * S * D
                p_end = p_row + BM if CAUSAL else S
                inline_pv_full_tile(v_l1_0, pv_ub0)
                p_key += BN
                if p_key >= p_end:
                    p_query += 1
                    p_key = 0
            if t >= 1 and t < task_count + 1:
                sm_pid = tl.program_id(0) + sm_query * CORES
                sm_bh = sm_pid // (S // BM)
                sm_row = (sm_pid % (S // BM) + sm_bh * ROTATION) % (S // BM) * BM
                sm_base = sm_bh * S * D
                sm_end = sm_row + BM if CAUSAL else S
                sm_state = sm_query % 3 * 64
                inline_softmax_full_tile()
                sm_key += BN
                if sm_key >= sm_end:
                    sm_query += 1
                    sm_key = 0
            if t >= 3 and t < task_count + 3:
                u_pid = tl.program_id(0) + u_query * CORES
                u_bh = u_pid // (S // BM)
                u_row = (u_pid % (S // BM) + u_bh * ROTATION) % (S // BM) * BM
                u_base = u_bh * S * D
                u_end = u_row + BM if CAUSAL else S
                u_state = u_query % 3 * 64
                inline_output_update(pv_ub1, 6, 12)
                u_key += BN
                if u_key >= u_end:
                    u_query += 1
                    u_key = 0
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 12)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)


# FA5: compact causal tail and balanced short D128 query assignment.
def fa_forward_phase4_tasks_mixed(Q, K, V, O, SM, SL, S: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                                  BM: tl.constexpr, CAUSAL: tl.constexpr, BN: tl.constexpr, TILES: tl.constexpr,
                                  CORES: tl.constexpr, FIRST_OPT: tl.constexpr, MASK_OPT: tl.constexpr,
                                  QUERY_DRAIN: tl.constexpr, ROTATION: tl.constexpr):
    PAD: tl.constexpr = 1
    VM: tl.constexpr = BM // 2
    SCALE: tl.constexpr = D**(-0.5)
    tl.static_assert(VM == 64 and BN >= 128 and (BN <= 512) and FIRST_OPT)
    p_ub = tla.alloc((VM // 16 * 2, (BN // 2 + PAD) * 16), tl.bfloat16, space='ub')
    tl.static_assert(BN == 256)
    p_tail = ext.prefix_view(p_ub, (VM // 16 * 2, (BM // 2 + PAD) * 16))
    acc_ub = tla.alloc((VM, D), tl.float32, space='ub')
    out_ub = tla.alloc((VM, D), tl.bfloat16, space='ub')
    m_ub = tla.alloc((192, 1), tl.float32, space='ub')
    l_ub = tla.alloc((192, 1), tl.float32, space='ub')
    m_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    l_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    stat_m = tla.alloc((VM, 8), tl.float32, space='ub')
    stat_l = tla.alloc((VM, 8), tl.float32, space='ub')
    pv_ub = tla.alloc((VM, D), tl.float32, space='ub')
    q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    s0 = tla.alloc((BN, VM), tl.float32, space='ub')
    p0 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf0 = tla.reshape(p0, (BM // 16, BN * 16))
    alpha0 = tla.alloc((64, 1), tl.float32, space='ub')
    s1 = tla.alloc((BN, VM), tl.float32, space='ub')
    p1 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf1 = tla.reshape(p1, (BM // 16, BN * 16))
    alpha1 = tla.alloc((64, 1), tl.float32, space='ub')
    p2 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf2 = tla.reshape(p2, (BM // 16, BN * 16))
    alpha2 = tla.alloc((64, 1), tl.float32, space='ub')
    with tla.scope.vector():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        with tla.scope.simd():
            zeros = tla.simd.vfull(0.0, tl.float32, 64)
            for state in tl.static_range(3):
                tla.simd.vsts(m_ub[state * 64, 0], zeros)
                tla.simd.vsts(l_ub[state * 64, 0], zeros)
            tla.simd.vsts(m_tmp[0, 0], zeros)
            tla.simd.vsts(l_tmp[0, 0], zeros)
    inline_balanced_query_schedule()
    task_count = 0
    for q_index in tl.range(query_count):
        count_pid = ((schedule_lo if q_index < 8 else schedule_hi) >> q_index % 8 * 8 & 255).to(tl.int32)
        count_bh = count_pid // (S // BM)
        count_row = count_pid % (S // BM) * BM
        count_end = count_row + BM if CAUSAL else S
        task_count += (count_end + BN - 1) // BN
    c_query = 0
    c_key = 0
    p_query = 0
    p_key = 0
    sm_query = 0
    sm_key = 0
    u_query = 0
    u_key = 0
    for step in tl.range(0, task_count + 3, 6):
        for PHASE in pipeline_phases(6):
            # Physical ring period lcm(2 score slots, 3 P/alpha slots) = 6.
            # Expand all phases before JIT; no runtime buffer selection.
            t = step + 0
            if t < task_count:
                c_pid = ((schedule_lo if c_query < 8 else schedule_hi) >> c_query % 8 * 8 & 255).to(tl.int32)
                c_bh = c_pid // (S // BM)
                c_row = c_pid % (S // BM) * BM
                c_base = c_bh * S * D
                c_end = c_row + BM if CAUSAL else S
                inline_qk_compact_tail(k_l1)
                c_key += BN
                if c_key >= c_end:
                    c_query += 1
                    c_key = 0
            if t >= 2 and t < task_count + 2:
                p_pid = ((schedule_lo if p_query < 8 else schedule_hi) >> p_query % 8 * 8 & 255).to(tl.int32)
                p_bh = p_pid // (S // BM)
                p_row = p_pid % (S // BM) * BM
                p_base = p_bh * S * D
                p_end = p_row + BM if CAUSAL else S
                inline_pv_compact_tail(v_l1, pv_ub)
                p_key += BN
                if p_key >= p_end:
                    p_query += 1
                    p_key = 0
            if t >= 1 and t < task_count + 1:
                sm_pid = ((schedule_lo if sm_query < 8 else schedule_hi) >> sm_query % 8 * 8 & 255).to(tl.int32)
                sm_bh = sm_pid // (S // BM)
                sm_row = sm_pid % (S // BM) * BM
                sm_base = sm_bh * S * D
                sm_end = sm_row + BM if CAUSAL else S
                sm_state = sm_query % 3 * 64
                inline_softmax_compact_tail()
                sm_key += BN
                if sm_key >= sm_end:
                    sm_query += 1
                    sm_key = 0
            if t >= 3 and t < task_count + 3:
                u_pid = ((schedule_lo if u_query < 8 else schedule_hi) >> u_query % 8 * 8 & 255).to(tl.int32)
                u_bh = u_pid // (S // BM)
                u_row = u_pid % (S // BM) * BM
                u_base = u_bh * S * D
                u_end = u_row + BM if CAUSAL else S
                u_state = u_query % 3 * 64
                inline_output_update(pv_ub, 2, 9)
                u_key += BN
                if u_key >= u_end:
                    u_query += 1
                    u_key = 0
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)


# FA6 and replayable short v4 baseline: single K/V and single PV.
def fa_forward_phase4_kqt_native(Q, K, V, O, SM, SL, S: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                                 BM: tl.constexpr, CAUSAL: tl.constexpr, BN: tl.constexpr, TILES: tl.constexpr,
                                 CORES: tl.constexpr, FIRST_OPT: tl.constexpr, MASK_OPT: tl.constexpr,
                                 QUERY_DRAIN: tl.constexpr, ROTATION: tl.constexpr):
    PAD: tl.constexpr = 1
    VM: tl.constexpr = BM // 2
    SCALE: tl.constexpr = D**(-0.5)
    tl.static_assert(VM == 64 and BN >= 128 and (BN <= 512) and FIRST_OPT)
    p_ub = tla.alloc((VM // 16 * 2, (BN // 2 + PAD) * 16), tl.bfloat16, space='ub')
    acc_ub = tla.alloc((VM, D), tl.float32, space='ub')
    out_ub = tla.alloc((VM, D), tl.bfloat16, space='ub')
    m_ub = tla.alloc((192, 1), tl.float32, space='ub')
    l_ub = tla.alloc((192, 1), tl.float32, space='ub')
    m_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    l_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    stat_m = tla.alloc((VM, 8), tl.float32, space='ub')
    stat_l = tla.alloc((VM, 8), tl.float32, space='ub')
    pv_ub = tla.alloc((VM, D), tl.float32, space='ub')
    q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    s0 = tla.alloc((BN, VM), tl.float32, space='ub')
    p0 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf0 = tla.reshape(p0, (BM // 16, BN * 16))
    alpha0 = tla.alloc((64, 1), tl.float32, space='ub')
    s1 = tla.alloc((BN, VM), tl.float32, space='ub')
    p1 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf1 = tla.reshape(p1, (BM // 16, BN * 16))
    alpha1 = tla.alloc((64, 1), tl.float32, space='ub')
    p2 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf2 = tla.reshape(p2, (BM // 16, BN * 16))
    alpha2 = tla.alloc((64, 1), tl.float32, space='ub')
    with tla.scope.vector():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        with tla.scope.simd():
            zeros = tla.simd.vfull(0.0, tl.float32, 64)
            for state in tl.static_range(3):
                tla.simd.vsts(m_ub[state * 64, 0], zeros)
                tla.simd.vsts(l_ub[state * 64, 0], zeros)
            tla.simd.vsts(m_tmp[0, 0], zeros)
            tla.simd.vsts(l_tmp[0, 0], zeros)
    query_count = (TILES - 1 - tl.program_id(0)) // CORES + 1
    task_count = 0
    for q_index in tl.range(query_count):
        count_pid = tl.program_id(0) + q_index * CORES
        count_bh = count_pid // (S // BM)
        count_row = (count_pid % (S // BM) + count_bh * ROTATION) % (S // BM) * BM
        count_end = count_row + BM if CAUSAL else S
        task_count += (count_end + BN - 1) // BN
    c_query = 0
    c_key = 0
    p_query = 0
    p_key = 0
    sm_query = 0
    sm_key = 0
    u_query = 0
    u_key = 0
    for step in tl.range(0, task_count + 3, 6):
        for PHASE in pipeline_phases(6):
            # Physical ring period lcm(2 score slots, 3 P/alpha slots) = 6.
            # Expand all phases before JIT; no runtime buffer selection.
            t = step + 0
            if t < task_count:
                c_pid = tl.program_id(0) + c_query * CORES
                c_bh = c_pid // (S // BM)
                c_row = (c_pid % (S // BM) + c_bh * ROTATION) % (S // BM) * BM
                c_base = c_bh * S * D
                c_end = c_row + BM if CAUSAL else S
                inline_qk_full_tile(k_l1)
                c_key += BN
                if c_key >= c_end:
                    c_query += 1
                    c_key = 0
            if t >= 2 and t < task_count + 2:
                p_pid = tl.program_id(0) + p_query * CORES
                p_bh = p_pid // (S // BM)
                p_row = (p_pid % (S // BM) + p_bh * ROTATION) % (S // BM) * BM
                p_base = p_bh * S * D
                p_end = p_row + BM if CAUSAL else S
                inline_pv_full_tile(v_l1, pv_ub)
                p_key += BN
                if p_key >= p_end:
                    p_query += 1
                    p_key = 0
            if t >= 1 and t < task_count + 1:
                sm_pid = tl.program_id(0) + sm_query * CORES
                sm_bh = sm_pid // (S // BM)
                sm_row = (sm_pid % (S // BM) + sm_bh * ROTATION) % (S // BM) * BM
                sm_base = sm_bh * S * D
                sm_end = sm_row + BM if CAUSAL else S
                sm_state = sm_query % 3 * 64
                inline_softmax_full_tile()
                sm_key += BN
                if sm_key >= sm_end:
                    sm_query += 1
                    sm_key = 0
            if t >= 3 and t < task_count + 3:
                u_pid = tl.program_id(0) + u_query * CORES
                u_bh = u_pid // (S // BM)
                u_row = (u_pid % (S // BM) + u_bh * ROTATION) % (S // BM) * BM
                u_base = u_bh * S * D
                u_end = u_row + BM if CAUSAL else S
                u_state = u_query % 3 * 64
                inline_output_update(pv_ub, 2, 9)
                u_key += BN
                if u_key >= u_end:
                    u_query += 1
                    u_key = 0
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)


# FA7: double PV, compact causal tail, reversed query order on odd heads.
def fa_forward_phase4_pv2_compact(Q, K, V, O, SM, SL, S: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                                  BM: tl.constexpr, CAUSAL: tl.constexpr, BN: tl.constexpr, TILES: tl.constexpr,
                                  CORES: tl.constexpr, FIRST_OPT: tl.constexpr, MASK_OPT: tl.constexpr,
                                  QUERY_DRAIN: tl.constexpr, ROTATION: tl.constexpr):
    PAD: tl.constexpr = 1
    VM: tl.constexpr = BM // 2
    SCALE: tl.constexpr = D**(-0.5)
    tl.static_assert(VM == 64 and BN >= 128 and (BN <= 512) and FIRST_OPT)
    p_ub = tla.alloc((VM // 16 * 2, (BN // 2 + PAD) * 16), tl.bfloat16, space='ub')
    tl.static_assert(BN == 256)
    p_tail = ext.prefix_view(p_ub, (VM // 16 * 2, (BM // 2 + PAD) * 16))
    acc_ub = tla.alloc((VM, D), tl.float32, space='ub')
    out_ub = tla.alloc((VM, D), tl.bfloat16, space='ub')
    m_ub = tla.alloc((192, 1), tl.float32, space='ub')
    l_ub = tla.alloc((192, 1), tl.float32, space='ub')
    m_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    l_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    stat_m = tla.alloc((VM, 8), tl.float32, space='ub')
    stat_l = tla.alloc((VM, 8), tl.float32, space='ub')
    tl.static_assert(D == 64)
    pv_ub0 = tla.alloc((VM, D), tl.float32, space='ub')
    pv_ub1 = tla.alloc((VM, D), tl.float32, space='ub')
    q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    s0 = tla.alloc((BN, VM), tl.float32, space='ub')
    p0 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf0 = tla.reshape(p0, (BM // 16, BN * 16))
    alpha0 = tla.alloc((64, 1), tl.float32, space='ub')
    s1 = tla.alloc((BN, VM), tl.float32, space='ub')
    p1 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf1 = tla.reshape(p1, (BM // 16, BN * 16))
    alpha1 = tla.alloc((64, 1), tl.float32, space='ub')
    p2 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf2 = tla.reshape(p2, (BM // 16, BN * 16))
    alpha2 = tla.alloc((64, 1), tl.float32, space='ub')
    with tla.scope.vector():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 12)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        with tla.scope.simd():
            zeros = tla.simd.vfull(0.0, tl.float32, 64)
            for state in tl.static_range(3):
                tla.simd.vsts(m_ub[state * 64, 0], zeros)
                tla.simd.vsts(l_ub[state * 64, 0], zeros)
            tla.simd.vsts(m_tmp[0, 0], zeros)
            tla.simd.vsts(l_tmp[0, 0], zeros)
    query_count = (TILES - 1 - tl.program_id(0)) // CORES + 1
    task_count = 0
    for q_index in tl.range(query_count):
        count_pid = tl.program_id(0) + q_index * CORES
        count_bh = count_pid // (S // BM)
        count_row = (S // BM - 1 - count_pid % (S // BM) if CAUSAL and count_bh % 2 else count_pid % (S // BM)) * BM
        count_end = count_row + BM if CAUSAL else S
        task_count += (count_end + BN - 1) // BN
    c_query = 0
    c_key = 0
    p_query = 0
    p_key = 0
    sm_query = 0
    sm_key = 0
    u_query = 0
    u_key = 0
    for step in tl.range(0, task_count + 3, 6):
        for PHASE in pipeline_phases(6):
            # Physical ring period lcm(2 score slots, 3 P/alpha slots) = 6.
            # Expand all phases before JIT; no runtime buffer selection.
            t = step + 0
            if t < task_count:
                c_pid = tl.program_id(0) + c_query * CORES
                c_bh = c_pid // (S // BM)
                c_row = (S // BM - 1 - c_pid % (S // BM) if CAUSAL and c_bh % 2 else c_pid % (S // BM)) * BM
                c_base = c_bh * S * D
                c_end = c_row + BM if CAUSAL else S
                inline_qk_compact_tail(k_l1)
                c_key += BN
                if c_key >= c_end:
                    c_query += 1
                    c_key = 0
            if t >= 2 and t < task_count + 2:
                p_pid = tl.program_id(0) + p_query * CORES
                p_bh = p_pid // (S // BM)
                p_row = (S // BM - 1 - p_pid % (S // BM) if CAUSAL and p_bh % 2 else p_pid % (S // BM)) * BM
                p_base = p_bh * S * D
                p_end = p_row + BM if CAUSAL else S
                inline_pv_compact_tail(v_l1, pv_ub0)
                p_key += BN
                if p_key >= p_end:
                    p_query += 1
                    p_key = 0
            if t >= 1 and t < task_count + 1:
                sm_pid = tl.program_id(0) + sm_query * CORES
                sm_bh = sm_pid // (S // BM)
                sm_row = (S // BM - 1 - sm_pid % (S // BM) if CAUSAL and sm_bh % 2 else sm_pid % (S // BM)) * BM
                sm_base = sm_bh * S * D
                sm_end = sm_row + BM if CAUSAL else S
                sm_state = sm_query % 3 * 64
                inline_softmax_compact_tail()
                sm_key += BN
                if sm_key >= sm_end:
                    sm_query += 1
                    sm_key = 0
            if t >= 3 and t < task_count + 3:
                u_pid = tl.program_id(0) + u_query * CORES
                u_bh = u_pid // (S // BM)
                u_row = (S // BM - 1 - u_pid % (S // BM) if CAUSAL and u_bh % 2 else u_pid % (S // BM)) * BM
                u_base = u_bh * S * D
                u_end = u_row + BM if CAUSAL else S
                u_state = u_query % 3 * 64
                inline_output_update(pv_ub1, 6, 12)
                u_key += BN
                if u_key >= u_end:
                    u_query += 1
                    u_key = 0
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 12)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)


# FA8: single K/V L1, double PV UB; dense attention needs no tail path.
def fa_forward_phase4_pv2(Q, K, V, O, SM, SL, S: tl.constexpr, D: tl.constexpr, H: tl.constexpr, BM: tl.constexpr,
                          CAUSAL: tl.constexpr, BN: tl.constexpr, TILES: tl.constexpr, CORES: tl.constexpr,
                          FIRST_OPT: tl.constexpr, MASK_OPT: tl.constexpr, QUERY_DRAIN: tl.constexpr,
                          ROTATION: tl.constexpr):
    PAD: tl.constexpr = 1
    VM: tl.constexpr = BM // 2
    SCALE: tl.constexpr = D**(-0.5)
    tl.static_assert(VM == 64 and BN >= 128 and (BN <= 512) and FIRST_OPT)
    p_ub = tla.alloc((VM // 16 * 2, (BN // 2 + PAD) * 16), tl.bfloat16, space='ub')
    acc_ub = tla.alloc((VM, D), tl.float32, space='ub')
    out_ub = tla.alloc((VM, D), tl.bfloat16, space='ub')
    m_ub = tla.alloc((192, 1), tl.float32, space='ub')
    l_ub = tla.alloc((192, 1), tl.float32, space='ub')
    m_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    l_tmp = tla.alloc((64, 1), tl.float32, space='ub')
    stat_m = tla.alloc((VM, 8), tl.float32, space='ub')
    stat_l = tla.alloc((VM, 8), tl.float32, space='ub')
    tl.static_assert(D == 64)
    pv_ub0 = tla.alloc((VM, D), tl.float32, space='ub')
    pv_ub1 = tla.alloc((VM, D), tl.float32, space='ub')
    q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
    k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    v_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    s0 = tla.alloc((BN, VM), tl.float32, space='ub')
    p0 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf0 = tla.reshape(p0, (BM // 16, BN * 16))
    alpha0 = tla.alloc((64, 1), tl.float32, space='ub')
    s1 = tla.alloc((BN, VM), tl.float32, space='ub')
    p1 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf1 = tla.reshape(p1, (BM // 16, BN * 16))
    alpha1 = tla.alloc((64, 1), tl.float32, space='ub')
    p2 = tla.alloc((BM // 16, BN // 16, 16, 16), tl.bfloat16, space='l1')
    pf2 = tla.reshape(p2, (BM // 16, BN * 16))
    alpha2 = tla.alloc((64, 1), tl.float32, space='ub')
    with tla.scope.vector():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 9)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 12)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 10)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 11)
        with tla.scope.simd():
            zeros = tla.simd.vfull(0.0, tl.float32, 64)
            for state in tl.static_range(3):
                tla.simd.vsts(m_ub[state * 64, 0], zeros)
                tla.simd.vsts(l_ub[state * 64, 0], zeros)
            tla.simd.vsts(m_tmp[0, 0], zeros)
            tla.simd.vsts(l_tmp[0, 0], zeros)
    query_count = (TILES - 1 - tl.program_id(0)) // CORES + 1
    task_count = 0
    for q_index in tl.range(query_count):
        count_pid = tl.program_id(0) + q_index * CORES
        count_bh = count_pid // (S // BM)
        count_row = (count_pid % (S // BM) + count_bh * ROTATION) % (S // BM) * BM
        count_end = count_row + BM if CAUSAL else S
        task_count += (count_end + BN - 1) // BN
    c_query = 0
    c_key = 0
    p_query = 0
    p_key = 0
    sm_query = 0
    sm_key = 0
    u_query = 0
    u_key = 0
    for step in tl.range(0, task_count + 3, 6):
        for PHASE in pipeline_phases(6):
            # Physical ring period lcm(2 score slots, 3 P/alpha slots) = 6.
            # Expand all phases before JIT; no runtime buffer selection.
            t = step + 0
            if t < task_count:
                c_pid = tl.program_id(0) + c_query * CORES
                c_bh = c_pid // (S // BM)
                c_row = (c_pid % (S // BM) + c_bh * ROTATION) % (S // BM) * BM
                c_base = c_bh * S * D
                c_end = c_row + BM if CAUSAL else S
                inline_qk_full_tile(k_l1)
                c_key += BN
                if c_key >= c_end:
                    c_query += 1
                    c_key = 0
            if t >= 2 and t < task_count + 2:
                p_pid = tl.program_id(0) + p_query * CORES
                p_bh = p_pid // (S // BM)
                p_row = (p_pid % (S // BM) + p_bh * ROTATION) % (S // BM) * BM
                p_base = p_bh * S * D
                p_end = p_row + BM if CAUSAL else S
                inline_pv_full_tile(v_l1, pv_ub0)
                p_key += BN
                if p_key >= p_end:
                    p_query += 1
                    p_key = 0
            if t >= 1 and t < task_count + 1:
                sm_pid = tl.program_id(0) + sm_query * CORES
                sm_bh = sm_pid // (S // BM)
                sm_row = (sm_pid % (S // BM) + sm_bh * ROTATION) % (S // BM) * BM
                sm_base = sm_bh * S * D
                sm_end = sm_row + BM if CAUSAL else S
                sm_state = sm_query % 3 * 64
                inline_softmax_full_tile()
                sm_key += BN
                if sm_key >= sm_end:
                    sm_query += 1
                    sm_key = 0
            if t >= 3 and t < task_count + 3:
                u_pid = tl.program_id(0) + u_query * CORES
                u_bh = u_pid // (S // BM)
                u_row = (u_pid % (S // BM) + u_bh * ROTATION) % (S // BM) * BM
                u_base = u_bh * S * D
                u_end = u_row + BM if CAUSAL else S
                u_state = u_query % 3 * 64
                inline_output_update(pv_ub1, 6, 12)
                u_key += BN
                if u_key >= u_end:
                    u_query += 1
                    u_key = 0
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 9)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 12)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 10)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 11)
