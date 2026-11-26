# 融合注意力（Fused Attention）

本节实现了一个基于 **Triton** 的 **Flash Attention v2风格的融合注意力前向传播内核**，适用于昇腾（Ascend）NPU平台。该实现支持：

- **因果（causal）与非因果注意力**
- **分块计算（tiling）以处理长序列**
- **数值稳定性优化（max-shifted softmax）**

整体结构包含两个核心Triton内核：

1. `_attn_fwd_inner`：执行单个query block与key/value blocks的attention计算（分阶段处理causal mask）
2. `_attn_fwd`：调度所有query blocks，并管理block指针、accumulator和归一化

并通过PyTorch `autograd.Function` 封装为可调用的 `attention` 函数，与 `torch_npu.npu_fusion_attention` 进行精度对齐验证。

```python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

DEVICE = "npu"


@triton.jit
def _attn_fwd_inner(acc_ptr, l_i, m_i, q,  # 累加器，局部l，局部m，查询向量
                    K_block_ptr, V_block_ptr,  # 当前阶段的Key和Value块指针
                    start_m, qk_scale,  # 当前query块的起始位置，qk缩放因子
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  # 块大小常量
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  # 当前阶段标志，m和n的偏移索引
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):  # 上下文总长度，是否对value启用FP8精度
    # 设置当前阶段的处理范围 [lo, hi)（以列块为单位）
    # causal = true
    # stage = 1
    # 因果注意力（causal attention）顾名思义，限制计算过程中的信息流动，
    # 只允许模型看到当前位置及之前的位置。
    # 换句话说，当前位置的输出只能依赖于该位置及之前的输入，
    # 不能访问未来位置的信息。
    # 因果注意力保证了顺序性，防止“未来信息泄露”。
    # 根据 STAGE 分阶段确定因果掩码的遍历范围 [lo, hi)：
    if STAGE == 1:
        # 阶段1：处理query块之前的所有token
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # 阶段2：处理当前query块
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # 对齐起始位置
    # causal = False（无需掩码）
    else:
        lo, hi = 0, N_CTX  # 处理整个上下文

    # 将K和V块指针调整到起始位置 `lo`
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))  # K的形状为 [HEAD_DIM, N_CTX]，沿第二维偏移lo
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))  # V的形状为 [N_CTX, HEAD_DIM]，沿第一维偏移lo

    # 累加器的索引映射，用于HEAD_DIM >= 256时的切片
    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
    block2d_acc = row * HEAD_DIM + col_head_dim

    # 遍历当前阶段的所有k、v块并累加输出
    for start_n in range(lo, hi, BLOCK_N):  # 每次处理BLOCK_N列
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 对齐列起始位置
        # -- 计算qk ----
        k = tl.load(K_block_ptr)
        # 转置K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # 在STAGE 2时应用因果掩码
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])  # 构造上三角掩码
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)  # 将无效位置设为 -∞
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # 更新m_ij = max(m_i, max(qk))
            qk -= m_ij[:, None]  # 减去最大值以保证softmax数值稳定性
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # 缩放后的最大值
            qk = qk - m_ij[:, None]  # 稳定化

        # softmax权重p = exp(qk)
        p = tl.math.exp(qk)

        # 根据FP8使用情况转换softmax权重类型
        if fp8_v:
            p_cast = p.to(tl.float8e5)  # 转换为FP8格式（节省内存）
        else:
            p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr)  # 加载对应的V块
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)  # softmax分母（每行求和）
        # -- 更新m_i和l_i
        alpha = tl.math.exp(m_i - m_ij)  # 更新因子：新旧最大值的exp差值
        l_i = l_i * alpha + l_ij  # 更新softmax分母
        # -- 更新输出累加器 --
        if HEAD_DIM < 256:
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)
        else:
            # 1. 加载累加器的当前切片
            acc = tl.load(acc_ptr + block2d_acc)
            # 2. 分片更新（按BLOCK_M的1/4切分以避免UB溢出）
            for i in range(4):
                # 计算当前切片的起始/结束行
                offset = i * (BLOCK_M // 4)
                # 提取切片数据
                acc_i = extension.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                alpha_i = extension.extract_slice(alpha, [offset], [BLOCK_M // 4], [1])
                pv_i = extension.extract_slice(pv, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                # 增量更新切片：acc = acc * alpha + pv
                acc_i = acc_i * alpha_i[:, None] + pv_i
                # 将更新后的切片写回累加器
                acc = extension.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            # 3. 更新后的累加器
            tl.store(acc_ptr + block2d_acc, acc)

        m_i = m_ij  # 更新当前块的最大值
        # 将V和K块指针推进到下一个BLOCK_N范围
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    # 返回累加输出acc_ptr、softmax分母l_i和最大值m_i
    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, sm_scale,
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # 序列维度（M）上的总块数
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # 总任务数 = 序列块数 × 批大小（Z）× 注意力头数（H）
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # 当前M维度的块索引
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, tl.num_programs(0)):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # 为Q、K、V、Output创建块指针
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # 初始化偏移量
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        # 初始化累加器
        if HEAD_DIM < 256:
            acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        else:
            acc_offset = (
                off_z.to(tl.int64) * stride_qz // stride_qm * HEAD_DIM +
                off_h.to(tl.int64) * stride_qh // stride_qm * HEAD_DIM +
                task_m_idx * BLOCK_M * HEAD_DIM
            )
            acc_ptr = acc + acc_offset

        q = tl.load(Q_block_ptr)

        # 阶段1：带外（off-band）
        # 当causal = True时，STAGE = 3，_attn_fwd_inner的STAGE取值为1
        # 当causal = False时，STAGE = 1，_attn_fwd_inner的STAGE取值为3
        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                                task_m_idx, sm_scale,  #
                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                                )
        # 阶段2：带内（on-band）
        if STAGE & 2:
            # barrier使得编译器更容易将
            # 两个循环独立调度
            acc_ptr, l_i, m_i = _attn_fwd_inner(acc_ptr, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                                task_m_idx, sm_scale,  #
                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                                )

        m_i += tl.math.log(l_i)
        if HEAD_DIM < 256:
            accumulator = acc_ptr / l_i[:, None]
        else:
            row = tl.arange(0, BLOCK_M)[:, None]
            col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
            block2d_acc = row * HEAD_DIM + col_head_dim
            accumulator = tl.load(acc_ptr + block2d_acc)
            accumulator = accumulator / l_i[:, None]

        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.dtype.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        """
        前向计算接口：
        参数：
            ctx: 上下文对象
            q: Query张量（Q），形状 [Z, H, N_CTX, HEAD_DIM]
            k: Key张量（K），形状 [Z, H, N_CTX, HEAD_DIM]
            v: Value张量（V），形状 [Z, H, N_CTX, HEAD_DIM]
            causal: 是否启用因果注意力
            sm_scale: QK乘积的缩放因子
            BM: Q块大小（BLOCK_M）
            BN: K/V块大小（BLOCK_N）
        返回：
            o: 注意力输出张量，形状 [Z, H, N_CTX, HEAD_DIM]
        """
        # 形状约束
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # 当v为float8_e5m2时会被转置
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}


        # NPU核数（根据硬件调整）
        num_cores = 20
        acc = torch.zeros((q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K), dtype=torch.float32, device=q.device)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attn_fwd[(num_cores,)](
            q, k, v, M, o, acc, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            STAGE=stage,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN", [
    (1, 1, 128, 128, False, torch.float16, 32, 128),
    (1, 1, 128, 128, False, torch.bfloat16, 64, 128),
    (1, 2, 256, 256, False, torch.bfloat16, 32, 256),
    (2, 2, 128, 256, False, torch.float16, 64, 128),
    (4, 32, 64, 64, False, torch.float16, 32, 64),
    (4, 32, 1024, 64, False, torch.bfloat16, 64, 128),
    (4, 32, 4096, 64, False, torch.float16, 128, 128),
])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    # 过滤非整除的情况；N_CTX必须能被BM和BN整除，HEAD_DIM必须能被16整除
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")

    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5

    tri_out = attention(q, k, v, causal, sm_scale, BM, BN)
    ref_out = torch_npu.npu_fusion_attention(
            q, k, v, H,
            padding_mask=None,
            atten_mask=None,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=0,
            )[0]

    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2, equal_nan=True)
    print(f"[PASSED] Attention shape:({Z}, {H}, {N_CTX}, {HEAD_DIM}), BM: {BM}, BN: {BN}, dtype: {dtype}")


if __name__ == "__main__":
    test_op(1, 1, 128, 128, causal=False, dtype=torch.float16, BM=32, BN=128)
    test_op(1, 1, 128, 128, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
    test_op(1, 2, 256, 256, causal=False, dtype=torch.bfloat16, BM=32, BN=256)
    test_op(2, 2, 128, 256, causal=False, dtype=torch.float16, BM=64, BN=128)
    test_op(4, 32, 64, 64, causal=False, dtype=torch.float16, BM=32, BN=64)
    test_op(4, 32, 1024, 64, causal=False, dtype=torch.bfloat16, BM=64, BN=128)
    test_op(4, 32, 4096, 64, causal=False, dtype=torch.float16, BM=128, BN=128)
```

## 输出结果

```output
[PASSED] Attention shape:(1, 1, 128, 128), BM: 32, BN: 128, dtype: torch.float16
[PASSED] Attention shape:(1, 1, 128, 128), BM: 64, BN: 128, dtype: torch.bfloat16
[PASSED] Attention shape:(1, 2, 256, 256), BM: 32, BN: 256, dtype: torch.bfloat16
[PASSED] Attention shape:(2, 2, 128, 256), BM: 64, BN: 128, dtype: torch.float16
[PASSED] Attention shape:(4, 32, 64, 64), BM: 32, BN: 64, dtype: torch.float16
[PASSED] Attention shape:(4, 32, 1024, 64), BM: 64, BN: 128, dtype: torch.bfloat16
[PASSED] Attention shape:(4, 32, 4096, 64), BM: 128, BN: 128, dtype: torch.float16
```

上面输出日志表明Triton和PyTorch上的输出结果在atol=1e-2、rtol=1e-2容差范围内一致。
