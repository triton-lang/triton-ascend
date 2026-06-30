# Cube 算子开发

Cube 算子以矩阵乘或批量矩阵乘为主要计算负载，Triton 代码中通常以 `tl.dot` 为核心。Cube 算子的关键是围绕 M/N/K 三个维度设计 tile，使 A/B tile 能高效搬运到片上并在 Cube Core 上完成累加。

## Cube 简单算子开发

简单 Cube 算子可参考本仓 [矩阵乘法样例](../examples/05_matrix_multiplication_example.md) 或 `third_party/ascend/tutorials/03-matrix-multiplication.py`。一个最小开发路径包括：

1. 明确输入输出 shape 和 stride，例如 `A[M, K]`、`B[K, N]`、`C[M, N]`。
2. 用 `tl.program_id` 映射当前 program 到输出矩阵的 `(pid_m, pid_n)` tile。
3. 用 `BLOCK_SIZE_M/N/K` 构造 A/B 的二维偏移。
4. 沿 K 维循环加载 A/B 子块，并用 `tl.dot` 累加到 fp32 accumulator。
5. 将 accumulator 转为输出 dtype，并用边界 mask 写回 C。

核心结构如下：

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  stride_am: tl.constexpr, stride_ak: tl.constexpr,
                  stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                  stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

简单 Cube 算子调参时优先关注：

- `BLOCK_M/N/K` 是否满足硬件支持和 UB/L1 容量限制。
- K 维循环是否可以开启 `multibuffer` 以形成搬运和计算流水。
- 输出 tile 是否包含额外 bias、scale、activation。如果后处理很轻，可以仍归为 Cube 算子；如果后处理包含明显 Vector 归约或跨核同步，应按 CV 融合算子组织。

## Cube 复杂算子开发

复杂 Cube 场景通常来自 attention、batched matmul、grouped matmul 或形状不规则的矩阵乘。当前 [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) 主分支的复杂案例集中在 `tutorial/best_practice/`，其中 [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) 可以作为复杂 Cube 核心逻辑的参考：它包含 QK、PV 两段 `tl.dot`，并展示了 KV cache 离散索引下如何重组 K/V 访存。

复杂 Cube 算子建议按以下顺序拆解：

1. **先抽出纯矩阵乘核心**：确认每次 `tl.dot` 的输入 tile shape、dtype、累加 dtype 和输出 tile shape。
2. **再处理不规则访存**：如果 K/V cache 低维离散、高维连续，直接二维 load 可能退化为标量访存。可先按连续维搬入 UB，再通过转置或 `tl.insert_slice` 重组为 `tl.dot` 需要的布局。
3. **把归约和归一化留到边界明确的位置**：例如 attention 中的 `max/sum/exp` 属于 Vector 逻辑，若和 `tl.dot` 放在同一 kernel，需要转到 [CV 融合算子开发](./cv_fusion_operator.md) 的思路。
4. **为长 K 或长序列设计内层循环**：K 维循环要控制单次 A/B tile 的片上占用；序列维循环要避免一次 load 过大的 K/V block。
5. **用 Autotune 管理候选 tile**：为常见 shape 准备多组 `BLOCK_M/N/K` 和 `multibuffer` 配置，让运行时选择最优组合。

复杂 Cube 算子的常见风险是把 GPU 上的大量 program 直接迁移到 NPU。若输出 tile 数远大于物理 Cube Core 数，可考虑让每个 program 通过内层循环处理多个 tile，或者在确认逻辑核相互独立时设置 `TRITON_ALL_BLOCKS_PARALLEL=1` 降低调度开销。
