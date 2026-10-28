# Cube算子开发

Cube算子以矩阵乘或批量矩阵乘为主要计算负载，Triton代码中通常以 `tl.dot` 为核心。Cube算子的关键是围绕M/N/K三个维度设计tile，使A/B tile能高效搬运到片上并在Cube Core上完成累加。

## Cube简单算子开发

简单Cube算子可参考本仓 [矩阵乘法样例](../examples/05_matrix_multiplication_example.md)。一个最小开发路径包括：

1. 明确输入输出shape和stride，例如 `A[M, K]`、`B[K, N]`、`C[M, N]`。
2. 用 `tl.program_id` 映射当前program到输出矩阵的 `(pid_m, pid_n)` tile。
3. 用 `BLOCK_SIZE_M/N/K` 构造A/B的二维偏移。
4. 沿K维循环加载A/B子块，并用 `tl.dot` 累加到fp32 accumulator。
5. 将accumulator转为输出dtype，并用边界mask写回C。

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

简单Cube算子调参时优先关注：

- `BLOCK_M/N/K` 是否满足硬件支持和UB/L1容量限制。
- K维循环是否可以开启 `multibuffer` 以形成搬运和计算流水。
- 输出tile是否包含额外bias、scale、activation。如果后处理很轻，可以仍归为Cube算子；如果后处理包含明显Vector归约或跨核同步，应按CV融合算子组织。

## Cube复杂算子开发

复杂Cube场景通常来自attention、batched matmul、grouped matmul或形状不规则的矩阵乘。当前 [`Ascend/triton-ascend-ops`](https://github.com/Ascend/triton-ascend-ops) 主分支的复杂案例集中在 `tutorial/best_practice/`，其中 [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) 可以作为复杂Cube核心逻辑的参考：它包含QK、PV两段 `tl.dot`，并展示了KV cache离散索引下如何重组K/V访存。

复杂Cube算子建议按以下顺序拆解：

1. **先抽出纯矩阵乘核心**：确认每次 `tl.dot` 的输入tile shape、dtype、累加dtype和输出tile shape。
2. **再处理不规则访存**：如果K/V cache低维离散、高维连续，直接二维load可能退化为标量访存。可先按连续维搬入UB，再通过转置或Ascend扩展接口 `extension.insert_slice` 重组为 `tl.dot` 需要的布局。
3. **把归约和归一化留到边界明确的位置**：例如attention中的 `max/sum/exp` 属于Vector逻辑，若和 `tl.dot` 放在同一kernel，需要转到 [CV融合算子开发](./cv_fusion_operator.md) 的思路。
4. **为长K或长序列设计内层循环**：K维循环要控制单次A/B tile的片上占用；序列维循环要避免一次load过大的K/V block。
5. **用Autotune管理候选tile**：为常见shape准备多组 `BLOCK_M/N/K` 和 `multibuffer` 配置，让运行时选择最优组合。

复杂Cube算子的常见风险是把GPU上的大量program直接迁移到NPU。对于逻辑核相互独立且通过IR安全分析的kernel，后端会自动把逻辑grid映射到可用物理核；不适用的kernel会被自动跳过。如果kernel存在顺序依赖或未通过安全分析，应让每个program通过内层循环处理多个tile。
