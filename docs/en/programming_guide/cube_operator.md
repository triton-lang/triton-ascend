# Cube Operator Development

Cube operators use matrix multiplication or batched matrix multiplication as the main workload. In Triton code, the core operation is usually `tl.dot`. The main task is to design M/N/K tiles so that A and B tiles can be moved on chip efficiently and accumulated on Cube Cores.

## Simple Cube Operator Development

For a simple Cube operator, refer to the [Matrix Multiplication example](../examples/05_matrix_multiplication_example.md). A minimal development path includes:

1. Define input/output shapes and strides, for example `A[M, K]`, `B[K, N]`, and `C[M, N]`.
2. Map `tl.program_id` to the output tile `(pid_m, pid_n)`.
3. Build 2D offsets for A and B using `BLOCK_SIZE_M/N/K`.
4. Loop over K, load A/B sub-blocks, and accumulate with `tl.dot` in fp32.
5. Cast the accumulator to the output dtype and store with boundary masks.

The core structure is as follows:

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

When tuning a simple Cube operator, prioritize the following:

- Check whether `BLOCK_M/N/K` meets hardware support and UB/L1 capacity limits.
- Check whether `multibuffer` can be enabled for the K-dimension loop to form a data movement and computation pipeline.
- Check whether the output tile includes extra bias, scale, or activation. If the post-processing is light, the operator can still be classified as a Cube operator; if the post-processing involves obvious Vector reduction or cross-core synchronization, it should be organized as a CV fusion operator.

## Complex Cube Operator Development

Complex Cube cases often come from attention, batched matmul, grouped matmul, or irregular shapes. In the current main branch of [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops), complex cases are mainly in `tutorial/best_practice/`. [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) is a useful reference for the Cube core because it contains QK and PV `tl.dot` stages and shows how to reorganize K/V memory access under discrete KV-cache indices.

Recommended decomposition:

1. **Extract the pure matmul core first**: confirm each `tl.dot`'s input tile shape, dtype, accumulator dtype, and output tile shape.
2. **Handle irregular memory access next**: if K/V cache access is discrete in a low dimension and contiguous in a high dimension, a direct 2D load may degrade into scalar access. Load along the contiguous dimension into UB first, then reorganize with transpose or the Ascend extension API `extension.insert_slice` into the layout required by `tl.dot`.
3. **Keep reductions and normalization at clear boundaries**: for example, `max/sum/exp` in attention is Vector logic; if it is placed in the same kernel as `tl.dot`, follow the [CV Fusion Operator Development](./cv_fusion_operator.md) guidance.
4. **Design inner loops for long K or long sequences**: the K-dimension loop should control the on-chip footprint of each A/B tile, and the sequence-dimension loop should avoid loading an oversized K/V block at once.
5. **Use Autotune to manage candidate tiles**: prepare multiple `BLOCK_M/N/K` and `multibuffer` configurations for common shapes and let the runtime choose the optimal combination.

A common migration risk is directly keeping a GPU-style large grid. If the output tile count is far larger than the physical Cube Core count, let each program process multiple tiles in an inner loop, or set `TRITON_ALL_BLOCKS_PARALLEL=1` when logical programs are independent.
