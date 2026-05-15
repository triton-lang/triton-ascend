# Cube Operator Development

Cube operators use matrix multiplication or batched matrix multiplication as the main workload. In Triton code, the core operation is usually `tl.dot`. The main task is to design M/N/K tiles so that A and B tiles can be moved on chip efficiently and accumulated on Cube Cores.

## Simple Cube Operator Development

For a simple Cube operator, refer to the [Matrix Multiplication example](../examples/05_matrix_multiplication_example.md) or `third_party/ascend/tutorials/03-matrix-multiplication.py`.

1. Define input/output shapes and strides, for example `A[M, K]`, `B[K, N]`, and `C[M, N]`.
2. Map `tl.program_id` to the output tile `(pid_m, pid_n)`.
3. Build 2D offsets for A and B using `BLOCK_SIZE_M/N/K`.
4. Loop over K, load A/B sub-blocks, and accumulate with `tl.dot` in fp32.
5. Cast the accumulator to the output dtype and store with boundary masks.

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

Tune `BLOCK_M/N/K` within hardware and UB/L1 limits, consider `multibuffer` for the K loop, and classify the operator as CV fusion if the epilogue becomes a non-trivial Vector reduction, softmax, or synchronized post-process.

## Complex Cube Operator Development

Complex Cube cases often come from attention, batched matmul, grouped matmul, or irregular shapes. In the current main branch of [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops), complex cases are mainly in `tutorial/best_practice/`. [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) is a useful reference for the Cube core because it contains QK and PV `tl.dot` stages and shows how to reorganize K/V memory access under discrete KV-cache indices.

Recommended decomposition:

1. Extract the pure matmul core first and verify tile shapes, dtypes, accumulator dtype, and output shape.
2. Handle irregular memory access next. If K/V cache access is discrete in a low dimension and contiguous in a high dimension, load along the contiguous dimension and reorganize in UB with transpose or `tl.insert_slice`.
3. Keep reductions and normalization at clear boundaries. If `max/sum/exp` or softmax is fused into the same kernel, follow the [CV Fusion Operator Development](./cv_fusion_operator.md) guidance.
4. Use inner loops for long K or long sequence dimensions to control on-chip usage.
5. Use autotune to manage candidate `BLOCK_M/N/K` and `multibuffer` configurations.

A common migration risk is directly keeping a GPU-style large grid. If the output tile count is far larger than the physical Cube Core count, let each program process multiple tiles in an inner loop, or set `TRITON_ALL_BLOCKS_PARALLEL=1` when logical programs are independent.
