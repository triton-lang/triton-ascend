# al.dot_s4 API Documentation

## 1. Operation Overview

`al.dot_s4` computes matrix multiplication over packed signed 4-bit integers
with an optional accumulator:

```text
D = A @ B + acc
```

Here, `A` and `B` are matrices with shapes `[M, K]` and `[K, N]`.
The result `D` has shape `[M, N]` and dtype `int32`. Omitting `acc` is equivalent
to starting with an all-zero `int32` accumulator.

## 2. Interface Description

```python
import triton.language.extra.cann.extension as al

al.dot_s4(input, other, acc=None)
```

### Parameters

| Parameter | Type                                  | Required | Description                                                                                  |
| --------- | ------------------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `input`   | 2D `tl.tensor`, `tl.int8`             | Yes      | Packed `A`, shape `[M, K/2]`; pairs are packed along `K`.                                    |
| `other`   | 2D `tl.tensor`, `tl.int8`             | Yes      | Packed `B`, shape `[K, N/2]`; pairs are packed along `N`.                                    |
| `acc`     | 2D `tl.tensor`, `tl.int32`, or `None` | No       | Initial accumulator of shape `[M, N]`. Defaults to `None`, which creates a zero accumulator. |

### Return Value

A new `tl.tensor` of shape `[M, N]` and dtype `tl.int32`, containing the product
plus the accumulator.

### Packed Representation

Each `int8` container holds two signed int4 values in the range `[-8, 7]`.
Each nibble uses a 4-bit two's-complement representation: the first value is in
bits `[3:0]` (the low nibble), and the second is in bits `[7:4]` (the high nibble).
For values `x0` and `x1`, the packed byte has the bit pattern
`(x0 & 0x0F) | ((x1 & 0x0F) << 4)`.

| Matrix        | Logical shape | Packed `int8` shape | Values in one byte                       |
| ------------- | ------------- | ------------------- | ---------------------------------------- |
| `A` (`input`) | `[M, K]`      | `[M, K/2]`          | Low: `A[m, 2*j]`; high: `A[m, 2*j + 1]`. |
| `B` (`other`) | `[K, N]`      | `[K, N/2]`          | Low: `B[k, 2*j]`; high: `B[k, 2*j + 1]`. |

For example, the pair `[-8, 7]` is packed as `0x78`. The `int8` container is a
storage format;

## 3. Constraints

- Both operands must be rank-2 `tl.tensor` values with dtype exactly `tl.int8`.
- `K` must be even.
- The logical output width is `N = 2 * other.shape[1]`.
- An explicit `acc` must have dtype exactly `tl.int32` and shape exactly
  `[input.shape[0], 2 * other.shape[1]]`.
- Register-based architectures are not supported

## 4. Example Usage

The example tiles `M` and packed `N`, accumulates three blocks along `K`, and
masks tail loads and stores. CPU inputs are generated with a fixed seed.
`BLOCK_N` counts packed bytes, so each output tile has `2 * BLOCK_N` columns.

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


def pack_int4(values):
    # Pack pairs along the last axis: low nibble first, high nibble second.
    if values.shape[-1] % 2:
        values = torch.nn.functional.pad(values, (0, 1), value=0)
    low = values[:, 0::2] & 0x0F
    high = values[:, 1::2] & 0x0F
    return (low | (high << 4)).to(torch.int8)


@triton.jit
def dot_s4_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_packed = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, 2 * BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        a_k = k_start // 2 + offs_k_packed
        a_ptrs = a_ptr + offs_m[:, None] * (K // 2) + a_k[None, :]
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (a_k[None, :] < K // 2),
            other=0,
        )

        b_k = k_start + offs_k
        b_ptrs = b_ptr + b_k[:, None] * tl.cdiv(N, 2) + offs_n_packed[None, :]
        b = tl.load(
            b_ptrs,
            mask=(b_k[:, None] < K) & (offs_n_packed[None, :] < tl.cdiv(N, 2)),
            other=0,
        )

        accumulator = al.dot_s4(a, b, accumulator)

    offs_n = tl.program_id(1) * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def main():
    m, n, k = 33, 32, 160
    block_m, block_n, block_k = 16, 8, 64
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-8, 8, (m, k), dtype=torch.int8, generator=generator)
    b = torch.randint(-8, 8, (k, n), dtype=torch.int8, generator=generator)
    expected = a.to(torch.int32) @ b.to(torch.int32)

    a_packed = pack_int4(a).npu()
    b_packed = pack_int4(b).npu()
    actual = torch.empty((m, n), dtype=torch.int32, device="npu")
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, 2 * block_n))
    dot_s4_kernel[grid](
        a_packed, b_packed, actual, m, n, k,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    main()
```
