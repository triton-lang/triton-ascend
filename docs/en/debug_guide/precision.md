# Precision Comparison and Error Analysis

This document describes how to perform precision comparison and error analysis for Triton operators on Ascend NPU, including comparison methods, evaluation criteria, and precautions.

## 1. Precision Comparison Workflow

### Basic Steps

1. **Obtain Reference Result (Golden)**: Compute results using equivalent Torch operators on CPU/GPU/NPU, or results from the same Triton operator on CPU/GPU

2. **Obtain Triton Result**: Run the Triton kernel on Ascend NPU to get the computation result

3. **Compare and Evaluate**: Use `torch.testing.assert_close` to determine whether the precision requirements are met

### Example: Vector Add

```python
import torch
import triton
import triton.language as tl


def test_vector_add(n, dtype):
    # 1. Input data
    x = torch.randn(n, dtype=dtype, device="cpu")
    y = torch.randn(n, dtype=dtype, device="cpu")

    # 2. Reference result (PyTorch CPU)
    torch_ref = x + y

    # 3. Triton kernel
    @triton.jit
    def add_kernel(in0_ptr, in1_ptr, out_ptr, n: tl.constexpr):
        idx = tl.arange(0, n)
        a = tl.load(in0_ptr + idx)
        b = tl.load(in1_ptr + idx)
        tl.store(out_ptr + idx, a + b)

    def triton_func(x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x.npu(), y.npu(), out, n=x.numel())
        return out

    triton_cal = triton_func(x, y)

    # 4. Precision comparison
    compare_precision(triton_cal.cpu(), torch_ref)
```

## 2. Precision Comparison Function

```python
def compare_precision(cal, ref):
    """
    Precision comparison function: selects the appropriate comparison
    strategy based on the data type.

    Args:
        cal: Computed result
        ref: Reference result
        rtol: Relative tolerance
        atol: Absolute tolerance

    Raises:
        AssertionError: Raised when precision does not meet requirements
    """
    assert cal.dtype == ref.dtype, f"dtype mismatch: {cal.dtype} vs {ref.dtype}"
    tensor_dtype = cal.dtype

    if tensor_dtype == torch.float16:
        torch.testing.assert_close(ref, cal, rtol=1e-3, atol=1e-3, equal_nan=True)

    elif tensor_dtype == torch.bfloat16:
        torch.testing.assert_close(ref, cal, rtol=5e-3, atol=5e-3, equal_nan=True)

    elif tensor_dtype == torch.float32:
        torch.testing.assert_close(ref, cal, rtol=1e-5, atol=1e-5, equal_nan=True)

    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        assert torch.equal(cal, ref), f"Integer tensors are not equal for dtype {tensor_dtype}"

    elif tensor_dtype == torch.bool:
        assert torch.equal(cal, ref), "Boolean tensors are not equal"

    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor_dtype}")

    print(f"dtype: {tensor_dtype} — Precision check passed.")
```

## 3. Precision Evaluation Criteria

### Evaluation Rules

`torch.testing.assert_close` / `torch.equal` does not raise an exception → **Pass**, otherwise **Fail**.

* `torch.testing.assert_close`: Passes (no exception raised) if tensors are approximately equal within the specified tolerance; otherwise fails (AssertionError raised).

* `torch.equal`: Returns True only if the two tensors have exactly the same shape and all elements are bitwise identical; otherwise return False.

Internal logic of `torch.testing.assert_close`:

```
|cal - ref| <= atol + rtol * |ref|
```

That is, the absolute error `|cal - ref|` must satisfy a dynamic error boundary composed of the relative tolerance and absolute tolerance combined.

### Recommended Tolerances by Data Type

| Data Type | rtol | atol | Notes |
|---|---|---|---|
| `float32` | 1e-5 | 1e-5 | Strict |
| `float16` | 1e-3 | 1e-3 | Lower precision, appropriately relaxed |
| `bfloat16` | 5e-3 | 5e-3 | Lower precision, appropriately relaxed |
| `int8/16/32/64` | — | — | Must be exactly equal (`torch.equal`) |
| `bool` | — | — | Must be exactly equal (`torch.equal`) |

## 4. Precautions

### NaN / Inf Handling

`equal_nan=True` treats NaN as equal. Set to `False` if strict detection of NaN differences is required.

### Integer Types

Integer and boolean types do not allow any error; they must be strictly identical. When comparing across devices, ensure the data has been moved to the same device (e.g., CPU) to avoid misjudgment caused by underlying representation differences.
