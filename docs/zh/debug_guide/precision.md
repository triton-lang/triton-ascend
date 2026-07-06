# 精度对比与误差分析

本文档介绍如何在昇腾 NPU 上进行 Triton 算子的精度对比与误差分析，包括对比方法、评判标准及注意事项。

## 1. 精度对比流程

### 基本步骤

1. **获取参考结果**（Golden）：使用等价 Torch 算子在 CPU/GPU/NPU 上计算结果或者相同 Triton 算子在 CPU/GPU 上的计算结果

2. **获取 Triton 结果**：在昇腾 NPU 上运行 Triton kernel 得到计算结果

3. **对比判断**：使用 `torch.testing.assert_close` 判断是否满足精度要求

### 示例：向量加法（Vector Add）

```python
import torch
import triton
import triton.language as tl


def test_vector_add(n, dtype):
    # 1. 输入数据
    x = torch.randn(n, dtype=dtype, device="cpu")
    y = torch.randn(n, dtype=dtype, device="cpu")

    # 2. 参考结果（PyTorch CPU）
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

    # 4. 精度对比
    compare_precision(triton_cal.cpu(), torch_ref)
```

## 2. 精度对比函数

```python
def compare_precision(cal, ref):
    """
    精度对比函数：根据数据类型选择合适的比对策略。

    参数:
        cal: 计算结果
        ref: 参考结果
        rtol: 相对误差容限
        atol: 绝对误差容限

    异常:
        AssertionError: 精度不达标时抛出
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

## 3. 精度判定标准

### 判定规则

`torch.testing.assert_close`/`torch.equal` 不抛出异常 → **通过**，否则 **不通过**。

* `torch.testing.assert_close`：如果张量在指定的容差范围内近似相等，则通过（不抛出异常）；否则不通过（抛出 AssertionError）。

* `torch.equal`：要求两个张量具有完全相同的形状，且所有元素在二进制级别上绝对相等，才返回 True，否则返回 False。

`torch.testing.assert_close` 的内部逻辑：

```
|cal - ref| <= atol + rtol * |ref|
```

即：绝对误差 `|cal - ref|` 需要满足相对误差容限和绝对误差容限相加共同构成一个动态的误差边界。

### 按数据类型推荐容限

| 数据类型 | rtol | atol | 说明 |
|---|---|---|---|
| `float32` | 1e-5 | 1e-5 | 严格 |
| `float16` | 1e-3 | 1e-3 | 精度较低，适当放宽 |
| `bfloat16` | 5e-3 | 5e-3 | 精度较低，适当放宽 |
| `int8/16/32/64` | — | — | 必须完全一致（`torch.equal`） |
| `bool` | — | — | 必须完全一致（`torch.equal`） |


## 4. 注意事项

### NaN / Inf 处理

`equal_nan=True` 将 NaN 视为相等。若需要严格检测 NaN 差异，设为 `False`。

### 整数类型

整数和布尔类型不允许任何误差，需严格一致。在跨设备比对时，务必确保数据已移至同一设备（如 CPU），以避免底层表示差异带来的误判。
