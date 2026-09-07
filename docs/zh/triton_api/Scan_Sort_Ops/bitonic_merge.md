# triton.language.bitonic_merge

## 1. OP 概述

简介：对输入张量`x`沿指定维度执行 bitonic merge（双调合并）操作，将一条 bitonic 序列合并为单调有序（升序或降序）序列。

> bitonic 序列指先单调递增后单调递减（或其循环移位后满足该性质）的序列，例如 `[1, 3, 5, 7, 6, 4, 2, 0]`。该操作假定输入已经满足 bitonic 序列性质，通常作为 `sort`、`topk` 排序网络的内部阶段使用，也可独立调用。

```python
triton.language.bitonic_merge(x, dim: constexpr = None, descending: constexpr = False)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名        | 类型       | 说明                                                             |
| ------------- | ---------- | ---------------------------------------------------------------- |
| `x`           | `tensor`   | 输入张量，沿指定维度必须是 bitonic 序列                          |
| `dim`         | `constexpr` | 合并维度，仅支持最后一个维度（默认为最后一维）                   |
| `descending`  | `constexpr` | 是否按降序合并，`False` 表示升序，`True` 表示降序                |

返回值：
`tensor`：输出张量的 shape 与输入`x`的 shape 相同，沿指定维度为合并后的单调有序序列。

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ | × | × | √ |
| Ascend A2/A3 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | √ | √ | √ |

结论：
- Ascend A2/A3 对比 GPU 缺失 fp64 的支持能力。
- Ascend 950 对比 GPU 缺失 fp64 的支持能力。

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| ------ | -------------------- |
| GPU    | 仅支持 1~8 维 tensor，且合并维度的长度必须为 2 的幂 |
| Ascend A2/A3 | 仅支持 1~8 维 tensor，且合并维度的长度必须为 2 的幂 |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 8 维张量，且合并维度的长度必须为 2 的幂。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

- 毕昇编译器限制，fp64 无法实现。
- 仅支持沿最后一个维度（minor dimension）执行合并，`dim` 传入其他维度会触发编译期断言。
- 输入在合并维度上必须是 bitonic 序列，否则合并结果为未定义行为。
- 合并维度的长度必须是 2 的幂。

### 2.4 使用方法

以下示例实现了对一条 bitonic 序列做 bitonic merge 运算：

```python
@triton.jit
def bitonic_merge_kernel(in_ptr0, out_ptr0, N: tl.constexpr, descending: tl.constexpr):
    off = tl.arange(0, N)
    x = tl.load(in_ptr0 + off)
    # 输入沿最后一维必须为 bitonic 序列，例如先增后减
    merged = tl.bitonic_merge(x, dim=0, descending=descending)
    tl.store(out_ptr0 + off, merged)
```

```python
import torch

# [1, 3, 5, 7, 6, 4, 2, 0] 为先增后减的 bitonic 序列，长度为 2 的幂
x = torch.tensor([1, 3, 5, 7, 6, 4, 2, 0], dtype=torch.float32, device="npu")
out = torch.empty_like(x)
bitonic_merge_kernel[(1, )](x, out, N=8, descending=False)
# 升序合并结果: [0, 1, 2, 3, 4, 5, 6, 7]
print(out)
```
