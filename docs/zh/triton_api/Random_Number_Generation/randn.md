# triton.language.randn

## 1. OP 概述

简介：给定 1 个 seed 标量和 1 个 offset 块，返回 1 个 在 **N**(**0**,**1**)中（服从标准正态分布）的 float32 类型的随机块。
原型：

```python
triton.language.randn(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `seed`        | `int`或 `tensor`           | 用于生成随机数的种子                                                   |
| `offset`       |`int`或 `tensor`     | 用于生成随机数的偏移量                     |
| `n_rounds`     | `constexpr`，默认值为10   | Philox 算法的迭代轮数 |

返回值：
1 个 float32 类型的随机块，shape与offset的相同，其值服从标准正态分布 `N(0, 1)`

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入seed的type：

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |

#### 2.2.2 Shape 支持

无特殊要求

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

### 2.4 使用方法

以下示例实现了对randn的调用：

```python
@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randn(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals) # 存储随机数

y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()
numel = y_calf.numel()
ncore = 1 if numel < 32 else 32
xblock = math.ceil(numel / ncore)
kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)
```
