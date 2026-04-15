# triton.language.randint
## 1. OP 概述

简介：给定 1 个 seed 标量和 1 个 offset 块，返回 1 个 int32 类型的随机块。
原型：

```python
triton.language.randint(
	seed,
	offset,
	n_rounds: constexpr = 10
)
```

如果需要多个随机数流，使用 randint4x 可能比连续调用 4 次 randint 更快。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `seed`        | `int`或 `tensor`           | 用于生成随机数的种子                                                   |
| `offset`       |`int`或 `tensor`     | 用于生成随机数的偏移量                     |
| `n_rounds`     | `constexpr `，默认值为10   | Philox 算法的迭代轮数 |

返回值：
1 个 int32 类型的随机块，shape与offset相同

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

以下示例实现了对randint的调用（调用时生成单个随机数）：

```python
@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals) # 存储随机数

y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)
```
