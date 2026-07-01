# triton.language.randint4x

## 1. OP 概述

简介：给定 1 个 seed 标量和 1 个 offset 块，返回 4 个 int32 类型的随机块。
Triton 的 Philox 伪随机数生成器的最高效入口点。
原型：

```python
triton.language.randint4x(
 seed, 
 offset, 
 n_rounds: constexpr = 10
)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `seed`        |`int`或 `tensor`          | 用于生成随机数的种子                                                   |
| `offset`       | `int`或 `tensor`      | 用于生成随机数的偏移量                     |
| `n_rounds`     | `constexpr`，默认值为10   | Philox 算法的迭代轮数 |

返回值：
4 个 int32 类型的随机块，每个块的shape与offset相同

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

以下示例实现了offset标量时对randint4x的调用：

```python
@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask) # 存储随机数

y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)
```

以下例子实现了offset非标量时对randint4x的调用，其中存储所用的tensor大小是offset的4倍：

```python
@triton.jit
def triton_randint4x1d(out_ptr, seed, L: tl.constexpr):
 idx = tl.arange(0, L)
 rnd0, rnd1, rnd2, rnd3 = tl.randint4x(seed, idx)
 pointers0 = out_ptr + idx
 pointers1 = out_ptr + L + idx
 pointers2 = out_ptr + 2 * L + idx
 pointers3 = out_ptr + 3 * L + idx
 tl.store(pointers0, rnd0)
 tl.store(pointers1, rnd1)
 tl.store(pointers2, rnd2)
 tl.store(pointers3, rnd3)
```
