# triton.language.dot_scaled
## 1. OP 概述

简介：**计算以缩放格式表示两个矩阵块的矩阵乘积**

```python
triton.language.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
						acc=None, lhs_k_pack=True, rhs_k_pack=True,
	 					out_dtype=triton.language.float32, _semantic=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `lhs`        | `tensor`          | 左矩阵张量的基指针（支持bf16、fp16格式）                                                      |
| `lhs_scale`        | `tensor`          | 左矩阵缩放张量的基指针（支持int8格式）                                                      |
| `lhs_format`        | `string`          | 左矩阵张量的存放格式 （支持"bf16"和"fp16"）                                                     |
| `rhs`        | `tensor`          | 右矩阵张量的基指针 （支持bf16、fp16格式）                                                     |
| `rhs_scale`        | `tensor`          | 右矩阵缩放张量的基指针（支持int8格式）                                                       |
| `rhs_format`        | `string`          | 右矩阵张量的存放格式 （支持"bf16"和"fp16"）                                                      |
| `acc`       | `tensor`    | 累积张量                                                        |
| `lhs_k_pack`     | `(bool, optional)`    | true	沿 K 维度打包<br>false	沿 M 维度打包<br>|
| `rhs_k_pack ` | `(bool, optional)` | true	沿 K 维度打包<br>false	沿 N 维度打包<br>|
| `_semantic`   | -                 | 保留参数，暂不支持外部调用                                                |

返回值：
`out`：tensor类型，计算缩放矩阵乘后输出的值

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |     fp4     |    fp8    |    bf16    |    fp16    |
| ------------- | --------- | -------- | -------- | -------- |
| GPU    | √    | √     | √     | √     |
| Ascend A2/A3 | ×    | ×     | √     | √    |

结论：
1、Ascend 对比 GPU 缺失fp4、fp8的支持能力（硬件限制）。
2、缩放张量的值为int8，GPU上为uint8。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 可支持 2~3维 tensor |
| Ascend | 可支持 2~3维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，lhs/rhs矩阵均支持 2 至 3 维张量，但scale矩阵只支持2维。

### 2.3 特殊限制说明

1、由于不支持fp8，左右矩阵不支持fp4、fp8格式，Ascend 对比 GPU 缺失lhs_k_pack、rhs_k_pack的矩阵解压缩支持能力（硬件限制）。
2、输入矩阵lhs、rhs推荐输入范围为[-5, 5]，超过可能会出现极值inf。
3、由于硬件存在对齐要求，需要限制scale矩阵做broadcast的倍数，至少应为16

4、当前支持的缩放矩阵格式为int8，社区为uint8



### 2.4 使用方法

以下示例实现了对输入张量 `x` 做就地绝对值计算：

```python@triton.jit
def dot_scale_kernel(a_base, stride_a0: tl.constexpr, stride_a1: tl.constexpr, a_scale, b_base, stride_b0: tl.constexpr,
                     stride_b1: tl.constexpr, b_scale, out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                     type_b: tl.constexpr):
    PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
    PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
    str_a0: tl.constexpr = stride_a0
    a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0,
                                                                            str_a0)[None, :] * stride_a1
    b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0,
                                                                                     BLOCK_N)[None, :] * stride_b1

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if a_scale is not None:
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                           SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(scale_a_ptr)
    if b_scale is not None:
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                           SCALE_BLOCK_K)[None, :]
        b_scale = tl.load(scale_b_ptr)
    accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b, acc=accumulator, out_dtype=tl.float32)

    out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr, accumulator.to(a.dtype))

x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
y = torch.randn(shape, dtype=torch.bfloat16, device="npu")
M, K = shape[0], shape[1]
scale_x = torch.randint(min_scale - 128, max_scale - 127, (M, K // 32), dtype=torch.int8, device="npu")
scale_y = torch.randint(min_scale - 128, max_scale - 127, (N, K // 32), dtype=torch.int8, device="npu")
type_a, type_b = "bf16", "bf16"
pgm = dot_scale_kernel[(1,)](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b)
```
