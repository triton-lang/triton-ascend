# triton.language.split
## 1 功能作用说明

将输入张量沿着最后一个维度分割成两个张量，输出张量的最后一个维度大小为输入张量的一半，其他维度保持不变。

**语法：**

- `triton.language.split(input)` - 函数调用形式
- `input.split()` - 成员函数形式

**功能：**

- 将输入张量沿着最后一个维度分割成两个张量
- 输出张量的最后一个维度大小为输入张量的一半，最后一个维度的大小必须为2
- 其他维度保持不变

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |

**返回值：**

- **类型：** Tuple[tensor, tensor]
- **形状：** 两个张量，形状相同，最后一个维度为输入的一半
- **数据类型：** 与输入张量相同
- **内存布局：** 分别包含输入张量的奇数和偶数位置元素

**约束条件：**

- 输入张量的最后一个维度大小必须为偶数
- 输出两个张量，形状相同

### 2.2 DataType支持表

| 支持情况 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape支持表

支持任意维度数、任意形状大小。

### 2.4 特殊限制说明

无

### 2.5 使用方法

```python
@triton.jit
def complex_split_kernel(complex_ptr, real_ptr, imag_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 加载复数数据
    complex_data = tl.load(complex_ptr + offsets, mask=mask)

    # 分割成实部和虚部
    real_part, imag_part = complex_data.split()

    # 存储实部和虚部
    tl.store(real_ptr + offsets, real_part, mask=mask)
    tl.store(imag_ptr + offsets, imag_part, mask=mask)
```
