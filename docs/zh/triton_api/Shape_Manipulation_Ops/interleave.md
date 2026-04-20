# triton.language.interleave

## 1 功能作用说明

将两个相同形状的输入张量在最后一个维度上交错排列，输出张量的最后一个维度大小为输入张量的2倍，其他维度保持不变。

**语法：**

- `triton.language.interleave(x, y)` - 函数调用形式
- `x.interleave(y)` - 成员函数形式

**功能：**

- 将两个相同形状的输入张量在最后一个维度上交错排列
- 输出张量的最后一个维度大小为输入张量的2倍
- 其他维度保持不变

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| x | tensor | 是 | 第一个输入张量 |
| y | tensor | 是 | 第二个输入张量，形状必须与x相同 |

**返回值：**

- **类型：** tensor
- **形状：** 输入形状的最后一个维度乘以2
- **数据类型：** 与输入张量相同
- **内存布局：** 交替排列x和y的元素

**约束条件：**

- 两个输入张量必须具有相同的形状和数据类型
- 输出张量的形状为输入形状的最后一个维度乘以2

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
import triton
import triton.language as tl

@triton.jit
def interleave_example():
    # 创建两个2x3的张量
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.ones([2, 3], dtype=tl.float32)

    # 交错排列，变成2x6
    z = tl.interleave(x, y)

    return z

## 调用示例
result = interleave_example()
print(result.shape)  # 输出: (2, 6)
```
