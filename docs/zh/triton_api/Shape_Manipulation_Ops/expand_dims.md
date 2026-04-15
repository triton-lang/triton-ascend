# triton.language.expand_dims
## 1 功能作用说明

在指定轴位置插入大小为1的维度，不改变张量的数据，仅增加维度数。支持负索引，从右向左计数。

**语法：**

- `triton.language.expand_dims(input, axis)` - 函数调用形式
- `input.expand_dims(axis)` - 成员函数形式

**功能：**

- 在指定轴位置插入大小为1的维度
- 不改变张量的数据，仅增加维度数
- 支持负索引，从右向左计数

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |
| axis | int \| Tuple[int] | 是 | 插入维度的位置，支持负索引 |

**返回值：**

- **类型：** tensor
- **形状：** 在指定axis位置插入大小为1的维度
- **数据类型：** 与输入张量相同
- **内存布局：** 通过tensor::ExpandShapeOp实现，无数据拷贝

**约束条件：**

- axis必须在[-rank-1, rank]范围内，其中rank为输入张量的维度数
- 插入的维度大小固定为1

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
def expand_dims_example():
    # 创建2x3的张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 在axis=1位置插入维度，变成2x1x3
    y = tl.expand_dims(x, axis=1)

    return y

## 调用示例
result = expand_dims_example()
print(result.shape)  # 输出: (2, 1, 3)
```
