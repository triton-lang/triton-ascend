# triton.language.permute
## 1 功能作用说明

根据dims参数重新排列张量的维度，不改变张量的数据，仅改变维度的顺序。支持任意维度的重新排列。

**语法：**

- `triton.language.permute(input, dims)` - 函数调用形式
- `input.permute(dims)` - 成员函数形式

**功能：**

- 根据dims参数重新排列张量的维度
- 不改变张量的数据，仅改变维度的顺序
- 支持任意维度的重新排列

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |
| dims | List[int] | 是 | 新的维度顺序 |

**返回值：**

- **类型：** tensor
- **形状：** 按照dims参数重新排列的维度
- **数据类型：** 与输入张量相同
- **内存布局：** 通过改变步长信息实现转置，无数据拷贝

**约束条件：**

- dims必须包含输入张量的所有维度索引

### 2.2 DataType支持表

| 支持情况 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape支持表

支持任意维度数和任意形状大小。

### 2.4 特殊限制说明



* 不支持维度高于8的转置

### 2.5 使用方法

```python
import triton
import triton.language as tl

@triton.jit
def permute_example():
    # 创建2x3x4的张量
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # 转置维度，变成4x2x3
    y = tl.permute(x, [2, 0, 1])

    return y

## 调用示例
result = permute_example()
print(result.shape)  # 输出: (4, 2, 3)
```
