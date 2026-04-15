# triton.language.reshape
## 1 功能作用说明

将张量重新解释为新的形状。

**语法：**

- `triton.language.reshape(input, shape, can_reorder=False)` - 函数调用形式
- `input.reshape(shape, can_reorder=False)` - 成员函数形式

**功能：**

- 将张量重新解释为新的形状

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |
| shape | List[int] | 是 | 目标形状 |
| can_reorder | bool | 否 | 是否允许重新排序元素，默认False |

**返回值：**

- **类型：** tensor
- **形状：** 与shape参数指定的目标形状相同
- **数据类型：** 与输入张量相同
- **内存布局：** 根据can_reorder参数决定

**约束条件：**

- 输入和输出张量的总元素数必须相等
- 所有tensor不允许某个shape的size小于1

### 2.2 DataType支持表

| 支持情况 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape支持表

支持任意维度数、任意形状大小。

### 2.4 特殊限制说明

* can_reorder参数仅支持False

### 2.5 使用方法

```python
import triton
import triton.language as tl

@triton.jit
def reshape_example():
    # 创建2x3x4的张量
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # reshape为6x4
    y = tl.reshape(x, [6, 4])

    return y

## 调用示例
result = reshape_example()
print(result.shape)  # 输出: (6, 4)
```
