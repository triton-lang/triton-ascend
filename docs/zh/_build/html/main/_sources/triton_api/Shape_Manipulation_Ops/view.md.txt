# triton.language.view

## 1 功能作用说明

创建张量的视图，改变形状但不复制数据，类似于reshape，但更强调视图的概念，保持数据在内存中的连续性。

**语法：**

- `triton.language.view(input, shape)` - 函数调用形式
- `input.view(shape)` - 成员函数形式

**功能：**

- 创建张量的视图，改变形状但不复制数据
- 类似于reshape，但更强调视图的概念
- 保持数据在内存中的连续性

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |
| shape | List[int] | 是 | 目标形状 |

**返回值：**

- **类型：** tensor
- **形状：** 与shape参数指定的目标形状相同
- **数据类型：** 与输入张量相同
- **内存布局：** 与输入张量在内存中连续

**约束条件：**

- 输入和输出张量的总元素数必须相等
- 输出张量必须与输入张量在内存中连续

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
def view_example():
    # 创建2x3x4的张量
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # 创建视图，变成6x4
    y = tl.view(x, [6, 4])

    return y

## 调用示例
result = view_example()
print(result.shape)  # 输出: (6, 4)
```
