# triton.language.ravel
## 1 功能作用说明

将输入张量展平为一维张量，保持元素在内存中的顺序，输出张量的总元素数与输入张量相同。

**语法：**

- `triton.language.ravel(input)` - 函数调用形式
- `input.ravel()` - 成员函数形式

**功能：**

- 将输入张量展平为一维张量
- 保持元素在内存中的顺序
- 输出张量的总元素数与输入张量相同

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |

**返回值：**

- **类型：** tensor
- **形状：** 一维张量，包含输入张量的所有元素
- **数据类型：** 与输入张量相同
- **内存布局：** 按行优先顺序展平

**约束条件：**

- 无特殊约束，支持任意形状的输入

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
def flatten_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # 加载2D数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 展平为一维
    x_flat = x.ravel()

    # 存储展平结果
    tl.store(output_ptr + offsets, x_flat, mask=mask)
```
