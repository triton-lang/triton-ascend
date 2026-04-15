# triton.language.broadcast
## 1 功能作用说明

将两个张量广播到共同兼容的形状，使它们可以进行逐元素操作。

**语法：**

- `triton.language.broadcast(input, other)` - 函数调用形式

**功能：**

- 自动对齐不同秩张量得到目标形状
- 将大小为1的维度扩展到目标形状中对应维度的大小

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 第一个输入张量，必须为RankedTensorType |
| other | tensor | 是 | 第二个输入张量，必须为RankedTensorType |

**返回值：**

- **类型：** tensor
- **形状：** 两个tensor共同兼容的目标形状
- **数据类型：** 每个返回的张量保持其输入的原始数据类型
- **内存布局：**返回新创建的张量

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

**基本用法：**

```python@triton.jit@triton.jit
def broadcast_kernel(
    output_ptr,
    BLOCK_SIZE: tl.constexpr
):
    # 创建一个标量（0维张量）
    scalar = 5.0

    # 创建一个向量（1维张量）
    vector = tl.arange(0, BLOCK_SIZE) * 1.0  # 形状: (BLOCK_SIZE,)

    # 使用 broadcast 将标量广播到与向量相同的形状
    # scalar: () -> (BLOCK_SIZE,)
    broadcasted_scalar = tl.broadcast(scalar, vector)

    result = vector + broadcasted_scalar

    # 存储结果
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + offsets, result)

```
