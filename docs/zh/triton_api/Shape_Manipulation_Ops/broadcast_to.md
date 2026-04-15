# triton.language.broadcast_to
## 1 功能作用说明

将张量广播到目标形状，自动处理维度对齐。广播操作不会复制数据，而是通过改变张量的形状和步长来实现。

**语法：**

- `triton.language.broadcast_to(input, shape)` - 函数调用形式
- `input.broadcast_to(shape)` - 成员函数形式

**功能：**

- 自动处理维度对齐，将大小为1的维度扩展到目标形状中对应维度的大小
- 保持数据不变，仅改变张量的形状信息

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
- **内存布局：** 通过改变步长信息实现广播，无数据拷贝

**约束条件：**

- 输入张量的维度数必须等于目标形状的维度数
- 所有维度必须满足广播规则

### 2.2 DataType支持表

| 支持情况 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape支持表

支持任意维度数、任意形状大小。

### 2.4 特殊限制说明

与broadcast不同，Triton社区实现的broadcast_to必须保证tensor的shape和目标shape的rank一致

### 2.5 使用方法

**基本用法：**

```python
@triton.jit
def matrix_add_bias_kernel(x_ptr, bias_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 加载数据块
    x = tl.load(x_ptr + offsets, mask=mask)

    # 广播bias到匹配的形状
    bias = tl.load(bias_ptr)
    bias_broadcast = bias.broadcast_to([BLOCK_M, BLOCK_N])

    # 执行加法
    output = x + bias_broadcast
    tl.store(output_ptr + offsets, output, mask=mask)
```
