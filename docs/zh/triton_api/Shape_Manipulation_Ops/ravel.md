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

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | × | √ |
| Ascend A2/A3 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ | √ | √ |


结论：
- Ascend A2/A3 对比 GPU 缺失 uint16、uint32、uint64、fp64 的支持能力。
- Ascend 950 对比 GPU 缺失 fp64 的支持能力。

### 2.3 Shape支持表

支持任意维度数、任意形状大小。

### 2.4 特殊限制说明

无

### 2.5 使用方法

```python
@triton.jit
def flatten_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    # 加载2D数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 展平为一维
    x_flat = x.ravel()

    # 存储展平结果
    tl.store(output_ptr + offsets, x_flat, mask=mask)
```
