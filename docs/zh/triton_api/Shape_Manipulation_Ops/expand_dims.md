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
import torch
import triton
import triton.language as tl

@triton.jit
def expand_dims_example(out_ptr):
    # 创建2x3的张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 在axis=1位置插入维度，变成2x1x3
    y = tl.expand_dims(x, axis=1)

    # 将结果写回外部张量
    offs = (
        tl.arange(0, 2)[:, None, None] * 3
        + tl.arange(0, 1)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, y)

## 调用示例
out = torch.empty((2, 1, 3), dtype=torch.float32, device="npu")
expand_dims_example[(1,)](out)
print(out.shape)  # 输出: torch.Size([2, 1, 3])
```
