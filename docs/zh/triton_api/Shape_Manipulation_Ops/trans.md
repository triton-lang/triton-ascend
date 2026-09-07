# triton.language.trans

## 1 功能作用说明

根据dims参数转置张量的维度，不改变张量的数据，仅改变维度的顺序。专门优化的转置操作。

**语法：**

- `triton.language.trans(input, dims)` - 函数调用形式
- `input.trans(dims)` - 成员函数形式

**功能：**

- 根据dims参数转置张量的维度
- 不改变张量的数据，仅改变维度的顺序
- 专门优化的转置操作

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| input | tensor | 是 | 输入张量 |
| dims | List[int] | 是 | 转置后的维度顺序 |

**返回值：**

- **类型：** tensor
- **形状：** 按照dims参数重新排列的维度
- **数据类型：** 与输入张量相同
- **内存布局：** 通过改变步长信息实现转置，无数据拷贝

**约束条件：**

- dims必须包含输入张量的所有维度索引

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

* 不支持维度高于8的转置

### 2.5 使用方法

```python
import triton
import triton.language as tl

@triton.jit
def trans_example():
    # 创建2x3x4的张量
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # 转置维度，变成4x2x3
    y = tl.trans(x, [2, 0, 1])

    return y

## 调用示例
result = trans_example()
print(result.shape)  # 输出: (4, 2, 3)
```
