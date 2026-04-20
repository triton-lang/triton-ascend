# triton.language.make_tensor_descriptor

## 1. OP 概述

简介：创建张量描述符对象
原型（Triton3.4.0版本）：

```python
triton.language.make_tensor_descriptor(
 base: tensor,
 shape: List[tensor],
 strides: List[tensor],
 block_shape: List[constexpr],
 _semantic=None
) → tensor_descriptor
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `base`        | `tensor`          | 张量的基指针                                                      |
| `shape`       | `List[tensor]`    | 张量的形状                                                        |
| `strides`     | `List[tensor]`    | 张量各维度的步长列表，约束如下：- 前面的维度必须是 16 字节的整数倍- 最后一维必须是连续存储的 |
| `block_shape` | `List[constexpr]` | 从全局内存加载 / 存储的块的形状                                              |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用                                                |

返回值：
`tensor_descriptor`：张量描述符对象（不可直接进行算术运算，需配合 `load` / `store` 使用）

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × |
|Ascend A2/A3| √ | √ | x | √ | × | √ | × | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend A2/A3 | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

结论：Ascend 对比 GPU 缺失uint16、uint32、uint64的支持能力（硬件限制）。

| 差异点                     | 描述                                                         | 解决途径                                               |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| 绑定使用限制               | `make/load/store_tensor_descriptor` 需配套使用，不能与 `tl.load()` / `tl.store()` 混用。 | 升级至 Triton 3.4.0 版本同步上游函数(如 `cast` )可解决 |
| 不支持`padding_option`入参 | 当前社区主线分支新增`padding_option`入参，用于越界元素填充策略。 | 可软件开发支持                                         |
| Triton 版本兼容性          | Triton 3.2.0 存在部分函数（如 `cast` ）的兼容性问题。建议升级 Triton版本至 3.4.0 来修复绑定限制。 | 升级至 Triton 3.4.0                                    |

### 2.4 使用方法

以下示例实现了对输入张量 `x` 做就地绝对值计算：

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    # 创建张量描述符
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
 # 计算当前线程对应的偏移量
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
 # 加载数据，计算绝对值，存储结果
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))
## 初始化张量
M, N = 256, 256
x = torch.randn(M, N, device="npu")
## 配置块大小和网络
M_BLOCK, N_BLOCK = 32, 32
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```
