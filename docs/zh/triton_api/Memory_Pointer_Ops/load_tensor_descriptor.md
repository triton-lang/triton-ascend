# triton.language.load_tensor_descriptor

## 1. OP 概述

简介：该函数用于从张量描述符加载数据块

```python
triton.language.load_tensor_descriptor(
 desc: tensor_descriptor_base,
 offsets: Sequence[constexpr | tensor],
 _semantic=None
) → tensor
```

## 2. OP 规格

### 2.1 参数说明

| 参数名         | 类型                              | 说明                                                         |
| ----------- | ------------------------------- | ---------------------------------------------------------- |
| `desc`      | `tensor_descriptor_base`        | 张量描述符对象，由 `make_tensor_descriptor` 创建，定义了内存布局（形状、步长、块大小等）。 |
| `offsets`   | `Sequence[constexpr \| tensor]` | 数据加载的起始偏移量序列，用于指定当前线程块要加载的数据位置                             |
| `_semantic` | -                               | 保留参数，暂不支持外部调用                                            |

返回值：`tensor` - 根据张量描述符内存布局信息，从指定偏移量处加载的数据块

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
| Ascend | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

结论：Ascend 对比 GPU 缺失uint16、uint32、uint64的支持能力（硬件限制）。

| 差异点            | 描述                                                         | 解决途径                                                |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| 绑定使用限制      | `make/load/store_tensor_descriptor` 需配套使用，不能与 `tl.load()` / `tl.store()` 混用。 | 升级至 Triton 3.4.0 版本同步上游函数（如 `cast`）可解决 |
| Triton 版本兼容性 | Triton 3.2.0 存在部分函数（如 `cast`）的兼容性问题。建议升级 Triton版本至 3.4.0 来修复绑定限制。 | 升级至 Triton 3.4.0                                     |

### 2.4 使用方法

`load_tensor_descriptor` 提供两种调用形式：

* 面向对象方法调用（推荐）

```python
value = desc.load(offsets)
```

* 函数式接口调用

```python
value = triton.language.load_tensor_descriptor(desc, offsets)
```

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
