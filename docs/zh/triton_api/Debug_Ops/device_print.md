# triton.language.device_print

## 1. 函数概述

`device_print` 用于在NPU运行时从设备端打印信息，与`static_print`不同，这是在内核执行时实时输出信息。 **使用`device_print`需要将环境变量`TRITON_DEVICE_PRINT`的值设置为`True`。**

```python
triton.language.device_print(prefix, *args, hex=False, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `prefix` | `str` | 必需 | 打印值之前的前缀字符串 |
| `args` | `tensor`/`scalar` | 必需 | 要打印的值，可以是任意张量或标量 |
| `hex` | `bool` | `False` | 是否以十六进制格式打印所有值 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

### 2.2.1 Data Type支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend | 仅支持 1~5维 tensor |

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

**DevicePrint 功能限制**

**现象描述：**
device_print 只能打印参与运算的结果值，无法打印纯粹用于访存的 offset 变量。

**根本原因：**
在访存分析优化阶段，编译器会将仅用于地址计算的 offset 优化掉，这些中间变量不会保留到最终的执行代码中。

**示例场景：**

```python
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               of_ptr,
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    tl.device_print("offsets:", offsets)// ❌ 无法打印，已被优化
```

另外在特定情况下，`device_print`会展开一些辅助dma代码，导致底层报错，相关功能还在优化中

### 2.4 使用方法

**注意**：`prefix`字符串前缀在使用`device_print`时是必需加上的，否则会导致编译错误。

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr):
    idx = tl.arange(0,3)
    idy = tl.arange(0,4)
    offset = idx[:,None] * 4 + idy[None,:]
    val = tl.load(x_ptr + offset)
    # 打印二维张量val的值
    tl.device_print("val:",val)
```
