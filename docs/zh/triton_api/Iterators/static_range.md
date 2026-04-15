# triton.language.static_range
## 1. 函数概述

`static_range` 是一个静态范围的迭代器，与 `range` 类似但会在编译时进行积极的循环展开优化。

```python
triton.language.static_range(arg1, arg2=None, step=None, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `arg1` | `constexpr` | 必需 | 起始值（单参数时作为结束值，从0开始） |
| `arg2` | `constexpr` | - | 结束值（不包含在范围内） |
| `step` | `constexpr` | `1` | 每次迭代的步长增量 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend 910系列 | ✓ | ✓ | ✓ | ×|×| × | × | ✓ | × | × | × | × | × |



### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

### 2.4 使用方法

```python
@triton.jit
def optimized_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # 使用static_range进行小规模循环展开，消除循环开销
    for i in tl.static_range(BLOCK_SIZE):
        # 当BLOCK_SIZE是编译时常量时，整个循环会被展开
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)

    # 对比：使用range会有循环控制开销
    for i in tl.range(BLOCK_SIZE):
        # 这个循环在运行时会有循环控制逻辑
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)
```

`static_range` 通过牺牲代码大小来换取运行时性能，适用于已知且较小的循环次数场景。
