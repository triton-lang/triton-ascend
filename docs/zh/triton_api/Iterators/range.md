# triton.language.range
## 1. 函数概述

`range` 是一个向上计数的迭代器，类似于 Python 的 `range()` 函数，但是允许传入更多的参数。

```python
triton.language.range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `arg1` | `int` /`constexpr`| 必需 | 起始值（单参数时作为结束值，从0开始） |
| `arg2` | `int`/`constexpr` | - | 结束值（不包含在范围内） |
| `step` | `int` /`constexpr`| `1` | 一个整数，每次迭代的步长增量|
| `num_stages` | `int` | - | 流水线阶段数（同时执行的迭代数量） |
| `loop_unroll_factor` | `int` | - | 循环展开因子（<2表示不展开） |
| `disallow_acc_multi_buffer` | `bool` | `False` | 禁止dot操作累加器的多缓冲优化 |
| `flatten` | `bool` | `False` | 自动展平嵌套循环为单层循环 |
| `warp_specialize` | `bool` | `False` | 启用warp专业化（仅Blackwell GPU） |
| `disable_licm` | `bool` | `False` | 禁用循环不变代码外提优化 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |


### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend A2/A3 | ✓ | ✓ | ✓ | ×|×| × | × | ✓ | × | × | × | × | × |



### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。
disallow_acc_multi_buffer, flatten, warp_specialize, disable_licm 相关功能还不全。

### 2.4 使用方法

```python
import triton.language as tl

@triton.jit
def basic_examples():
    # 单参数：0到9
    for i in tl.range(10):
        # i = 0, 1, 2, ..., 9
        pass

    # 双参数：2到9
    for i in tl.range(2, 10):
        # i = 2, 3, ..., 9
        pass

    # 三参数：0到10，步长为2
    for i in tl.range(0, 10, 2):
        # i = 0, 2, 4, 6, 8
        pass
```

```python
@triton.jit
def advanced_examples():
    # 使用循环优化参数
    for i in tl.range(0, 100, num_stages=3, loop_unroll_factor=4):
        # 流水线阶段数为3，循环展开因子为4
        pass

    # 嵌套循环展平
    for i in tl.range(0, 10, flatten=True):
        for j in tl.range(0, 20, flatten=True):
            # 两个循环会被自动展平为单层循环
            pass
```
