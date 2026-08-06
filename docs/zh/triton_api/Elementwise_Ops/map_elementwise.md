# triton.language.map_elementwise

## 1. OP 概述

简介：`triton.language.map_elementwise` 将一个使用 `@triton.jit` 标记的标量函数映射到输入张量的每个元素上执行。与 `tl.where` 仅支持二元选择不同，`map_elementwise` 支持在标量函数内编写 `if/elif/else` 多分支控制流以及 `for` 循环，提供了更灵活的元素级计算表达能力。

```python
triton.language.map_elementwise(scalar_fn, *args, pack=1, _semantic=None, _generator=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `scalar_fn` | `Callable` | 必须使用 `@triton.jit` 标记的标量函数，接收标量参数，返回一个或多个标量结果。当 `pack > 1` 时，每个参数接收 `pack` 个连续元素（交织排列）。 |
| `*args` | `tensor` | 输入张量，会被隐式广播到相同形状 |
| `pack` | `int` | 每次函数调用处理的元素组大小，必须是 2 的幂，默认为 1。在 NPU 后端上，最终实现会转为向量化运算，因此 `pack` 参数在 NPU 上不产生实际语义差异。 |
| `_semantic` | - | 框架内部参数，不支持外部传参 |
| `_generator` | - | 框架内部参数，不支持外部传参 |

**返回值：**

- 若 `scalar_fn` 返回单个值，则 `map_elementwise` 返回一个 `tensor`
- 若 `scalar_fn` 返回多个值（元组），则 `map_elementwise` 返回一个 `tuple of tensor`
- 输出形状为所有输入张量广播后的公共形状

### 2.2 Shape 支持

`map_elementwise` 要求输入为 RankedTensorType（秩 ≥ 1），所有输入张量会被隐式广播为相同形状。在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

`while` 循环不支持：在 Ascend 平台上，标量函数内部不能使用 `while` 循环。

### 2.4 使用方法

以下示例将三路比较标量函数映射到两个张量的每个元素上：`x < y` 时返回 `-1`、`x == y` 时返回 `0`、`x > y` 时返回 `1`：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _compare(x, y):
    if x < y:
        return -1
    elif x == y:
        return 0
    else:
        return 1


@triton.jit
def kernel(X, Y, Z, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.load(Y + tl.arange(0, BLOCK))
    z = tl.map_elementwise(_compare, x, y)
    tl.store(Z + tl.arange(0, BLOCK), z)


shape = (128, )
x = torch.randint(-100, 100, shape, dtype=torch.int32, device='npu')
y = torch.randint(-100, 100, shape, dtype=torch.int32, device='npu')
z = torch.zeros(shape, dtype=torch.int32, device='npu')
kernel[(1, )](x, y, z, BLOCK=shape[0])
```
