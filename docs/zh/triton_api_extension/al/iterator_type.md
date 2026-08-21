# al.IteratorType

## 1. 概述

`al.IteratorType` 用于描述定制算子每个逻辑循环维度的迭代语义。它是编译期元数据，
不是 Python 迭代器、NPU 指令或运行时参数。

`iterator_types` 与 `indexing_map` 的分工如下：

- `iterator_types` 描述每个逻辑循环的性质；
- `indexing_map` 描述输入和输出如何使用这些逻辑循环；
- `symbol` 或 `bitcode` 提供真实计算实现。

## 2. 枚举值

| 枚举值 | 底层编码 | 含义 |
| --- | ---: | --- |
| `Parallel` | 0 | 不同迭代之间可独立执行 |
| `Broadcast` | 1 | 广播维度 |
| `Transpose` | 2 | 转置或维度置换 |
| `Reduction` | 3 | 归约维度 |
| `Interleave` | 4 | 交织维度 |
| `Deinterleave` | 5 | 解交织维度 |
| `Inverse` | 6 | 反向或逆序维度 |
| `Pad` | 7 | 填充维度 |
| `Concat` | 8 | 拼接维度 |
| `Gather` | 9 | Gather 索引维度 |
| `Cumulative` | 10 | 累积或扫描维度 |
| `Opaque` | 99 | 不按上述通用模式解释的特殊维度 |

`Opaque` 的真实底层编码是 99。

## 3. 逻辑循环示例

以下示意只描述循环语义，不会自动生成对应计算：

```text
# 二维逐元素计算
iterator_types = [Parallel, Parallel]

# 对每一行沿 j 维归约
iterator_types = [Parallel, Reduction]

# C[M, N] = A[M, K] @ B[K, N]
iterator_types = [Parallel, Parallel, Reduction]
```

矩阵乘的输出是二维，但逻辑循环是 `M`、`N`、`K` 三维，因此 `iterator_types` 的长度不一定等于
输出张量的 rank。

## 4. 定制算子中的用法

下面的配置将一维定制算子声明为逐元素并行计算：

```python
@al.register_custom_op
class elementwise_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    symbol = "elementwise_custom_impl"
    bitcode = "/path/to/elementwise_custom_impl.bc"
    iterator_types = [al.IteratorType.Parallel]

    def __init__(self, x, out=None):
        identity = al.affine_map.get_identity(1)
        self.indexing_map = [identity, identity]
```

对应的关键 IR 属性为：

```mlir
iterator_types = [#hivm.iterator_type<parallel>]
```

`iterator_types` 是可选属性。省略时不会生成该属性，编译器也不会自动补成 `Parallel`。

## 5. 约束说明

- 列表顺序必须与定制算子的逻辑循环顺序一致。
- 声明必须与 `indexing_map`、输入输出形状以及设备侧实现的真实语义一致。
- 当前前端不能完整证明这些元数据与 bitcode 一致；错误声明可能导致后续分析或变换产生错误结果。
- 枚举值存在不表示任意组合都被所有后端 pass 和目标芯片支持。

## 6. 验证范围

仓库的 `test_custom.py` 覆盖了 12 个枚举值到 HIVM IR 的序列化，但使用的是占位 bitcode，
只验证编译期属性生成，不验证这些迭代语义对应的真实 NPU 计算。
