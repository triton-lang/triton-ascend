# al.IteratorType 接口文档

## 1. 背景

`CustomOp` 和 `CustomMacro` 是通过 Ascend 扩展接口接入 Triton 的设备侧操作。
一个操作可以包含多个逻辑循环维度，例如二维逐元素计算包含行、列两个维度。

`al.IteratorType` 是编译期枚举，用于说明每个逻辑循环维度的作用，例如并行或归约。

## 2. 接口说明

### 2.1 枚举定义

```python
class IteratorType(enum.Enum):
    Parallel = ascend_ir.IteratorType.Parallel
    Broadcast = ascend_ir.IteratorType.Broadcast
    Transpose = ascend_ir.IteratorType.Transpose
    Reduction = ascend_ir.IteratorType.Reduction
    Interleave = ascend_ir.IteratorType.Interleave
    Deinterleave = ascend_ir.IteratorType.Deinterleave
    Inverse = ascend_ir.IteratorType.Inverse
    Pad = ascend_ir.IteratorType.Pad
    Concat = ascend_ir.IteratorType.Concat
    Gather = ascend_ir.IteratorType.Gather
    Cumulative = ascend_ir.IteratorType.Cumulative
    Opaque = ascend_ir.IteratorType.Opaque
```

这些枚举值通过 `al.IteratorType` 访问，例如 `al.IteratorType.Parallel`。

### 2.2 枚举值说明

| 枚举值 | 底层编码 | 含义 |
| --- | ---: | --- |
| `Parallel` | 0 | 不同迭代之间可以独立执行 |
| `Broadcast` | 1 | 广播维度 |
| `Transpose` | 2 | 转置维度 |
| `Reduction` | 3 | 归约维度 |
| `Interleave` | 4 | 交织维度 |
| `Deinterleave` | 5 | 解交织维度 |
| `Inverse` | 6 | 逆序维度 |
| `Pad` | 7 | 填充维度 |
| `Concat` | 8 | 拼接维度 |
| `Gather` | 9 | Gather 索引维度 |
| `Cumulative` | 10 | 累积维度 |
| `Opaque` | 99 | 不按上述通用类型解释的维度 |

### 2.3 列表含义

在 `CustomOp` 或 `CustomMacro` 的注册类中，`iterator_types` 按逻辑循环顺序保存
`al.IteratorType` 枚举值。以下代码只用于说明列表含义，不是完整用例：

```python
# 二维逐元素计算：行、列两个维度都可以并行
iterator_types = [
    al.IteratorType.Parallel,
    al.IteratorType.Parallel,
]

# 按行归约：行维度并行，列维度归约
iterator_types = [
    al.IteratorType.Parallel,
    al.IteratorType.Reduction,
]

# 矩阵乘的逻辑循环顺序为 M、N、K
iterator_types = [
    al.IteratorType.Parallel,
    al.IteratorType.Parallel,
    al.IteratorType.Reduction,
]
```

矩阵乘的输出是二维，但逻辑循环包含 `M`、`N`、`K` 三个维度。因此，
`iterator_types` 的长度不一定等于输出张量的维数。

## 3. 约束说明

- `iterator_types` 中的每个元素都应是 `al.IteratorType` 枚举值。
- 列表顺序必须与操作的逻辑循环顺序一致。
- 声明的循环语义必须与设备侧实现的实际计算保持一致。
- `iterator_types` 是可选配置。省略时，当前前端不会生成该属性，也不会自动补成
  `al.IteratorType.Parallel`。

## 4. 用例示例

以下代码是 `@al.register_custom_op` 注册类中的配置片段。为突出 `iterator_types`，省略了
其他必需的注册字段和设备侧实现，因此不是完整可运行用例：

```python
@al.register_custom_op
class row_sum_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    # 逻辑循环顺序为行、列
    iterator_types = [
        al.IteratorType.Parallel,
        al.IteratorType.Reduction,
    ]
```

这里的 `Parallel` 表示不同的行可以独立处理，`Reduction` 表示每一行需要沿列方向归约。
设备侧实现仍负责完成实际的求和计算。

## 5. 编译输出结果

上述配置生成的关键 IR 属性为：

```mlir
iterator_types = [
  #hivm.iterator_type<parallel>,
  #hivm.iterator_type<reduction>
]
```

该属性记录两个逻辑循环维度的语义，本身不包含求和计算。
