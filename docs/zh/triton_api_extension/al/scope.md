# al.scope 接口文档

## 1. 硬件背景

昇腾处理器包含多种类型的计算核心（例如，用于矩阵运算的 **Cube Unit** 和用于向量/标量运算的 **Vector Unit**）。`al.scope` 允许内核开发者显式告知 Triton 编译器：某一代码块应当运行在哪种硬件核心上，从而实现更精细的性能调优与资源调度。

在 Triton IR 中，`al.scope` 会被 lowering 为 `scope.scope` 操作，并通过属性 `hivm.tcore_type` 标记目标核心（CUBE 或 VECTOR）。编译器在后续的 pass 中据此生成对应核心的指令序列，并在需要时在 CUBE 与 VECTOR 之间自动插入同步操作。

## 2. 接口说明

`al.scope` 是一个 Python **上下文管理器（Context Manager）**，使用 `with` 语句进入作用域：

```python
with al.scope(core_mode: str, *, disable_auto_sync: bool = False,
              noinline: bool = True, vec_mode: str | None = None):
    # 此代码块内的 Triton 语句（tl.load / tl.store / tl.dot / 算术运算等）
    # 将按指定的 core_mode 编译到对应硬件核心上执行。
    ...
```

### 参数

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `core_mode` | `str` | 是 | — | 目标核心类型，取值为 `"vector"` 或 `"cube"`（其他值会抛出 `ValueError`） |
| `disable_auto_sync` | `bool` | 否 | `False` | 若为 `True`，编译器不在此 scope 边界自动插入 CUBE↔VECTOR 同步指令，由开发者自行保证数据依赖 |
| `noinline` | `bool` | 否 | `True` | 是否给 `scope.scope` 操作附加 `noinline` 标记，提示编译器不要把该 scope 内联掉 |
| `vec_mode` | `str \| None` | 否 | `None` | 向量核心执行模式字符串（具体取值由后端决定），透传为 IR 属性 `vec_mode` |
| `**kwargs` | — | 否 | — | 其余关键字参数会被转换为 MLIR 属性附加到 `scope.scope` 操作上（高级用法，供内部实验性参数使用） |
| `_builder` / `_semantic` | — | 内部 | — | 编译器自动注入，用户不要手动传 |

### core_mode 取值说明

| 值 | 目标核心 | 典型用途 |
|----|----------|----------|
| `"vector"` | Vector Unit（向量核心） | 元素级操作（Element-wise）：加/乘等算术运算、激活函数（ReLU / Sigmoid 等）、`tl.load` / `tl.store` 等内存访问 |
| `"cube"` | Cube Unit（矩阵核心） | 矩阵计算密集型操作，尤其是矩阵乘法（GEMM / `tl.dot`）、卷积等张量收缩操作 |

> **注意**：`core_mode` 仅接受 `"vector"` 和 `"cube"` 两个字符串。`"SIMT"` / `"SIMD"` 等并非该接口的合法取值。

## 3. 约束说明

- **仅可在 `@triton.jit` 内核内部使用**：在 Triton kernel 外部调用 `al.scope` 会抛出 `RuntimeError: scope can only be used inside a Triton kernel`。
- **核心匹配**：`core_mode="cube"` 的 scope 内建议放置矩阵乘类操作（`tl.dot` 等）；将普通逐元素运算放入 cube scope 可能导致编译器生成效率低下的代码或直接编译失败。
- **作用域嵌套**：`al.scope` 支持嵌套，但内层 core_mode 必须与目标硬件的调度语义一致；过深或不必要的嵌套会导致 IR 中出现多层 `scope.scope`，影响优化。
- **跨 scope 数据依赖与同步**：
  - 默认情况下（`disable_auto_sync=False`），编译器会在 scope 的入口/出口自动插入必要的 CUBE↔VECTOR 同步，以保证前一 scope 写入的数据对下一 scope 可见。
  - 设置 `disable_auto_sync=True` 后，自动同步被关闭；开发者需要通过 Ascend 提供的同步原语（如 `al.fixpipe` 等）显式保证数据一致性，否则可能读到过期数据。
- **SSA 变量传出**：scope 内定义/修改的 Triton 张量可以在 scope 外部继续使用；编译器通过 `scope.return` 机制把这些值透传到外层，无需手动"搬出"。

## 4. 用例示例

### 4.1 基础用法：分别指定 Vector / Cube 核心

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < n

    # 整个基本块都在 Vector 核心上执行
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=mask)
        y = tl.load(y_ptr + i, mask=mask)
        result = x + y
        tl.store(out_ptr + i, result, mask=mask)


@triton.jit
def cube_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # ... 构造 tile 指针 rm / rn / rk ...

    # 矩阵乘法部分放在 Cube 核心上
    with al.scope(core_mode="cube"):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + rm[:, None] * K + (rk + k)[None, :])
            b = tl.load(b_ptr + (rk + k)[:, None] * N + rn[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)
```

### 4.2 Scope Escape：从作用域中"带出"值

在 scope 内计算得到的张量可以在 scope 外继续使用，编译器会自动插入 `scope.return`：

```python
@triton.jit
def scope_escape_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < n

    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=mask)

    # x 从 vector scope 中"逃逸"出来，可在外部继续参与运算
    a = x + 1.0
    tl.store(out_ptr + i, a, mask=mask)
```

### 4.3 关闭自动同步（高级用法）

当开发者明确知道 scope 之间不存在数据依赖，或已经手动插入了同步操作时，可以关闭自动同步以减少同步开销：

```python
@triton.jit
def scope_no_auto_sync_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < n

    with al.scope(core_mode="vector", disable_auto_sync=True):
        x = tl.load(x_ptr + i, mask=mask)
        y = tl.load(y_ptr + i, mask=mask)
        result = x + y
        tl.store(out_ptr + i, result, mask=mask)
```

## 5. 编译输出结果

### 5.1 Vector Scope

```mlir
// Vector core scope
%5:3 = scope.scope : () -> (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) {
  ^entry:
    // ... tl.load / arith.addf / tl.store 等指令在此 region 内 ...
    scope.return %10, %15, %16 : tensor<256xf32>, tensor<256xf32>, tensor<256xf32>
} {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
```

### 5.2 Cube Scope

```mlir
// Cube core scope
%5:3 = scope.scope : () -> (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) {
  ^entry:
    // ... tl.dot / tl.load / tl.store 等指令在此 region 内 ...
    scope.return %10, %15, %16 : tensor<256xf32>, tensor<256xf32>, tensor<256xf32>
} {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
```

### 5.3 关闭自动同步

```mlir
// disable_auto_sync=True 时，属性中出现 hivm.disable_auto_sync = true
%5:3 = scope.scope : () -> (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) {
  ^entry:
    // ...
    scope.return %10, %15, %16 : tensor<256xf32>, tensor<256xf32>, tensor<256xf32>
} {hivm.disable_auto_sync = true, hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
```

### 输出要点说明

- `al.scope` 在 IR 中对应 `scope.scope` 操作，带一个 region（`{ ... }` 内为该 scope 内的指令列表）。
- 目标核心类型通过属性 `hivm.tcore_type` 传递：`#hivm.tcore_type<VECTOR>` 对应 Vector 核心，`#hivm.tcore_type<CUBE>` 对应 Cube 核心。
- 默认会附加 `noinline` 属性（除非显式传 `noinline=False`），防止编译器把 scope 边界优化掉。
- 当 `disable_auto_sync=True` 时，属性中会出现 `hivm.disable_auto_sync = true`，告诉后端跳过该 scope 边界上的自动同步指令。
- scope 内被修改的 SSA 值通过 `scope.return` 作为操作结果返回，scope 外继续使用这些值时，引用的就是 `scope.scope` 的结果 SSA（如 `%5#0`、`%5#1`）。

## 6. 相关接口

- [`al.ascend_address_space`](ascend_address_space.md)：昇腾片上地址空间枚举（UB / L1 / L0A / L0B / L0C），常与 buffer 分配配合使用
- [`al.fixpipe`](fixpipe.md)：L0C → UB 的专用数据搬运通路，常在 Cube scope 产出结果、Vector scope 需要消费结果时用于显式搬运与同步
- [`bl.alloc`](../bl/alloc.md)：在片上地址空间分配 buffer，常作为 scope 间数据传递的中转
