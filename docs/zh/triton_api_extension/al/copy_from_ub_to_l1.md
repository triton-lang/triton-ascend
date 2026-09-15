# al.copy_from_ub_to_l1 接口文档

> **已弃用**：`al.copy_from_ub_to_l1` 自 triton-ascend 引入通用 `al.copy` 接口后被标记为 deprecated，调用时会触发 Python `DeprecationWarning`。新代码请使用 [`al.copy`](al.copy.md)，它同时支持 UB→UB 和 UB→L1 两种目标。本文仅为兼容既有代码保留说明。

## 1. 硬件背景

昇腾硬件 A5 及后续架构支持**直接从 UB（Unified Buffer）搬运数据到 L1（Cube 输入缓存）**，无需经过 GM（Global Memory）中转。传统路径需要"UB→GM→L1"两次 DMA 搬运，直接通路只需要一次 DMA（MTE3 + MTE2 流水协同），能显著降低数据搬运延迟、节省 GM 带宽、减少 L1 填充等待时间。

`al.copy_from_ub_to_l1` 用于显式触发这一 UB→L1 直接通路，常见场景是 Cube 核做 GEMM/Conv 之前，由 Vector 核把预处理好的权重/数据直接从 UB 推到 L1，避免 Cube 等待 GM→L1 搬运。

## 2. 接口说明

```python
def copy_from_ub_to_l1(
    src: Union[tl.tensor, bl.buffer],
    dst: Union[tl.tensor, bl.buffer],
    _semantic=None,
) -> None:
```

### 2.1 参数

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `src` | `tl.tensor` \| `bl.buffer` | 是 | 源数据，必须位于 UB 地址空间；既可以是 tensor（前端会自动 to_buffer 到 UB），也可以是通过 `bl.alloc(..., al.ascend_address_space.UB)` 分配的 buffer |
| `dst` | `tl.tensor` \| `bl.buffer` | 是 | 目标缓冲区，必须位于 L1（cbuf）地址空间；通常是通过 `bl.alloc(..., al.ascend_address_space.L1)` 分配的 buffer |
| `_semantic` | - | 内部 | JIT 编译器自动传参，用户不要手动传 |

### 2.2 返回值

无返回值。数据通过 DMA 异步写入 `dst` 所指 L1 buffer（实际完成时刻由硬件流水保证，必要时需用 `al.sync_block_set`/`al.sync_block_wait` 做同步）。

## 3. 约束说明

- `src` 和 `dst` **必须同时是 tensor 或同时是 buffer**，不允许 tensor 与 buffer 混用。
- `src` 的地址空间必须是 **UB**（可通过 `bl.to_buffer(tensor, al.ascend_address_space.UB)` 显式指定）。
- `dst` 的地址空间必须是 **L1**（cbuf）（通过 `bl.alloc(..., al.ascend_address_space.L1)` 分配）。
- `src` 和 `dst` 的 **element type 与 shape 必须完全一致**（同 bitwidth、同维度、同大小）。
- L1 buffer 通常作为 Cube 核的输入，复制完成后应通过 `al.sync_block_set/wait` 通知 Cube 核可安全读取；否则 Cube 可能读到未更新的旧数据。
- 本接口已弃用，新代码请使用 `al.copy(src, dst)`（当 `dst` 是 L1 buffer 时，`al.copy` 会自动选择 UB→L1 的直接通路）。

## 4. 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al


@triton.jit
def preload_to_l1_kernel(A_ptr, A1_ptr, M: tl.constexpr, N: tl.constexpr):
    """把 GM 上的两个矩阵加载到 UB、相加后直接搬到 L1 供 Cube 使用。"""
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    offs = offs_m * N + offs_n

    # 1) 从 GM 读两块数据到 tensor
    a_val  = tl.load(A_ptr + offs)
    a1_val = tl.load(A1_ptr + offs)

    # 2) Vector 上做预处理（这里简单相加），结果放到 UB
    add = a_val + a1_val
    add_ub = bl.to_buffer(add, al.ascend_address_space.UB)

    # 3) 分配 L1 buffer 并把 UB 数据直接搬到 L1（推荐用新接口 al.copy）
    A_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)
    al.copy_from_ub_to_l1(add_ub, A_l1)   # deprecated：请改用 al.copy(add_ub, A_l1)

    # 4) 可选：在 UB 上分配一份副本，供 Vector 后续使用
    A_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.copy(add_ub, A_ub)
```

## 5. 编译输出结果

以 `M=N=16, dtype=fp32` 为例，核心 IR 片段：

```mlir
// ... 前置 tl.load / arith.addf 等指令省略 ...

// 把 add 结果放到 UB：tensor → memref<..., ub>
%14 = arith.addf %10, %13 : tensor<16x16xf32>
%15 = bufferization.to_memref %14 : memref<16x16xf32>
%memspacecast = memref.memory_space_cast %15
    : memref<16x16xf32> to memref<16x16xf32, #hivm.address_space<ub>>

// 在 L1（cbuf）分配目标 buffer
%alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
annotation.mark %alloc {effects = ["write", "read"]}
    : memref<16x16xf32, #hivm.address_space<cbuf>>

// UB → L1 直接搬运：对应 hivm.hir.copy
hivm.hir.copy
    ins(%memspacecast : memref<16x16xf32, #hivm.address_space<ub>>)
    outs(%alloc       : memref<16x16xf32, #hivm.address_space<cbuf>>)

// 对照：UB→UB 同地址空间内拷贝
%alloc_1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
annotation.mark %alloc_1 {effects = ["write", "read"]}
    : memref<16x16xf32, #hivm.address_space<ub>>
hivm.hir.copy
    ins(%memspacecast : memref<16x16xf32, #hivm.address_space<ub>>)
    outs(%alloc_1     : memref<16x16xf32, #hivm.address_space<ub>>)
```

### 输出要点说明

- `al.copy_from_ub_to_l1(src, dst)` 底层统一降低为 `hivm.hir.copy ins(...) outs(...)` 指令，搬运方向由 `ins`（源 memref 的地址空间）与 `outs`（目标 memref 的地址空间）决定。
- 如果 `src` 是 tensor，前端会先通过 `bufferization.to_memref` + `memref.memory_space_cast` 把 tensor 显式落到 UB memref（即等价于隐式 `bl.to_buffer(src, al.ascend_address_space.UB)`）。
- L1 buffer 必须通过 `memref.alloc() : memref<..., #hivm.address_space<cbuf>>` 预先分配，且带有 `annotation.mark {effects = ["write", "read"]}` 标记。
- UB→L1 搬运本身在硬件上是异步的（MTE 流水），如果后续 Cube/Vector 需要依赖 L1 中的新数据，必须通过 `al.sync_block_set`/`al.sync_block_wait` 保证数据已到达。
- 对比 UB→UB 的 `hivm.hir.copy`，IR 结构完全一致，唯一区别是 outs 端 memref 的地址空间属性（`<cbuf>` vs `<ub>`）。这也是为什么新接口 `al.copy` 可以统一两种路径。

## 6. 相关接口

- [`al.copy`](al.copy.md)：**推荐替代**，统一支持 UB→UB 与 UB→L1 两种搬运
- [`al.ascend_address_space`](ascend_address_space.md)：昇腾地址空间枚举（UB / L1 / L0A / L0B / L0C）
- [`bl.alloc`](../bl/alloc.md)：分配 L1 / UB 目标 buffer
- [`bl.to_buffer`](../bl/to_buffer.md)：将 tensor 搬运/绑定到指定地址空间 buffer
- [`al.sync_block_set`](sync_block_set.md) / [`al.sync_block_wait`](sync_block_wait.md)：搬运完成后通知 Cube 核可读取 L1
