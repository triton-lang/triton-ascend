# al.ascend_address_space 接口文档

## 1. 背景

为了支持 Ascend 级编程的需要，TritonAscend 扩展暴露了 `al.ascend_address_space` 枚举对象，用于在 buffer 分配（`bl.alloc` / `bl.to_buffer` 等）时手动指定内存所在的硬件地址空间。该枚举底层对接 C++ 侧 `hivm::AddressSpace` 枚举。

## 2. 接口说明

`al.ascend_address_space` 不是一个函数，而是一个预定义的**枚举常量组**，可直接作为常量使用：

```python
al.ascend_address_space.UB
al.ascend_address_space.L1
al.ascend_address_space.L0A
al.ascend_address_space.L0B
al.ascend_address_space.L0C
```

每个值都是一个 `bl.address_space` 子类实例，可以直接传给接受 `space` / `_address_space` 参数的 buffer 接口。

### 2.1 地址空间一览

| 枚举值 | IR 助记符 | 对应硬件区域 | 典型用途 |
|--------|-----------|--------------|----------|
| `al.ascend_address_space.UB` | `#hivm.address_space<ub>` | Unified Buffer（片上共享内存，Vector 单元输入/输出） | 通用临时 buffer、vector 计算中间结果、`bl.to_buffer` 默认目标 |
| `al.ascend_address_space.L1` | `#hivm.address_space<cbuf>` | L1 Cache Buffer（Cube 单元输入缓存） | 矩阵乘前的权重/数据预加载 |
| `al.ascend_address_space.L0A` | `#hivm.address_space<ca>` | L0 Buffer A（Cube 单元左矩阵输入） | Cube 矩阵乘 A 矩阵缓冲 |
| `al.ascend_address_space.L0B` | `#hivm.address_space<cb>` | L0 Buffer B（Cube 单元右矩阵输入） | Cube 矩阵乘 B 矩阵缓冲 |
| `al.ascend_address_space.L0C` | `#hivm.address_space<cc>` | L0 Buffer C（Cube 单元输出） | Cube 矩阵乘结果缓冲（通常由 `al.fixpipe` 搬到 UB） |

> **注意**：GM（Global Memory，即 HBM 显存）对应枚举值 `GM`（IR 助记符 `gm`），不在 Python 侧暴露为常量，因为 GM 内存由 Triton 张量本身（`tl.tensor` / `!tt.ptr`）隐式表示，不需要显式通过 `bl.alloc` 分配。

### 2.2 返回值

不涉及（枚举常量组，无函数返回值）。

### 2.3 入参

不涉及（无函数调用）。

## 3. 约束说明

- 需要配合 `bl.alloc` / `bl.to_buffer` / `bl.subview` 等接受 `bl.address_space` 参数的接口使用
- 指定地址空间后，分配出的 buffer 大小必须**不超过**对应硬件区域的容量（UB / L1 / L0A / L0B / L0C 各有固定大小，具体容量依赖具体芯片型号）
- 同一地址空间内的 buffer 生命周期由 PlanMemory pass 自动进行内存复用；若不希望被复用，请在 `bl.alloc` 时使用 `is_mem_unique=True`
- L0A / L0B / L0C 是 Cube 专用缓冲，只能通过专用通路（如 fixpipe / matrix multiply pipeline）访问，不能直接用 `tl.load`/`tl.store` 读写

## 4. 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al


@triton.jit
def allocate_buffers_demo(XBLOCK: tl.constexpr):
    # 默认地址空间（不指定）：由编译器自动决定
    bl.alloc(tl.float32, [XBLOCK])

    # 显式指定各 Ascend 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)

    # 指定 UB 且标记为独占内存（不参与 PlanMemory 复用）
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB,
             is_mem_unique=True)
```

`bl.to_buffer` 也接受相同的地址空间参数：

```python
@triton.jit
def to_buffer_demo():
    a = tl.full((32, 32), 0, dtype=tl.float32)
    a_ub = bl.to_buffer(a, al.ascend_address_space.UB)
    b = tl.full((32, 32), 0, dtype=tl.float32)
    b_l1 = bl.to_buffer(b, al.ascend_address_space.L1)
```

## 5. 编译输出结果

以 `XBLOCK=256` 为例，不同地址空间在 IR 中体现为 memref 类型上的 `#hivm.address_space<...>` 属性：

```mlir
// 默认地址空间：不携带 address_space 属性
%alloc = memref.alloc() : memref<256xf32>
annotation.mark %alloc {effects = ["write", "read"]} : memref<256xf32>

// UB → <ub>
%alloc_0 = memref.alloc() : memref<256x256xf32, #hivm.address_space<ub>>
annotation.mark %alloc_0 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<ub>>

// L1 → <cbuf>
%alloc_1 = memref.alloc() : memref<256x256xf32, #hivm.address_space<cbuf>>
annotation.mark %alloc_1 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<cbuf>>

// L0A → <ca>
%alloc_2 = memref.alloc() : memref<256x256xf32, #hivm.address_space<ca>>
annotation.mark %alloc_2 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<ca>>

// L0B → <cb>
%alloc_3 = memref.alloc() : memref<256x256xf32, #hivm.address_space<cb>>
annotation.mark %alloc_3 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<cb>>

// L0C → <cc>
%alloc_4 = memref.alloc() : memref<256x256xf32, #hivm.address_space<cc>>
annotation.mark %alloc_4 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<cc>>

// 独占内存：除 effects 外额外携带 {mem_unique}
%alloc_5 = memref.alloc() : memref<256x256xf32, #hivm.address_space<ub>>
annotation.mark %alloc_5 {mem_unique}
    : memref<256x256xf32, #hivm.address_space<ub>>
annotation.mark %alloc_5 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<ub>>
```

### 输出要点说明

- 地址空间属性附加在 memref 类型上，格式为 `#hivm.address_space<mnemonic>`
- IR 中的助记符与 Python 枚举名**不完全一致**：L1 在 IR 中显示为 `cbuf`，L0A/L0B/L0C 分别为 `ca`/`cb`/`cc`，UB 保持 `ub`
- 不指定地址空间时，memref 类型不携带任何 `address_space` 属性
- `is_mem_unique=True` 通过独立的 `annotation.mark {mem_unique}` 表达，而不是地址空间属性的一部分

## 6. 相关接口

- [`bl.alloc`](../bl/alloc.md)：在指定地址空间分配 buffer
- [`bl.to_buffer`](../bl/to_buffer.md)：将 tensor 搬运/绑定到指定地址空间的 buffer
- [`bl.subview`](../bl/subview.md)：在已有 buffer 上创建子视图（地址空间沿用源 buffer）
- [`al.fixpipe`](./fixpipe.md)：L0C → UB 的专用搬运通路
