# BiShengIR MultiBuffer — MarkMultiBuffer 标记流程与适用场景

Phase 1 负责自动识别循环内可被多缓冲优化的 buffer 并打上标记。以下是完整的匹配逻辑、过滤链和适用场景分析。

---

## 1. MarkMultiBuffer 标记流程

**入口**：bishengir-compile 编译流水线中的 MarkMultiBuffer pass

**核心代码路径**：

| 函数 | 文件:行号 | 职责 |
|------|----------|------|
| `MarkMultiBufferPass::runOnOperation()` | MarkMultiBuffer.cpp:271 | pass 入口，注册各 pattern |
| `MarkMultiBuffer::matchAndRewrite()` | MarkMultiBuffer.cpp:174 | 匹配 copy-like op 并标记 buffer |
| `MarkScopeMultiBuffer::matchAndRewrite()` | MarkMultiBuffer.cpp:111 | 处理 scope preload 场景 |
| `mark()` | MarkMultiBuffer.cpp:89 | 创建 annotation.mark |
| `tracebackMemRef()` | Utils/Util.cpp:993 | 沿 SSA 链回溯 buffer 来源 |
| `isAllocLikeOp()` | Utils/Util.cpp:627 | 判断是否 alloc-like |
| `isMarked()` | MarkMultiBuffer.cpp:82 | 检查是否已标记 |

### 1.1 matchAndRewrite 的触发逻辑

`MarkMultiBuffer<CopyOpType>` 模板化的 `matchAndRewrite`（MarkMultiBuffer.cpp:174-228）不关心"数据从哪来"，只关心"数据写到哪"：

- 调用 `copyLikeOp.hasPureBufferSemantics()`，非纯 buffer 语义则跳过
- 检查 src/dst 的 MemorySpace 是否存在

来源/目的地地址空间检查（MarkMultiBuffer.cpp:219-227）：

| 条件 | 行为 |
|------|------|
| src 不是 GM 地址空间 | 对 src 调用 markBufferFunc（数据从非 GM load 到 UB） |
| dst 不是 GM 地址空间 | 对 dst 调用 markBufferFunc（数据 store/写到非 GM） |
| src 和 dst 都是 GM | failure，不标记（纯 GM↔GM 搬运不需要 multibuffer） |

**Fixpipe 场景**也会命中。Fixpipe 可以是：
- GM → UB：数据搬入 UB，dst 是 UB 的 alloc → 进入 markBufferFunc
- UB → GM：搬出，src 是 UB → 进入 markBufferFunc
- GM → L0C：搬入 Cube 累加器，dst 在 L0C 地址空间
- L0C → GM：Cube 结果搬出

但 L0C 场景在后面被额外的策略开关排除（见 1.3）。

### 1.2 markBufferFunc 过滤链

进入 markBufferFunc 后，顺序执行七层过滤（MarkMultiBuffer.cpp:176-203）：

| 步骤 | 检查项 | 不通过时 | 代码位置 |
|------|--------|---------|---------|
| (a) | `tracebackMemRef(v)` 追溯到 buffer 来源定义点 | 跳过 | Utils/Util.cpp:993 |
| (b) | `isAllocLikeOp(allocOp)` 判断是否是 alloc-like | 跳过（如 BlockArgument） | Utils/Util.cpp:627 |
| (c) | `isMarked(allocOp)` 是否已标记 | 跳过（防重复） | MarkMultiBuffer.cpp:82 |
| (d) | `getParentLoop()` 是否在循环内 | 跳过 | — |
| (e) | 循环类型是 scf::ForOp 或 scf::WhileOp | 跳过（scf.parallel 等） | — |
| (f) | 全部通过 | 调用 `mark(allocOp, rewriter)` | MarkMultiBuffer.cpp:89 |

**isAllocLikeOp 识别的三种 op**（Utils/Util.cpp:627）：
- `memref::AllocOp`
- `memref::AllocaOp`
- `bishengir::memref_ext::AllocWorkspaceOp`

追溯到 BlockArgument（函数参数 arg）→ 不是 alloc-like → 跳过。原因：函数参数是调用方传入的外部 memref，地址固定，无法扩 slot。

**过滤判定链总结**：

```
tracebackMemRef → isAllocLikeOp? → isMarked? → getParentLoop? → 循环类型 for/while?
    ↓ 失败              ↓ 失败          ↓ 重复       ↓ 不在循环        ↓ 不是
   arg/其他          跳过            跳过           跳过            跳过
                                      ↓ 全部通过
                                  mark(allocaOp) → annotation.mark {multi_buffer = 2}
```

### 1.3 Fixpipe 与 L0C 的特殊处理

`runOnOperation()` 中（MarkMultiBuffer.cpp:293-295）：

```cpp
if (limitAutoMultiBufferOfLocalBuffer != MultiBufferStrategy::CUBE_NO_L0C) {
    patterns.insert<MarkMultiBuffer<hivm::FixpipeOp>>(...)
}
```

| 策略 | Fixpipe 行为 |
|------|-------------|
| `CUBE_NO_L0C` | Fixpipe 全部不标记 |
| 其他策略 | Fixpipe 照常匹配，但需通过 markBufferFunc 过滤链 |

设计原因：L0C 是 Cube 的累加器地址空间，有独立硬件流水线调度，不需要 round-robin 来解跨迭代串行。而 Fixpipe 搬运数据到 UB 的 alloca → 会正常通过过滤链 → 标记。

### 1.4 Load/Store 的注册策略

`runOnOperation()` 中（MarkMultiBuffer.cpp:297-301）：

```cpp
if (!isMixFuncCore ||
    !(limitMixAutoMultiBufferBuffer == MultiBufferStrategy::ONLY_CUBE)) {
    patterns.insert<MarkMultiBuffer<hivm::LoadOp>>(...)
    patterns.insert<MarkMultiBuffer<hivm::StoreOp>>(...)
}
```

| 核类型 | 策略 | LoadOp/StoreOp | ND2NZ/Fixpipe |
|--------|------|---------------|---------------|
| 非 MIX 核 | — | 注册 | 注册 |
| MIX 核 | ONLY_VECTOR | 注册 | 不注册 |
| MIX 核 | ONLY_CUBE | 不注册 | 注册 |

### 1.5 Scope Preload 场景

`MarkScopeMultiBuffer::matchAndRewrite()` 处理 scope preload：

| 步骤 | 检查 | 不通过 |
|------|------|--------|
| (a) | `scopeOp.preload_num > 0` | 跳过 |
| (b) | scope 类型非 CUBE | 跳过（CUBE 输出存 GM） |
| (c) | 返回值被外层 V1 消费 | 跳过 |
| (d) | 满足条件 | `mark(allocOpPtr, rewriter, 4, true)` |

- `multi_buffer = 4`（preload 需要更深缓冲深度）
- `isPreload = true`（额外设置 `PreloadLocalBufferAttr`）

---

## 2. Phase 2: 中间形态 — alloca → pointer_cast 转换

这个阶段不在 multibuffer 体系内，由其他 pass 完成：

```mlir
// 变换前
%buf = memref.alloca() : memref<8xf16, ub>
annotation.mark %buf {hivm.multi_buffer = 2 : i32}

// 变换后
%buf = hivm.hir.pointer_cast(%addr0, %addr1) []
         : memref<8xf16, ub>
```

关键：`pointer_cast` 将 N 个物理地址（对应 N 个 slot）打包为一个逻辑 memref。后续 EnableMultiBuffer 将其拆分。

---

## 3. 适用场景分析

multibuffer 不是万能优化，它解决的是一个非常具体的问题：循环内每轮迭代 alloc 新 buffer，上一轮的 store 和下一轮的 load 打到同一地址造成的 WAR 假依赖。

### 3.1 纯 Vector 逐元素计算（最佳场景）

典型 kernel：load → vadd → store，全部在 Vector 核上执行。

Triton 层面：

```python
@triton.jit
def elemwise_add(x_ptr, y_ptr, out_ptr, N):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    for i in range(0, N // BLOCK):
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        z = x + y
        tl.store(out_ptr + offsets, z)
        offsets += BLOCK * grid_size
```

编译后 IR 形态：

```mlir
scf.for %iv = %c0 to %cN step %cBLOCK {
  %buf_x = memref.alloc() : memref<8xf16, ub>
  %buf_y = memref.alloc() : memref<8xf16, ub>
  hivm.hir.load  ins(%x_ptr) outs(%buf_x)   // DMA写buf_x
  hivm.hir.load  ins(%y_ptr) outs(%buf_y)   // DMA写buf_y
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  %z = hivm.hir.vadd ins(%buf_x, %buf_y)     // Vector计算
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  hivm.hir.store ins(%z) outs(%out_ptr)       // DMA读buf_x/buf_y的结果
}
```

**WAR 依赖分析**：
- `buf_x`：iter1 的 load 写 buf_x 地址 ← 冲突 → iter0 的 store 读同一 buf_x 地址
- `buf_y`：同上
- `z`（vadd 输出）：store 读 z 地址，但 z 是 alloc 产物，vadd 写 z，store 读 z——这是同迭代内的 RAW 真依赖，multibuffer 不管这个

**适用判断**：buf_x 和 buf_y 每轮迭代在循环内 alloc，被 load 写入、被 store 读出，存在跨迭代 WAR 假依赖。满足条件，MarkMultiBuffer 会标记 `multi_buffer=2`。

**变换后效果**：
- buf_x 和 buf_y 各拆为 2 个 slot
- iter0 的 load/store 用 slot0，iter1 的 load/store 用 slot1
- MTE2 的 iter1.load(slot1) 和 MTE3 的 iter0.store(slot0) 地址不同 → Scoreboard 不阻塞 → 并行发射

**预期收益**：DMA 耗时约等于 compute 耗时时，理论加速比接近 2x。compute 远大于 DMA 时收益有限，DMA 远大于 compute 时收益也有限（瓶颈在 DMA 带宽）。

### 3.2 纯 Cube 矩阵计算（不适用）

典型 kernel：matmul，全部在 Cube 核上执行。

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K):
    a = tl.load(A_block)   # GM→L1
    b = tl.load(B_block)   # GM→L1
    c = tl.dot(a, b)       # L1→L0C→计算→L0C
    tl.store(C_block, c)   # L0C→GM
```

**为什么不适用**：
- L0C 是 Cube 单元的累加器空间，有独立的硬件流水和累加器重用机制
- Cube 的数据流是"多次累加到 L0C → 一次性写出 L0C"，不像 UB 那样每轮迭代 load→compute→store 轮转
- L0C 上不存在"上一轮 store 和下一轮 load 打同一地址"的问题
- MarkMultiBuffer 中 Fixpipe 的 L0C buffer 在 `CUBE_NO_L0C` 模式下不标记

Cube 的流水由 CVPipelining 单独处理。

### 3.3 Mixed CV（Cube + Vector 混合，部分适用）

典型 kernel：flash attention 中的 fused 操作——matmul（Cube）+ scale/softmax（Vector）交替。

```python
@triton.jit
def flash_attn_fwd(Q, K, V, O, ...):
    for i in range(num_blocks):
        s = tl.dot(q_block, k_block.T)      # Cube 核
        s = s * scale                         # Vector: scale
        m = tl.max(s, axis=1)                 # Vector: max
        s = s - m[:, None]                    # Vector: sub
        s = tl.exp(s)                         # Vector: exp
        o = tl.dot(s, v_block)               # Cube 核
```

**multibuffer 的两个作用**：

| 作用 | 机制 | 负责 pass |
|------|------|----------|
| Vector scope 内的 WAR 消除 | 和纯 Vector 场景一样，`multi_buffer=2` | MarkMultiBuffer |
| Scope preload 的深缓冲 | 预取下一轮 Cube 输出，`multi_buffer=4` | MarkScopeMultiBuffer |
| 确定 pipeline depth | CVPipelining 读取 multibuffer 信息 | CVPipelining |
| 扩展输出维度 | `expandOutputInits()` 按 numMultibuffer 扩展 | CVPipelining |

**不适合的部分**：
- Cube scope 内部的 Fixpipe（L0C）不标记
- Cube scope 的输出直接存 GM，不涉及 UB buffer 的跨迭代 WAR
- 跨核同步由 AllocMultiCache/OuterScope 处理，不是 MarkMultiBuffer 的职责

### 3.4 scf.while 循环（适用）

和 scf.for 不同，scf.while 没有 iv/lb/step 的概念，不能用 affine-based 计数器。BiShengIR 的 alloca-based 计数器方案天然支持：

```mlir
scf.while (%arg = %init) : (f32) → f32 {
  %cond = arith.cmpf ogt, %arg, %cst : f32
  scf.condition(%cond) %arg : f32
} do {
^bb(%arg: f32):
  %buf = memref.alloca() : memref<8xf16, ub>
  hivm.hir.load  ins(%in) outs(%buf)
  hivm.hir.store ins(%buf) outs(%out)
  %next = arith.subf %arg, %cst : f32
  scf.yield %next : f32
}
```

MarkMultiBuffer 检测到 while 循环内的 `memref.alloca`，标记 `multi_buffer=2`。EnableMultiBuffer 通过 MultiBufferLoopAdapter 插入 alloca 计数器 + 循环内 load/remui/store。

### 3.5 嵌套循环 + yield 传递（适用）

典型场景：buffer 在内层循环 alloc，通过 yield 传到外层循环使用。

```mlir
scf.for %outer = ... {
  %result = scf.for %inner = ... {
    %buf = memref.alloc() : memref<8xf16, ub>
    hivm.hir.load  ins(%in) outs(%buf)
    hivm.hir.store ins(%buf) outs(%out)
    scf.yield %buf : memref<8xf16, ub>
  }
  use(%result)
}
```

`getParentLoop()` 的处理：
- 从 pointer_cast 的定义点出发，找到最近的父循环（内层 scf.for）
- 检查 buffer 是否在内层循环被消费：被 yield 传出，不算"在内层消费"
- 追踪 `scf.yield` 到外层循环的 `%result`，递归检查
- 最终返回真正消费 buffer 的最外层循环作为 multibuffer 锚点

### 3.6 跨核数据传输（不适用 MarkMultiBuffer）

CUBE 核 → VECTOR 核的数据传输涉及跨核同步（SyncBlockSet/SyncBlockWait），MarkMultiBuffer 不管这个。

AllocMultiCache 的 OuterScope pass 处理跨核数据传输的双缓冲：
- 为每组传输创建 output buffer（memref::AllocOp）
- 创建 output sync（SyncBlockSetOp/SyncBlockWaitOp）
- 通过 `scf.if` polling 选择 input buffer 还是 output buffer

两套独立机制：OuterScope 处理跨核传输的"发送/接收"双缓冲，MarkMultiBuffer 处理核内循环的 WAR 消除。

### 3.7 不适用场景速查

| 场景 | 原因 | 代码位置 |
|------|------|---------|
| buffer 在 L0C 空间 | Cube 累加器有独立流水 | MarkMultiBuffer.cpp，CUBE_NO_L0C 策略 |
| buffer 动态大小 `memref.alloca(%dyn)` | PlanMemory 无法分配确定大小的 slot | markBufferFunc lambda |
| buffer 不在任何循环内 | 无跨迭代依赖 | getParentLoop() 返回 nullptr |
| 循环是 `scf.parallel` | 并行迭代无固定顺序 | isa<scf::ForOp, scf::WhileOp> 失败 |
| buffer 已标记过 | 防重复 | isMarked() 检查 |
| 命令行禁用地址空间 | — | --disable-multi-buffer-on-ub/l1/l0c |

### 3.8 收益预估

| DMA vs Compute | 收益 | 原因 |
|----------------|------|------|
| DMA ≈ compute | 最大，接近 2x | iter0 store 和 iter1 load 完全重叠 |
| DMA << compute | 有限 | 瓶颈在 VEC 计算，DMA 本来就不阻塞 |
| DMA >> compute | 有限 | 瓶颈在 DMA 带宽，VEC 经常空等 |

**内存开销**：每个 multibuffer 的 buffer 占用 N 倍 UB 空间（默认 N=2，preload N=4）。UB 空间紧张时可能得不偿失，可通过命令行选项按地址空间关闭。
