# BiShengIR MultiBuffer — 核内同步与联合工作机制

multibuffer 变换后，原有的 PipeBarrier 还需要改吗？不同迭代如何在不加新 barrier 的前提下实现并行？答案是：完全不需要改 barrier，并行由"不同 slot 地址 + 硬件 Scoreboard 不阻塞 + 三条独立执行队列"自动实现。

---

## 1. 整体流程概览

BiShengIR multibuffer 分为四个阶段，数据在各 pass 间通过 MLIR 属性和 IR 结构传递：

```
MarkMultiBuffer → (中间 pass: alloca→pointer_cast) → EnableMultiBuffer → PlanMemory
     │                                                        │                  │
     └── MultiBufferAttr (annotation.mark) ───────────────────┘                  │
                                                                                  │
     └── MultiBufferAttr ──→ UpdateMultiBufferInfo() ──→ buffer2MultiNum ──→ ExpandMultiBufferStorageEntry()
                                   (PlanMemory.cpp:677)              (PlanMemory.cpp:1734)
```

每个阶段的关键产出和对下游的影响：

| 阶段 | 产出 | 下游消费 |
|------|------|---------|
| MarkMultiBuffer | `annotation.mark {hivm.multi_buffer = 2 : i32}` | PlanMemory 读取多缓冲深度 |
| 中间 pass | `hivm.hir.pointer_cast(%addr0, %addr1)` 打包多地址 | EnableMultiBuffer 拆分 |
| EnableMultiBuffer | 拆分的单地址 pointer_cast + round-robin 选槽 + alloca 计数器 | PlanMemory 地址分配 |
| PlanMemory | 每个 slot 的独立物理地址偏移量 | 写入 pointer_cast 地址参数 |

---

## 2. 硬件执行模型

Ascend NPU 有三条独立的执行队列：

| 队列 | 职责 | 同步机制 |
|------|------|---------|
| MTE2 | DMA 读（GM→UB） | Scoreboard 跟踪目标地址 |
| PIPE_V | Vector 计算（vadd/vmul/vdiv 等） | PipeBarrier 阻挡后续指令发射 |
| MTE3 | DMA 写（UB→GM） | Scoreboard 跟踪源地址 |

**Scalar 核**负责向三条队列派发指令。派发后 Scalar 不等结果，继续派发下一条。三条队列各自独立执行，理想情况下可以并行。

**Scoreboard 机制**：每个 DMA 操作在发射时向 Scoreboard 注册目标/源地址范围。后续 DMA 操作发射前查询 Scoreboard——如果地址范围与在途操作冲突，阻塞发射直到在途操作完成。

**PipeBarrier`<PIPE_ALL>` 的语义**：Scalar 暂停向指定管道发射后续指令，直到该管道之前所有指令执行完毕。barrier 是"同管道内"的，不跨管道生效。

---

## 3. PipeBarrier 是核内的，不影响跨迭代并行

`PlanMemory.cpp` 第 163-173 行的注释明确指出：

```cpp
// Record positions of cross-core RECEIVE sync ops. Only
// SyncBlockWaitOp guarantees that the OTHER core has made progress past
// a known signal point. sync_block_set is a one-way signal that doesn't
// wait; pipe_barrier is intra-core. SyncBlockOp ALL_*-mode also doesn't
// cross cores. Conservative count keeps reuse safe.
int64_t scopeTime = 0;
for (size_t i = 0; i < linearOperation.size(); ++i) {
    Operation *op = linearOperation[i]->operation;
    if (isa<hivm::SyncBlockWaitOp>(op))
        syncBlockPositions.push_back(scopeTime);  // 只追踪跨核同步
    scopeTime++;
}
```

PlanMemory 在构建内存活性分析时，只追踪 `SyncBlockWaitOp`（跨核接收等待），不追踪 `pipe_barrier`。因为 `pipe_barrier` 只在同一核内生效，不影响跨核的内存复用决策。

---

## 4. Barrier 只拦同一迭代内的指令，不拦不同迭代

变换前的 IR：

```mlir
scf.for %iv = %c0 to %c128 step %c8 {
  %buf = hivm.hir.pointer_cast(%addr0, %addr1) []
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  hivm.hir.load  ins(%in)  outs(%buf)     // DMA 写 buf
  hivm.hir.pipe_barrier[<PIPE_ALL>]        // 确保 load 完成后再 vadd
  hivm.hir.vadd  ins(%buf, %buf) outs(%res)
  hivm.hir.pipe_barrier[<PIPE_ALL>]        // 确保 vadd 完成后再 store
  hivm.hir.store ins(%res) outs(%out)      // DMA 读 buf
}
```

**关键点**：barrier 拦的是"同一迭代内"的指令发射顺序。Scalar 派发完 iter0 的全部指令后，继续派发 iter1 的指令——Scalar 不等执行结果。

---

## 5. 单缓冲下的跨迭代串行原因

```
iter0: Scalar派发 DMA_load(buf, addr)  → MTE2队列
       Scalar派发 barrier              → MTE2队列标记
       Scalar派发 VEC_vadd             → V队列
       Scalar派发 barrier              → V队列标记
       Scalar派发 DMA_store(buf, addr) → MTE3队列  ← 目标地址 = addr

iter1: Scalar派发 DMA_load(buf, addr)  → MTE2队列  ← 目标地址 = addr（相同！）
              ↑
       MTE2引擎硬件Scoreboard检测：addr == addr，和MTE3在途的store冲突
       → 数据冒险，DMA_load 必须等 DMA_store 完成 → 被迫串行
```

问题根源：硬件 Scoreboard 跟踪在途操作的目标地址。iter1 的 load 和 iter0 的 store 打到同一地址，Scoreboard 检测到冲突，阻塞 iter1 的 load。

---

## 6. 双缓冲下 barrier 不变，并行自动发生

变换后的 IR（EnableMultiBuffer 之后）：

```mlir
// 函数入口
%buf0 = hivm.hir.pointer_cast(%addr0) []   // slot 0, 物理地址 = addr0
%buf1 = hivm.hir.pointer_cast(%addr1) []   // slot 1, 物理地址 = addr1

%counter = memref.alloca() : memref<1xi64>
memref.store %c0, %counter[]

scf.for %iv = %c0 to %c128 step %c8 {
  // round-robin 选槽
  %iter = memref.load %counter[]
  %idx  = arith.remui %iter, %c2
  %cond = arith.cmpi eq, %idx, %c1
  %buf  = arith.select %cond, %buf1, %buf0   // iter0选buf0, iter1选buf1

  // === barrier 完全不变 ===
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  hivm.hir.load  ins(%in)  outs(%buf)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  hivm.hir.vadd  ins(%buf, %buf) outs(%res)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  hivm.hir.store ins(%res) outs(%out)

  %next = arith.addi %iter, %c1
  memref.store %next, %counter[]
}
```

硬件执行视角：

```
iter0: Scalar派发 DMA_load(buf0, addr0)  → MTE2队列
       Scalar派发 barrier                 → MTE2队列标记
       Scalar派发 VEC_vadd                → V队列
       Scalar派发 barrier                 → V队列标记
       Scalar派发 DMA_store(buf0, addr0)  → MTE3队列  ← 目标地址 = addr0

iter1: Scalar派发 DMA_load(buf1, addr1)   → MTE2队列  ← 目标地址 = addr1（不同！）
              ↑
       MTE2引擎硬件Scoreboard检测：addr1 ≠ addr0，和MTE3在途的store不冲突
       → 无冒险，DMA_load 直接发射，不等 DMA_store 完成
```

结果：三条执行队列并行跑不同迭代的指令：

```
MTE2:   [load(buf0,addr0)]  [load(buf1,addr1)]  [load(buf0,addr0)]  ...
PIPE_V:         [vadd0]             [vadd1]             [vadd2]      ...
MTE3:                 [store0,addr0]      [store1,addr1]      ...
```

---

## 7. 核心结论

multibuffer 变换不需要修改任何 PipeBarrier。原有的 barrier 保证同一迭代内 `load → vadd → store` 的顺序不变。不同迭代间的并行由以下三者自动实现：

1. **不同 slot 地址**：buf₀ ≠ buf₁，Scoreboard 不阻塞
2. **硬件 Scoreboard**：自动检测地址冲突，不冲突则不阻塞
3. **三条独立执行队列**：MTE2、PIPE_V、MTE3 各自独立发射和调度
