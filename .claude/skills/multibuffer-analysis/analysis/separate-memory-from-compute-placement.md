# DynamicCVPipeline 中计算访存分离的放置位置：SplitDataflow 之后 vs 控制流之后

## 流水线上下文

```
1. PreCheckAvailable → 2. StandardizeOp → 3. PlanComputeBlock → 4. ComputeBlockOpt
→ 5. SplitDataflow      ← 按 block_id 拆 SSA 作用域，插入 scf.if
→ 6. AnalyzeDataFlow
→ 7. [当前] SeparateMemoryFromCompute
→ 8. AllocMultiCache
→ 9. AddControlFlowCondition  ← 插入控制流条件（scf.if / iter_args 等）
→ 10. RemoveSsbufAttr
```

比较的两种位置：

| | SplitDataflow 之后（当前） | 控制流之后 |
|---|---|---|
| **位置** | Step 7 | Step 9 之后 |
| **scf.if 状态** | SplitDataflow 刚拆完，未加控制流条件 | 所有控制流已稳定，最终形态 |
| **后续 pass** | AllocMultiCache、AddControlFlowCondition | 只剩 RemoveSsbufAttr |

---

## 一、SplitDataflow 之后（当前位置）

### 利

1. **循环体干净**：SplitDataflow 刚完成按 block_id 的拆分，`scf.if` 结构简单——每个 if 内部是同一 block_id 的 op，没有复杂嵌套。拆体（Producer/Consumer）可以直接在这层 if 之上操作。

2. **依赖信息全**：AnalyzeDataFlow 已跑完，`DependencyAnalysis.cpp` 的 `computeLoadChain` 能拿到完整的 SSA 依赖链，知道哪个 load 喂给了哪个 compute。

3. **block_id 完整**：AsyncLoadHoisting 可以按 block_id 分组扫描 load op，区分哪些是 Cube 搬运、哪些是 Vector 搬运。

4. **AllocMultiCache 可接收 slot 数**：multibuffer 决定了需要几个 buffer slot，这个信息可以直接传给后面的 AllocMultiCache，不用 Guess。

### 弊

1. **后续 pass 必须适配**：拆体改了循环签名（+4 个 iter_args：2 flags + prod + cons），AllocMultiCache 和 AddControlFlowCondition 需要感知这些新 iter_args，不能按原来的循环签名写逻辑。

2. **与后续 pass 耦合**：SeparateMemoryFromCompute → AllocMultiCache → AddControlFlowCondition 三者本质上在做同一件事（计算访存分离），但被拆成了三个 pass，靠隐式约定传递信息。

---

## 二、控制流之后

### 利

1. **对后续零冲击**：只剩 RemoveSsbufAttr 清理属性，不需要任何 pass 适配我们的变换。

2. **不改变控制流逻辑**：不会因为拆体引入的 flag 握手干扰 AddControlFlowCondition 插入的控制流条件，不导致性能退化。

3. **buffer 地址已分配**：AllocMultiCache 已跑完，multibuffer 的 slot 直接对应已分配的物理 buffer。

4. **前序 pass 完全封闭**：所有块划分、数据流分析、控制流插入都不需要感知 multibuffer 的存在。

### 弊（需要解决的两个关键问题）

#### 问题 1：不同 iter 不知道是否会走入 if

SplitDataflow 之后的 `scf.if` 是按 block_id 决定是否执行的——不同迭代可能走不同分支。

**当前拆体方案（Producer/Consumer）放在控制流之后**：

```mlir
scf.for ... iter_args(flag0, flag1, prod, cons) {
  // Producer 段：检查 flag → 填 slot
  scf.if %cond {   // 填 slot 逻辑
  }

  scf.if %is_compute_block {
    // Consumer 段：消费 slot → 释放 flag
    // 如果这个迭代不走 compute 分支 → flag 不释放 → 死锁
    flag = false   // ← 不执行！
  }

  scf.yield ..., flag0, flag1, prod, cons
}
```

Consumer 释放 flag 的逻辑在 `scf.if` 内部。SplitDataflow 保证同一个迭代中 load block 和 compute block 的 if 条件相关，但 **AddControlFlowCondition 可能改变条件语义**，导致某些迭代走了 load 分支但不走 compute 分支。flag 一旦不释放，Producer 就永远等不到空槽。

**改用 round-robin 方案**：

```mlir
scf.for %iv = ... {
  %idx = iter % 2                                    // 无条件
  %buf = select(%idx, slot1, slot0)                  // 无条件

  scf.if %is_load_block {                             // 可能跳过
    memref.copy %src, %buf
  }
  scf.if %is_compute_block {                          // 可能跳过
    linalg.matmul ins(%buf) ...
  }
  counter++                                           // 无条件
}
```

`select` 在循环头、`counter++` 在循环尾，都在 `scf.if` 外部，**无条件执行**。即使某次迭代不走任何分支，slot 也正常轮转。没有 flag，没有状态机，不怕分支差异化执行。

**结论：拆体方案无法解决此问题，round-robin 方案对此免疫。**

---

#### 问题 2：跨 if 场景需要额外处理

SplitDataflow 之后的典型形态——buffer 在 if 外部，load 和 compute 分属两个 `scf.if`：

```mlir
scf.for %iv = ... {
  %buf = memref.alloc() : memref<128x128xf16>    // ← 在 if 外部

  scf.if %is_block_0 {                            // Load
    memref.copy %src, %buf
  }

  %t = bufferization.to_tensor %buf {gm_load_bufferable}

  scf.if %is_block_1 {                            // Compute
    %r = linalg.matmul ins(%t, ...) ...
  }
}
```

**当前拆体方案（Producer/Consumer）放在控制流之后**：

Producer 需要把 `memref.copy` 提前，Consumer 需要把 `matmul` 推迟——但这俩 op 现在被 `scf.if` 包着。把 copy 从 if 里提到 if 外面，就改变了 SplitDataflow 的 scope 语义（copy 和 matmul 原本各属一个 block scope，现在 copy 被提到 scope 外）。破坏 scope 结构会连带影响后续的 sync 注入和 block 调度。

**改用 round-robin 方案**：

`%buf = memref.alloc()` 的位置在 `scf.if` 外部，这就是插入 select 的位置：

```mlir
%slot0 = memref.alloc() : memref<128x128xf16>    // hoist 到函数入口
%slot1 = memref.alloc() : memref<128x128xf16>

scf.for %iv = ... {
  %buf = select(iter%2, slot1, slot0)             // ← 替代原来的 memref.alloc

  scf.if %is_block_0 {                            // 不动
    memref.copy %src, %buf
  }
  %t = bufferization.to_tensor %buf                // 不动

  scf.if %is_block_1 {                            // 不动
    %r = linalg.matmul ins(%t, ...) ...
  }
  counter++
}
```

**`%buf` 是同一个 SSA 值**，同时流向两个 `scf.if`。select 替代的是 alloc 的位置（在 if 外），load 和 compute 该怎么用还怎么用，不需要跨 if 做任何额外处理。

**结论：拆体方案会破坏 scope 结构，round-robin 方案天然适应——因为 select 直接替代了 buffer 定义点。**

---

## 总结

| | SplitDataflow 之后（当前） | 控制流之后（拆体） | 控制流之后（round-robin） |
|---|---|---|---|
| **问题 1：不同分支** | 不涉及（if 简单） | 死锁 | 无影响 |
| **问题 2：跨 if** | 不涉及（未加控制流） | 破坏 scope | 无影响 |
| **后续适配** | 8/9 必须感知新签名 | 不需要 | 不需要 |
| **前序耦合** | 依赖 5/6 的输出 | 依赖全部 | 无 |
| **可行性** | 当前方案 | 不可行 | 可行 |

**核心差异不在"放哪个位置"，而在"用什么方案"**：拆体方案只能放 SplitDataflow 之后（还没加复杂控制流）；round-robin 方案放哪都行，放控制流之后好处最多（零耦合、零冲击）。代价是需要从拆体迁移到 round-robin 的改造工作。
