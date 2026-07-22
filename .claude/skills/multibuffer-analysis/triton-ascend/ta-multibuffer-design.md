# TA 层 Multibuffer 设计方案

> 将 BiShengIR HIVM 层的 MarkMultiBuffer + EnableMultiBuffer 逻辑上移到 Triton-ascend (TA) 层，PlanMemory 保留在 HIVM 层。

---

## 1. 动机

### 1.1 问题

当前 triton-ascend 的 multibuffer 由 `SeparateMemoryFromCompute`（DynamicCVPipeline 第 7 步）实现，采用软件流水方式：拆循环体为 Producer/Consumer，flag 握手控制，4 阶段 SSA 追溯依赖链。复杂度高，且与 BiShengIR 的 round-robin 方案是两套独立实现。

BiShengIR 的 multibuffer（MarkMultiBuffer → EnableMultiBuffer → PlanMemory）更简洁：不拆循环体，只做 round-robin 选槽，靠硬件 DMA/VEC 独立队列自动并行。但它运行在 HIVM 层，无法被 DynamicCVPipeline 利用。

### 1.2 目标

将 BiShengIR multibuffer 的核心逻辑（MarkMultiBuffer + EnableMultiBuffer）上移到 TA 层，替换 `SeparateMemoryFromCompute`，让 DynamicCVPipeline 直接享受 round-robin 多缓冲。PlanMemory（物理地址分配）保留在 HIVM 层不变。

### 1.3 可行性论证

**核心洞察**：PlanMemory 已经会为每个独立的 `memref.alloc` 分配不重叠的物理地址。如果在 TA 层为 buffer 创建 N 个独立的 `memref.alloc`（对应 N 个 slot），PlanMemory 会自动保证它们物理地址不冲突——这正是 multibuffer 需要的。

**数据流**：

```
TA 层:  memref.alloc(buf₀) + memref.alloc(buf₁)  →  arith.select 选槽
         ↓ (lowering)
HIVM层: hivm.hir.pointer_cast(addr₀) + hivm.hir.pointer_cast(addr₁)
         ↓ (PlanMemory)
硬件:   两块不重叠的物理地址 → DMA/VEC 自动并行
```

---

## 2. 架构总览

### 2.1 分层职责

| 层 | 职责 | 对应 BiShengIR Pass |
|----|------|-------------------|
| **TA 层** | MarkMultiBuffer: 识别可多缓冲的 buffer + 标记 | MarkMultiBuffer |
| **TA 层** | EnableMultiBuffer: 拆分 buffer slot + 插入 round-robin 选槽 | EnableMultiBuffer |
| **HIVM 层** | PlanMemory: 为多个 `memref.alloc` 分配不重叠物理地址 | PlanMemory（不变） |

### 2.2 TA 层 IR 构造映射

BiShengIR 使用 HIVM 方言特有的操作，TA 层需要用通用 MLIR/Triton 操作替代：

| BiShengIR (HIVM) | TA 层替代 | 说明 |
|------------------|----------|------|
| `hivm.hir.pointer_cast(%addr)` | `memref.alloc()` | TA 层不感知物理地址，用 memref.alloc 创建逻辑 buffer |
| `hivm.hir.load` / `hivm.hir.store` | `tt.load` / `tt.store` | TA 层使用 Triton 方言的 load/store |
| `annotation.mark {hivm.multi_buffer = N}` | 自定义属性或 metadata | 标记可多缓冲的 buffer |
| 其他（`arith.select`, `arith.remui`, `memref.alloca`） | 相同 | 通用 MLIR 操作，TA 层直接可用 |

### 2.3 在 DynamicCVPipeline 中的位置

```
PreCheckAvailable → StandardizeOp → PlanComputeBlock → ComputeBlockOpt
→ SplitDataflow → AnalyzeDataFlow → [★ MultiBufferPass] → AllocMultiCache
→ AddControlFlowCondition → RemoveSsbufAttr
```

替换 `SeparateMemoryFromCompute`（第 7 步），位置不变。

---

## 3. Phase 1: TA 层 MarkMultiBuffer

### 3.1 目标

自动识别循环内可被多缓冲优化的 buffer，并打上标记。

### 3.2 标记条件

对循环内的 `memref.alloc`（或 `tt.load` 的 dst buffer），满足以下条件才标记：

1. **buffer 的 alloc 在循环内**：`scf.for` 或 `scf.while` 的 body 内
2. **buffer 被 load 写入且被 store（或 compute）读出**：存在跨迭代 WAR 依赖
3. **buffer 非 Cube 累加器空间**：L0C 有独立流水机制，不需要此优化
4. **未被标记过**：避免重复

### 3.3 标记方式

在 `memref.alloc` 后插入属性标记：

```mlir
// 标记前
%buf = memref.alloc() : memref<8xf16>
// 标记后
%buf = memref.alloc() {triton_ascend.multi_buffer = 2 : i32} : memref<8xf16>
```

或者通过 `ssbuffer` 属性体系（与现有 DynamicCVPipeline 属性一致）：

```mlir
%buf = memref.alloc() {ssbuffer.multi_buffer = 2 : i32} : memref<8xf16>
```

### 3.4 跳过条件

| 条件 | 原因 |
|------|------|
| buffer 不在循环内 | 无跨迭代 WAR 依赖 |
| buffer 动态大小 | 编译期无法确定 slot 数量 |
| 循环不是 scf.for/while | scf.parallel 语义不支持 |
| 已标记过 | 避免重复 |
| buffer 已带 `gm_load_bufferable` | 已被 AsyncLoadHoisting 处理，跳过 |

---

## 4. Phase 2: TA 层 EnableMultiBuffer

### 4.1 目标

将标记的 buffer 拆分为 N 个独立 slot，插入 round-robin 选槽逻辑。

### 4.2 核心变换

**输入 IR**（MarkMultiBuffer 后）：

```mlir
scf.for %iv = %c0 to %c128 step %c8 {
  %buf = memref.alloc() {ssbuffer.multi_buffer = 2} : memref<8xf16>
  tt.load %in, %buf                    // WAR: buf 同一地址
  %res = tt.compute(%buf)
  tt.store %res, %out
}
```

**输出 IR**（EnableMultiBuffer 后）：

```mlir
// ★ 函数入口：为每个 slot 创建独立 buffer
%buf0 = memref.alloc() : memref<8xf16>
%buf1 = memref.alloc() : memref<8xf16>

// ★ 迭代计数器
%counter = memref.alloca() : memref<1xi64>
memref.store %c0, %counter[]

scf.for %iv = %c0 to %c128 step %c8 {
  // ★ Round-Robin 选槽
  %iter = memref.load %counter[]
  %idx  = arith.remui %iter, %c2
  %cond = arith.cmpi eq, %idx, %c1
  %buf  = arith.select %cond, %buf1, %buf0

  // === 原有逻辑不变 ===
  tt.load %in, %buf
  %res = tt.compute(%buf)
  tt.store %res, %out

  // ★ 计数器递增
  %next = arith.addi %iter, %c1
  memref.store %next, %counter[]
}
```

### 4.3 选槽级联（N > 2）

```mlir
// 默认选 buf₀
%selected = %buf0
// 级联比较
%c1 = arith.constant 1 : i64
%cond1 = arith.cmpi eq, %idx, %c1
%selected = arith.select %cond1, %buf1, %selected
// ... 更多 slot ...
```

### 4.4 迭代计数器设计

采用与 BiShengIR 相同的 **alloca-based 计数器**：

- 函数入口分配 `memref.alloca<1xi64>`，初始化为 0
- 循环体头部 `memref.load` 读取
- 循环体尾部 `arith.addi +1` 后 `memref.store` 写回
- `slot_idx = counter % N`

**优势**：
- 统一 `scf.for` 和 `scf.while`
- 不修改循环签名（iter_args / yields 不变）
- 比 affine-based 方案更通用

### 4.5 嵌套循环处理

buffer 可能在嵌套循环中被 yield 传递。需要追踪 yielded value：

1. 从 buffer 的定义点出发，找到最近的 `LoopLikeOpInterface` 父节点
2. 检查 buffer 是否在循环内被消费
3. 如果 buffer 被 yield，追踪循环结果，在更外层循环中继续查找

---

## 5. PlanMemory（HIVM 层，保持不变）

TA 层的 `memref.alloc` 在 lowering 到 HIVM 后变为 `hivm.hir.pointer_cast`（带物理地址）。PlanMemory 为每个独立的 `pointer_cast` 分配不重叠的物理地址。

**关键**：因为 TA 层已经创建了 N 个独立的 `memref.alloc`（对应 N 个 slot），lowering 后会有 N 个独立的 `pointer_cast`，PlanMemory 会自动为它们分配 N 块不重叠的物理地址。无需修改 PlanMemory。

```
TA 层         memref.alloc → buf₀    memref.alloc → buf₁
                ↓ (lowering)            ↓ (lowering)
HIVM 层       pointer_cast(addr₀)     pointer_cast(addr₁)
                ↓ (PlanMemory)          ↓ (PlanMemory)
硬件          物理地址 0x000-0x00F    物理地址 0x010-0x01F  (不重叠 ✓)
```

---

## 6. 与现有 DynamicCVPipeline 的集成

### 6.1 替换方案

| 改动项 | 说明 |
|--------|------|
| 新增 `MarkMultiBufferPass` | TA 层标记可多缓冲的 buffer |
| 新增 `EnableMultiBufferPass` | TA 层拆分 slot + 插入 round-robin |
| 删除 `SeparateMemoryFromComputePass` | 不再需要软件流水方式 |
| 删除 `AsyncLoadHoistingPass` | 不再需要 `gm_load_bufferable` 标记 |
| 删除 `AddMultiBufferToGMLoadPass` | 不再需要循环体拆分 |
| `CMakeLists.txt` | 替换编译源文件 |

### 6.2 不冲突的保证

1. **与 AllocMultiCache 不冲突**：AllocMultiCache 处理**跨 block 的 tensor 依赖**（inner scope）和**跨核数据传输**（outer scope），而 TA multibuffer 处理**同循环内的 memref WAR 依赖**。两者作用于不同粒度和不同 IR 类型（tensor vs memref）。

2. **与 PlanComputeBlock 不冲突**：PlanComputeBlock 分配 `ssbuffer.block_id`，TA multibuffer 在 block_id 分配之后运行，可以利用 block_id 信息判断是否需要多缓冲。

3. **与后续 pass 不冲突**：`AllocMultiCache`、`AddControlFlowCondition`、`RemoveSsbufAttr` 均不依赖 `SeparateMemoryFromCompute` 的输出。

---

## 7. 与 BiShengIR 方案的关键差异

| 维度 | BiShengIR HIVM 层 | TA 层（本方案） |
|------|------------------|----------------|
| **Buffer 创建** | `hivm.hir.pointer_cast(%addr0, %addr1)` 打包 | 多个独立 `memref.alloc()` |
| **Slot 拆分** | `pointer_cast` → 多个单地址 `pointer_cast` | 天然独立，无需拆分 |
| **物理地址** | MarkMultiBuffer 后中间 pass 转为 pointer_cast | lowering 时自动转为 pointer_cast |
| **PlanMemory** | `ExpandMultiBufferStorageEntry()` 为多 slot 分配 | 独立 `memref.alloc` 自动获得独立地址 |
| **IR 层次** | HIVM（低层，接近硬件） | Triton IR（高层，tensor/memref） |

**本质简化**：TA 层方案省去了 BiShengIR 中 `pointer_cast` 打包/拆包的中间步骤——直接从独立 `memref.alloc` 开始，选槽用 `arith.select`，地址分配交给 PlanMemory。

---

## 8. 实现计划

### 8.1 文件清单

| 文件 | 说明 |
|------|------|
| `lib/DynamicCVPipeline/TAMultiBuffer/MarkMultiBuffer.cpp` | MarkMultiBuffer pass |
| `lib/DynamicCVPipeline/TAMultiBuffer/EnableMultiBuffer.cpp` | EnableMultiBuffer pass |
| `include/DynamicCVPipeline/TAMultiBuffer/MarkMultiBuffer.h` | MarkMultiBuffer 头文件 |
| `include/DynamicCVPipeline/TAMultiBuffer/EnableMultiBuffer.h` | EnableMultiBuffer 头文件 |
| `include/DynamicCVPipeline/Passes.td` | 注册新 pass |

### 8.2 实现步骤

1. **MarkMultiBufferPass**
   - 遍历所有 `scf.for` / `scf.while`
   - 对循环内的 `memref.alloc`：检查是否被 load/store 使用
   - 满足条件则添加 `ssbuffer.multi_buffer = 2` 属性
   - 跳过 L0C buffer、动态大小 buffer、已标记 buffer

2. **EnableMultiBufferPass**
   - 扫描带 `ssbuffer.multi_buffer` 的 `memref.alloc`
   - 在函数入口创建 N 个 `memref.alloc`（hoist）
   - 创建迭代计数器 `memref.alloca<1xi64>`
   - 在循环体头部插入选槽级联
   - 替换原 buffer 使用点为 `arith.select` 结果
   - 在循环体尾部插入计数器递增
   - 处理嵌套循环中的 yield 追踪

3. **集成到 DynamicCVPipeline**
   - 在 `AddDynamicCVPipeline.cpp` 中：删除 `SeparateMemoryFromComputePass`，添加 `MarkMultiBufferPass` + `EnableMultiBufferPass`
   - 在 `CMakeLists.txt` 中：替换源文件

### 8.3 验证方式

- 单元测试：构造带 `scf.for` + `memref.alloc` + `tt.load/store` 的 IR，验证变换后 IR 结构正确
- 端到端测试：用实际 Triton kernel 验证编译通过且性能不退化

---

## 9. 总结

本方案将 BiShengIR multibuffer 的核心思想（round-robin 选槽 + 硬件自动并行）从 HIVM 层上移到 TA 层，替换当前的软件流水方案。核心优势：

- **简单**：不拆循环体，不引入 flag 握手，变换局部可预测
- **统一**：for/while 共享同一套计数器机制
- **自动**：PlanMemory 天然保证独立 `memref.alloc` 物理地址不重叠
- **低耦合**：PlanMemory 完全不需要修改
