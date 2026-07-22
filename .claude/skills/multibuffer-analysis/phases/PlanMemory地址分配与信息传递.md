# BiShengIR MultiBuffer — PlanMemory 地址分配与信息传递

Phase 4 为多缓冲的每个 slot 分配独立的、不重叠的物理地址。同时汇总各 pass 之间的信息传递链。

---

## 1. PlanMemory 地址分配

**入口**：bishengir-compile 编译流水线中的 PlanMemory pass

**核心代码路径**：

| 函数 | 文件:行号 | 职责 |
|------|----------|------|
| `MemLivenessAnalysis::build()` | PlanMemory.cpp:151 | 构建活性分析，只追踪 SyncBlockWaitOp |
| `MemLivenessAnalysis::UpdateMultiBufferInfo()` | PlanMemory.cpp:677 | 从 markOp 读取 MultiBufferAttr |
| `MemPlan::PlanLocalMemAddress()` | PlanMemory.cpp:1653 | 本地内存规划 |
| `MemPlan::ExpandMultiBufferStorageEntry()` | PlanMemory.cpp:1734 | 按 multiBufferNum 扩展 StorageEntry |
| `MemPlan::VerifyConflictStage2()` | PlanMemory.cpp:2375 | 检查 pipeline 冲突 |
| `PipeConflict()` | PlanMemory.cpp:2412 | 检查两个 StorageEntry 的 pipeline 冲突 |

### 1.1 执行流程

**第1步**：`MemLivenessAnalysis::build()` 构建活性分析
- 遍历函数内所有操作，按线性顺序记录时间戳
- 只追踪 `SyncBlockWaitOp` 作为跨核同步点（`pipe_barrier` 是核内的，不追踪）
- 记录每个 buffer 的"出生"和"死亡"时间点

**第2步**：`UpdateMultiBufferInfo()` 读取 multibuffer 标记
- 扫描 `annotation.mark` 或 `pointer_cast` 上的 `MultiBufferAttr`
- 将 `multiBufferNum` 填入 `buffer2MultiNum` map
- 对于 scope preload，额外通过 `UpdatePreloadBuffers()` 追踪 preload buffer 生命周期

**第3步**：`ExpandMultiBufferStorageEntry()` 扩展内存条目

```cpp
// PlanMemory.cpp:1734-1753
void MemPlan::ExpandMultiBufferStorageEntry() {
    size_t size = StorageEntryVec.size();
    for (size_t i = 0; i < size; i++) {
        if (StorageEntryVec[i]->multiBufferNum > 1) {
            for (uint32_t j = 1; j < StorageEntryVec[i]->multiBufferNum; j++) {
                std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
                entry->bufInfo = StorageEntryVec[i]->bufInfo;
                entry->bufferLifeVec = StorageEntryVec[i]->bufferLifeVec;
                StorageEntryVec[i]->otherBufferRelationEntries.push_back(entry.get());
                StorageEntryVec.push_back(std::move(entry));
            }
        }
    }
}
```

- `multiBufferNum` 来自 `buffer2MultiNum` map，该 map 由 `UpdateMultiBufferInfo()` 从 `MultiBufferAttr` 填充
- 每个 multibuffer entry 复制相同的 `bufferInfo` 和 `bufferLifeVec`
- 通过 `otherBufferRelationEntries` 关联主 entry

**第4步**：`PlanLocalMemAddress()` 规划本地内存地址
- 为所有 `StorageEntry`（包括 multibuffer 扩展的）分配地址偏移量
- 通过 `otherBufferRelationEntries` 保证关联 entry 使用连续且不重叠的偏移量

**第5步**：`PipeConflict()` 检测流水线冲突
- 通过 `dmaFirstPipelineOpt.BufferPipeConflict()` 检查两个 buffer 是否有 DMA/标量流水线冲突
- 确保被复用地址的 buffer 不会产生硬件流水线冲突

**第6步**：地址写回 IR
- 将分配的地址偏移量写入 `pointer_cast` 的地址参数
- 每个 slot 的 `pointer_cast` 获得独立的物理地址

### 1.2 内存布局示意

`multi_buffer=2` 时 UB 空间布局：

```
offset=0x000: buf_slot0 (8×f16 = 16 bytes)
offset=0x010: buf_slot1 (8×f16 = 16 bytes)   ← 与 slot0 不重叠
offset=0x020: result   (8×f16 = 16 bytes)
...
```

### 1.3 PipeBarrier 在 PlanMemory 中的处理

`PlanMemory.cpp` 第 163-173 行：

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
        syncBlockPositions.push_back(scopeTime);
    scopeTime++;
}
```

`pipe_barrier` 被明确排除在跨核同步追踪之外，因为它是核内的。PlanMemory 只追踪 `SyncBlockWaitOp`（跨核接收等待）。

---

## 2. 各 Pass 间信息传递总结

### 2.1 传递链 1：multibuffer 深度

```
MarkMultiBuffer.mark() → annotation.mark {hivm.multi_buffer = N}
  → PlanMemory.UpdateMultiBufferInfo() → buffer2MultiNum[N]
  → ExpandMultiBufferStorageEntry() → N 个 StorageEntry → N 个地址偏移量
```

这条链从标记阶段一直贯穿到地址分配阶段。`multi_buffer` 属性的值（2 或 4）决定了最终分配的 slot 数量。

### 2.2 传递链 2：slot 拆分

```
MarkMultiBuffer.mark() → annotation.mark {hivm.multi_buffer = N}
  → (中间 pass) → pointer_cast(%addr0, %addr1, ...)
  → EnableMultiBuffer.createPtrCastOps() → N 个单地址 pointer_cast
  → EnableMultiBuffer 选槽级联 → arith.select 结果替换原引用
```

这条链描述了从标记到 IR 变换的全过程。中间 pass 将 alloca+mark 转为 pointer_cast，EnableMultiBuffer 拆分为独立 slot 并插入选槽逻辑。

### 2.3 传递链 3：迭代计数器

```
EnableMultiBuffer → MultiBufferLoopAdapter.ensureCounterMaterialized()
  → memref.alloca<1xi64> + hivm.multi_buffer_counter_for = loopId
  → 循环体头部 load + 尾部 store
```

计数器是 EnableMultiBuffer 独立创建的，通过 Loop ID 系统与循环绑定，支持幂等发现。

### 2.4 传递链 4：CVPipelining 消费

```
CVPipelining.wlBuilder.build() → resolvedMultibuffer
  → expandOutputInits() → 扩展输出维度
  → createNewLoops() → kMultibufferUnrollAttrName 写入循环属性
  → migrateOps() → SetAtomicOp 包裹原子操作
```

CVPipelining 是 multibuffer 信息的主要消费者之一，利用它确定流水线深度和输出维度。

### 2.5 传递链 5：Preload

```
MarkScopeMultiBuffer.mark(allocOp, rewriter, 4, true)
  → annotation.mark {hivm.multi_buffer = 4, hivm.preload_local_buffer}
  → PlanMemory.UpdatePreloadBuffers() → preload buffer 生命周期追踪
  → PlanMemory → 4 个 StorageEntry → 4 个地址偏移量
```

Preload 场景的传递链与普通 multibuffer 类似，但 `multi_buffer=4` 且额外携带 `preload_local_buffer` 标记。

### 2.6 信息流全景图

```
MarkMultiBuffer ──→ annotation.mark (hivm.multi_buffer 属性)
                          │
          ┌───────────────┘
          ▼
   (中间 pass: pointer_cast 转换)
          │
          ▼
EnableMultiBuffer ──→ MultiBufferLoopAdapter (迭代计数器)
          │                    │
          │            memref.alloca<1xi64>
          │            循环体头部 load + 尾部 store
          │
          ▼
     PlanMemory ──→ StorageEntry.multiBufferNum
                     ExpandMultiBufferStorageEntry()
                     PlanLocalMemAddress() → N 个偏移量
                          │
          ┌───────────────┘
          ▼
   CVPipelining ──→ resolvedMultibuffer
                     expandOutputInits() → 扩展维度
                     createNewLoops() → kMultibufferUnrollAttrName
```

### 2.7 属性传递汇总

| 属性 | 设置方 | 读取方 | 用途 |
|------|--------|--------|------|
| `hivm.multi_buffer = N` | MarkMultiBuffer | PlanMemory, EnableMultiBuffer, CVPipelining | 多缓冲深度 |
| `hivm.preload_local_buffer` | MarkScopeMultiBuffer | PlanMemory | preload buffer 追踪 |
| `hivm.multi_buffer_loop_id` | EnableMultiBuffer | EnableMultiBuffer（幂等） | 循环标识 |
| `hivm.multi_buffer_counter_for` | MultiBufferLoopAdapter | MultiBufferLoopAdapter（幂等） | 计数器归属 |
| `kMultibufferUnrollAttrName` | CVPipelining | 下游 pass | 展开因子 |
| `hivm.cv_pipelined_multi_buffer` | CVPipelining | PlanMemory | 流水线化后的 buffer |
