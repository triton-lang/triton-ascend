# BiShengIR MultiBuffer — EnableMultiBuffer 变换流程与 CVPipelining 集成

Phase 3 将多地址 pointer_cast 拆分为独立的单地址 buffer，并插入迭代轮转逻辑。同时包含迭代计数器设计（MultiBufferLoopAdapter）和 CVPipelining 的集成。

---

## 1. EnableMultiBuffer — Round-Robin 轮转

**入口**：bishengir-compile 编译流水线中的 EnableMultiBuffer pass

**核心代码路径**：

| 函数 | 文件:行号 | 职责 |
|------|----------|------|
| `EnableMultiBufferPass::runOnOperation()` | EnableMultiBuffer.cpp:405 | pass 入口 |
| `MultiBufferPattern::matchAndRewrite()` | EnableMultiBuffer.cpp:374 | 匹配带标记的 pointer_cast |
| `MultiBufferHelper::extMultiBuffer()` | EnableMultiBuffer.cpp:228 | 核心变换逻辑 |
| `MultiBufferHelper::createPtrCastOps()` | EnableMultiBuffer.cpp:286 | 拆分 pointer_cast |
| `getParentLoop()` | EnableMultiBuffer.cpp:80 | yield 链追踪找锚点循环 |
| `MultiBufferLoopAdapter::ensureCounterMaterialized()` | MultiBufferLoopAdapter.cpp:149 | 创建迭代计数器 |

### 1.1 执行流程

**第1步**：`MultiBufferPattern::matchAndRewrite()` 匹配 pointer_cast
- 查找带有 `annotation.mark {hivm.multi_buffer}` 的 `hivm.hir.pointer_cast`
- 或者带有 `hivm.multi_buffer` 属性的 pointer_cast

**第2步**：`MultiBufferHelper::extMultiBuffer()` 执行核心变换：

| 子步骤 | 操作 | 代码位置 |
|--------|------|---------|
| 2a | 从 `MultiBufferAttr` 读取深度 N（默认2，preload 为4） | — |
| 2b | 通过 `getParentLoop()` 找到锚点循环 | EnableMultiBuffer.cpp:80 |
| 2c | `createPtrCastOps()` 拆分 pointer_cast，hoist 到函数入口 | EnableMultiBuffer.cpp:286 |
| 2d | 通过 `MultiBufferLoopAdapter` 创建迭代计数器 | MultiBufferLoopAdapter.cpp:149 |
| 2e | 插入 round-robin 选槽级联（`arith.cmpi eq` + `arith.select`） | EnableMultiBuffer.cpp:255 |
| 2f | 循环体内所有引用替换为 `arith.select` 的结果 | — |
| 2g | 标记循环和计数器属性 | — |

### 1.2 输入输出 IR 对比

**输入**（单 buffer，多地址 pointer_cast）：

```mlir
scf.for %iv = ... {
  %buf = hivm.hir.pointer_cast(addr0, addr1) []  // 两个地址打包
  use(%buf)
}
```

**输出**（双 buffer + round-robin 选槽）：

```mlir
%counter = memref.alloca() {hivm.multi_buffer_counter_for = 0}
memref.store 0, %counter[]

%buf0 = hivm.hir.pointer_cast(addr0) []   // slot 0
%buf1 = hivm.hir.pointer_cast(addr1) []   // slot 1

scf.for %iv = ... {hivm.multi_buffer_loop_id = 0} {
  %iter = memref.load %counter[]
  %idx  = arith.remui %iter, 2
  %cond = arith.cmpi eq, %idx, 1
  %buf  = arith.select %cond, %buf1, %buf0   // 偶数轮选 buf0, 奇数轮选 buf1

  use(%buf)

  %next = arith.addi %iter, 1
  memref.store %next, %counter[]
}
```

### 1.3 getParentLoop() 的 yield 追踪机制

`getParentLoop()`（EnableMultiBuffer.cpp:80-145）实现了跨越嵌套循环和 `scf.if` 的 yield 链追踪：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 从定义 op 获取父循环 | 找到定义该值的 op 的最近父 `LoopLikeOpInterface` |
| 2 | 判断是否在当前循环被消费 | `isConsumedInLoop()` 检查：有非 terminator、非 annotation 的 use 则更新 `consumerLoop` |
| 3 | 追踪 `scf.for`/`scf.while` 的 yield | 值被 yield 出去 → 追踪对应循环结果 → 递归 |
| 4 | 追踪 `scf.if` 的 yield | 父 op 是 `scf::IfOp` → 检查 then/else 分支 yield → 递归 |
| 5 | 返回最外层消费循环 | 真正消费该值的**最外层**循环作为 multibuffer 锚点 |

---

## 2. 迭代计数器详细设计（MultiBufferLoopAdapter）

### 2.1 为什么用 alloca-based 而非 affine-based

| 方案 | 计数器计算方式 | 支持 scf.for | 支持 scf.while |
|------|-------------|-------------|---------------|
| affine-based（上游 MLIR） | `affine.apply((iv-lb)/step) mod N` | ✅ | ❌（无 iv/lb/step） |
| alloca-based（BiShengIR） | 函数入口 `memref.alloca<1xi64>` + 循环内 load/remui/store | ✅ | ✅ |

BiShengIR 选择 alloca-based 方案以统一 `scf.for` 和 `scf.while` 的处理。

### 2.2 ensureCounterMaterialized() 的三阶段原子创建

`MultiBufferLoopAdapter.cpp:149-232`：

| 阶段 | 位置 | 操作 | 行号 |
|------|------|------|------|
| Phase 1 | 函数体入口 | 创建 `memref.alloca() : memref<1xi64>` + `hivm.multi_buffer_counter_for = loopId` + 初始化 store 0 | 187-198 |
| Phase 2 | 循环 body 开头 | 插入 `memref.load` 读取计数器，缓存为 `cachedLoad_` | 201-208 |
| Phase 3 | 循环 body 末尾 | 插入 `arith.addi %cachedLoad, %c1` + `memref.store` | 214-226 |

### 2.3 幂等性保证

`MultiBufferLoopAdapter.cpp:154-177`：

构造时先调用 `findExistingCounterAlloca()` 查找是否已有同 loop_id 的计数器。如果已存在，直接复用 alloca 和 load，不重复创建。保证"alloca 存在 ⟺ load+store 存在"的不变性。

### 2.4 Loop ID 系统

| 属性 | 附着对象 | 值 | 用途 |
|------|---------|-----|------|
| `hivm.multi_buffer_loop_id` | 循环 op | 唯一整数 ID | 标识拥有 multibuffer 的循环 |
| `hivm.multi_buffer_counter_for` | 计数器 alloca | 匹配的 loopId | 跨 pass 幂等发现 |

---

## 3. CVPipelining 与 Multibuffer 的集成

### 3.1 pipeline depth 的确定

`CVPipelining.cpp:1810`：

```cpp
numMultibuffer = buildResult->resolvedMultibuffer;
```

`numMultibuffer` 由 `wlBuilder.build()` 的返回值覆盖，该值从 multibuffer 标记中解析出来。最终写入每个 WorkItem 的 `scf.for` 属性：

```cpp
kMultibufferUnrollAttrName, builder.getI32IntegerAttr(numMultibuffer)
```

### 3.2 输出维度扩展

`CVPipelining.cpp:939`：

`expandOutputInits()` 使用 `numMultibuffer` 作为新 buffer 的第一维大小：

| 原 buffer | numMultibuffer | 扩展后 |
|-----------|---------------|--------|
| `memref<8xf16>` | 2 | `memref<2x8xf16>` |
| `memref<8xf16>` | 4 | `memref<4x8xf16>` |

使 Cube 和 Vector 之间的数据传递也能通过多缓冲实现 overlap。

### 3.3 跨阶段同步

CVPipelining 将循环体拆分为 VECTOR stage 和 CUBE stage，每个 stage 在自己的循环中顺序执行完所有迭代后再进入下一个 stage。同步靠 `scf.for` 的顺序执行语义，不引入额外的 barrier。

### 3.4 关键函数速查（CVPipelining）

| 函数 | 行号 | 职责 |
|------|------|------|
| `CVPipelineImpl::run()` | 1795 | pass 主入口 |
| `collectAtomicEffects()` | 467 | 收集 SetAtomicOp 作用域内的原子操作 |
| `wlBuilder.build()` | 1801 | 构建 worklist，解析 multibuffer 深度 |
| `absorbMergerOpsIntoWorkItems()` | 512 | 吸收 yield 链上的 merge op |
| `markOutputs()` | 668 | 标记每个 WorkItem 的 localOutputs 和 yieldedOutputs |
| `checkWorkItemDependencies()` | 777 | 检测跨核 loop-carried 依赖 |
| `expandOutputInits()` | 930 | 按 numMultibuffer 扩展输出 |
| `createNewLoops()` | 1087 | 创建 unrolled 外层循环 + 内层 jam 循环 |
| `migrateOps()` | 1231 | 将 op 克隆到对应的 WorkItem 循环 |
| `createNewLoopsForPreloadWithScopes()` | 1534 | preload 模式 scope 创建 |
