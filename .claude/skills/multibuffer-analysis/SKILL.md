---
name: "multibuffer-analysis"
description: "Analyzes the MultiBuffer implementation in AscendNPU-IR-Dev (BiShengIR). Invoke when the user wants to understand how multibuffer/double-buffering works in the HIVM dialect, including pass pipelines, code locations, and design rationale."
---

# MultiBuffer 实现分析

## 目标项目

`/home/zk/bishengir-compile/AscendNPU-IR-Dev`

这是基于 LLVM/MLIR 的 BiShengIR，为 Ascend NPU 编译器定制的 IR 基础设施。MultiBuffer（双缓冲/多缓冲）是其核心优化之一，用于消除跨迭代的 WAR（写后读）假依赖，释放 DMA 与计算单元的并行能力。

## 分析流程

当用户询问 multibuffer 相关问题时，按以下步骤分析：

### 1. 先读文档

首先读取 `/home/zk/bishengir-compile/AscendNPU-IR-Dev/docs/MultiBuffer-CVPipeline-HIVMLevel.md`，获取完整的设计动机和变换流程概览。

核心概念：
- **问题**：单缓冲下，第 i 轮 DMA_store 和第 i+1 轮 DMA_load 打到同一物理地址，形成 WAR 依赖，DMA 和 VEC 无法并行
- **方案**：分配两块不重叠的物理地址（buf₀ / buf₁），迭代轮流使用，消除地址冲突
- **机制**：IR 保持单 `scf.for` 不变，不拆循环体，靠 round-robin 选槽 + 硬件调度器自动并行

### 2. 核心文件及职责

| 文件 | 绝对路径 | 职责 |
|------|----------|------|
| 设计文档 | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/docs/MultiBuffer-CVPipeline-HIVMLevel.md` | 完整变换流程、动机、与 triton-ascend 对比 |
| MarkMultiBuffer | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/lib/Dialect/HIVM/Transforms/MarkMultiBuffer.cpp` | Pass 1：自动分析 buffer liveness，标记可多缓冲的 buffer |
| EnableMultiBuffer | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/lib/Dialect/HIVM/Transforms/EnableMultiBuffer.cpp` | Pass 3：拆分 pointer_cast + 插入 round-robin select 级联 |
| MultiBufferLoopAdapter | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/lib/Dialect/HIVM/Utils/MultiBufferLoopAdapter.cpp` | 统一迭代计数器：alloca-based，同时支持 scf.for 和 scf.while |
| MultiBufferLoopAdapter 头文件 | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/include/bishengir/Dialect/HIVM/Utils/MultiBufferLoopAdapter.h` | 计数器策略文档、API 声明 |
| PlanMemory | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp` | Pass 4：为多缓冲 slot 分配不重叠的物理地址 |
| PlanMemory 头文件 | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/include/bishengir/Dialect/HIVM/Transforms/PlanMemory.h` | StorageEntry（multiBufferNum 等字段）、MemLivenessAnalysis |
| CVPipelining | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/lib/Dialect/HIVM/Transforms/CVPipelining.cpp` | Cube/Vector 流水，利用 multibuffer 确定 pipeline depth |
| Pass Pipeline | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/lib/Tools/bishengir-compile/PassPipeline.cpp` | 将 multibuffer passes 编排进编译流水线 |
| 属性定义 | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td` | `hivm.multi_buffer` 和 `hivm.cv_pipelined_multi_buffer` 属性的 TableGen 定义 |
| 常量定义 | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h` | `kMultiBufferCounterAttr`、`kMultiBufferLoopIdAttr` 字符串常量 |
| Pass 声明 | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td` | `MarkMultiBuffer` 和 `EnableMultiBuffer` 的 TableGen pass 定义 |
| Pass 头文件 | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.h` | C++ pass 创建函数声明 |

### 3. 测试文件

| 测试文件 | 绝对路径 |
|----------|----------|
| MarkMultiBuffer (scf.for) | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/test/Dialect/HIVM/mark-multi-buffer.mlir` |
| MarkMultiBuffer (scf.while) | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/test/Dialect/HIVM/mark-multi-buffer-while.mlir` |
| EnableMultiBuffer (scf.for) | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/test/Dialect/HIVM/enable-multi-buffer.mlir` |
| EnableMultiBuffer (scf.while) | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/test/Dialect/HIVM/enable-multi-buffer-while.mlir` |
| Mixed for/while nesting | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/test/Dialect/HIVM/enable-multi-buffer-mixed-for-while-nesting.mlir` |
| CVPipelining regression | `/home/zk/bishengir-compile/AscendNPU-IR-Dev/bishengir/test/Dialect/HIVM/cv-pipelining-no-multibuffer-attr.mlir` |

### 4. 四阶段变换流程

**Phase 1: MarkMultiBuffer** — 自动标记
- 扫描 `hivm.hir.load`、`hivm.hir.store`、`hivm.ND2NZ`、`hivm.Fixpipe`
- 对每个 buffer operand：若 alloc 在 `scf.for`/`scf.while` 内 + 非 L0C + 非动态大小 + 未标记 → 插入 `annotation.mark {hivm.multi_buffer = 2 : i32}`
- 通过命令行选项可按地址空间禁用：`--disable-multi-buffer-on-ub`、`--disable-multi-buffer-on-l0c`、`--disable-multi-buffer-on-l1`
- 对 Scope preload 场景标记 `multi_buffer = 4`

**Phase 2: 中间步骤** — 缓冲展开
- 其他 pass 将 `memref.alloca` + `annotation.mark` 转为 `hivm.hir.pointer_cast`（携带多个物理地址）

**Phase 3: EnableMultiBuffer** — Round-Robin 轮转
- `MultiBufferHelper::extMultiBuffer()` 核心逻辑：
  1. 将多地址 `pointer_cast` 拆分为多个单地址 `pointer_cast`，hoist 到函数入口
  2. 通过 `MultiBufferLoopAdapter` 创建函数级 `memref.alloca<1xi64>` 计数器
  3. 在循环体头部插入 `memref.load` + `arith.remui` + `arith.cmpi eq` + `arith.select` 级联选槽
  4. 在循环体尾部插入 `arith.addi` + `memref.store` 计数器递增
- `getParentLoop()` 递归追踪 yielded value，穿越嵌套 loop 和 `scf.if`，找到真正消费 buffer 的循环作为轮转锚点

**Phase 4: PlanMemory** — 地址分配
- `MemLivenessAnalysis::UpdateMultiBufferInfo()` 读取 `hivm.multi_buffer` 属性
- `MemPlan::ExpandMultiBufferStorageEntry()` 为每个额外 slot 创建独立 `StorageEntry`
- 确保所有 slot 的物理地址不重叠

### 5. 关键设计决策

- **Alloca-based 计数器**（非 affine-based）：`MultiBufferLoopAdapter` 使用函数级 `memref.alloca<1xi64>` + 循环体内 load/store，统一 `scf.for` 和 `scf.while` 的处理，不修改循环签名
- **Loop ID 系统**：每个拥有 multibuffer 的循环获得唯一的 `hivm.multi_buffer_loop_id` 整数属性，计数器 alloca 带有匹配的 `hivm.multi_buffer_counter_for` 属性，支持跨 pass 幂等发现
- **幂等性**：多次构造 `MultiBufferLoopAdapter` 不会重复创建计数器基础设施

### 6. 禁用条件

以下情况**不会**自动 multibuffer：
- buffer 在 L0C 空间（`#hivm.address_space<cc>`）
- buffer 为动态大小（`memref.alloca(%dyn)`）
- buffer 不在任何循环内
- buffer 所在循环不是 `scf::ForOp` 或 `scf::WhileOp`
- buffer 已标记过
- 对应的地址空间被命令行选项禁用

### 7. 上游 MLIR MemRef MultiBuffer

项目还包含上游 MLIR 的 MemRef MultiBuffer 实现（位于 `third-party/llvm-project/`），采用不同的 affine-based 方案（`affine.apply((iv-lb)/step) mod factor`），不被 BiShengIR 直接使用，仅作为参考。

### 8. 分析方法论

当用户提出具体问题时：
1. **理解问题场景**：确定是标记阶段、展开阶段还是地址分配阶段
2. **定位相关代码**：使用上述文件表找到对应源文件
3. **阅读关键函数**：重点关注 `MarkMultiBuffer::matchAndRewrite`、`MultiBufferHelper::extMultiBuffer`、`MultiBufferLoopAdapter::ensureCounterMaterialized`、`MemPlan::ExpandMultiBufferStorageEntry`
4. **对照测试用例**：在 `test/Dialect/HIVM/` 下找到对应 `.mlir` 文件，理解输入输出 IR 变换
5. **追踪 Pass Pipeline**：在 `PassPipeline.cpp` 中查看 pass 注册顺序
