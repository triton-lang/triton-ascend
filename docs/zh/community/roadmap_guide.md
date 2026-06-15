# 社区路线图编写与发布指南

使用 GitHub Issue 来跟踪和管理各组织的计划及中长期目标。本文档为社区项目编写路线图类 Issue 提供参考和规范，以帮助创建和维护高质量的路线图。

下面是一个完整的路线图 Issue 示例，展示了所有推荐元素的实际应用。建议先查看此示例获得整体印象，再阅读后续的详细规范说明。

```markdown
创建 Issue 标题：[Roadmap] Triton-Ascend Roadmap 2026 Q2

---
# Triton-Ascend Roadmap 2026 Q2

This quarter focuses on upstream Triton compatibility alignment, Ascend NPU backend performance optimization, and operator coverage expansion, continuously improving compiler stability and ecosystem integration capability.

## Focus

- Upstream Compatibility: Align with upstream Triton compiler frontend and IR changes, reduce fork divergence
- Backend Performance: Optimize Ascend NPU kernel generation and memory scheduling strategy
- Operator Coverage: Expand supported Triton ops and data types, improve end-to-end model coverage
- Usability: Improve debugging tools and error reporting for Ascend backend

## Upstream Compatibility

- [ ] **Triton 3.x IR and frontend alignment**
Goal: Align forked Triton frontend with upstream Triton 3.x IR changes, reduce merge conflict and divergence
Issue: [Related Issue link]

- [ ] **Triton Language feature parity check**
Goal: Systematically check and supplement missing Triton language features (e.g., tl.dot with new dtypes, constexpr enhancements) on Ascend backend
Issue: [Related Issue link]

## Backend Performance

- [ ] **Ascend NPU kernel auto-tuning support**
Goal: Support Triton autotune mechanism on Ascend backend, enable dynamic kernel configuration selection
Owner: @contributor-a
Issue: [Related Issue link]

- [ ] **Memory scheduling and L2 cache optimization**
Goal: Optimize memory allocation strategy and L2 cache utilization in generated Ascend kernels for large-scale training scenarios
Issue: [Related Issue link]

## Operator Coverage

- [ ] **FP8 dtype and mixed-precision ops support [🙋 Help Wanted]**
Goal: Support FP8 (E4M3/E5M2) dtype in tl.dot and related ops on Ascend backend, enabling FP8 training workflows
Owner: TBD
Issue: [Related Issue link]

- [ ] **Additional Triton built-in ops support**
Goal: Add missing built-in ops (e.g., tl.cumsum, tl.reduce with custom axis, advanced indexing) on Ascend backend
Issue: [Related Issue link]

## Usability

- [ ] **Ascend backend debugging and error reporting improvement**
Goal: Improve Ascend backend error messages, add device-side debug print and kernel profiling support
Issue: [Related Issue link]

## Sub-issues

[Triton-Ascend Roadmap 2026 Q1 #xxx](link)
[FP8 Support Phase 2 #xxx](link)
```

## 1. 标题格式

**格式：** `[Roadmap] <Project Name> Roadmap <Time Range>`，季度发布使用 Q1/Q2/Q3/Q4 标记，半年度发布使用 H1/H2 标记

**示例：**

- `[Roadmap] Triton-Ascend Roadmap 2026 Q2`
- `[Roadmap] Triton-Ascend Roadmap 2026 H1`

## 2. 顶层内容

### 2.1 开篇描述（可选）

提供项目概述、愿景或整体方向的简要总结。例如，简要描述 Triton-Ascend 本季度在上游对齐、后端性能和算子覆盖方面的目标。

### 2.2 重点方向章节

列出本周期最关键的 3-5 个重点方向，建议按项目的**功能领域**或**技术模块**分组，覆盖全局视角：

```markdown
## Focus

• Upstream Compatibility: Align with upstream Triton compiler frontend and IR changes, reduce fork divergence
• Backend Performance: Optimize Ascend NPU kernel generation and memory scheduling strategy
• Operator Coverage: Expand supported Triton ops and data types, improve end-to-end model coverage
• Usability: Improve debugging tools and error reporting for Ascend backend
• Ecosystem Integration: Enhance integration with PyTorch, vLLM, and MindSpeed on Ascend backend
```

**特征：**

- 高层次概括和描述当前周期项目的主要开发方向，无需详细展开

## 3. 主要功能模块章节

### 3.1 章节划分原则

按项目的**功能领域**或**技术模块**分组，例如：

- **Upstream Compatibility** - Upstream Triton frontend and IR alignment
- **Backend Performance** - Ascend NPU backend kernel generation and memory optimization
- **Operator Coverage** - Triton built-in ops and dtype support on Ascend
- **Usability** - Debugging tools and error reporting
- **Ecosystem Integration** - Integration with training/inference frameworks

### 3.2 各模块结构

每个模块包含多个**具体工作项**，格式如下：

```markdown
## [Module Name]

- [ ] **Work item name/feature description**
Goal: [Goal description]
Owner: @GitHubID      [optional]
Issue: [Related Issue link]   [optional]
PR: [Related PR link]         [optional]

- [ ] **Another work item**
Goal: [Goal description]
Owner: @GitHubID      [optional]
Issue: [Related Issue link]   [optional]
PR: [Related PR link]         [optional]
```

## 4. 关键元数据字段

每个工作项应包含以下关键信息：

### 4.1 目标（Goal）

- **含义**：工作目标或简要描述
- **用法**：说明该工作项的目标
- **示例**：`目标：在昇腾后端 tl.dot 中支持 FP8（E4M3/E5M2）数据类型`

### 4.2 负责人（Owner）

- **含义**：责任人
- **格式**：`负责人：@GitHubID`
- **用法**：明确谁负责或主导该工作项
- **示例**：`负责人：@contributor-a`

### 4.3 Issue

- **含义**：关联的 GitHub Issue
- **格式**：`Issue：<Issue 链接>`
- **用法**：跟踪详细设计和讨论
- **示例**：`Issue：https://github.com/triton-lang/triton-ascend/issues`

### 4.4 PR（Pull Request）

- **含义**：相关实现 PR
- **格式**：`PR：<PR 链接>`
- **用法**：关联实现工作
- **示例**：`PR：https://github.com/triton-lang/triton-ascend/pulls`

## 5. 可选补充内容

### 5.1 🙋 欢迎贡献标记

对于特别欢迎社区开发者贡献的工作项，建议使用 **[🙋 欢迎贡献]** 标记来标识：

```markdown
- [ ] **FP8 dtype and mixed-precision ops support [🙋 Help Wanted]**
Goal: Support FP8 (E4M3/E5M2) dtype in tl.dot and related ops on Ascend backend
Owner: TBD
Issue: #123
```

### 5.2 子 Issue

在路线图 Issue 底部列出跨周期的相关路线图 Issue 或大工作项的分解 Issue。

**与工作项中 Issue 字段的区别：**

- **工作项中的 Issue 字段**：链接到该工作项的具体设计、讨论或跟踪 Issue
- **子 Issue 章节**：用于关联其他周期的路线图 Issue（如上季度未完成的工作），或将大工作项分解为多个独立跟踪的子 Issue

```markdown
## Sub-issues

[Triton-Ascend Roadmap 2026 Q1 #xxx](link)  <!-- Related previous quarter Roadmap -->
[FP8 Support Phase 2 #xxx](link)            <!-- Breakdown of a large work item -->
```
