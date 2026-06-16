# Community Roadmap Writing and Publishing Guide
Use GitHub Issues to track and manage each organization's plans and mid-to-long-term goals. This document provides references and specifications for community projects on writing Roadmap-type Issues, to help create and maintain high-quality Roadmaps.

Below is a complete Roadmap Issue example demonstrating the practical application of all recommended elements. It is recommended to review this example first for an overall impression, then read the detailed specification instructions that follow.

```markdown
Create Issue title: [Roadmap] Triton-Ascend Roadmap 2026 Q2

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

## 1. Title Format

**Format:** `[Roadmap] <Project Name> Roadmap <Time Range>`, quarterly releases use Q1/Q2/Q3/Q4 markers, semi-annual releases use H1/H2 markers

**Examples:**

- `[Roadmap] Triton-Ascend Roadmap 2026 Q2`
- `[Roadmap] Triton-Ascend Roadmap 2026 H1`

## 2. Top-level Content

### 2.1 Opening Description (optional)

Provide a project overview, vision, or brief summary of the overall direction. For example, a brief description of Triton-Ascend's current quarter goals in upstream alignment, backend performance, and operator coverage.

### 2.2 Focus Section

List the 3-5 most critical focus areas for this cycle, recommended to be grouped by the project's **functional domains** or **technical modules**, covering a holistic perspective:

```markdown
## Focus

• Upstream Compatibility: Align with upstream Triton compiler frontend and IR changes, reduce fork divergence
• Backend Performance: Optimize Ascend NPU kernel generation and memory scheduling strategy
• Operator Coverage: Expand supported Triton ops and data types, improve end-to-end model coverage
• Usability: Improve debugging tools and error reporting for Ascend backend
• Ecosystem Integration: Enhance integration with PyTorch, vLLM, and MindSpeed on Ascend backend
```

**Characteristics:**

- Summarize and describe the main development directions of the current project for the current cycle at a high level, no need to elaborate in detail

## 3. Major Functional Module Sections

### 3.1 Section Division Principles

Group by the project's **functional domains** or **technical modules**, such as:

- **Upstream Compatibility** - Upstream Triton frontend and IR alignment
- **Backend Performance** - Ascend NPU backend kernel generation and memory optimization
- **Operator Coverage** - Triton built-in ops and dtype support on Ascend
- **Usability** - Debugging tools and error reporting
- **Ecosystem Integration** - Integration with training/inference frameworks

### 3.2 Structure of Each Module

Each module contains multiple **specific work items**, formatted as follows:

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

## 4. Key Metadata Fields

Each work item should contain the following key information:

### 4.1 Goal

- **Meaning**: Work objective or brief description
- **Usage**: Explain the goal of this work item
- **Example**: `Goal: Support FP8 (E4M3/E5M2) dtype in tl.dot on Ascend backend`

### 4.2 Owner

- **Meaning**: Responsible person
- **Format**: `Owner: @GitHubID`
- **Usage**: Clarify who is responsible for or leading this work item
- **Example**: `Owner: @contributor-a`

### 4.3 Issue

- **Meaning**: Associated GitHub Issue
- **Format**: `Issue: <Issue link>`
- **Usage**: Track detailed design and discussion
- **Example**: `Issue: https://github.com/triton-lang/triton-ascend/issues`

### 4.4 PR (Pull Request)

- **Meaning**: Related implementation PR
- **Format**: `PR: <PR link>`
- **Usage**: Link implementation work
- **Example**: `PR: https://github.com/triton-lang/triton-ascend/pulls`

## 5. Optional Supplementary Content

### 5.1 🙋 Help Wanted Marker

For work items where community developer contributions are especially welcome, it is recommended to use the **[🙋 Help Wanted]** marker to indicate:

```markdown
- [ ] **FP8 dtype and mixed-precision ops support [🙋 Help Wanted]**
Goal: Support FP8 (E4M3/E5M2) dtype in tl.dot and related ops on Ascend backend
Owner: TBD
Issue: #123
```

### 5.2 Sub-issues

List cross-cycle related Roadmap Issues or breakdown Issues for large work items at the bottom of the Roadmap Issue.

**Difference from the Issue field in work items:**

- **Issue field in work items**: Links to the specific work item's detailed design, discussion, or tracking Issue
- **Sub-issues section**: Used to associate Roadmap Issues from other cycles (e.g., unfinished work from the previous quarter), or to break down large work items into multiple independently tracked sub-Issues

```markdown
## Sub-issues

[Triton-Ascend Roadmap 2026 Q1 #xxx](link)  <!-- Related previous quarter Roadmap -->
[FP8 Support Phase 2 #xxx](link)            <!-- Breakdown of a large work item -->
```
