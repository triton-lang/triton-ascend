# Triton-Ascend 版本发布

Triton-Ascend 版本提供了一个稳定的代码基础快照，封装成可以通过 PyPI 轻松安装的二进制包。此外，版本发布代表着开发团队可以向社区正式宣布新功能的可用性、已完成的改进以及可能影响用户的变化（例如破坏性变更）。

## 发布兼容性矩阵

以下是 Triton-Ascend 版本的发布兼容性矩阵：

| Triton-Ascend 版本  | Manylinux 版本 | 硬件平台 | 硬件产品 |
| --- | --- | --- | --- |
| 3.6.0 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品|
| 3.2.2.post1 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品|
| 3.2.2 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品|
| 3.2.1 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品|
| 3.2.0 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2系列产品、Atlas A3系列产品|

## 发布计划

以下是 Triton-Ascend 的发布计划。请注意：补丁版本是可选的。

| 主版本 | 发布分支切出时间 | 发布日期 | 补丁发布日期 |
| --- | --- | --- | --- |
| 3.6.0 | TBD | TBD | --- |
| 3.2.0 | 2025年12月08日 | 2026年1月 | --- |

## 版本亮点

### Triton-Ascend 3.2.0

**首次发布：Ascend NPU 支持**

Triton-Ascend 3.2.0 是第一个正式支持华为 Ascend NPU 的 Triton 版本。此版本基于 Triton 3.2.0 社区版本，专门适配 Ascend NPU 硬件架构。

#### 主要特性

1. **Ascend NPU 全栈支持**
   - 完整的 Triton IR 到 NPU 指令集编译流水线
   - 支持全部 Triton Ops

2. **性能优化**
   - NPU 特定内核优化
   - CV 计算优化

3. **开发者工具**
   - 支持全面的调试输出
   - 编译中间产物转储

#### 已知限制

1. **数据类型**: 部分数据类型支持仍在完善中
2. **算子覆盖**: 正在持续扩展支持的算子集合

#### 迁移指南

对于现有 Triton GPU 用户迁移到 Ascend NPU，详见 [GPU Triton算子迁移](./migration_guide/migrate_from_gpu.md)

## 版本发布策略

### 版本编号

Triton-Ascend 遵循 [PEP 440](https://peps.python.org/pep-0440/) 版本规范，版本号与上游 Triton 对齐：`vMAJOR.MINOR.PATCH[rcN][.postN]`

- **MAJOR.MINOR**：与上游 Triton 版本一一对应，例如 Triton-Ascend `3.2` 基于 Triton `3.2`
- **PATCH**：Triton-Ascend 的 `PATCH` 版本可能高于上游 Triton，用于 `MAJOR.MINOR` 级别的问题修复或改进，例如 Triton-Ascend `3.2.0` 和 `3.2.1` 均基于 Triton `3.2.0`
- **rcN**：发布候选版本，根据需要发布以供社区早期测试和反馈
- **postN**：已发布版本的后续补丁，根据需要发布以修复稳定版本中的问题

### 分支策略

- `main` 分支为最新开发分支，跟踪最新上游 Triton 版本
- 每个发布版本创建对应的发布开发分支（如 `release/3.2.x`），与社区发布具有相同的 commit id
- 功能开发应在 fork 仓库中进行，通过 `PR` 合入 Triton-Ascend 仓库

**`main` 分支映射：**

| Triton-Ascend | Triton commit hash                                           | Python    | CANN    | PyTorch | LLVM commit hash                                             | 补丁                                                        |
| ------------- | ------------------------------------------------------------ | --------- |---------| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `main`        | [85400f8](https://github.com/triton-lang/triton-ascend/commit/85400f8) | `3.9~3.13` | `9.1.0` | `2.10.0`   | [f6ded0b](https://github.com/llvm/llvm-project/commit/f6ded0b) | [llvm_patch_f6ded0b.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch) |

### 维护分支与生命周期

维护分支状态包括：

- **活跃（Active）**：持续接受缺陷修复、功能改进和安全补丁；将继续演进功能或发布新版本
- **维护（Maintenance）**：仅接受关键缺陷修复和安全补丁；不再发布功能改进
- **停止维护（End of Life）**：不再接受任何修复；分支维护已停止

| 分支              | 状态     | Triton 版本 | Triton-Ascend 发布                         | 维护截止 |
|-----------------| -------- | ------------ |------------------------------------------| -------- |
| `main`          | `活跃`   | `3.6.0`      | /                                        | /        |
| `release/3.2.2` | `活跃`   | `3.2.0`      | `3.2.2`                                  | /        |
| `release/3.2.1` | `维护`   | `3.2.0`      | `3.2.1`                                  | /        |
| `release/3.2.x` | `维护`   | `3.2.0`      | `3.2.0rc2`，`3.2.0rc3`，`3.2.0rc4`，`3.2.0` | /        |

## 发布周期

- **稳定版本**：按照项目版本节奏发布，并非每个上游 Triton 版本都会有对应的稳定发布
- **rc 版本**：与上游 Triton 版本节奏同步发布，供用户早期测试
- **post 版本**：根据需要发布，用于修复现有稳定版本中的问题

### 发布时间线

| 日期         | 事件                |
|------------|-------------------|
| 2026-07-31 | 发布稳定版本 `3.2.2`    |
| 2026-05-06 | 发布稳定版本 `3.2.1`    |
| 2026-01-21 | 发布稳定版本 `3.2.0`    |
| 2025-11-14 | 发布预览版本 `3.2.0rc4` |
| 2025-11-12 | 发布预览版本 `3.2.0rc3` |
| 2025-05-26 | 发布预览版本 `3.2.0rc2` |

<a id="version-compatibility-matrix"></a>

## 版本配套说明

| Triton-Ascend | Triton  | PyTorch  | TorchNPU安装包 | CANN    | LLVM commit hash | Python              | LLVM 补丁 |
|---------------|---------|----------|----------|---------|------------------|---------------------|-----------|
| `main`        | `3.6.0` | `2.10.0` | `2.10.0.post7` | `9.2.0` | `f6ded0b`        | `3.9~3.13`          | [llvm_patch_f6ded0b.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch) |
| `3.6.0`       | `3.6.0` | `2.10.0` | `2.10.0.post7` | `9.2.0` | `f6ded0b`        | `3.10~3.13`          | - |
| `3.2.2.post1`       | `3.2.0` | `2.7.1`  | `2.7.1.post11` | `9.1.0` | `b5cc222`        | `3.9`(x86), `3.10-3.13` | -         |
| `3.2.2`       | `3.2.0` | `2.7.1`  | `2.7.1.post8` | `9.1.0` | `b5cc222`        | `3.9`(x86), `3.10-3.13` | -         |
| `3.2.1`       | `3.2.0` | `2.7.1`  | `2.7.1.post8` | `9.0.0` | `b5cc222`        | `3.9`(x86), `3.10-3.13` | -         |
| `3.2.0`       | `3.2.0` | `2.6.0`  | `2.6.0` | `8.5.0` | `b5cc222`        | `3.9-3.11`          | -         |

## 版本兼容性说明

> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表1** Triton-Ascend与TorchNPU安装包版本兼容

<table class="table">
  <thead>
    <tr>
      <th rowspan="2">Triton-Ascend</th>
      <th colspan="8" style="text-align:left">TorchNPU安装包版本</th>
    </tr>
    <tr>
      <td>2.6.0</td>
      <td>2.7.1.post8</td>
      <td>2.7.1.post11</td>
      <td>2.9.0.post6</td>
      <td>2.9.0.post9</td>
      <td>2.10.0.post4</td>
      <td>2.10.0.post7</td>
      <td>2.13.0</td>
    </tr>
  </thead>
  <tbody>
    <tr><td>3.2.0</td><td>Y</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.1</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>/</td><td>/</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.2</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>/</td><td>/</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.2.post1</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>/</td></tr>
    <tr><td>3.6.0</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td></tr>
  </tbody>
</table>

**表2** Triton-Ascend与CANN版本兼容

<table class="table">
  <thead>
    <tr>
      <th rowspan="2">Triton-Ascend</th>
      <th colspan="4" style="text-align:left">CANN版本</th>
    </tr>
    <tr>
      <td>8.5.0</td>
      <td>9.0.0</td>
      <td>9.1.0</td>
      <td>9.2.0</td>
    </tr>
  </thead>
  <tbody>
    <tr><td>3.2.0</td><td>Y</td><td>/</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.1</td><td>Y</td><td>Y</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.2</td><td>Y</td><td>Y</td><td>Y</td><td>/</td></tr>
    <tr><td>3.2.2.post1</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td></tr>
    <tr><td>3.6.0</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td></tr>
  </tbody>
</table>

注：**Ascend 950PR&950DT products**硬件产品不支持CANN 8.5.0版本。
