# Triton-Ascend 版本发布

Triton-Ascend 版本提供了一个稳定的代码基础快照，封装成可以通过 PyPI 轻松安装的二进制包。此外，版本发布代表着开发团队可以向社区正式宣布新功能的可用性、已完成的改进以及可能影响用户的变化（例如破坏性变更）。

## 发布兼容性矩阵

以下是 Triton-Ascend 版本的发布兼容性矩阵：

| Triton-Ascend 版本 | Python 版本 | Manylinux 版本 | 硬件平台 | 硬件产品 |
| --- | --- | --- | --- | --- |
| 3.2.0 | >=3.9, <=3.11 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2/A3|

## 发布计划

以下是 Triton-Ascend 的发布计划。请注意：补丁版本是可选的。

| 主版本 | 发布分支切出时间 | 发布日期 | 补丁发布日期 |
| --- | --- | --- | --- |
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
