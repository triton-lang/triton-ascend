# Triton-Ascend算子调试与调优

概述：本文档介绍 Triton-Ascend 算子的调试方法、性能调优技术和 Warmup 预热方法，帮助开发者定位和解决算子开发过程中的问题。

## 文档组织

- [调试指南](./debugging.md) — Triton-Ascend算子调试方法
- [性能分析](./profiling.md) — 使用性能分析工具定位性能瓶颈
- [精度分析](./precision.md) — 算子精度问题分析与解决
- [msdebug 使用指南](./triton_msdebug_user_guide.md) — 使用 msdebug 调试 Triton Kernel
- [do_bench_npu使用指南](./do_bench_npu_guide.md) — do_bench_npu使用指南
- [Warmup使用指南](./warmup_guide.md) — 在服务启动或 CI 阶段提前编译普通 JIT 和 Autotune 算子

```{toctree}
:maxdepth: 3
:hidden:

debugging.md
profiling.md
precision.md
triton_msdebug_user_guide.md
ub_overflow.md
do_bench_npu_guide.md
warmup_guide.md
```
