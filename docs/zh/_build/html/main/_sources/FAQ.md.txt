# Triton-Ascend FAQ

## 1. 安装与环境配置

**Q: 如何正确安装 Triton-Ascend？是否支持 pip 直接安装？**

A: 可以直接使用pip 安装

```Python
pip install triton-ascend
```

**Q: 社区 Triton 和 Triton-Ascend 能否同时存在？**

A: 不可以。需要先卸载社区 Triton，再安装 Triton-Ascend。

- 注：在安装依赖Triton的其他软件时，会自动安装社区 Triton，将覆盖掉已安装的 Triton-Ascend 目录。
此时也需要先卸载社区 Triton 和 Triton-Ascend，再安装 Triton-Ascend。

```Python
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

**Q: 能否在非 Ascend 硬件（如 CUDA AMD）上使用 Triton-Ascend？**

A: 不可以，只能在 Ascend NPU 硬件环境使用 Triton-Ascend

## 2. 精度与数值一致性问题

**Q: NPU 运行结果和 PyTorch/CPU/GPU 参考结果不一致，如何排查？**

A: 用例请参考 [07_accuracy_comparison_example.md](../zh/examples/07_accuracy_comparison_example.md)
调试方法请参考 [解释器模式调试方法](./debug_guide/debugging.md#4-解释器模式)

## 3. 错误代码与异常处理

**Q: 为什么 kernel 编译时报 MLIRCompilationError？如何定位具体失败的 Pass？**

A: 请参考 [编译错误调试方法](./debug_guide/debugging.md#52-编译错误调试方法)

## 4. 调试与日志

**Q: 如何开启详细日志输出？TRITON_DEBUG=1 输出在哪？**

A: 可以使用 TRITON_DEBUG=1 获取详细的调试转储文件，请参考 [调试转储文件（Dump Files）](./debug_guide/debugging.md#32-调试转储文件dump-files)

**Q: 能否在 kernel 中打印中间张量值？tl.device_print 是否可用？**

A: 可以使用 tl.device_print 打印 kernel 中的张量，请参考 [打印调试方法](debug_guide/debugging.md#51-打印调试方法)

## 5. 开发与贡献

**Q: 如何本地构建并测试 Triton-Ascend？**

A: 本地构建和测试方法，请参考 [通过源码安装Triton-Ascend](./installation_guide.md#通过源码安装triton-ascend)

**Q: 提交 PR 需要通过哪些 CI 检查？**

A: PR 的 CI 检查包括：编码安全与规范检查、开源片段检查、恶意代码检查、编译构建、开发者测试

## 6. 性能调优

**Q: 有没有性能分析工具（profiler）可以使用？**

A: 有集成性能分析工具（profiler），请参考 [算子性能调优方法](./debug_guide/profiling.md)
