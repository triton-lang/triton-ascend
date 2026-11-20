# Triton-Ascend FAQ

## 1. 安装与环境配置

**Q: 如何正确安装Triton-Ascend？是否支持pip直接安装？**

A: 可以直接使用pip安装。

```bash
pip install triton-ascend
```

**Q: 社区Triton和Triton-Ascend能否同时存在？**

A: `triton-ascend 3.2.0` 及以下不可以。需要先卸载社区Triton，再安装Triton-Ascend。
`triton-ascend 3.2.1` 及以上，Triton-Ascend通过将Triton声明为安装依赖来缓解安装覆盖问题。
安装Triton-Ascend时会先安装社区Triton，再由Triton-Ascend覆盖同名目录，从而避免后续安装其他依赖Triton的软件包时再次安装Triton而覆盖Triton-Ascend。
x86与ARM使用不同版本的社区Triton安装包的原因是社区从3.5版本开始才提供ARM版本安装包：x86依赖 `triton==3.2.0`，ARM依赖 `triton==3.5.0`。

- 注：若安装Triton-Ascend后，再安装依赖Triton的第三方软件包或者Triton本身，将覆盖掉已安装的Triton-Ascend目录。此时需要先卸载社区Triton和Triton-Ascend，再安装Triton-Ascend。

```bash
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

**Q: 能否在非Ascend硬件（如CUDA、AMD）上使用Triton-Ascend？**

A: 不可以，只能在Ascend NPU硬件环境使用Triton-Ascend。

## 2. 精度与数值一致性问题

**Q: NPU运行结果和PyTorch/CPU/GPU参考结果不一致，如何排查？**

A: 用例请参考 [07_accuracy_comparison_example.md](../zh/examples/07_accuracy_comparison_example.md)；调试方法请参考 [解释器模式调试方法](./debug_guide/debugging.md#debug-interpreter-mode)。

## 3. 错误代码与异常处理

**Q: 为什么Kernel编译时报MLIRCompilationError？如何定位具体失败的Pass？**

A: 请参考 [编译错误调试方法](./debug_guide/debugging.md#debug-compilation-error)。

## 4. 调试与日志

**Q: 如何开启详细日志输出？TRITON_DEBUG=1输出在哪？**

A: 可以使用TRITON_DEBUG=1获取详细的调试转储文件，请参考 [调试转储文件（Dump Files）](./debug_guide/debugging.md#debug-dump-files)。

**Q: 能否在kernel中打印中间张量值？tl.device_print是否可用？**

A: 可以使用tl.device_print打印kernel中的张量，请参考 [打印调试方法](./debug_guide/debugging.md#debug-printing)。

## 5. 开发与贡献

**Q: 如何本地构建并测试Triton-Ascend？**

A: 本地构建和测试方法，请参考 [通过源码安装Triton-Ascend](./installation_guide.md#install-from-source)。

**Q: 提交PR需要通过哪些CI检查？**

A: PR的CI检查包括：编码安全与规范检查、开源片段检查、恶意代码检查、编译构建、开发者测试。

## 6. 性能调优

**Q: 有没有性能分析工具（profiler）可以使用？**

A: 有集成性能分析工具（profiler），请参考 [算子性能调优方法](./debug_guide/profiling.md)。

## 7. UB Overflow常见问题

**Q: 编译时报 "UB Overflow" 错误，如何解决？**

A: UB Overflow是Triton-Ascend开发中常见的问题，请参考 [UB Overflow问题排查指南](./debug_guide/ub_overflow.md) 排查问题。如果不知道如何减小Tiling来减少UB占用，可以使用Autotune来自动选择最优配置，Autotune的使用请参考[Triton-Ascend autotune使用指南](./autotune_guide.md)。
此外，Ascend 950PR&950DT系列产品上可运行的算子迁移到Atlas A2系列产品、Atlas A3系列产品时，由于UB大小的差异可能导致UB Overflow。如果手动排查不出问题，也可采用Autotune自动选择最优配置。

## 8. Triton使用限制

**Q: Triton Kernel中指针参数有哪些使用限制？**

A: Triton-Ascend编译器假设所有外部输入的指针参数本质上指向不同的内存区域，无法识别指针别名（Pointer Alias）场景。当多个指针参数在运行时实际指向同一块内存，但编译器无法获知这一事实时，可能导致优化失效或运行结果异常。例如：

```python
@triton.jit
def func(ptr0, ptr1):
    # load from ptr0 and do something
    # store to ptr0
    # load from ptr1 and do something
    # store to ptr1

in_out_tensor = torch.randn(shape)
func[grid](in_out_tensor, in_out_tensor)
```

上述代码中 `ptr0` 和 `ptr1` 实际指向同一块内存（即同一个 `in_out_tensor`），但编译器无法识别这种指针别名关系，因此这种同一个张量同时作为多个指针参数传入的写法是不受支持的，对应的Kernel将无法启用相关优化。
