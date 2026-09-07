# Triton-Ascend `do_bench_npu` 指导文档

## 文档定位

本指南面向已经掌握如何编写 Triton 算子，并已了解社区 `triton.testing.do_bench` 基本概念的用户。主要介绍在昇腾 NPU 上进行性能基准测试的推荐用法：

- Triton-Ascend 上 `do_bench_npu` 的推荐用法；
- 在 Ascend 后端中“纯 Device 侧执行时间”的含义；
- 何时使用默认的 `mspti` 路径，以及何时需要回退到 `torch_npu.profiler` 路径。

## 快速上手

在 Triton-Ascend 中，当您需要构建算子性能基线或测量 kernel 的纯硬件执行时间时，推荐导入并使用 `do_bench_npu`：

```python
import torch
import torch_npu
from triton.backends.ascend.testing import do_bench_npu

@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    out_ptr
):

def fn():
    kernel[grid](x, y, out)

ms = do_bench_npu(fn)
print(f"Kernel execution time: {ms} ms")
```

这表示：

- `fn` 的语义与社区 `do_bench` 相同，代表要进行基准测试的可调用函数；
- `do_bench_npu` 在 Triton-Ascend 上的意思是“让 NPU 性能分析工具捕获纯 Device 侧 kernel 执行时间”，从而有效消除 Host 侧开销对测量的影响。

## 前提条件

### 1. 理解 Host 侧和 Device 侧计时的区别

社区 `triton.testing.do_bench` 使用`Event`来测量时间，记录的开始和结束时间会包含 Host 侧启动开销。

`do_bench_npu` 利用 NPU 性能分析工具（`MSPTI` 或 `torch_npu.profiler`）,保证返回的时间严格是 Device 侧执行时间。

### 2. 快速路径（`mspti`）的环境要求

为了实现与社区 `do_bench` 相当的运行速度，`do_bench_npu` 默认尝试使用轻量级的 `mspti` 库。要启用此功能：

- 必须可以导入 `mspti` 包。
- 如果 CANN 版本低于 9.1.0，必须在 LD_PRELOAD 环境变量中设置 `libmspti.so`。否则，系统将抛出 `RuntimeError` 提示您进行设置。
- 如果 `mspti` 不可用，`do_bench_npu` 将打印警告并自动回退到 `torch_npu.profiler` 路径，该路径较慢但功能完整。

## 使用说明

### 1. 传入单个函数与多个函数的区别

您可以传入单个可调用对象，也可以传入可调用对象的列表：

```python
# 单个函数: 返回浮点数
time_ms = do_bench_npu(fn_A)

# 多个函数: 返回浮点数列表
times_ms = do_bench_npu([fn_A, fn_B, fn_C])
```

约束： 当传入函数列表时，`do_bench_npu` 只有在每个函数只包含一个 kernel 时，才能准确返回每个函数的 Device 侧时间。如果函数包含多个 kernel，`do_bench_npu` 无法正确区分哪个 kernel 属于哪个函数。请避免测试复杂函数的列表。

### 2. 清除 L2 Cache

要测量最坏情况下的性能或模拟冷启动内存场景，可以在每次迭代前清除 L2 缓存：

```python
ms = do_bench_npu(fn, clear_l2_cache=True)
```

启用后，`do_bench_npu` 将在每次 fn 调用前执行一个虚拟缓冲区读取操作。 `do_bench_npu` 会自动过滤掉此缓存清除操作消耗的时间。

### 3. 按名称对特定 kernel 进行基准测试

如果您的 `fn` 包含多个 kernel，但您只想测量其中一个的执行时间，请使用 `target_kernel_name` 参数：

```python
ms = do_bench_npu(fn, target_kernel_name="triton_matmul_kernel")
```

指定 `target_kernel_name` 后，`do_bench_npu` 将强制回退到 `torch_npu.profiler` 路径。profiler 将解析详细的 CSV 并严格过滤确切的 kernel 名称。
如果捕获到的 kernel 数量与预期的 `(warmup + active)` 迭代次数不匹配，它将抛出 `ProfilerResultMismatchError` 以确保数据完整性。

### 4. 保留原始 profiler 结果

如果您正在调试性能异常，需要检查原始 profiler 输出文件（如 `task_time*.csv` 或 `kernel_details.csv`），请使用：

```python
ms = do_bench_npu(fn, prof_dir="./my_prof_results", keep_res=True)
```

设置 `prof_dir` 或 `keep_res=True` 中的任意一个都将强制使用 `torch_npu.profiler` 路径，因为 `mspti` 不会写入这些 CSV 文件。

### 5. 设置预热和运行次数

可以自主传入预热和实际运行次数：

```python
ms = do_bench_npu(fn, warmup=1, active=1)
```

`warmup`代表在实际计时前运行fn的次数，用于预热。`active`代表fn实际的运行次数，最终结果是这active次迭代的平均值。

## 计时收集的范围与行为

`do_bench_npu` 相对于社区 `do_bench` 的核心扩展集中在准确性和硬件亲和性上。

### 1. 纯 Device 侧时间收集

收集机制聚焦于片上执行时间。此功能不记录 Host 侧开销或启动延迟。它保证返回的时间严格是 Device 侧 kernel 执行时间的总和。

### 2. 自动过滤缓存清除操作

当使用 `clear_l2_cache=True` 时，profiler 会自动识别并过滤掉用于清除缓存的虚拟 kernel（例如包含 “ReduceSum” 或 “zero” 的操作），确保它们不会污染最终的计时计算。

### 3. 单function中含有多kernel的时间收集

在实际 NPU 执行中，单次 Triton kernel 启动可能会被底层调度程序拆分为多个硬件任务。`do_bench_npu` 会自动检测捕获到的实际行数（任务数）是否为预期迭代次数的倍数。如果是，它会计算倍数并正确平均时间，从而返回逻辑函数的准确总时间。

## 高级用法：与 Triton Autotune 集成

Triton-Ascend 的 Autotune 扩展支持切换 autotune 基准测试期间使用的性能收集方法。通过设置环境变量 `TRITON_BENCH_METHOD="npu"`，Autotune 机制将在内部调用 `do_bench_npu` 来评估候选配置。

```python
export TRITON_BENCH_METHOD="npu"
```

对于执行时间短的 kernel，建议使用此方法，否则 Host 侧启动延迟将主导社区 `do_bench` 的测量结果，从而导致选择配置非device侧最优。

## 总结

`do_bench_npu` 相对于社区 `do_bench` 的关键扩展在于从基于事件的计时过渡到基于 profiler 的计时。对于大多数用户，推荐用法是：

- 调用  `do_bench_npu(fn)` 获取纯 Device 侧执行时间；
- 如果需要在复杂函数中隔离特定 kernel，请使用 `target_kernel_name`；
- 如果需要稳定的最坏情况性能基线，请启用 `clear_l2_cache=True`；
- 确保安装了 `mspti` ，以保持与社区 `do_bench` 的运行时性能持平。
