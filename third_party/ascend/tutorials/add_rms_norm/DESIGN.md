# AddRmsNorm 算子设计文档

## 1. 设计目标

本实现面向 Ascend NPU 上的 BF16 推理张量，使用 Triton-Ascend 实现 AddRmsNorm。核心语义为：

```text
z = x1 + x2
yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma
```

本交付件只输出 `yOut`。CANN API 中可能存在的 `xOut`、`rstdOut` 等辅助输出不属于本次交付范围。

## 2. 接口定义

```python
add_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float = 1e-6) -> Tensor
```

输入约束：三个输入均为 NPU 上的连续 BF16 三维张量，形状相同，`epsilon` 为正数。输出为 BF16 连续 ND 张量。

## 3. 形状覆盖与分发

公开风格验证覆盖 80 个 case：`B in {1,8,16,32,64}`、`S in {1,8,32,128}`、`H in {3584,4096,5120,8192}`。交付正确性验证还包含固定 seed 随机生成的 40 个非公开 shape case。实现不硬编码公开 case id、随机 case id，也不读取 workload 文件名。

- `H <= 8192`：单个融合 Triton kernel，在一个 kernel 中完成 `x1+x2`、FP32 RMS reduction、`gamma` 乘法和 BF16 写回。
- `next_power_of_2(H) > 8192`：8192 hidden 元素分块 partial-sum/reduce/apply 路径，用于泛化形状。

## 4. Kernel 设计

逻辑张量展平成 `B*S` 行，每行沿 hidden 维做 RMS 归一化。`H <= 8192` 的主路径使用单个 `@triton.jit` kernel，先加载 `x1/x2` 并计算 `z`，完成 FP32 平方和归约与 `rstd` 后再加载 `gamma`，减少归约前寄存器压力。分块路径为更宽 hidden 泛化保留 partial-sum、reduce、apply 三段同后端实现。

## 5. 精度设计

验证脚本复刻 BF16 检查口径：阈值 `2^-7 = 0.0078125`，普通位置要求 `MERE < threshold` 且 `MARE < 10 * threshold`，小值或抵消位置使用 checker 的 ErrorCount 规则。`eval_20260615_154048` 显示 80/80 公开 case 精度通过，mismatch 总数为 0，最大 MARE 为 `0.00781249960938`，最大 diff 为 `0.015625`。

当前交付目录的 `logs/add_rms_norm_random_generalization_20260615.jsonl` 记录了 80 个 public case 加 40 个 fixed-seed random generalization case，Triton 候选为 120/120 通过，其中 random generalization 为 40/40 通过。

## 6. 无 fallback 边界

被测 AddRmsNorm 函数只使用本地 Triton-Ascend JIT kernel。Python 封装只做元数据校验、输出/工作区分配和 launch 参数组织，不在 Python 中计算被测结果，也不调用任务 golden/reference、PyTorch 等价实现、`torch_npu.npu_add_rms_norm`、CANN/vendor AddRmsNorm、CPU fallback、peer 或其他后端代码。

## 7. 当前性能证据

性能主证据以 OpForge `eval_20260615_154048` 为准。主口径为 active-window，kernel-sum 只作为诊断。

| 指标 | 数值 |
| --- | --- |
| OpForge run id | `eval_20260615_154048` |
| OpForge 状态 | `PERF_REGRESSION` |
| public 正确性 | 80/80 PASS |
| active/active geomean | 15.578318x |
| kernel/kernel geomean | 9.171818x |
| candidate active mean | 50.138 us |
| candidate kernel mean | 50.130 us |
| candidate mean gap | 0.036 us |
| active regression cases | 2 |
| baseline source split | task_npu_baseline=47, pytorch_fallback=33 |
| OpForge generalization audit | 40/40 PASS |
| delivery correctness | 120/120 PASS; public 80/80, random 40/40 |
| delivery benchmark timing completeness | Triton active 74/80; missing cases 75-80 due blank Step Id |

OpForge summary 的 baseline split 为 `task_npu_baseline=47`、`pytorch_fallback=33`，timing source 均为 `torch_npu.profiler.kernel_details.csv`。`active_regression_cases=2`，因此当前状态仍为 `PERF_REGRESSION`。

## 8. 风险与后续工作

- 官方 full-public run 中 cases 1 和 3 对选中基线存在 active-window regression，是 `PERF_REGRESSION` 的主要原因。
- delivery `--benchmark` 日志在 cases 75-80 的候选 profiler CSV 上遇到 blank `Step Id`，严格策略下写为 `N/A`；完整性能数字引用官方 OpForge traces。
- `torch_npu` helper 只在交付脚本的 61/80 public case 上通过同一 BF16 精度检查，不能作为 80 case 全覆盖 helper 结论。
- random generalization 只作为正确性泛化证据，不替代 80 public timing matrix 或官方 OpForge public scoring。
