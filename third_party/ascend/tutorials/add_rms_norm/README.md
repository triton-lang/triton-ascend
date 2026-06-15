# AddRmsNorm Triton-Ascend

本目录存放 AddRmsNorm 算子的 Triton-Ascend 交付材料，内容按当前私有 OpForge/CANN-Bench 口径整理。

## 需求概述

- 后端：Ascend NPU 上的 Triton-Ascend。
- 算子：`addRmsNorm`。
- 输入：`x1`、`x2`、`gamma` 均为 BF16 连续 ND 张量，形状 `[B, S, H]`。
- 输出：`yOut`，BF16 连续 ND 张量，形状 `[B, S, H]`。
- 公式：`z = x1 + x2; yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma`。

公开风格验证覆盖 `B in {1, 8, 16, 32, 64}`、`S in {1, 8, 32, 128}`、`H in {3584, 4096, 5120, 8192}`，共 80 个 case。交付验证另含固定 seed 的随机泛化 case，用于检查非公开 shape 的正确性。

## 文件说明

- `add_rms_norm.py`：Triton-Ascend kernel 及 Python 调用封装。
- `validate_add_rms_norm.py`：功能验证脚本，并支持 CANN-Bench 对齐的三路双时间采集。
- `DESIGN.md`：算子设计说明。
- `SELF_VALIDATION_REPORT.md`：自验证报告、80 public 性能明细和随机泛化明细。
- `AddRmsNorm算子设计方案.docx`：设计方案模板填充件。
- `AddRmsNorm算子自验证报告.xlsx`：80 public 自验证表和随机泛化工作表。
- `OPFORGE_EVIDENCE.json`：结构化证据副本。
- `logs/add_rms_norm_timing_matrix_20260615.jsonl`：本地 delivery benchmark 日志，精度完整，严格 profiler CSV 解析会把坏 Step Id 行记录为 `N/A`。
- `logs/add_rms_norm_random_generalization_20260615.jsonl`：80 public 加 40 random generalization 的正确性验证记录。

## 运行验证

```bash
python3 validate_add_rms_norm.py --public --random-generalization 40 --random-seed 20260613
```

三路性能采集命令：

```bash
python3 validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5   --jsonl logs/add_rms_norm_timing_matrix_$(date +%Y%m%d_%H%M%S).jsonl   --profiler-data-dir logs/prof_data_timing_matrix
```

正确性日志不采集 profiler，所以 `active=N/A` 是预期现象。需要非 `N/A` 的 timing 字段必须加 `--benchmark`。如果 benchmark 日志仍出现 `N/A`，含义是对应 profiler CSV 未通过严格解析；本次 `20260615` delivery benchmark 中候选 cases 75-80 因 blank `Step Id` 被拒绝，官方 OpForge traces 仍提供完整 80 case 计时。

## 计时口径

当前主比较使用 **active-window**：每个 Step Id 内可见 NPU kernel 的 `max(end)-min(start)`，case 时间取 measured step 的中位数。`kernel-sum` 只作为诊断字段。性能主证据来自 OpForge `eval_20260615_154048` 的 `timing_artifacts/main_results.csv` 和 `summary.json`。

## 当前证据

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

`torch_npu` helper 在交付正确性脚本中通过 61/80 个 public case；未通过或未能完整 profiler 的 case 不作为 helper 全覆盖结论。OpForge 选中基线为 `task_npu_baseline` 47 case、`pytorch_fallback` 33 case。状态仍为 `PERF_REGRESSION`，不能表述为性能门完全通过。

## 实现边界

实现根据运行时 dtype/rank/shape/contiguity 和 hidden 维能力边界选择路径，不根据 case id、workload 文件名、公开输入值、观测输出或计时特征分支。被测函数内不调用 PyTorch 等价计算、`torch_npu.npu_add_rms_norm`、CANN/vendor AddRmsNorm、CPU 代码或任务 golden 路径。
