# KvRmsNormRopeCache 算子设计方案

## 1. 需求分析

- 功能：Decode KV split, RoPE, RMSNorm, and cache update; outputs updated K cache and CKV cache.
- 输入：kv/gamma/cos/sin/index/cache tensors; BF16 data path with dynamic Dk/Dv validation
- 输出：updated K cache and CKV cache tensors
- 覆盖：20 public cases plus fixed-seed random dynamic-split cases.

## 2. 实现策略

Fast 64/64 split path plus generic dynamic Dk/Dv split path; unsupported metadata fails loudly.

被测 candidate 路径只调用本目录 Triton-Ascend 实现，不调用 Torch、torch_npu 高阶等价算子、外部 golden 或历史 baseline 文件。Torch/torch_npu 仅在 `run_inference.py` 校验流程中作为本地 baseline。

## 3. 精度和 baseline 策略

本目录生成 Torch 语义参考和 torch_npu baseline probe。torch_npu 只有在运行成功且通过同一 checker 时才作为 selected baseline；否则 selected baseline 为 Torch semantic。candidate 最终精度始终对 Torch 语义参考判定。

## 4. 性能统计

性能统计来自本目录 `run_inference.py --benchmark --benchmark-torch` 的 torch_npu.profiler kernel_details.csv active-window 计时。主速度验收口径为 `torch_npu runnable all` active/active 几何平均 >= 1.2x；所有 torch_npu 可计时 case 都纳入，精度通过和失败都计入。Torch semantic 计时只作为辅助 Torch 对比，不参与主速度 gate；selected baseline 仅用于语义标杆选择说明。

## 5. 统计汇总

| 指标 | 数值 |
| --- | --- |
| evidence source | logs/kv_rms_norm_rope_cache_validation.jsonl |
| total cases | 60 |
| public cases | 20 |
| random/generalization cases | 40 |
| candidate pass | 60/60 |
| main speed sample | torch_npu runnable all (0 cases) |
| main speed candidate active geomean | N/A |
| main speed torch_npu active geomean | N/A |
| main speed active/active geomean speedup | N/A |
| main speed gate | N/A (no torch_npu runnable timed case) |
| overall selected baseline split | torch=60 |
| public selected baseline split | torch=20 |
| overall torch_npu runnable all | 0 |
| overall torch_npu accuracy pass/fail | 0/0 |
| torch_npu runnable-all active speedup geomean | N/A |
| torch_npu runnable-all speed gate | N/A (no torch_npu runnable timed case) |
| public torch_npu runnable all | 0 |
| public torch_npu accuracy pass/fail | 0/0 |
| aux torch semantic timed all | 57 |
| aux torch semantic candidate active geomean | 105.836 us |
| aux torch semantic baseline active geomean | 3519.494 us |
| aux torch semantic active/active geomean speedup | 32.082368x |
| aux public torch semantic timed | 20 |
| aux public torch semantic candidate active geomean | 76.228 us |
| aux public torch semantic baseline active geomean | 2423.042 us |
| aux public torch semantic active/active geomean speedup | 31.067939x |
| aux public candidate active geomean | 76.228 us |
| aux public selected baseline active geomean | 2423.042 us |
| aux public selected active/active geomean speedup | 31.067939x |
| max candidate RMSE | 1.9895e-05 |
| commercial standard | references/commercial_standard.md @ c260c8ab7a9be4823ac8f8a07c60442de9bf141e |

## 6. 无 fallback / 无 hacking 声明

实现调度只依赖 dtype、rank、shape、属性、contiguity 等合法运行时元数据，不依赖 case id、workload 文件名、输入取值、输出模式或 timing signature。unsupported contract fail loudly。
