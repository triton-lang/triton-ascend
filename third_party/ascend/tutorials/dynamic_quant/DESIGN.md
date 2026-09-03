# DynamicQuant 算子设计方案

## 1. 需求分析

- 功能：Per-token symmetric dynamic quantization on BF16 [B,S,H], output quantized tensor plus FP32 scale.
- 输入：x: BF16 contiguous [B,S,H]; dst_type in {int8,int4}
- 输出：output quantized tensor and FP32 scale [B,S]
- 覆盖：40 public int8/int4 cases plus fixed-seed random generalization cases.

## 2. 实现策略

Triton row/row-stride kernel with chunked large-H path; quantized values are stored with the current Triton-Ascend int8 cast semantics.

被测 candidate 路径只调用本目录 Triton-Ascend 实现，不调用 Torch、torch_npu 高阶等价算子、外部 golden 或历史 baseline 文件。Torch/torch_npu 仅在 `run_inference.py` 校验流程中作为本地 baseline。

## 3. 精度和 baseline 策略

本目录生成 Torch 语义参考和 torch_npu baseline probe。torch_npu 只有在运行成功且通过同一 checker 时才作为 selected baseline；否则 selected baseline 为 Torch semantic。candidate 最终精度始终对 Torch 语义参考判定。

## 4. 性能统计

性能统计来自本目录 `run_inference.py --benchmark --benchmark-torch` 的 torch_npu.profiler kernel_details.csv active-window 计时。主速度验收口径为 `torch_npu runnable all` active/active 几何平均 >= 1.2x；所有 torch_npu 可计时 case 都纳入，精度通过和失败都计入。Torch semantic 计时只作为辅助 Torch 对比，不参与主速度 gate；selected baseline 仅用于语义标杆选择说明。

## 5. 统计汇总

| 指标 | 数值 |
| --- | --- |
| evidence source | logs/dynamic_quant_validation.jsonl |
| total cases | 80 |
| public cases | 40 |
| random/generalization cases | 40 |
| candidate pass | 80/80 |
| main speed sample | torch_npu runnable all (80 cases) |
| main speed candidate active geomean | 4.006 us |
| main speed torch_npu active geomean | 4.879 us |
| main speed active/active geomean speedup | 1.217857x |
| main speed gate | PASS >= 1.2x |
| overall selected baseline split | torch_npu=80 |
| public selected baseline split | torch_npu=40 |
| overall torch_npu runnable all | 80 |
| overall torch_npu accuracy pass/fail | 80/0 |
| torch_npu runnable-all active speedup geomean | 1.217857x |
| torch_npu runnable-all speed gate | PASS >= 1.2x |
| public torch_npu runnable all | 40 |
| public torch_npu accuracy pass/fail | 40/0 |
| aux torch semantic timed all | 80 |
| aux torch semantic candidate active geomean | 4.006 us |
| aux torch semantic baseline active geomean | 226.114 us |
| aux torch semantic active/active geomean speedup | 56.446296x |
| aux public torch semantic timed | 40 |
| aux public torch semantic candidate active geomean | 3.804 us |
| aux public torch semantic baseline active geomean | 200.789 us |
| aux public torch semantic active/active geomean speedup | 52.784675x |
| aux public candidate active geomean | 3.804 us |
| aux public selected baseline active geomean | 4.522 us |
| aux public selected active/active geomean speedup | 1.188669x |
| max candidate RMSE (all outputs) | 0.715618 |
| max candidate output RMSE | 0.715618 |
| max candidate scale RMSE | 0 |
| commercial standard | references/commercial_standard.md @ c260c8ab7a9be4823ac8f8a07c60442de9bf141e |

## 6. 无 fallback / 无 hacking 声明

实现调度只依赖 dtype、rank、shape、属性、contiguity 等合法运行时元数据，不依赖 case id、workload 文件名、输入取值、输出模式或 timing signature。unsupported contract fail loudly。
