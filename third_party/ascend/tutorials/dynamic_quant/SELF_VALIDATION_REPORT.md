# DynamicQuant 算子自验证报告

## 1. 报告说明

- 单一数值证据源：`logs/dynamic_quant_validation.jsonl`
- 本报告由当前目录 `generate_delivery.py` 生成，只读取本目录 logs/templates/references。
- L1 是商业精度等级，不是 L1 norm；本表对齐商业标准中的 MARE/MERE/RMSE 指标口径。
- 本报告是本目录自验证，不等同于完整商业 L1 认证；完整商业认证还要求标准规定的用例规模和执行轮次。
- RMSE/MERE/MARE 由本目录 checker 输出。
- 速度门槛按 `torch_npu runnable all` active/active 几何平均 >= 1.2x；无 torch_npu 可计时 case 标记为 N/A，不用 selected baseline 或 Torch semantic timing 代替。
- Torch semantic timing 由 `--benchmark-torch` 显式开启，只作为辅助 Torch 速度对比。
- 截图证据未外置，日志内容嵌入 XLSX `日志证据` 工作表。
- 不保留、不读取外部历史对照文件。

## 2. 性能总体对比

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

## 3. 性能口径汇总

| Scope | Cases | Candidate active geomean | Baseline active geomean | Active/active geomean speedup | Precision pass | Note |
| --- | --- | --- | --- | --- | --- | --- |
| main torch_npu timed sample | 80 | 4.006 us | 4.879 us | 1.217857x | 80/80 | 主速度验收口径；candidate 和 torch_npu 均只在这同一批有 torch_npu active 计时的 case 上取几何平均；gate >= 1.2x |
| torch_npu accuracy-pass | 80 | 4.006 us | 4.879 us | 1.217857x | 80/80 | 全量 torch_npu 有效计时且本地 checker PASS 子集 |
| torch_npu accuracy-fail | 0 | N/A | N/A | N/A | 0/0 | 全量 torch_npu 有效计时但本地 checker FAIL 子集 |
| aux all torch semantic timed baseline | 80 | 4.006 us | 226.114 us | 56.446296x | 80/80 | 补充 Torch 语义参考计时；由 --benchmark-torch 开启，不参与主速度 gate |
| aux public torch semantic timed baseline | 40 | 3.804 us | 200.789 us | 52.784675x | 40/40 | 补充 public Torch 语义参考计时；由 --benchmark-torch 开启，不参与主速度 gate |
| aux public selected baseline | 40 | 3.804 us | 4.522 us | 1.188669x | 40/40 | 补充语义标杆口径；torch_npu 仅在本地 checker 通过时选中，否则选 Torch |
| aux public torch semantic correctness baseline | 40 | 3.804 us | N/A | N/A | 40/40 | Torch 语义参考覆盖全部 public case；latency/speedup 只看上面的 timed baseline 行 |

## 4. Baseline 校验明细

| Case | Selected implementation | Selection rule | Torch pass | torch_npu runnable | torch_npu pass | torch_npu MERE | torch_npu MARE | torch_npu RMSE | torch_npu max diff | Reason | Seed/attrs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/dynamic_quant_1 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 7.87402e-09 | 7.87402e-09 | 7.87402e-15 | 7.87402e-15 |  | {"seed": 290946555, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_2 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 517281410, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_3 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1553757808, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_4 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1519782940, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_5 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1693678812, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_6 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 295208901, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_7 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 715257418, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_8 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 7.87402e-09 | 7.87402e-09 | 7.87402e-15 | 7.87402e-15 |  | {"seed": 1533416434, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_9 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 788628000, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_10 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 958483133, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_11 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1953412913, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_12 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 861259166, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_13 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 603824714, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_14 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1234808994, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_15 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 7.87402e-09 | 7.87402e-09 | 7.87402e-15 | 7.87402e-15 |  | {"seed": 1122073957, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_16 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1572320534, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_17 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 484232680, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_18 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1107036625, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_19 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 366523540, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_20 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 9819431, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int8"}, "dst_type": "int8", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_21 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 1.42857e-07 | 1.42857e-07 | 1.42857e-13 | 1.42857e-13 |  | {"seed": 1872290961, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_22 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 754747769, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_23 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1497542212, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_24 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 525113986, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_25 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 833686457, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_26 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1169915283, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_27 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 574724603, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_28 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 1.42857e-07 | 1.42857e-07 | 1.42857e-13 | 1.42857e-13 |  | {"seed": 1477014784, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_29 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 607314940, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_30 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 437729554, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_31 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 977094068, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_32 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 516721451, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_33 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1170163291, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_34 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1909845499, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_35 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 1.42857e-07 | 1.42857e-07 | 1.42857e-13 | 1.42857e-13 |  | {"seed": 1543752005, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_36 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 817577730, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/dynamic_quant_37 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 783582863, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/dynamic_quant_38 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 654189849, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/dynamic_quant_39 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 543681792, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/dynamic_quant_40 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 156727975, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1, "dst_type": "int4"}, "dst_type": "int4", "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |

## 5. Public 逐Case速度

| Case | Kind | Shape | DType | Selected baseline | Triton active | Torch active | torch_npu active | Selected active speedup | Torch active speedup | torch_npu active speedup | Triton precision | Torch precision | torch_npu precision | MERE | MARE | RMSE | Max diff | torch_npu error/note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/dynamic_quant_1 | public | [1, 1, 3584] | bfloat16 | torch_npu | 2.500 us | 151.250 us | 2.750 us | 1.100000x | 60.500000x | 1.100000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/dynamic_quant_2 | public | [1, 1, 4096] | bfloat16 | torch_npu | 2.500 us | 158.500 us | 3.000 us | 1.200000x | 63.400000x | 1.200000x | PASS | PASS | PASS | 0 | 0 | 0.703993 | 1 |  |
| custom/dynamic_quant_3 | public | [1, 1, 5120] | bfloat16 | torch_npu | 2.750 us | 156.750 us | 3.000 us | 1.090909x | 57.000000x | 1.090909x | PASS | PASS | PASS | 0 | 0 | 0.715618 | 1 |  |
| custom/dynamic_quant_4 | public | [1, 1, 8192] | bfloat16 | torch_npu | 3.750 us | 158.250 us | 2.500 us | 0.666667x | 42.200000x | 0.666667x | PASS | PASS | PASS | 0 | 0 | 0.704686 | 1 |  |
| custom/dynamic_quant_5 | public | [8, 1, 3584] | bfloat16 | torch_npu | 3.750 us | 138.750 us | 3.750 us | 1.000000x | 37.000000x | 1.000000x | PASS | PASS | PASS | 0 | 0 | 0.705082 | 1 |  |
| custom/dynamic_quant_6 | public | [8, 1, 4096] | bfloat16 | torch_npu | 3.750 us | 230.500 us | 5.000 us | 1.333333x | 61.466667x | 1.333333x | PASS | PASS | PASS | 0 | 0 | 0.71201 | 1 |  |
| custom/dynamic_quant_7 | public | [8, 1, 5120] | bfloat16 | torch_npu | 3.750 us | 257.750 us | 5.000 us | 1.333333x | 68.733333x | 1.333333x | PASS | PASS | PASS | 0 | 0 | 0.711564 | 1 |  |
| custom/dynamic_quant_8 | public | [8, 1, 8192] | bfloat16 | torch_npu | 4.750 us | 195.750 us | 4.250 us | 0.894737x | 41.210526x | 0.894737x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/dynamic_quant_9 | public | [16, 1, 3584] | bfloat16 | torch_npu | 3.000 us | 209.500 us | 3.750 us | 1.250000x | 69.833333x | 1.250000x | PASS | PASS | PASS | 0 | 0 | 0.707057 | 1 |  |
| custom/dynamic_quant_10 | public | [16, 1, 4096] | bfloat16 | torch_npu | 3.000 us | 195.500 us | 4.500 us | 1.500000x | 65.166667x | 1.500000x | PASS | PASS | PASS | 0 | 0 | 0.70742 | 1 |  |
| custom/dynamic_quant_11 | public | [16, 1, 5120] | bfloat16 | torch_npu | 3.000 us | 141.250 us | 4.500 us | 1.500000x | 47.083333x | 1.500000x | PASS | PASS | PASS | 0 | 0 | 0.709958 | 1 |  |
| custom/dynamic_quant_12 | public | [16, 1, 8192] | bfloat16 | torch_npu | 4.250 us | 225.000 us | 4.500 us | 1.058824x | 52.941176x | 1.058824x | PASS | PASS | PASS | 0 | 0 | 0.707727 | 1 |  |
| custom/dynamic_quant_13 | public | [32, 1, 3584] | bfloat16 | torch_npu | 4.000 us | 160.750 us | 4.750 us | 1.187500x | 40.187500x | 1.187500x | PASS | PASS | PASS | 0 | 0 | 0.707828 | 1 |  |
| custom/dynamic_quant_14 | public | [32, 1, 4096] | bfloat16 | torch_npu | 3.750 us | 242.500 us | 5.250 us | 1.400000x | 64.666667x | 1.400000x | PASS | PASS | PASS | 0 | 0 | 0.707932 | 1 |  |
| custom/dynamic_quant_15 | public | [32, 1, 5120] | bfloat16 | torch_npu | 3.750 us | 211.750 us | 5.000 us | 1.333333x | 56.466667x | 1.333333x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/dynamic_quant_16 | public | [32, 1, 8192] | bfloat16 | torch_npu | 5.000 us | 189.500 us | 7.000 us | 1.400000x | 37.900000x | 1.400000x | PASS | PASS | PASS | 0 | 0 | 0.705973 | 1 |  |
| custom/dynamic_quant_17 | public | [64, 1, 3584] | bfloat16 | torch_npu | 4.750 us | 197.000 us | 5.750 us | 1.210526x | 41.473684x | 1.210526x | PASS | PASS | PASS | 0 | 0 | 0.707708 | 1 |  |
| custom/dynamic_quant_18 | public | [64, 1, 4096] | bfloat16 | torch_npu | 5.000 us | 284.250 us | 5.750 us | 1.150000x | 56.850000x | 1.150000x | PASS | PASS | PASS | 0 | 0 | 0.708842 | 1 |  |
| custom/dynamic_quant_19 | public | [64, 1, 5120] | bfloat16 | torch_npu | 5.000 us | 219.750 us | 6.500 us | 1.300000x | 43.950000x | 1.300000x | PASS | PASS | PASS | 0 | 0 | 0.707025 | 1 |  |
| custom/dynamic_quant_20 | public | [64, 1, 8192] | bfloat16 | torch_npu | 7.500 us | 194.750 us | 9.500 us | 1.266667x | 25.966667x | 1.266667x | PASS | PASS | PASS | 0 | 0 | 0.708947 | 1 |  |
| custom/dynamic_quant_21 | public | [1, 1, 3584] | bfloat16 | torch_npu | 2.500 us | 158.500 us | 2.750 us | 1.100000x | 63.400000x | 1.100000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/dynamic_quant_22 | public | [1, 1, 4096] | bfloat16 | torch_npu | 2.500 us | 237.250 us | 2.750 us | 1.100000x | 94.900000x | 1.100000x | PASS | PASS | PASS | 0 | 0 | 0.714321 | 1 |  |
| custom/dynamic_quant_23 | public | [1, 1, 5120] | bfloat16 | torch_npu | 2.500 us | 161.500 us | 2.750 us | 1.100000x | 64.600000x | 1.100000x | PASS | PASS | PASS | 0 | 0 | 0.7089 | 1 |  |
| custom/dynamic_quant_24 | public | [1, 1, 8192] | bfloat16 | torch_npu | 3.750 us | 237.250 us | 3.000 us | 0.800000x | 63.266667x | 0.800000x | PASS | PASS | PASS | 0 | 0 | 0.702604 | 1 |  |
| custom/dynamic_quant_25 | public | [8, 1, 3584] | bfloat16 | torch_npu | 3.500 us | 221.750 us | 4.000 us | 1.142857x | 63.357143x | 1.142857x | PASS | PASS | PASS | 0 | 0 | 0.708585 | 1 |  |
| custom/dynamic_quant_26 | public | [8, 1, 4096] | bfloat16 | torch_npu | 3.750 us | 192.750 us | 4.250 us | 1.133333x | 51.400000x | 1.133333x | PASS | PASS | PASS | 0 | 0 | 0.709326 | 1 |  |
| custom/dynamic_quant_27 | public | [8, 1, 5120] | bfloat16 | torch_npu | 3.750 us | 181.750 us | 4.500 us | 1.200000x | 48.466667x | 1.200000x | PASS | PASS | PASS | 0 | 0 | 0.706813 | 1 |  |
| custom/dynamic_quant_28 | public | [8, 1, 8192] | bfloat16 | torch_npu | 4.750 us | 181.500 us | 4.250 us | 0.894737x | 38.210526x | 0.894737x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/dynamic_quant_29 | public | [16, 1, 3584] | bfloat16 | torch_npu | 3.000 us | 249.750 us | 4.250 us | 1.416667x | 83.250000x | 1.416667x | PASS | PASS | PASS | 0 | 0 | 0.706749 | 1 |  |
| custom/dynamic_quant_30 | public | [16, 1, 4096] | bfloat16 | torch_npu | 3.000 us | 236.750 us | 4.500 us | 1.500000x | 78.916667x | 1.500000x | PASS | PASS | PASS | 0 | 0 | 0.707042 | 1 |  |
| custom/dynamic_quant_31 | public | [16, 1, 5120] | bfloat16 | torch_npu | 3.000 us | 163.750 us | 4.500 us | 1.500000x | 54.583333x | 1.500000x | PASS | PASS | PASS | 0 | 0 | 0.705941 | 1 |  |
| custom/dynamic_quant_32 | public | [16, 1, 8192] | bfloat16 | torch_npu | 4.250 us | 236.250 us | 4.500 us | 1.058824x | 55.588235x | 1.058824x | PASS | PASS | PASS | 0 | 0 | 0.704496 | 1 |  |
| custom/dynamic_quant_33 | public | [32, 1, 3584] | bfloat16 | torch_npu | 3.500 us | 238.500 us | 5.000 us | 1.428571x | 68.142857x | 1.428571x | PASS | PASS | PASS | 0 | 0 | 0.707341 | 1 |  |
| custom/dynamic_quant_34 | public | [32, 1, 4096] | bfloat16 | torch_npu | 3.750 us | 379.500 us | 4.750 us | 1.266667x | 101.200000x | 1.266667x | PASS | PASS | PASS | 0 | 0 | 0.706389 | 1 |  |
| custom/dynamic_quant_35 | public | [32, 1, 5120] | bfloat16 | torch_npu | 3.500 us | 189.750 us | 4.750 us | 1.357143x | 54.214286x | 1.357143x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/dynamic_quant_36 | public | [32, 1, 8192] | bfloat16 | torch_npu | 5.000 us | 170.000 us | 7.000 us | 1.400000x | 34.000000x | 1.400000x | PASS | PASS | PASS | 0 | 0 | 0.708037 | 1 |  |
| custom/dynamic_quant_37 | public | [64, 1, 3584] | bfloat16 | torch_npu | 4.750 us | 182.750 us | 5.250 us | 1.105263x | 38.473684x | 1.105263x | PASS | PASS | PASS | 0 | 0 | 0.707258 | 1 |  |
| custom/dynamic_quant_38 | public | [64, 1, 4096] | bfloat16 | torch_npu | 5.000 us | 189.500 us | 6.000 us | 1.200000x | 37.900000x | 1.200000x | PASS | PASS | PASS | 0 | 0 | 0.706537 | 1 |  |
| custom/dynamic_quant_39 | public | [64, 1, 5120] | bfloat16 | torch_npu | 5.250 us | 281.250 us | 6.000 us | 1.142857x | 53.571429x | 1.142857x | PASS | PASS | PASS | 0 | 0 | 0.708368 | 1 |  |
| custom/dynamic_quant_40 | public | [64, 1, 8192] | bfloat16 | torch_npu | 7.750 us | 208.000 us | 9.250 us | 1.193548x | 26.838710x | 1.193548x | PASS | PASS | PASS | 0 | 0 | 0.706784 | 1 |  |

## 6. 商业L1精度对比

| Case | Output | DType | Shape | Reference | Criterion | Candidate AE | Candidate MARE | Candidate MERE | Candidate RMSE | Baseline AE | Baseline MARE | Baseline MERE | Baseline RMSE | L1 metric status | Checker status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/dynamic_quant_1 | output | torch.int8 | [1, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_1 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 7.87402e-15 | 7.87402e-09 | 7.87402e-09 | 7.87402e-15 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_2 | output | torch.int8 | [1, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.703993 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_2 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_3 | output | torch.int8 | [1, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.715618 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_3 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_4 | output | torch.int8 | [1, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.704686 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_4 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_5 | output | torch.int8 | [8, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.705082 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_5 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_6 | output | torch.int8 | [8, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.71201 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_6 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_7 | output | torch.int8 | [8, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.711564 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_7 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_8 | output | torch.int8 | [8, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_8 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 7.87402e-15 | 7.87402e-09 | 7.87402e-09 | 7.87402e-15 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_9 | output | torch.int8 | [16, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707057 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_9 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_10 | output | torch.int8 | [16, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.70742 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_10 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_11 | output | torch.int8 | [16, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.709958 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_11 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_12 | output | torch.int8 | [16, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707727 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_12 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_13 | output | torch.int8 | [32, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707828 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_13 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_14 | output | torch.int8 | [32, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707932 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_14 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_15 | output | torch.int8 | [32, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_15 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 7.87402e-15 | 7.87402e-09 | 7.87402e-09 | 7.87402e-15 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_16 | output | torch.int8 | [32, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.705973 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_16 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_17 | output | torch.int8 | [64, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707708 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_17 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_18 | output | torch.int8 | [64, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.708842 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_18 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_19 | output | torch.int8 | [64, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707025 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_19 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_20 | output | torch.int8 | [64, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.708947 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_20 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_21 | output | torch.int8 | [1, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_21 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 1.42857e-13 | 1.42857e-07 | 1.42857e-07 | 1.42857e-13 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_22 | output | torch.int8 | [1, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.714321 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_22 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_23 | output | torch.int8 | [1, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.7089 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_23 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_24 | output | torch.int8 | [1, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.702604 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_24 | scale | torch.float32 | [1, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_25 | output | torch.int8 | [8, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.708585 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_25 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_26 | output | torch.int8 | [8, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.709326 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_26 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_27 | output | torch.int8 | [8, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.706813 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_27 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_28 | output | torch.int8 | [8, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_28 | scale | torch.float32 | [8, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 1.42857e-13 | 1.42857e-07 | 1.42857e-07 | 1.42857e-13 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_29 | output | torch.int8 | [16, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.706749 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_29 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_30 | output | torch.int8 | [16, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707042 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_30 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_31 | output | torch.int8 | [16, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.705941 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_31 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_32 | output | torch.int8 | [16, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.704496 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_32 | scale | torch.float32 | [16, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_33 | output | torch.int8 | [32, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707341 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_33 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_34 | output | torch.int8 | [32, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.706389 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_34 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_35 | output | torch.int8 | [32, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_35 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 1.42857e-13 | 1.42857e-07 | 1.42857e-07 | 1.42857e-13 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_36 | output | torch.int8 | [32, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.708037 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_36 | scale | torch.float32 | [32, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_37 | output | torch.int8 | [64, 1, 3584] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.707258 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_37 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_38 | output | torch.int8 | [64, 1, 4096] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.706537 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_38 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_39 | output | torch.int8 | [64, 1, 5120] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.708368 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_39 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_40 | output | torch.int8 | [64, 1, 8192] | Torch semantic reference generated by this directory | quantized integer AE <= 1 | 1 | 0 | 0 | 0.706784 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/dynamic_quant_40 | scale | torch.float32 | [64, 1] | Torch semantic reference generated by this directory | floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |

## 7. 随机泛化明细

| Case | Shape | Category/dst | Seed | Status | Mismatch | Max diff | MERE | MARE | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/dynamic_quant_random_001 | [1, 1, 8192] | int8 | 1015585109 | PASS | 0 | 1 | 0 | 0 | 0.712267 |
| custom/dynamic_quant_random_002 | [1, 1, 5120] | int4 | 1822950086 | PASS | 0 | 1 | 0 | 0 | 0.706969 |
| custom/dynamic_quant_random_003 | [64, 1, 4096] | int8 | 583040935 | PASS | 0 | 1 | 0 | 0 | 0.70715 |
| custom/dynamic_quant_random_004 | [8, 1, 5120] | int4 | 1099779966 | PASS | 0 | 1 | 0 | 0 | 0.707469 |
| custom/dynamic_quant_random_005 | [32, 1, 8192] | int8 | 926932448 | PASS | 0 | 1 | 0 | 0 | 0.708223 |
| custom/dynamic_quant_random_006 | [64, 1, 3584] | int4 | 1095327137 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/dynamic_quant_random_007 | [32, 1, 8192] | int8 | 413481519 | PASS | 0 | 1 | 0 | 0 | 0.708209 |
| custom/dynamic_quant_random_008 | [8, 1, 4096] | int4 | 483908426 | PASS | 0 | 1 | 0 | 0 | 0.707215 |
| custom/dynamic_quant_random_009 | [64, 1, 5120] | int8 | 119331736 | PASS | 0 | 1 | 0 | 0 | 0.70704 |
| custom/dynamic_quant_random_010 | [32, 1, 3584] | int4 | 1678906645 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/dynamic_quant_random_011 | [8, 1, 8192] | int8 | 711742197 | PASS | 0 | 1 | 0 | 0 | 0.706113 |
| custom/dynamic_quant_random_012 | [64, 1, 3584] | int4 | 1484537696 | PASS | 0 | 1 | 0 | 0 | 0.706965 |
| custom/dynamic_quant_random_013 | [32, 1, 4096] | int8 | 1719934111 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/dynamic_quant_random_014 | [64, 1, 8192] | int4 | 2063819961 | PASS | 0 | 1 | 0 | 0 | 0.706207 |
| custom/dynamic_quant_random_015 | [1, 1, 8192] | int8 | 380564658 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/dynamic_quant_random_016 | [32, 1, 5120] | int4 | 107577838 | PASS | 0 | 1 | 0 | 0 | 0.705019 |
| custom/dynamic_quant_random_017 | [64, 1, 5120] | int8 | 1524914795 | PASS | 0 | 1 | 0 | 0 | 0.708032 |
| custom/dynamic_quant_random_018 | [1, 1, 5120] | int4 | 1495431921 | PASS | 0 | 1 | 0 | 0 | 0.711101 |
| custom/dynamic_quant_random_019 | [32, 1, 4096] | int8 | 1002509468 | PASS | 0 | 1 | 0 | 0 | 0.709052 |
| custom/dynamic_quant_random_020 | [32, 1, 8192] | int4 | 921778545 | PASS | 0 | 1 | 0 | 0 | 0.706707 |
| custom/dynamic_quant_random_021 | [32, 1, 4096] | int8 | 1016052845 | PASS | 0 | 1 | 0 | 0 | 0.708826 |
| custom/dynamic_quant_random_022 | [8, 1, 5120] | int4 | 1385093510 | PASS | 0 | 1 | 0 | 0 | 0.703021 |
| custom/dynamic_quant_random_023 | [8, 1, 4096] | int8 | 884936993 | PASS | 0 | 1 | 0 | 0 | 0.707948 |
| custom/dynamic_quant_random_024 | [1, 1, 4096] | int4 | 1696159923 | PASS | 0 | 1 | 0 | 0 | 0.706934 |
| custom/dynamic_quant_random_025 | [64, 1, 5120] | int8 | 1230597722 | PASS | 0 | 1 | 0 | 0 | 0.708502 |
| custom/dynamic_quant_random_026 | [8, 1, 5120] | int4 | 1082962109 | PASS | 0 | 1 | 0 | 0 | 0.704409 |
| custom/dynamic_quant_random_027 | [64, 1, 3584] | int8 | 1249361969 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/dynamic_quant_random_028 | [64, 1, 8192] | int4 | 2126356694 | PASS | 0 | 1 | 0 | 0 | 0.706836 |
| custom/dynamic_quant_random_029 | [32, 1, 5120] | int8 | 2030633306 | PASS | 0 | 1 | 0 | 0 | 0.707633 |
| custom/dynamic_quant_random_030 | [32, 1, 4096] | int4 | 580124394 | PASS | 0 | 1 | 0 | 0 | 0.707242 |
| custom/dynamic_quant_random_031 | [64, 1, 5120] | int8 | 1163552118 | PASS | 0 | 1 | 0 | 0 | 0.708663 |
| custom/dynamic_quant_random_032 | [32, 1, 4096] | int4 | 1805025668 | PASS | 0 | 1 | 0 | 0 | 0.706135 |
| custom/dynamic_quant_random_033 | [32, 1, 8192] | int8 | 360348971 | PASS | 0 | 1 | 0 | 0 | 0.70846 |
| custom/dynamic_quant_random_034 | [32, 1, 3584] | int4 | 392170129 | PASS | 0 | 1 | 0 | 0 | 0.705749 |
| custom/dynamic_quant_random_035 | [64, 1, 8192] | int8 | 1902112164 | PASS | 0 | 1 | 0 | 0 | 0.70824 |
| custom/dynamic_quant_random_036 | [16, 1, 3584] | int4 | 1156290584 | PASS | 0 | 1 | 0 | 0 | 0.695294 |
| custom/dynamic_quant_random_037 | [16, 1, 4096] | int8 | 454063838 | PASS | 0 | 1 | 0 | 0 | 0.706492 |
| custom/dynamic_quant_random_038 | [64, 1, 8192] | int4 | 613794047 | PASS | 0 | 1 | 0 | 0 | 0.696795 |
| custom/dynamic_quant_random_039 | [16, 1, 8192] | int8 | 2000343471 | PASS | 0 | 1 | 0 | 0 | 0.706896 |
| custom/dynamic_quant_random_040 | [64, 1, 3584] | int4 | 247847709 | PASS | 0 | 1 | 0 | 0 | 0.696522 |
