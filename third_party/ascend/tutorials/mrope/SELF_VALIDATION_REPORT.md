# MRoPE 算子自验证报告

## 1. 报告说明

- 单一数值证据源：`logs/mrope_validation.jsonl`
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
| evidence source | logs/mrope_validation.jsonl |
| total cases | 60 |
| public cases | 20 |
| random/generalization cases | 40 |
| candidate pass | 60/60 |
| main speed sample | torch_npu runnable all (32 cases) |
| main speed candidate active geomean | 11.707 us |
| main speed torch_npu active geomean | 14.214 us |
| main speed active/active geomean speedup | 1.214162x |
| main speed gate | PASS >= 1.2x |
| overall selected baseline split | torch=28, torch_npu=32 |
| public selected baseline split | torch=8, torch_npu=12 |
| overall torch_npu runnable all | 32 |
| overall torch_npu accuracy pass/fail | 32/0 |
| torch_npu runnable-all active speedup geomean | 1.214162x |
| torch_npu runnable-all speed gate | PASS >= 1.2x |
| public torch_npu runnable all | 12 |
| public torch_npu accuracy pass/fail | 12/0 |
| aux torch semantic timed all | 60 |
| aux torch semantic candidate active geomean | 17.163 us |
| aux torch semantic baseline active geomean | 1145.951 us |
| aux torch semantic active/active geomean speedup | 66.769734x |
| aux public torch semantic timed | 20 |
| aux public torch semantic candidate active geomean | 15.579 us |
| aux public torch semantic baseline active geomean | 1096.435 us |
| aux public torch semantic active/active geomean speedup | 70.378307x |
| aux public candidate active geomean | 15.579 us |
| aux public selected baseline active geomean | 79.229 us |
| aux public selected active/active geomean speedup | 5.085547x |
| max candidate RMSE | 0 |
| commercial standard | references/commercial_standard.md @ c260c8ab7a9be4823ac8f8a07c60442de9bf141e |

## 3. 性能口径汇总

| Scope | Cases | Candidate active geomean | Baseline active geomean | Active/active geomean speedup | Precision pass | Note |
| --- | --- | --- | --- | --- | --- | --- |
| main torch_npu timed sample | 32 | 11.707 us | 14.214 us | 1.214162x | 32/32 | 主速度验收口径；candidate 和 torch_npu 均只在这同一批有 torch_npu active 计时的 case 上取几何平均；gate >= 1.2x |
| torch_npu accuracy-pass | 32 | 11.707 us | 14.214 us | 1.214162x | 32/32 | 全量 torch_npu 有效计时且本地 checker PASS 子集 |
| torch_npu accuracy-fail | 0 | N/A | N/A | N/A | 0/0 | 全量 torch_npu 有效计时但本地 checker FAIL 子集 |
| aux all torch semantic timed baseline | 60 | 17.163 us | 1145.951 us | 66.769734x | 60/60 | 补充 Torch 语义参考计时；由 --benchmark-torch 开启，不参与主速度 gate |
| aux public torch semantic timed baseline | 20 | 15.579 us | 1096.435 us | 70.378307x | 20/20 | 补充 public Torch 语义参考计时；由 --benchmark-torch 开启，不参与主速度 gate |
| aux public selected baseline | 20 | 15.579 us | 79.229 us | 5.085547x | 20/20 | 补充语义标杆口径；torch_npu 仅在本地 checker 通过时选中，否则选 Torch |
| aux public torch semantic correctness baseline | 20 | 15.579 us | N/A | N/A | 20/20 | Torch 语义参考覆盖全部 public case；latency/speedup 只看上面的 timed baseline 行 |

## 4. Baseline 校验明细

| Case | Selected implementation | Selection rule | Torch pass | torch_npu runnable | torch_npu pass | torch_npu MERE | torch_npu MARE | torch_npu RMSE | torch_npu max diff | Reason | Seed/attrs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_1 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1084307072, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_2 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1477511722, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_3 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:23:10 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 597351877, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_4 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 462559110, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_5 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 1479209576, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_6 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 2071566058, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_7 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 187621401, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_8 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:26:26 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 42610454, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_9 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 21837909, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_10 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 1564414684, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_11 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1837934397, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_12 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1772772694, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_13 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:29:42 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 5610152, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_14 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1347296229, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_15 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 177205156, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_16 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 333733719, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_17 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1428411558, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_18 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:33:02 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 2109847545, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_19 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 795094805, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_20 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 1599279944, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |

## 5. Public 逐Case速度

| Case | Kind | Shape | DType | Selected baseline | Triton active | Torch active | torch_npu active | Selected active speedup | Torch active speedup | torch_npu active speedup | Triton precision | Torch precision | torch_npu precision | MERE | MARE | RMSE | Max diff | torch_npu error/note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_1 | public | [[1], [1, 3584], [1, 3584], [2048, 128]] | bfloat16 | torch_npu | 3.500 us | 862.750 us | 8.000 us | 2.285714x | 246.500000x | 2.285714x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_2 | public | [[3, 1], [1, 4096], [1, 4096], [2048, 128]] | bfloat16 | torch_npu | 7.000 us | 1302.500 us | 8.750 us | 1.250000x | 186.071429x | 1.250000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_3 | public | [[3, 1], [1, 5120], [1, 5120], [2048, 128]] | bfloat16 | torch | 8.250 us | 1105.750 us | N/A | 134.030303x | 134.030303x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:23:10 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_4 | public | [[4, 1], [1, 8192], [1, 8192], [2048, 128]] | bfloat16 | torch_npu | 7.250 us | 1096.000 us | 14.000 us | 1.931034x | 151.172414x | 1.931034x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_5 | public | [[3, 8], [8, 3584], [8, 3584], [2048, 64]] | bfloat16 | torch | 12.750 us | 1392.750 us | N/A | 109.235294x | 109.235294x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |
| custom/mrope_6 | public | [[8], [8, 4096], [8, 4096], [2048, 128]] | bfloat16 | torch_npu | 7.250 us | 880.500 us | 10.000 us | 1.379310x | 121.448276x | 1.379310x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_7 | public | [[3, 8], [8, 5120], [8, 5120], [2048, 128]] | bfloat16 | torch_npu | 9.000 us | 1339.000 us | 11.500 us | 1.277778x | 148.777778x | 1.277778x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_8 | public | [[3, 8], [8, 8192], [8, 8192], [2048, 128]] | bfloat16 | torch | 23.750 us | 1080.750 us | N/A | 45.505263x | 45.505263x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:26:26 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_9 | public | [[4, 16], [16, 3584], [16, 3584], [2048, 128]] | bfloat16 | torch_npu | 10.000 us | 1185.000 us | 12.500 us | 1.250000x | 118.500000x | 1.250000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_10 | public | [[3, 16], [16, 4096], [16, 4096], [2048, 64]] | bfloat16 | torch | 16.500 us | 1050.000 us | N/A | 63.636364x | 63.636364x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |
| custom/mrope_11 | public | [[16], [16, 5120], [16, 5120], [2048, 128]] | bfloat16 | torch_npu | 14.500 us | 846.000 us | 12.000 us | 0.827586x | 58.344828x | 0.827586x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_12 | public | [[3, 16], [16, 8192], [16, 8192], [2048, 128]] | bfloat16 | torch_npu | 9.500 us | 1007.250 us | 16.500 us | 1.736842x | 106.026316x | 1.736842x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_13 | public | [[3, 32], [32, 3584], [32, 3584], [2048, 128]] | bfloat16 | torch | 35.000 us | 1105.250 us | N/A | 31.578571x | 31.578571x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:29:42 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_14 | public | [[4, 32], [32, 4096], [32, 4096], [2048, 128]] | bfloat16 | torch_npu | 10.750 us | 1227.250 us | 15.500 us | 1.441860x | 114.162791x | 1.441860x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_15 | public | [[3, 32], [32, 5120], [32, 5120], [2048, 64]] | bfloat16 | torch | 32.000 us | 1008.750 us | N/A | 31.523438x | 31.523438x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |
| custom/mrope_16 | public | [[32], [32, 8192], [32, 8192], [2048, 128]] | bfloat16 | torch_npu | 39.250 us | 1069.750 us | 17.500 us | 0.445860x | 27.254777x | 0.445860x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_17 | public | [[3, 64], [64, 3584], [64, 3584], [2048, 128]] | bfloat16 | torch_npu | 18.250 us | 953.250 us | 19.250 us | 1.054795x | 52.232877x | 1.054795x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_18 | public | [[3, 64], [64, 4096], [64, 4096], [2048, 128]] | bfloat16 | torch | 73.500 us | 1154.000 us | N/A | 15.700680x | 15.700680x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-20-11:33:02 (PID:1554668, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_19 | public | [[4, 64], [64, 5120], [64, 5120], [2048, 128]] | bfloat16 | torch_npu | 19.000 us | 1172.250 us | 22.250 us | 1.171053x | 61.697368x | 1.171053x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_20 | public | [[3, 64], [64, 8192], [64, 8192], [2048, 64]] | bfloat16 | torch | 89.500 us | 1303.500 us | N/A | 14.564246x | 14.564246x | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |

## 6. 商业L1精度对比

| Case | Output | DType | Shape | Reference | Criterion | Candidate AE | Candidate MARE | Candidate MERE | Candidate RMSE | Baseline AE | Baseline MARE | Baseline MERE | Baseline RMSE | L1 metric status | Checker status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_1 | query_out | torch.bfloat16 | [1, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_1 | key_out | torch.bfloat16 | [1, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_2 | query_out | torch.bfloat16 | [1, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_2 | key_out | torch.bfloat16 | [1, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_3 | query_out | torch.bfloat16 | [1, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_3 | key_out | torch.bfloat16 | [1, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_4 | query_out | torch.bfloat16 | [1, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_4 | key_out | torch.bfloat16 | [1, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_5 | query_out | torch.bfloat16 | [8, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_5 | key_out | torch.bfloat16 | [8, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_6 | query_out | torch.bfloat16 | [8, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_6 | key_out | torch.bfloat16 | [8, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_7 | query_out | torch.bfloat16 | [8, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_7 | key_out | torch.bfloat16 | [8, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_8 | query_out | torch.bfloat16 | [8, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_8 | key_out | torch.bfloat16 | [8, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_9 | query_out | torch.bfloat16 | [16, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_9 | key_out | torch.bfloat16 | [16, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_10 | query_out | torch.bfloat16 | [16, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_10 | key_out | torch.bfloat16 | [16, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_11 | query_out | torch.bfloat16 | [16, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_11 | key_out | torch.bfloat16 | [16, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_12 | query_out | torch.bfloat16 | [16, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_12 | key_out | torch.bfloat16 | [16, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_13 | query_out | torch.bfloat16 | [32, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_13 | key_out | torch.bfloat16 | [32, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_14 | query_out | torch.bfloat16 | [32, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_14 | key_out | torch.bfloat16 | [32, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_15 | query_out | torch.bfloat16 | [32, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_15 | key_out | torch.bfloat16 | [32, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_16 | query_out | torch.bfloat16 | [32, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_16 | key_out | torch.bfloat16 | [32, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_17 | query_out | torch.bfloat16 | [64, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_17 | key_out | torch.bfloat16 | [64, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_18 | query_out | torch.bfloat16 | [64, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_18 | key_out | torch.bfloat16 | [64, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_19 | query_out | torch.bfloat16 | [64, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_19 | key_out | torch.bfloat16 | [64, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_20 | query_out | torch.bfloat16 | [64, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_20 | key_out | torch.bfloat16 | [64, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |

## 7. 随机泛化明细

| Case | Shape | Category/dst | Seed | Status | Mismatch | Max diff | MERE | MARE | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_random_001 | [[8], [8, 5120], [8, 5120], [256, 128]] | rope_half_default | 357155024 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_002 | [[4, 8], [8, 5120], [8, 5120], [256, 128]] | mrope4_half_default | 317870542 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_003 | [[3, 8], [8, 3584], [8, 3584], [1024, 128]] | mrope3_interleaved_default | 262279943 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_004 | [[3, 64], [64, 5120], [64, 5120], [512, 128]] | mrope3_interleaved_default | 1991089026 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_005 | [[3, 64], [64, 8192], [64, 8192], [256, 64]] | mrope3_half_interleave64 | 1376064627 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_006 | [[3, 1], [1, 5120], [1, 5120], [256, 128]] | mrope3_interleaved_default | 784605869 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_007 | [[4, 8], [8, 3584], [8, 3584], [1024, 128]] | mrope4_half_default | 189440073 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_008 | [[3, 1], [1, 3584], [1, 3584], [512, 64]] | mrope3_half_interleave64 | 948197677 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_009 | [[3, 64], [64, 8192], [64, 8192], [256, 128]] | mrope3_interleaved_default | 1588184522 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_010 | [[3, 8], [8, 4096], [8, 4096], [256, 64]] | mrope3_half_interleave64 | 1892553638 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_011 | [[3, 32], [32, 8192], [32, 8192], [1024, 64]] | mrope3_half_interleave64 | 1825869526 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_012 | [[4, 8], [8, 3584], [8, 3584], [2048, 128]] | mrope4_half_default | 347532505 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_013 | [[4, 64], [64, 4096], [64, 4096], [512, 128]] | mrope4_half_default | 1700093221 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_014 | [[3, 1], [1, 3584], [1, 3584], [512, 128]] | mrope3_half_default | 1537592907 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_015 | [[16], [16, 8192], [16, 8192], [1024, 128]] | rope_half_default | 1948272746 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_016 | [[1], [1, 8192], [1, 8192], [1024, 128]] | rope_half_default | 749951944 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_017 | [[3, 1], [1, 4096], [1, 4096], [2048, 64]] | mrope3_half_interleave64 | 1314172540 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_018 | [[3, 1], [1, 4096], [1, 4096], [256, 64]] | mrope3_half_interleave64 | 2088918334 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_019 | [[3, 32], [32, 8192], [32, 8192], [256, 64]] | mrope3_half_interleave64 | 1662217404 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_020 | [[3, 64], [64, 3584], [64, 3584], [1024, 128]] | mrope3_interleaved_default | 1237494549 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_021 | [[64], [64, 4096], [64, 4096], [1024, 128]] | rope_half_default | 1117345193 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_022 | [[4, 16], [16, 8192], [16, 8192], [512, 128]] | mrope4_half_default | 1405715249 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_023 | [[3, 64], [64, 5120], [64, 5120], [256, 128]] | mrope3_interleaved_default | 506179563 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_024 | [[3, 64], [64, 4096], [64, 4096], [1024, 128]] | mrope3_half_default | 1122512817 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_025 | [[3, 1], [1, 8192], [1, 8192], [512, 128]] | mrope3_half_default | 630834576 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_026 | [[3, 16], [16, 4096], [16, 4096], [256, 64]] | mrope3_half_interleave64 | 882373969 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_027 | [[3, 8], [8, 3584], [8, 3584], [2048, 128]] | mrope3_half_default | 1070317775 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_028 | [[4, 1], [1, 5120], [1, 5120], [2048, 128]] | mrope4_half_default | 1299025417 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_029 | [[4, 8], [8, 4096], [8, 4096], [1024, 128]] | mrope4_half_default | 2098293120 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_030 | [[3, 8], [8, 5120], [8, 5120], [256, 64]] | mrope3_half_interleave64 | 1607265919 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_031 | [[3, 64], [64, 8192], [64, 8192], [512, 128]] | mrope3_interleaved_default | 1938822793 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_032 | [[3, 64], [64, 4096], [64, 4096], [256, 128]] | mrope3_half_default | 272046855 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_033 | [[4, 32], [32, 8192], [32, 8192], [1024, 128]] | mrope4_half_default | 1374793886 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_034 | [[64], [64, 5120], [64, 5120], [2048, 128]] | rope_half_default | 277126696 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_035 | [[3, 8], [8, 3584], [8, 3584], [512, 64]] | mrope3_half_interleave64 | 1472817882 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_036 | [[3, 32], [32, 5120], [32, 5120], [1024, 128]] | mrope3_interleaved_default | 1533341534 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_037 | [[3, 8], [8, 5120], [8, 5120], [2048, 128]] | mrope3_interleaved_default | 1236142190 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_038 | [[3, 32], [32, 4096], [32, 4096], [512, 128]] | mrope3_half_default | 88831201 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_039 | [[64], [64, 3584], [64, 3584], [256, 128]] | rope_half_default | 21920662 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_040 | [[3, 1], [1, 4096], [1, 4096], [1024, 128]] | mrope3_interleaved_default | 1673357719 | PASS | 0 | 0 | 0 | 0 | 0 |
