# AddRmsNorm 自验证报告

## 1. 环境信息

证据来源为本地 Ascend NPU 工作区中的 OpForge/CANN-Bench 评测记录、交付目录 80 public timing matrix，以及本次固定 seed 随机泛化验证。

| 项目 | 数值 |
|---|---|
| 评测器 run id | `eval_20260615_154048` |
| 后端 | Triton-Ascend |
| 芯片 | Ascend910_9382 |
| CANN | 9.0.0 |
| Driver | 25.5.2 |
| Python | 3.11.14 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| 官方 public timing source | `results/eval_records/eval_20260615_154048/timing_artifacts/main_results.csv` |
| 80 public delivery benchmark | `logs/add_rms_norm_timing_matrix_20260615.jsonl` |
| public+random correctness | `logs/add_rms_norm_random_generalization_20260615.jsonl` |

## 2. 测试范围

公开验证集包含 80 个 BF16 case：`B in {1,8,16,32,64}`，`S in {1,8,32,128}`，`H in {3584,4096,5120,8192}`。输入范围为 `x1/x2 in [-1,1]`，`gamma in [0.5,1.5]`，`epsilon=1e-6`。

随机泛化集包含 40 个固定 seed 非公开 shape case，`seed=20260613`，`shape_policy=seeded_non_public_bsh_v1`。

## 3. 精度结果

| 范围 | Triton 候选 | Torch 语义实现 | torch_npu helper | 总数 |
|---|---:|---:|---:|---:|
| 80 public | 80 | 80 | 61 | 80 |
| 40 random generalization | 40 | 40 | 25 | 40 |

Triton 候选在 80 个 public case 和 40 个 random generalization case 上均满足 BF16 检查。public 最大 MARE 为 `0.00781249960938`，random generalization 最大 MARE 为 `0.00781249921875`，最大 diff 均为 `0.015625`。

## 4. 性能结果

### 4.1 OpForge 官方记录

`eval_20260615_154048` 是当前本地 OpForge 评测器的最新 full-public 80-case 记录：

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

该 run 的状态为 `PERF_REGRESSION`，因为 `active_regression_cases=2`。kernel-sum 仅作为诊断，不改变 active-window 主口径。

### 4.2 delivery benchmark 完整性

`logs/add_rms_norm_timing_matrix_20260615.jsonl` 的 public 精度为 80/80 PASS。它使用严格 profiler CSV 解析；出现 blank `Step Id` 时不会兼容成 latency 数字，而是保留 `N/A` 和 `profile_error`。

| 路径 | active timed | active N/A cases | profile error |
| --- | ---: | --- | --- |
| Triton | 74/80 | [75, 76, 77, 78, 79, 80] | blank Step Id |
| Torch | 66/80 | [13, 55, 56, 59, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80] | blank Step Id |
| torch_npu | 33/80 | [10, 15, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80] | blank Step Id |

因此逐 case 性能表和 XLSX 的 `逐Case速度` 工作表使用官方 `eval_20260615_154048` 的完整 `main_results.csv`，而不是用 delivery benchmark 的 N/A 行补数。

### 4.3 80 个公开用例逐项明细

下表由 `results/eval_records/eval_20260615_154048/timing_artifacts/main_results.csv` 和同 run 的 `traces.jsonl` 生成。

| Case | Shape | Baseline | Triton active | Triton kernel | Baseline active | Baseline kernel | active/active | kernel/kernel | Risk | MERE | MARE | Max diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [1, 1, 3584] | task_npu_baseline | 3.750 us | 3.740 us | 3.000 us | 3.060 us | 0.800000x | 0.818182x | active_regression | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 2 | [1, 1, 4096] | task_npu_baseline | 3.250 us | 3.200 us | 3.500 us | 3.440 us | 1.076923x | 1.075000x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 3 | [1, 1, 5120] | task_npu_baseline | 4.500 us | 4.440 us | 3.750 us | 3.640 us | 0.833333x | 0.819820x | active_regression | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 4 | [1, 1, 8192] | task_npu_baseline | 4.000 us | 4.040 us | 4.250 us | 4.140 us | 1.062500x | 1.024752x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 5 | [8, 1, 3584] | task_npu_baseline | 4.000 us | 4.100 us | 10.250 us | 10.200 us | 2.562500x | 2.487805x |  | 1.743861e-07 | 4.999999e-03 | 1.953125e-03 |
| 6 | [8, 1, 4096] | task_npu_baseline | 3.500 us | 3.400 us | 10.250 us | 10.340 us | 2.928571x | 3.041176x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 7 | [8, 1, 5120] | task_npu_baseline | 5.000 us | 4.880 us | 12.000 us | 12.020 us | 2.400000x | 2.463115x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 8 | [8, 1, 8192] | task_npu_baseline | 4.500 us | 4.420 us | 18.000 us | 18.060 us | 4.000000x | 4.085973x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 9 | [16, 1, 3584] | pytorch_fallback | 4.500 us | 4.420 us | 231.750 us | 54.420 us | 51.500000x | 12.312217x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 10 | [16, 1, 4096] | pytorch_fallback | 4.000 us | 4.040 us | 352.750 us | 53.800 us | 88.187500x | 13.316832x | baseline_severe_gap_gt_80pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 11 | [16, 1, 5120] | task_npu_baseline | 5.250 us | 5.260 us | 20.000 us | 20.020 us | 3.809524x | 3.806084x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 12 | [16, 1, 8192] | task_npu_baseline | 5.000 us | 4.940 us | 34.000 us | 34.020 us | 6.800000x | 6.886640x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 13 | [32, 1, 3584] | pytorch_fallback | 5.500 us | 5.560 us | 286.750 us | 57.740 us | 52.136364x | 10.384892x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 14 | [32, 1, 4096] | pytorch_fallback | 5.250 us | 5.140 us | 221.750 us | 57.480 us | 42.238095x | 11.182879x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 15 | [32, 1, 5120] | pytorch_fallback | 6.500 us | 6.440 us | 245.750 us | 58.420 us | 37.807692x | 9.071429x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 16 | [32, 1, 8192] | task_npu_baseline | 6.250 us | 6.240 us | 65.000 us | 65.120 us | 10.400000x | 10.435897x |  | 1.749861e-08 | 4.587155e-03 | 1.953125e-03 |
| 17 | [64, 1, 3584] | pytorch_fallback | 9.000 us | 9.020 us | 297.000 us | 60.400 us | 33.000000x | 6.696231x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 18 | [64, 1, 4096] | pytorch_fallback | 7.500 us | 7.440 us | 265.250 us | 60.040 us | 35.366667x | 8.069892x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 19 | [64, 1, 5120] | pytorch_fallback | 10.500 us | 10.540 us | 207.250 us | 62.380 us | 19.738095x | 5.918406x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 20 | [64, 1, 8192] | task_npu_baseline | 9.500 us | 9.480 us | 128.000 us | 128.000 us | 13.473684x | 13.502110x |  | 8.514941e-09 | 4.464282e-03 | 4.882812e-04 |
| 21 | [1, 8, 3584] | task_npu_baseline | 3.750 us | 3.820 us | 9.500 us | 9.520 us | 2.533333x | 2.492147x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 22 | [1, 8, 4096] | task_npu_baseline | 3.250 us | 3.320 us | 10.250 us | 10.240 us | 3.153846x | 3.084337x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 23 | [1, 8, 5120] | task_npu_baseline | 4.750 us | 4.780 us | 12.250 us | 12.220 us | 2.578947x | 2.556485x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 24 | [1, 8, 8192] | task_npu_baseline | 4.250 us | 4.340 us | 18.000 us | 18.080 us | 4.235294x | 4.165899x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 25 | [8, 8, 3584] | pytorch_fallback | 8.500 us | 8.560 us | 307.750 us | 60.420 us | 36.205882x | 7.058411x | baseline_severe_gap_gt_80pct | 1.937624e-08 | 4.444443e-03 | 1.953125e-03 |
| 26 | [8, 8, 4096] | pytorch_fallback | 7.750 us | 7.700 us | 282.000 us | 59.360 us | 36.387097x | 7.709091x | baseline_high_gap_gt_50pct | 2.217847e-08 | 5.813953e-03 | 7.812500e-03 |
| 27 | [8, 8, 5120] | pytorch_fallback | 10.500 us | 10.400 us | 208.500 us | 62.280 us | 19.857143x | 5.988462x | baseline_high_gap_gt_50pct | 1.481436e-08 | 4.854369e-03 | 7.812500e-03 |
| 28 | [8, 8, 8192] | task_npu_baseline | 10.000 us | 9.920 us | 128.000 us | 128.020 us | 12.800000x | 12.905242x |  | 5.522093e-08 | 6.666666e-03 | 7.812500e-03 |
| 29 | [16, 8, 3584] | pytorch_fallback | 10.750 us | 10.680 us | 239.500 us | 75.440 us | 22.279070x | 7.063670x | baseline_high_gap_gt_50pct | 5.508171e-08 | 5.714284e-03 | 3.906250e-03 |
| 30 | [16, 8, 4096] | pytorch_fallback | 8.750 us | 8.760 us | 209.250 us | 74.180 us | 23.914286x | 8.468037x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 31 | [16, 8, 5120] | pytorch_fallback | 12.500 us | 12.420 us | 301.000 us | 65.880 us | 24.080000x | 5.304348x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 32 | [16, 8, 8192] | pytorch_fallback | 12.000 us | 11.940 us | 208.750 us | 75.540 us | 17.395833x | 6.326633x | baseline_high_gap_gt_50pct | 1.115447e-08 | 5.882352e-03 | 7.812500e-03 |
| 33 | [32, 8, 3584] | pytorch_fallback | 15.500 us | 15.600 us | 227.750 us | 73.100 us | 14.693548x | 4.685897x | baseline_high_gap_gt_50pct | 1.583981e-08 | 5.102040e-03 | 7.812500e-03 |
| 34 | [32, 8, 4096] | pytorch_fallback | 11.750 us | 11.820 us | 323.750 us | 76.780 us | 27.553192x | 6.495770x | baseline_high_gap_gt_50pct | 1.919926e-08 | 6.451612e-03 | 7.812500e-03 |
| 35 | [32, 8, 5120] | pytorch_fallback | 20.500 us | 20.560 us | 257.000 us | 85.620 us | 12.536585x | 4.164397x | baseline_high_gap_gt_50pct | 7.865354e-09 | 5.154639e-03 | 7.812500e-03 |
| 36 | [32, 8, 8192] | task_npu_baseline | 19.000 us | 18.880 us | 500.750 us | 500.710 us | 26.355263x | 26.520657x |  | 1.278560e-08 | 6.849315e-03 | 1.562500e-02 |
| 37 | [64, 8, 3584] | pytorch_fallback | 25.250 us | 25.240 us | 271.000 us | 97.580 us | 10.732673x | 3.866086x | baseline_high_gap_gt_50pct | 2.008836e-08 | 7.299269e-03 | 1.562500e-02 |
| 38 | [64, 8, 4096] | pytorch_fallback | 18.000 us | 18.120 us | 353.250 us | 107.380 us | 19.625000x | 5.926049x | baseline_high_gap_gt_50pct | 1.225157e-08 | 7.518796e-03 | 7.812500e-03 |
| 39 | [64, 8, 5120] | pytorch_fallback | 33.000 us | 32.940 us | 247.750 us | 116.480 us | 7.507576x | 3.536126x | baseline_high_gap_gt_50pct | 8.863666e-09 | 6.493506e-03 | 7.812500e-03 |
| 40 | [64, 8, 8192] | task_npu_baseline | 28.750 us | 28.800 us | 953.000 us | 952.980 us | 33.147826x | 33.089583x |  | 1.253281e-08 | 7.352941e-03 | 1.562500e-02 |
| 41 | [1, 32, 3584] | pytorch_fallback | 5.500 us | 5.580 us | 304.250 us | 56.500 us | 55.318182x | 10.125448x | baseline_severe_gap_gt_80pct | 9.051100e-08 | 5.405404e-03 | 1.953125e-03 |
| 42 | [1, 32, 4096] | pytorch_fallback | 5.250 us | 5.140 us | 255.750 us | 57.980 us | 48.714286x | 11.280156x | baseline_high_gap_gt_50pct | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 43 | [1, 32, 5120] | task_npu_baseline | 6.500 us | 6.420 us | 38.500 us | 38.480 us | 5.923077x | 5.993769x |  | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 44 | [1, 32, 8192] | task_npu_baseline | 6.250 us | 6.160 us | 66.250 us | 66.200 us | 10.600000x | 10.746753x |  | 2.445318e-08 | 6.410254e-03 | 1.953125e-03 |
| 45 | [8, 32, 3584] | pytorch_fallback | 15.000 us | 15.120 us | 252.750 us | 73.400 us | 16.850000x | 4.854497x | baseline_high_gap_gt_50pct | 3.066111e-08 | 6.756756e-03 | 3.906250e-03 |
| 46 | [8, 32, 4096] | pytorch_fallback | 11.000 us | 10.960 us | 375.000 us | 76.820 us | 34.090909x | 7.009124x | baseline_high_gap_gt_50pct | 1.317454e-08 | 4.901960e-03 | 7.812500e-03 |
| 47 | [8, 32, 5120] | pytorch_fallback | 18.500 us | 18.420 us | 287.000 us | 85.820 us | 15.513514x | 4.659066x | baseline_high_gap_gt_50pct | 2.053856e-08 | 6.024089e-03 | 7.812500e-03 |
| 48 | [8, 32, 8192] | pytorch_fallback | 18.000 us | 17.940 us | 246.750 us | 95.220 us | 13.708333x | 5.307692x | baseline_high_gap_gt_50pct | 1.089529e-08 | 7.518797e-03 | 1.562500e-02 |
| 49 | [16, 32, 3584] | pytorch_fallback | 21.250 us | 21.340 us | 219.500 us | 98.040 us | 10.329412x | 4.594189x | baseline_high_gap_gt_50pct | 2.334679e-08 | 6.097560e-03 | 1.562500e-02 |
| 50 | [16, 32, 4096] | pytorch_fallback | 16.250 us | 16.320 us | 316.750 us | 101.140 us | 19.492308x | 6.197304x | baseline_high_gap_gt_50pct | 1.205086e-08 | 6.849314e-03 | 7.812500e-03 |
| 51 | [16, 32, 5120] | pytorch_fallback | 30.000 us | 30.080 us | 283.250 us | 113.960 us | 9.441667x | 3.788564x | baseline_high_gap_gt_50pct | 1.569169e-08 | 6.493506e-03 | 7.812500e-03 |
| 52 | [16, 32, 8192] | task_npu_baseline | 28.500 us | 28.540 us | 952.250 us | 952.220 us | 33.412281x | 33.364401x |  | 1.934633e-08 | 7.462686e-03 | 1.562500e-02 |
| 53 | [32, 32, 3584] | task_npu_baseline | 40.500 us | 40.480 us | 905.250 us | 905.160 us | 22.351852x | 22.360672x |  | 1.102406e-08 | 6.024096e-03 | 7.812500e-03 |
| 54 | [32, 32, 4096] | task_npu_baseline | 29.250 us | 29.180 us | 953.750 us | 953.820 us | 32.606838x | 32.687457x |  | 1.922937e-08 | 6.329113e-03 | 1.562500e-02 |
| 55 | [32, 32, 5120] | task_npu_baseline | 62.000 us | 61.900 us | 1194.750 us | 1194.680 us | 19.270161x | 19.300162x |  | 1.610848e-08 | 7.575757e-03 | 7.812500e-03 |
| 56 | [32, 32, 8192] | task_npu_baseline | 61.250 us | 61.300 us | 1924.000 us | 1924.080 us | 31.412245x | 31.387928x |  | 1.893420e-08 | 7.751935e-03 | 7.812500e-03 |
| 57 | [64, 32, 3584] | task_npu_baseline | 85.000 us | 84.920 us | 1682.250 us | 1682.370 us | 19.791176x | 19.811234x |  | 3.768288e-08 | 7.633587e-03 | 1.562500e-02 |
| 58 | [64, 32, 4096] | task_npu_baseline | 60.000 us | 60.000 us | 1922.750 us | 1922.800 us | 32.045833x | 32.046667x |  | 2.133124e-08 | 7.407407e-03 | 1.562500e-02 |
| 59 | [64, 32, 5120] | task_npu_baseline | 123.500 us | 123.580 us | 2400.000 us | 2400.070 us | 19.433198x | 19.421185x |  | 2.260155e-08 | 7.751937e-03 | 1.562500e-02 |
| 60 | [64, 32, 8192] | task_npu_baseline | 110.000 us | 110.020 us | 3843.500 us | 3843.540 us | 34.940909x | 34.934921x |  | 2.118879e-08 | 7.751937e-03 | 1.562500e-02 |
| 61 | [1, 128, 3584] | pytorch_fallback | 10.500 us | 10.520 us | 232.500 us | 74.040 us | 22.142857x | 7.038023x | baseline_high_gap_gt_50pct | 4.712945e-08 | 7.246375e-03 | 7.812500e-03 |
| 62 | [1, 128, 4096] | pytorch_fallback | 8.750 us | 8.680 us | 257.000 us | 75.260 us | 29.371429x | 8.670507x | baseline_high_gap_gt_50pct | 3.644398e-08 | 7.142856e-03 | 7.812500e-03 |
| 63 | [1, 128, 5120] | pytorch_fallback | 12.500 us | 12.400 us | 242.000 us | 65.040 us | 19.360000x | 5.245161x | baseline_high_gap_gt_50pct | 3.844586e-08 | 5.263158e-03 | 7.812500e-03 |
| 64 | [1, 128, 8192] | task_npu_baseline | 11.250 us | 11.140 us | 252.000 us | 252.070 us | 22.400000x | 22.627469x |  | 1.883042e-08 | 7.575756e-03 | 1.562500e-02 |
| 65 | [8, 128, 3584] | task_npu_baseline | 39.500 us | 39.480 us | 904.000 us | 904.000 us | 22.886076x | 22.897670x |  | 3.013531e-08 | 7.518796e-03 | 1.562500e-02 |
| 66 | [8, 128, 4096] | task_npu_baseline | 28.250 us | 28.260 us | 956.000 us | 955.920 us | 33.840708x | 33.825902x |  | 1.862101e-08 | 7.751937e-03 | 7.812500e-03 |
| 67 | [8, 128, 5120] | task_npu_baseline | 62.000 us | 62.040 us | 1195.000 us | 1194.880 us | 19.274193x | 19.259832x |  | 2.338870e-08 | 7.299270e-03 | 1.562500e-02 |
| 68 | [8, 128, 8192] | task_npu_baseline | 61.000 us | 60.880 us | 1924.000 us | 1923.960 us | 31.540984x | 31.602497x |  | 1.936018e-08 | 7.751937e-03 | 1.562500e-02 |
| 69 | [16, 128, 3584] | task_npu_baseline | 89.250 us | 89.200 us | 1679.000 us | 1679.030 us | 18.812325x | 18.823206x |  | 2.179647e-08 | 7.751937e-03 | 1.562500e-02 |
| 70 | [16, 128, 4096] | task_npu_baseline | 56.500 us | 56.620 us | 1926.000 us | 1926.040 us | 34.088496x | 34.016955x |  | 2.726430e-08 | 7.812499e-03 | 1.562500e-02 |
| 71 | [16, 128, 5120] | task_npu_baseline | 123.750 us | 123.780 us | 2397.750 us | 2397.630 us | 19.375758x | 19.370092x |  | 4.455534e-08 | 7.812498e-03 | 1.562500e-02 |
| 72 | [16, 128, 8192] | task_npu_baseline | 113.750 us | 113.740 us | 3843.000 us | 3842.900 us | 33.784615x | 33.786707x |  | 1.644043e-08 | 7.633585e-03 | 1.562500e-02 |
| 73 | [32, 128, 3584] | task_npu_baseline | 175.000 us | 175.060 us | 3363.750 us | 3363.870 us | 19.221429x | 19.215526x |  | 2.485518e-08 | 7.751937e-03 | 1.562500e-02 |
| 74 | [32, 128, 4096] | task_npu_baseline | 110.500 us | 110.420 us | 3841.500 us | 3841.400 us | 34.764706x | 34.788987x |  | 2.787566e-08 | 7.812500e-03 | 1.562500e-02 |
| 75 | [32, 128, 5120] | task_npu_baseline | 244.250 us | 244.130 us | 4804.000 us | 4804.040 us | 19.668373x | 19.678204x |  | 1.737641e-08 | 7.633587e-03 | 1.562500e-02 |
| 76 | [32, 128, 8192] | task_npu_baseline | 224.500 us | 224.560 us | 7714.500 us | 7714.550 us | 34.363029x | 34.354070x |  | 1.986338e-08 | 7.692306e-03 | 1.562500e-02 |
| 77 | [64, 128, 3584] | task_npu_baseline | 338.750 us | 338.730 us | 6749.000 us | 6748.900 us | 19.923247x | 19.924128x |  | 2.065520e-08 | 7.751937e-03 | 1.562500e-02 |
| 78 | [64, 128, 4096] | task_npu_baseline | 220.750 us | 220.860 us | 7711.000 us | 7710.950 us | 34.930917x | 34.913294x |  | 2.313105e-08 | 7.812500e-03 | 1.562500e-02 |
| 79 | [64, 128, 5120] | task_npu_baseline | 481.750 us | 481.830 us | 10054.500 us | 10054.380 us | 20.870784x | 20.867069x |  | 2.005021e-08 | 7.812499e-03 | 1.562500e-02 |
| 80 | [64, 128, 8192] | task_npu_baseline | 465.250 us | 465.290 us | 16186.500 us | 16186.540 us | 34.790973x | 34.788068x |  | 2.089717e-08 | 7.751937e-03 | 1.562500e-02 |

## 5. 随机泛化明细

下表只由 `logs/add_rms_norm_random_generalization_20260615.jsonl` 的 `random_generalization` 记录生成，和 xlsx 的 `随机泛化明细` 工作表同源。

| Case | Shape | Category | Seed | Triton acc | Torch acc | torch_npu acc | MERE | MARE | Max diff |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 81 | [64, 8, 3200] | near_hidden | 605294706 | PASS | PASS | FAIL | 7.320653e-09 | 7.092197e-03 | 7.812500e-03 |
| 82 | [8, 32, 7876] | contract_shape | 776602979 | PASS | PASS | PASS | 1.312956e-08 | 7.352941e-03 | 1.562500e-02 |
| 83 | [3, 17, 804] | small_tail | 1610810472 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 84 | [5, 2, 371] | small_tail | 523483389 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 85 | [16, 8, 12204] | wide_hidden | 608003707 | PASS | PASS | PASS | 3.002375e-08 | 7.812499e-03 | 7.812500e-03 |
| 86 | [4, 32, 10811] | wide_hidden | 957215991 | PASS | PASS | PASS | 2.011221e-08 | 6.849314e-03 | 7.812500e-03 |
| 87 | [1, 32, 3968] | near_hidden | 1728411603 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 88 | [4, 3, 6007] | contract_shape | 628828860 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 89 | [4, 16, 8774] | wide_hidden | 1608803918 | PASS | PASS | PASS | 4.214981e-08 | 5.917159e-03 | 7.812500e-03 |
| 90 | [3, 5, 850] | small_tail | 1599198054 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 91 | [8, 1, 4480] | near_hidden | 313083522 | PASS | PASS | PASS | 1.594387e-07 | 5.714282e-03 | 9.765625e-04 |
| 92 | [24, 128, 265] | contract_shape | 1421982534 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 93 | [8, 1, 8128] | near_hidden | 213516464 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 94 | [24, 8, 1685] | contract_shape | 37829643 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 95 | [2, 8, 9539] | wide_hidden | 309758314 | PASS | PASS | PASS | 2.589743e-08 | 3.952569e-03 | 3.906250e-03 |
| 96 | [1, 8, 10263] | wide_hidden | 848680703 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 97 | [5, 17, 290] | small_tail | 1928669891 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 98 | [4, 1, 8968] | wide_hidden | 1516450686 | PASS | PASS | PASS | 1.171298e-07 | 4.201680e-03 | 7.812500e-03 |
| 99 | [3, 33, 384] | small_tail | 123132269 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 100 | [4, 32, 10387] | wide_hidden | 993302822 | PASS | PASS | PASS | 3.798696e-09 | 5.050504e-03 | 1.953125e-03 |
| 101 | [32, 8, 8064] | near_hidden | 1155435423 | PASS | PASS | PASS | 2.960832e-08 | 7.352941e-03 | 1.562500e-02 |
| 102 | [5, 17, 386] | small_tail | 385055200 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 103 | [32, 8, 8384] | near_hidden | 1000357303 | PASS | PASS | PASS | 2.484116e-08 | 5.813952e-03 | 7.812500e-03 |
| 104 | [12, 8, 535] | contract_shape | 1263814717 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 105 | [1, 31, 15] | small_tail | 1696919022 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 106 | [16, 2, 10668] | wide_hidden | 549151188 | PASS | PASS | PASS | 1.246519e-08 | 4.255318e-03 | 1.953125e-03 |
| 107 | [5, 128, 5298] | contract_shape | 884061470 | PASS | PASS | PASS | 3.032280e-08 | 7.575757e-03 | 1.562500e-02 |
| 108 | [1, 8, 10057] | wide_hidden | 562746875 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 109 | [64, 32, 3839] | near_hidden | 1716956553 | PASS | PASS | PASS | 1.844011e-08 | 7.751938e-03 | 1.562500e-02 |
| 110 | [64, 1, 4863] | near_hidden | 1653950118 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 111 | [8, 8, 4352] | near_hidden | 1437109909 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 112 | [1, 31, 623] | small_tail | 33695092 | PASS | PASS | FAIL | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 113 | [1, 1, 4223] | near_hidden | 1947097670 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 114 | [8, 31, 4415] | contract_shape | 118644070 | PASS | PASS | PASS | 1.265855e-08 | 5.319148e-03 | 3.906250e-03 |
| 115 | [16, 1, 8448] | near_hidden | 1439727298 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 116 | [4, 4, 10823] | wide_hidden | 105776654 | PASS | PASS | PASS | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 117 | [64, 32, 3840] | near_hidden | 1446628988 | PASS | PASS | FAIL | 2.152251e-08 | 7.812497e-03 | 1.562500e-02 |
| 118 | [8, 8, 10000] | wide_hidden | 1304375406 | PASS | PASS | PASS | 5.460658e-08 | 6.060604e-03 | 7.812500e-03 |
| 119 | [7, 31, 6989] | contract_shape | 1992472482 | PASS | PASS | PASS | 1.206657e-08 | 7.633582e-03 | 7.812500e-03 |
| 120 | [16, 5, 935] | small_tail | 344353821 | PASS | PASS | FAIL | 1.430639e-07 | 6.134969e-03 | 7.812500e-03 |

## 6. 无 fallback 检查

本交付实现的被测路径使用 `@triton.jit` kernel。Python 侧只进行元数据校验、输出/工作区分配和 launch 参数组织，不调用 PyTorch 等价实现、`torch_npu.npu_add_rms_norm`、CANN/vendor AddRmsNorm、CPU fallback、任务 golden 路径或 peer 解法。

## 7. 复现命令

```bash
python3 validate_add_rms_norm.py --public --random-generalization 40 --random-seed 20260613   --jsonl logs/add_rms_norm_random_generalization_20260615.jsonl
python3 validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5   --jsonl logs/add_rms_norm_timing_matrix_20260615.jsonl   --profiler-data-dir logs/prof_data_20260615
```
