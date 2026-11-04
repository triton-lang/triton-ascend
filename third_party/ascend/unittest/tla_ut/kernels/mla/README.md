# Paged MLA decode

使用公共 TLA 接口实现 BF16 MLA，测试入口和 profiling 工具沿用 `kernels/dsa` 的组织方式。
当前只覆盖以下四种规格：

| Batch | Q length | KV length | Q heads | KV heads | QK / V | Page size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 131072 | 96 / 48 / 24 / 12 | 1 | 576 / 512 | 128 |

默认 `softmax_scale=0.04`，使用右下对齐 causal mask：query `i` 可见的逻辑 KV 位置满足
`j < 131072 - 4 + i + 1`。KV 前 512 维同时作为 K 的内容部分和 V，后 64 维只参与 QK；
RoPE 变换由上游完成。

- `q`：连续 BF16 `[1, 4, H, 576]`，BSND。
- `kv_cache`：连续 BF16 `[physical_pages, 128, 1, 576]`，PA_BBND。
- `block_table`：连续 int32 `[1, 1024]`，包含合法物理页号，允许乱序和重复页。
- 返回 BF16 output `[1, 4, H, 512]` 和 FP32 logsumexp `[1, 4, H]`。

## 实现

`mla_kernel.py` 包含两个 kernel 和 `forward` 入口：

1. Split-KV：运行时查询 Cube/Vector 核数，根据行块数划分逻辑页，再把工作项循环分配到 Cube。
   不固定物理核数，支持少核时一个核处理多个工作项。当前要求 Ascend950 的 1:2 Cube/Vector 配置。
2. Q 常驻 L1，两份 KV L1 buffer 交替预取。分页 KV 直接通过 `load_gm_to_l1` 完成 ND2NZ。
3. QK 通过 fixpipe ROW_SPLIT 分给两个 AIV。FP32 online softmax 后，BF16 P 用 `vsstb`
   转为 NZ，再通过 `copy_ub_to_l1` 交给 Cube；P 不经过 GM。
4. PV 保持完整 512 列。QK/PV 交换 buffer 的生命周期不重叠，复用 UB 并通过 block flag 保护。
   H96/48/24 使用 M=96，以减少重复读取 KV；H12 使用 M=16。
5. SQ=4 的 causal mask 只影响最后一页，因此前 1023 页省去逐行位置计算和比较。
   最后一页仍严格执行完整 mask，测试中有专门的 causal 边界输入。
6. 第二个纯 AIV kernel 在 FP32 中重缩放、合并各 split 的分子和分母，最后才转 BF16。

每次执行都分配新的 workspace/output，并执行两个 kernel；不缓存输入、KV 数据、softmax 或输出。
`core_limit` 只用于诊断核数映射，默认不限制运行时查询的核数。

## 测试

`test_mla.py` 提供 IR、compile、run 三阶段和可选 profiling，复用现有 `TLA_TEST_STAGE`：

```bash
TLA_TEST_STAGE=ir pytest -q third_party/ascend/unittest/tla_ut/kernels/mla
TLA_TEST_STAGE=compile pytest -q third_party/ascend/unittest/tla_ut/kernels/mla
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/mla

# 也可单独执行；设备沿用现有 ASCEND_RT_VISIBLE_DEVICES 配置。
TLA_TEST_STAGE=run python third_party/ascend/unittest/tla_ut/kernels/mla/test_mla.py --heads 96 48
TLA_TEST_STAGE=run python third_party/ascend/unittest/tla_ut/kernels/mla/test_mla.py \
    --operator both --profile --samples 40 --rounds 3 --out mla-results
```

pytest 配置位于文件顶部：`HEADS`、`SEEDS`、`FAMILIES`、`ACCURACY_OPERATOR`、`CORE_LIMIT`，
以及 `RUN_PERF`、`PERF_OPERATOR`、`PERF_ROUNDS`、`PERF_SAMPLES`、`PERF_COUNTERS`、`PERF_OUTPUT`。
默认只验证 TLA，性能测试关闭；启用 AscendC 对比需要可用的官方 Flash MLA 算子及 Python 封装。
`ASCENDC_PACKAGE` 可指定其安装包名，不会用其他 attention 算子替代。

IR 检查覆盖 MIX/AIV core 类型、ND2NZ、MMAD、ROW_SPLIT fixpipe、UB→L1 数据路径和完整 PV 宽度。
离线 IR 使用 29 核作为诊断配置；真实执行始终查询 runtime。
compile 检查这次 specialization 对应的两个 binary 和缓存文件，不通过搜索其他用例的旧产物判断成功。

精度覆盖 4 heads × 2 seeds × 5 类输入，共 40 组：normal、uniform、peaked、repeated_pages、
causal_boundary。参考是独立的分页 gather + FP32 dense attention，关闭 HF32，并检查所有 output/LSE 元素。
同时要求 finite、最大归一化误差和 relative RMS 满足阈值：output 0.005，LSE 0.0001。

性能测试先验证精度，预热后按轮交替执行顺序，保留每轮原始样本和设备占用快照。
完整调用时间包含分配、核数查询、launch、全部设备工作和同步。`ascendc` 包含官方 metadata 生成，
`ascendc_prepared` 复用仅与长度/形状有关的 metadata，两种路径都经过精度检查并分别报告，
不会只选择较慢的基线口径。固定输入长度和 causal mask 在计时外准备。
AI Core/Vector 任务时间由 profiler 单独采集，
不能代替完整调用耗时；AscendC 的 AICPU metadata 工作仍包含在完整调用计时和原始 trace 中。
占用检查失败时拒绝本轮结果，不发布混跑性能。

性能测试请串行运行，不使用 pytest-xdist。`PERF_COUNTERS=True` 额外导出 Level2 PipeUtilization，
计数器采集与性能计时分开。环境、源码 hash、精度、原始 trace 和结果保存在每次独立结果目录中。
