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

`mla_kernel.py` 包含两阶段计算和 `forward` 入口，每次仍只 launch 两个 kernel：

1. Split-KV：运行时查询 Cube/Vector 核数，根据行块数划分逻辑页，再把工作项循环分配到 Cube。
   不固定物理核数，支持少核时一个核处理多个工作项。当前要求 Ascend950 的 1:2 Cube/Vector 配置。
2. Q 常驻 L1，两份 KV L1 buffer 交替预取。分页 KV 直接通过 `load_gm_to_l1` 完成 ND2NZ。
3. QK 通过 fixpipe ROW_SPLIT 分给两个 AIV。FP32 online softmax 后，BF16 P 用 `vsstb`
   转为 NZ，再通过 `copy_ub_to_l1` 交给 Cube；P 不经过 GM。
4. PV 保持完整 512 列。QK/PV 交换 buffer 的生命周期不重叠，复用 UB 并通过 block flag 保护。
   H96/48/24 使用 M=96，以减少重复读取 KV；H12 使用 M=48，覆盖全部有效行。
5. SQ=4 的 causal mask 只影响最后一页，因此前 1023 页省去逐行位置计算和比较。
   最后一页仍严格执行完整 mask，测试中有专门的 causal 边界输入。
6. M96 使用延迟一页的 PV 调度：Cube 做 PV(t-1)、预取 K(t+1)、QK(t+1)，
   Vector 做 softmax(t) 后合并 PV(t-1)。两份 P 和 alpha 保留延迟输入，score/PV 仍复用 UB。
   flag 4/5 分别保护 score 已读、PV 已读，最后单独 drain；一页 split 有明确的收尾分支。
   M48 的短循环更快，H12 保留原顺序；选择只依赖公开的 tile 形状。
7. 数据就绪的跨核等待放在消费者流水线：Vector V、Cube MTE1、Cube F；工作项结束仍用 S 握手。
8. 第二个纯 AIV kernel 批量读取各 split 的 FP32 原始分子、最大值和分母，先求公共最大值，
   每个 split 的缩放权重只计算一次，再按 128 列分组在寄存器累加。最后才转 BF16，并保留 FP32 LSE。
   超过 64 个 split 时保留逐 split 的有界 UB 回退。当前包的 nested TLA JIT helper 会丢失 TLA
   builder，因此两种 split 调度保留独立入口；没有修改编译器。


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

## 本次验证结果

2026-11-04，Ascend950，runtime 28 Cube / 56 Vector。TA 1103 `0ee9f1a3`，
NPU-IR `b6af414f2289`，均未修改。AscendC 为官方 ops-transformer
`74dc469b5744bea3b5bd06d249fb59747195bddd` 的 Flash MLA，源码未改。
只在允许的卡上运行，未使用物理 4/5/6；主配对实验用物理 2，独立计数器用物理 1。

最终文件直接通过 40 组全量 output/LSE 精度（TLA 与两种官方 AscendC 调用路径均检查），
以及 4 组单 Cube、3 组强制 65/1023/1024 splits 和 2 组 noncausal 诊断。
强制 split 仅改变诊断调度，用同样的目标输入在实际 28 核硬件验证归约回退和单页收尾。
四规格 1/29/192 核离线编译以及 H24 的 1024 核单页配置通过；
离线配置不代表在相应核数的硬件上测过。pytest IR 4 passed、12 skipped，standalone IR 通过。

以下是最终逐字节源码在同一进程、相同普通随机输入下的完整调用中位数，
每项 3×40 样本、逐样本轮换调用顺序。原 PR 指 `8f5bea4e4`，不是历史不同轮次结果。
包含分配、runtime 核数查询、launch、全部设备工作和同步；prepared 为主要 AscendC 基线。

| H | 原 PR (us) | 本次 (us) | 降低 | AscendC prepared (us) | AscendC regenerate (us) | prepared / 本次 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 614.402 | 569.940 | 7.2% | 382.158 | 455.233 | 0.671x |
| 48 | 352.774 | 339.980 | 3.6% | 240.701 | 319.013 | 0.708x |
| 24 | 209.494 | 191.918 | 8.4% | 180.495 | 239.750 | 0.940x |
| 12 | 151.762 | 139.659 | 8.0% | 187.637 | 250.060 | 1.344x |

H96/H48/H24 仍未超过 prepared。H12 本轮快于 prepared，但官方 H12 基线从重启前
约 130 us 变为本轮约 188 us，原因尚未确定，不能把跨轮次变化归因于本次 TLA 优化。
重新核对的官方源码、输入/输出形状和核数一致；独立 profiler 中官方 H12 的 MTE2
活跃时间约 116 us、比率约 0.919，说明当前执行以搬运为主，但不是历史差异的因果证明。

H48 加入延迟 PV 后相对只改等待位置的中间版，wall 338.868→339.980 us，约增加 0.3%；
独立 AI-task 总和 288.072→281.694 us。保留这一差异，不用 task 时间替代完整调用。
最终仍比原 PR 的完整调用快约 3.6%。

Level2 H96 split 中位数：原 PR 545.431 us，本次 517.415 us，官方 324.066 us。
三者 MAC 活跃时间都约 310 us，而活跃比例约 0.569/0.601/0.959；主要剩余方向是
跨页流水重叠和缓冲区周转。计数器可以重叠，不能相加，也不是 MFU。
现有 llvm-objdump 仍不能解码 A5 指令；报告使用 LLVM/HIVM 与 msprof 证据，不作指令级 stall 结论。
此前报告中“UB 预算 216 KiB，无法独立分配 score”有误：最终 HIVM 声明为 248 KiB，
独立 score 候选已编译执行通过，但未比复用 UB 更快，故未采用。

完整原始样本、占用快照、源码 hash、计数器及未采用的候选随本次任务报告提供。
