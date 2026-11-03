# DSA / Sparse MLA prefill

使用 TLA 实现的稀疏注意力 prefill 算子，支持 pytest 精度检测、完整调用计时和 NPU profiling。
当前仅有四个 prefill 规格：B=1、SQ=KV length=8192、TopK=2048，Q heads 分别为 16、32、64、128。

## 性能

BF16，B=1，SQ=KV length=8192，TopK=2048，内容维 512、RoPE 维 64；
normal 输入，seed=1103。测量日期 2026-11-03，物理卡 7，编译目标 `Ascend950PR_9579`。
每个实现预热 5 次，3 轮各 40 个样本，逐样本交替执行顺序，取全部 120 个样本中位数。

| Q heads | TLA 完整调用 / ms | AscendC 完整调用 / ms | 完整调用加速比 | TLA 设备任务 / ms | AscendC 设备任务 / ms | 设备任务加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 30.236 | 32.165 | 1.064× | 30.587 | 32.668 | 1.068× |
| 64 | 15.865 | 19.995 | 1.260× | 15.956 | 20.078 | 1.258× |
| 32 | 13.447 | 19.082 | 1.419× | 13.403 | 19.193 | 1.432× |
| 16 | 12.841 | 19.032 | 1.482× | 14.018 | 20.993 | 1.498× |

加速比均为 AscendC/TLA，基线是 `torch_npu.npu_sparse_flash_attention`。
完整调用计时包含 KV/RoPE 合并、内存分配、核数查询、全部 kernel 和设备同步。
设备任务时间单独通过 profiler 采集，TLA 包含 `_pack_kv` 和 `_sparse_mla` 两个任务；
它与完整调用不来自同一组采样，不能相减推算 host 开销。

四种规格的完整调用与设备任务均快于 AscendC。精度数据覆盖 5 类分布 × 2 个种子 × 4 种 heads，
共 40 组、120 条输出指标全部通过。
上述数据来自精简前的实现，精简后尚未重新上卡复测。
当前 pytest 工具按轮轮换执行顺序，每轮完整调用采样最多 10 次，与上表的逐样本交替口径不同。

## 文件与环境

- [dsa_kernel.py](dsa_kernel.py)：KV/RoPE 合并、稀疏 attention 和运行时入口。
- [test_dsa.py](test_dsa.py)：DSA 输入、参考计算、精度和性能测试。
- [../utils.py](../utils.py)：各 kernel 共享的模块加载、设备快照、profiling、计数器和结果保存。

运行需要兼容的 Ascend950、CANN、torch_npu 和 TLA 构建，包含 `copy_ub_to_l1` 等公共接口。
先激活 Python 环境并加载 CANN 的 `set_env.sh`。
本目录不提供 IR/compile 检测；运行设备用例仍会触发必要的 JIT 编译。

## 配置与运行

统一通过 pytest 执行，配置直接修改 [test_dsa.py](test_dsa.py) 顶部的全局变量，不提供 CLI 或 DSA 专用环境变量。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `HEADS` | `(128, 64, 32, 16)` | Q head 数 |
| `SEED` | `1103` | 固定随机种子 |
| `ACCURACY_FAMILIES` | 五种分布 | normal、uniform、peaked、duplicates、invalid；仅精度检测使用 |
| `ACCURACY_OPERATOR` | `'tla'` | 精度检测对象：tla、ascendc、both |
| `CORE_LIMIT` | `None` | 可选的 TLA 启动核数上限 |
| `RUN_PERF` | `False` | 是否启用性能测试 |
| `PERF_OPERATOR` | `'both'` | 性能比较对象 |
| `PERF_ROUNDS` / `PERF_SAMPLES` | `3` / `40` | 采样轮数与每轮设备采样次数 |
| `PERF_COUNTERS` | `False` | 额外采集 Level2 PipeUtilization，需要 msprof |
| `PERF_OUTPUT` | `Path('dsa-results')` | 性能结果目录，每个用例创建唯一子目录 |

从仓库根目录运行：

```bash
# 全量精度检测；默认性能用例跳过
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/dsa

# 仅精度
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/dsa -k test_run

# 先在脚本中设置 RUN_PERF = True，再运行性能测试
TLA_TEST_STAGE=run pytest -q -s third_party/ascend/unittest/tla_ut/kernels/dsa -k test_perf
```

沿用公共 `TLA_TEST_STAGE`：`run` 执行设备测试，`ir`/`compile` 跳过，默认 `all` 在有 NPU 时运行。
设备由父目录 fixture 分配；可通过已有的 `ASCEND_RT_VISIBLE_DEVICES` 选择可见物理卡。
性能测试不使用 xdist，启动采样前检查设备上是否有其他进程。
pytest 收集仍需父目录公共工具依赖的 Triton/TLA 扩展。

### 精度

`test_run` 只参数化 heads，内部遍历 `ACCURACY_FAMILIES`，失败信息包含分布和种子。
每个分布检查全部 query/head/输出元素，使用关闭 HF32 的 NPU FP32 参考，结束后恢复设置。
检查输出有限性、最大归一化误差及 relative RMS。最大归一化误差为
`max(abs(actual-reference)) / max(abs(reference))`，output 门限 0.005，max/sum 门限 0.0001。
失败由 pytest 显示误差指标。大 prefill 占用较多显存，建议串行执行。
当前测量环境的 AscendC 对 `-1` 索引与屏蔽参考存在行为差异，`invalid` 建议只测试 TLA。

### 性能与 profiling

`test_perf` 只参数化 heads，固定使用 `normal` 分布与 `SEED`，独立执行精度预检查，通过后才开始采样，不依赖 `test_run` 的执行顺序。
每轮预热 5 次，TLA/AscendC 按轮轮换执行顺序，始终包含 KV/RoPE 合并和 attention 两个 kernel。

| 文件或字段 | 内容 |
| --- | --- |
| `environment.json` | 配置、版本、设备核数和算子源码 SHA256 |
| `results.json` | 当前用例的精度、状态和逐轮采样结果 |
| `<operator>-<round>.json` | Level1 Chrome trace |
| `samples_us` | 每次调用全部设备任务时长之和 |
| `wall_samples_us` | 无 profiler 的完整 Python 调用耗时，每轮 `min(10, PERF_SAMPLES)` 次 |
| counters 子目录 | 可选的 Level2 数据和 msprof CSV |

输入和编译预热不计入采样，完整调用包含分配、核数查询、发射及同步。
设备任务时间和完整调用时间分别采集；流水线计数时间可能重叠，不能相加作为总延迟。
每个用例结束都会保存结果，包括精度或采样失败。入口不导出 IR 或二进制。

## 输入输出

固定 B=1、SQ=KV length=8192、TopK=2048，H 支持 16、32、64、128。
所有输入必须连续并位于同一 NPU：

| 输入 | 形状 | dtype |
| --- | --- | --- |
| q | `[1,8192,H,512]` | BF16 |
| kv | `[1,8192,1,512]` | BF16 |
| qr | `[1,8192,H,64]` | BF16 |
| kr | `[1,8192,1,64]` | BF16 |
| indices | `[1,8192,1,2048]` | int32 |

```python
from dsa_kernel import forward
output, maximum, denominator = forward(q, kv, qr, kr, indices)
```

返回 BF16 output `[1,8192,H,512]`，以及 FP32 maximum/denominator `[1,1,8192,H]`。
重复索引保留其权重，范围外索引屏蔽；每个 query 至少需要一个有效索引。
不包含 indexer、TopK 生成、因果 mask、paged cache、sinks 或后续模型投影。
缩放系数固定为 1/24，接入其他模型时须核对其实际 scale。

## 算法原理

### 稀疏 attention 与 MLA

普通 attention 对全部 KV 做 `QKᵀ → softmax → PV`；本算子按上游提供的索引选出 2048 个 KV，
再做同样的计算。Q 决定关注什么，K 用于匹配，V 是最终汇总的内容。
多个 Q heads 共享一个 KV head，但各自计算不同的权重和输出。

MLA absorb 将 key 的投影吸收到 Q，将 value 的投影放到 attention 之后。设 latent cache 为 c：

$$q^T(W_Kc)=(W_K^Tq)^Tc,\qquad
\sum_j p_j(W_Vc_j)=W_V\left(\sum_jp_jc_j\right).$$

因此同一份 512 维 KV 在 QK 中作为内容 key，在 PV 中作为 value。
64 维 RoPE 位置向量已由上游完成旋转，只参加 QK 打分，通过权重间接影响输出：

$$s_j=(q\cdot kv_j+q_r\cdot k_{r,j})/24,\qquad
m=\max_j s_j,\quad l=\sum_j e^{s_j-m},\quad o=\sum_j e^{s_j-m}kv_j/l.$$

### Online softmax

每次只处理 BN=128 个选中 KV，不保存完整 score/P 矩阵。
在片上维护最大值 m、指数和 l、未归一化分子 u。新 tile 到来时：

$$m'=\max(m,\max_{j\in tile}s_j),\qquad a=e^{m-m'},$$
$$l'=al+\sum_{j\in tile}e^{s_j-m'},\qquad
u'=au+\sum_{j\in tile}e^{s_j-m'}v_j.$$

最终输出 u/l。分子、分母必须同时重缩放，才能保留之前 token 的正确权重。
max/sum/correction 和输出累加为 FP32；P 转 BF16 参加 PV，最终输出也转 BF16。

## TLA 实现

```text
KV + RoPE → _pack_kv → GM 中连续的 576 维行
索引 → GM 离散读取 → UB → NZ 排列 → L1
QK = Q_lat × K_latᵀ + Q_rope × K_ropeᵀ → Fixpipe → softmax
P（BF16）→ L1 → PV512 → Fixpipe → FP32 原地累加 → 归一化输出
```

每次调用重新生成约 9 MiB 的 packed KV，不缓存选中的 KV 或结果。
一个 Cube 配两个 AIV：gather 按 KV 行分工，softmax/输出按 heads 分工。

### 三槽流水与滚动 gather

L1 的 512 维 latent KV 使用三个槽，64 维 RoPE 使用两个槽。QK 分成两次 MMAD，
先计算内容项，再累加 RoPE 项，使 latent 三槽、RoPE 两槽、Q 和 P 能容纳在 L1 中。

稳态 Vector 处理 gather(t+1)、softmax(t)、accumulator(t-1)；Cube 处理 QK(t)、PV(t-1)。
SCORE_FREE、P_FREE、PV_FREE 分别保护 score、P 和 PV 缓冲的复用，最后显式排空末 tile。
这些先后顺序是正确性的一部分，不能仅为消除重复代码而移动 gather 或 flag。

每个 AIV gather 64 行，按 16 行分四组。稳态先发出前两组 DMA，再打包第 0/1 组并分别
发出第 2/3 组 DMA，最后打包剩余两组，让行搬运与 NZ 排列、UB→L1 拷贝重叠。
首次预取保留独立的 prologue，它与稳态的流水安排不同。

### 索引、缓冲与数值

`indices.view(torch.int64)` 仅建立存储别名；一次 64-bit load 拆出两个 int32 索引，
不改变外部输入 dtype，也不分配或缓存另一份索引。固定 L=8192 时使用 `index & (L-1)`
产生安全搬运地址，softmax 再用原 int32 索引的 unsigned 比较屏蔽范围外元素。
这不是支持任意长度的通用取模方案，不能直接推广到非 2 的幂长度。

`key0/key1` 每行保留 640 个 BF16：512 个 KV、64 个 RoPE、64 个 padding。
RoPE 与 KV 共享 allocation；DMA 只搬 576 个有效元素。末尾 SIMD load 使用 padding，
NZ 写入通过 mask 只保留 64 个 RoPE 元素。NZ staging 的 `GR+1` pitch 也保留不变。

PV 全宽 512，输出改为一份 FP32 accumulator 原地更新；correction 使用两槽，
使延迟一拍的 PV 使用相应 tile 的缩放因子。softmax 使用显式 `vsub + vexp`，
不替换成 `vexpdif`。

| Q heads | Head tile（BM） | Key tile（BN） | PV 宽度 | Gather batch |
| --- | ---: | ---: | ---: | ---: |
| 128 / 64 | 64 | 128 | 512 | 16 |
| 32 | 32 | 128 | 512 | 16 |
| 16 | 16 | 128 | 512 | 16 |

核数由 runtime 查询，按 `core_id, core_id + core_count, ...` 分配作业。
H128 仍拆成两个独立 BM64 作业，会重复 gather 同一 query 的 KV；本版本没有跨核共享 gather。

### MMAD 接口

QK 使用正式接口 `tla.cube.mmad(..., transpose_b=True)`，B 的逻辑形状为
`(N, K)`。512 维 latent 乘积指定 `clear_accum=True`，64 维 RoPE 乘积指定
`clear_accum=False`，累加到同一 L0C。转置由 MMAD 原生处理，无需额外搬运。
运行时需要安装当前分支的 TLA 前端；旧安装包可能缺少转置形状检查修复。

## 验证

四种 heads 均通过离线编译；忽略源码位置、SSA 名称及常量声明顺序后，
ttadapter 边界 IR 与精简前一致。
四种 heads × 三种核数设置（默认/7/13）的宿主调用对比通过，输入输出、分配和 launch 参数一致。
