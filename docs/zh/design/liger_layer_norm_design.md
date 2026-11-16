> **状态**：设计评审阶段（Draft）
> **关联任务**：[triton-lang/triton-ascend#1800](https://github.com/triton-lang/triton-ascend/issues/1800)（任务池 [#1796](https://github.com/triton-lang/triton-ascend/issues/1796)，编号 08-03「Liger LayerNorm 性能优化」）
> **更新日期**：2026-10-21

---

# LigerLayerNorm 昇腾适配与性能优化设计文档（A2 / A3 / A5）

## 1. 任务与验收标准

### 1.1 任务概述

在昇腾 NPU 上基于 Triton-Ascend，完成 Liger-Kernel v0.8.2 中 `LigerLayerNorm` 算子的亲和适配与性能优化，
前向、反向结果与 `torch.nn.LayerNorm` 对齐，性能不慢于 torch 原生实现，覆盖 A2 / A3 / A5 三类平台。

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| liger_kernel | 0.8.2 |
| triton-ascend | 3.2.2 |
| CANN | 9.1.0 |
| torch_npu | 与 CANN 9.1.0 配套版本 |

### 1.3 验收标准（摘自 issue 原文）

1. `benchmark/scripts/benchmark_layer_norm.py` 全部配置（forward / backward / full 各 `kernel_operation_mode`）
   正常产出数据，**无崩溃、无超时**；
2. **性能达标**：所有配置下 liger kernel 耗时均不慢于 baseline（**ratio ≤ 1.0**），提交优化前后逐配置对比数据表；
3. **平台覆盖**：在 **Ascend A2、A3、950** 平台上均可运行且满足上述性能标准；
   如因平台硬件差异存在限制，须在 PR 中说明原因；
4. **精度不回归**：`tests/test_layer_norm.py` 全量通过（0 failed / 0 error），**不得修改断言与容差**；
5. PR 附 profiling 根因分析、优化手段说明与复测命令及结果。

### 1.4 与 #1796 任务池的关系

`#1796` 是任务池索引，**不引入新的技术标准**。横向核对同批任务（#1800 LayerNorm / #1804 GroupNorm）正文，
任务标准五条、测试标准、验收标准逐字一致。任务池仅追加流程约定：

- 领取后 2 周内完成设计文档并评论「申请文档验收」；
- 整体完成后评论「申请验收」；
- 验收通过后在 `Ascend/triton-ascend-kernels` 的 `experimental` 分支提 PR。

---

## 2. 算子接口与约束

### 2.1 接口

| 参数 | 方向 | 类型 | shape | 说明 |
|---|---|---|---|---|
| `x` | 输入 | FP16 / BF16 / FP32 | `[B, T, H]`（实际支持任意 `...×H`） | 归一化输入 |
| `weight` | 输入 | 同 x | `[H]` | 可学习缩放 |
| `bias` | 输入 | 同 x | `[H]` | 可学习偏置 |
| `eps` | 属性 | FLOAT | - | 默认 1e-6 |
| `out` | 输出 | 同 x | `[B, T, H]` | 归一化结果 |

反向产出 `dx [B,T,H]`、`dweight [H]`、`dbias [H]`。

### 2.2 约束

- 支持任意 `hidden_size`，含非 2 的幂（测试覆盖 64 / 1023 / 1500 / 256 / 128，benchmark 覆盖 2048~7168）；
- 前向与反向均需与 `torch.nn.LayerNorm` 在容差内一致；
- **不允许修改测试用例断言与容差**。

### 2.3 数学定义

对每一行 `x ∈ R^H`：

```
mean = (1/H) Σ x_i
var  = (1/H) Σ (x_i − mean)²
rstd = 1 / sqrt(var + eps)
y_i  = (x_i − mean) · rstd · γ_i + β_i
```

反向（LayerNorm 标准形式）：

```
x̂_i  = (x_i − mean) · rstd
c1   = (1/H) Σ x̂_i · γ_i · dy_i
c2   = (1/H) Σ γ_i · dy_i
dx_i = (γ_i·dy_i − (x̂_i·c1 + c2)) · rstd
dγ_i = Σ_batch dy_i · x̂_i
dβ_i = Σ_batch dy_i
```

---

## 3. 上游实现分析

### 3.1 Liger-Kernel v0.8.2 已自带 Ascend 后端

`src/liger_kernel/ops/backends/_ascend/ops/layer_norm.py` 已存在，并通过 `select_impl()` 在
`infer_device() == "npu"` 时自动替换默认 GPU 实现：

```
liger_kernel.ops.__init__ → select_impl(device) → liger_kernel.ops.backends._ascend.ops.layer_norm
```

**因此本任务不是从零移植，而是性能优化。**

### 3.2 现有 Ascend 实现结构

| kernel | 触发条件 | 策略 | 问题 |
|---|---|---|---|
| `_layer_norm_forward_fused_2d` | `BLOCK == n_cols` 且 UB 容纳多行 | 2D tile，单次 load，`E[x²]−mean²` | 仅适用于 2 的幂 H |
| `_layer_norm_forward_row` | 同上但只容 1 行 | grid-stride 逐行 | 同上 |
| `_layer_norm_forward_tiled` | 整行放不进 UB | **三遍**列 tile（均值 / 方差 / 仿射） | **每遍都从 GM 重读 X，共 3 遍** |
| `_layer_norm_backward_rows` | `BLOCK >= n_cols` | 逐行 + `tl.atomic_add` 累加 dW/dB | atomic 串行 |
| `_layer_norm_backward_tiled` | 列 tile | 两遍 + 逐 tile atomic | 多遍 + 原子 |

host 侧用 `_fused_forward_tile()` 按 UB × `memory_multiplier`(6.0/4.0/8.0) × 0.85 反推 tile；
当 `next_pow2(n_cols) > n_cols` 时直接退回三遍路径。

### 3.3 问题归纳

| 编号 | 问题 | 影响 |
|---|---|---|
| R1 | 非 2 的幂 `hidden_size`（5120 / 3584 / 7168）立即退化到三遍列 tile | X 从 GM 读 3 次，带宽浪费 2 倍 |
| R2 | 反向每 program 维护两个 `BLOCK_SIZE` fp32 累加器 + atomic | UB 压力大，`BLOCK` 被压小 → 落入多遍路径 |
| R3 | `memory_multiplier=8`（反向）过度保守 | tile 过小，循环次数增多 |
| R4 | 全局限定 `BLOCK >= n_cols` 无显式校验 | 若 `BLOCK < n_cols` 会静默截断统计量 |

---

## 4. 平台差异分析

### 4.1 硬件规格

| 平台 | 产品 / SoC | 架构 | UB | L1 | L0C | L2 | Cube:Vector |
|---|---|---|---|---|---|---|---|
| **A2** | Atlas 800T/800I A2、910B1~B4、910_93xx | arch22 | 192 KB | 512 KB | 128 KB | 192 MB | 1:2 |
| **A3** | Atlas 800T/800I A3、910_93 | arch22 代际 | 192 KB | 512 KB | 128 KB | 192 MB | 1:2 |
| **A5** | Ascend 950PR / 950DT（910_95 / 950） | arch35 | **248 KB** | 512 KB | **256 KB** | **128 MB** | 1:2 |

数据来源：`triton/backends/ascend/runtime/cv_autotune/hardware_consts.py`、`runtime/utils.py`
（910_95 / 950 判定为 `ub_size_in_kbytes=256`、`rf_size_in_kbytes=128`）。

### 4.2 对 LayerNorm 的影响

LayerNorm 是**纯 Vector 访存密集型算子**（每元素 2~4 flop，无 Cube 路径），平台差异只落在
「片上容量 → tile 尺寸」和「核数 → grid」两处，**算法路径完全一致**。

| 差异维度 | A2 | A3 | A5 | 设计应对 |
|---|---|---|---|---|
| UB 容量 | 192 KB | 192 KB | 248 KB | 运行时探测，**下限以 A2 校准**（A2 能过的配置 A3/A5 必过） |
| Vector 核数 | 按产品 | 按产品 | 按产品 | `num_vectorcore` 运行时探测，不写死 |
| 尾轴对齐 | VV 类需 32B | 同 | 同 | `BLOCK == n_cols` 且 32B 对齐才走无掩码快路径 |
| 寄存器文件 | 无 | 无 | 128 KB RF | A5 可尝试更大 row tile（可选增强，非必需） |
| 编译路径 | `c100` / `ascend_910` | 同 | 同 | 无需平台分支 |

**结论：一套算法 + host 侧按平台调参**，不做三份 kernel。与同批 #1804 的表述
（「平台差异通过每平台 BLOCK_SIZE 上限与 tile 配置吸收，算法路径一致」）保持一致。

---

## 5. 设计方案

### 5.1 总体思路

```
host 侧平台探测（ub_kb / num_vectorcore / soc）
        │
        ▼
tile 规划器（按 UB 反推，A2 为下限）
        │
        ├── 前向 F1：单遍 2D tile（mask / 无 mask 两种特化）
        ├── 前向 F2：大 H 兜底（2 遍，整行放不进 UB 时）
        ├── 反向 B1：单遍 row kernel + atomic 累加 dW/dB
        └── 反向 B2：大 H 两段列切分
```

三条主线：

1. **X 只从 GM 读一遍**（消除 R1）；
2. **非 2 的幂 hidden_size 不退化**（消除 R1）；
3. **反向减少 launch 与归约**（消除 R2/R3）。

### 5.2 前向 F1（主路径）

```python
s1   = tl.sum(xf, axis=0)                     # 行内 fp32 求和
s2   = tl.sum(xf * xf, axis=0)                # 行内 fp32 平方和
mean = s1 * inv
var  = s2 * inv - mean * mean
rstd = rsqrt(tl.maximum(var, 0.0) + eps)      # 防抵消产生负值
y    = (xf - mean) * rstd * W + B
```

- **单次 load、单次 pass**，与 `n_cols` 是否为 2 的幂无关；
- 采用 `E[x²] − mean²` 而非两遍求方差，换取单遍读；
- `tl.maximum(var, 0.0)` 防止浮点抵消产生负数导致 `rsqrt` 出 NaN；
- 当 `BLOCK == n_cols` 且 `n_cols × dtype_size` 为 32B 整数倍时，编译期常量化列掩码，走无掩码特化。

**前置约束**：`BLOCK` 必须覆盖整行（`BLOCK >= n_cols`）。否则统计量只累加了部分列，
因此实现中显式校验，不满足时抛错而不是静默产生错误结果。

### 5.3 前向 F2（大 H 兜底）

当 `next_pow2(n_cols) × 4B` 超出 UB 预算（A2 上约 `n_cols > 8192`）时，整行无法装入片上内存，
必须做列切分。F2 用两遍：

- pass-1：累加 `sum`、`sumsq`（只读 X 一遍）；
- pass-2：读 X 写 Y（第二遍）。

相比上游的三遍，少读一遍。

### 5.4 反向 B1（主路径）

- 每 program 处理一组行，`x̂` / `wdy` / `c1` / `c2` / `dx` 全程 fp32；
- `dγ` / `dβ` 在 program 内以 fp32 累加，结束时用 `tl.atomic_add` 直接累加到
  `(n_cols,)` 的 fp32 buffer；
- 相比「每 program 写 `(grid, n_cols)` 部分和 + host 侧 `sum(0)`」，少一次 kernel launch 和一次归约。

**原子性说明**：不同 program 累加到同一 buffer 的不同位置只发生在最后一个 program 结束时，
各 program 一次性 `atomic_add` 自己负责的 `dγ`/`dβ` 向量，累加次数等于 program 数，
精度上无问题（fp32 累加）。

**注意**：`dγ`/`dβ` buffer 必须以 `zeros` 初始化；且**不能跨调用复用**，
因为 `atomic_add` 是累加语义，复用会带入上次结果。

### 5.5 反向 B2（大 H）

反向 kernel 活跃 fp32 tile 比前向多，UB 上限更低（A2 上 `BLOCK ≤ 4096`）。当 `n_cols > 4096` 时：

- kernel-1：按列 chunk 遍历，累加 `dγ`/`dβ`；
- kernel-2：按行遍历，先归约 `c1`/`c2`（列 chunk 循环），再写 `dx`。

### 5.6 平台参数化

```python
ub_kb  = runtime.utils.ub_size_in_kbytes          # A2/A3=192, A5=256（可用 248）
cores  = driver.utils.get_device_properties()["num_vectorcore"]
block  = next_pow2(n_cols)
assert block * 4B * 8 * mult * 0.85 <= ub_kb*1024*8   # mult 按 kernel 类型区分
```

- **不写死平台字符串**，A2 / A3 / A5 自动走正确参数；
- `mult` 按 kernel 区分：前向 5.0、反向 10.0（活跃 fp32 tile 更多）；
- `ub_bits_effective()` 取运行时探测值，并以 A2 的 192 KB 作为下限兜底。

### 5.7 grid 选择

层归一化每个 program 处理多行，grid 决定并行度与循环次数的平衡：

- 候选集：`{cores, 2×cores, 4×cores, 8×cores}`，且不超过 `n_rows`；
- **不把 `n_rows` 作为候选**（会让每 program 只处理一行，循环退化）；
- 首次遇到某个 shape 时实测选取，**取 3 轮测量的中位数**（单次采样易受节点负载干扰），
  之后按 `(kernel, n_cols, n_rows, dtype)` 缓存。

**autotune 安全约束**：若被测 kernel 会累加 `dγ`/`dβ`，探针必须使用**独立的临时 buffer**，
不能复用正式 buffer，否则探针的多次执行会把结果累加多倍。

---

## 6. 精度设计（不可退让项）

1. 统计量与反向归约全程 fp32；fp16/bf16 载入后立即 `.to(tl.float32)`，store 前转回；
2. `var = E[x²] − mean²` 结果与 0 取 `tl.maximum` 后再加 eps，防止抵消产生负数导致 `rsqrt` 出 NaN；
3. `rsqrt` 使用 `triton.language.math.rsqrt`（NPU 路径）；A5 若 lowering 异常退化为 `1/tl.sqrt(...)`；
4. `dγ`/`dβ` 用 fp32 累加，结束后一次性 cast 回权重 dtype；
5. **不改断言、不改容差、不 skip 用例**。

---

## 7. 三平台验证矩阵

| 平台 | `pytest tests/test_layer_norm.py` | `benchmark_layer_norm.py` | 定位 |
|---|---|---|---|
| **A2**（910B3 基准卡） | 0 failed / 0 error | fwd / bwd / full × 7 模型全部 ratio ≤ 1.0 | **性能结论主口径** |
| **A3**（910_93） | 同上 | 同上 | 证明可运行且达标 |
| **A5**（950PR/950DT） | 同上 | 同上；硬件差异须在 PR 说明 | 证明可运行且达标 |

> 若某平台因硬件差异无法达标，须在 PR 中说明原因并附实测数据，不得静默跳过。

---

## 8. 环境可行性

### 8.1 已确认的技术前提

| 项 | 结论 | 依据 |
|---|---|---|
| NPU 可用 tile 上限 | 前向 masked 单遍 `BLOCK ≤ 8192`；反向 row kernel `BLOCK ≤ 4096` | 编译探针（超出时报 ub overflow） |
| `tl.atomic_add` 语义 | 标量与向量累加均正确 | 最小 kernel 验证（各 grid 下累加值精确） |
| 无掩码特化可行性 | 2 的幂 hidden_size 且 32B 对齐时可消去掩码 | 编译期常量折叠 |
| 非 2 的幂 hidden_size | 单遍 masked 路径可覆盖，无需退回三遍 | 与上游三遍路径对比 |

> 上述为前提验证，量化结果见配套的《验证报告》，本文不展开。

### 8.2 已知环境问题

| 问题 | 现象 | 解决 |
|---|---|---|
| git 全局 `insteadOf` 指向失效代理 | clone 失败 | `git -c url."https://github.com/".insteadOf= clone ...` |
| `kernels` 0.14.1 与 `huggingface_hub` 类型校验冲突 | `StrictDataclassFieldValidationError: 'import_name': str \| None` | 将 `kernels/deps.py` 的 `str \| None` 改为 `Optional[str]` |
| `peft` / `kernels` 对 hub 版本互相拉扯 | 导入失败 | 保持 `huggingface_hub 0.36.2` + 上述补丁 |
| `benchmark_layer_norm.py` 默认不覆盖旧数据 | 新结果不写入 CSV | 运行需加 `--overwrite` |

### 8.3 A3 / A5 设备状态

- A3 容器（22005 / 22006）驱动 `rtGetSocVersion` 上报 `Ascend910_9382`，与 910_93 同属
  triton-ascend 的同一编译路径（`c100` / `ascend_910`），UB 同为 192 KB，tile 规划口径一致；
- A5（950PR/950DT）当前无可用设备，相关章节按硬件规格设计，**待设备到位实测**。

---

## 9. 实施计划

| 阶段 | 内容 | 产出 |
|---|---|---|
| P0 | 环境搭建 + 上游代码获取 | 可运行环境 |
| P1 | A2/A3 功能验证（不改代码先跑通） | pytest 全量通过 |
| P2 | A2 性能基线 | 逐配置 ratio 表 |
| P3 | 实现前向 F1（单遍 + 任意 H） | 前向达标 |
| P4 | 实现反向 B1/B2（atomic + 大 H 两段） | 反向达标 |
| P5 | A2/A3 全量复测 + profiling 根因 | 优化前后对比表 |
| P6 | A5 复测（248 KB UB，复核 rsqrt lowering） | A5 测试记录 |
| P7 | PR 提交至 `triton-ascend-kernels` experimental 分支 | PR |

## 10. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 小 shape（`n_rows` 小）时 grid < 核数，吃不满 AI Core | 单配置 ratio 抖动 | `n_rows` 自适应 grid + 逐 shape 选参 |
| H 极大（> UB 单行容量）必须多遍 | 该配置难达标 | 两遍方案 + 与 torch 同策略对齐 |
| A5 上 `rsqrt` / `atomic` lowering 差异 | 精度或崩溃 | 保留 `1/sqrt` 回退开关；A5 单独验证 |
| A3 / A5 无设备 | 无法完成平台覆盖验收 | 需申请设备 |

---

## 11. 参考资料

1. [liger-kernel/liger-kernel](https://github.com/linkedin/Liger-Kernel) v0.8.2
2. [Triton-Ascend 文档](https://triton-ascend.readthedocs.io/zh-cn/latest/)
3. [Triton-Ascend 算子开发指南](https://triton-ascend.readthedocs.io/zh-cn/latest/programming_guide/index.html)
4. [Triton-Ascend 安装指南（版本配套矩阵）](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html)
5. 上游测试：`test/transformers/test_layer_norm.py`
6. 上游 Benchmark：`benchmark/scripts/benchmark_layer_norm.py`
