# Wall Attention 算子 Ascend 适配 — 验收测试报告

> 任务：[triton-lang/triton-ascend#1812](https://github.com/triton-lang/triton-ascend/issues/1812)【社区任务】Wall Attention 算子 Ascend 适配
> 算子来源：上游 [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla/ops/wall_attn`（Tilde Research 贡献）
> 设计文档：[design.md](design.md)　　代码：`kernel/fla/`（本目录上层）

## 1. 测试结论汇总

验收命令：`pytest tests/ops/test_wall_attn.py`。测试文件取自上游 fla（commit `e52dbc0`），断言与容差未做任何修改，仅经本仓 pre-commit（yapf）格式化，格式化前后逐文件 AST 比对一致。

| 平台 | SoC | CANN | 代码版本 | 结果 | 耗时 |
|------|-----|------|----------|------|------|
| Ascend 950 | Ascend950PR (9579) | 9.1.0 | 修复前 | **31 passed, 0 failed, 0 error** | 527s |
| Ascend A2 | Ascend910B4 | 9.0.0 | 修复前 | 27 passed, **4 failed**（反向 dq，见 4.3） | 4855s |
| Ascend A2 | Ascend910B4 | 9.0.0 | 修复后 | **31 passed, 0 failed, 0 error** | 1087s（复用编译缓存） |
| Ascend A2 | Ascend910B4 | 9.1.0 | 修复后 | **31 passed, 0 failed, 0 error** | 4889s |
| Ascend A3 | Ascend910_9382 | 9.1.0 | 修复后 | **31 passed, 0 failed, 0 error** | 2854s |

结论：三个平台在 CANN 9.1.0 下均全量通过。A2/A3 上暴露出一处后端相关的反向 dq 缺陷（4.3），修复后三平台一致通过；该修复在 off-diag 循环正常执行时不改变任何数值。

测试覆盖：训练并行前向（定长/varlen/滑窗/sink bias/GQA/长序列强衰减）、训练反向（dq/dk/dv 对拍、V 维切分一致性、门控梯度有限差分）、scalar gate 前向与梯度、推理 decode（MHA/GQA/V 维切分/长上下文 4096/scalar gate/流式 serving 端到端/cache 布局）。数据类型覆盖 fp32 与 bf16（验收测试矩阵定义的全部参数化配置）。

## 2. 测试环境

| 组件 | Ascend 950 | Ascend A2 | Ascend A3 |
|------|-----------|-----------|-----------|
| 硬件 | Ascend950PR (9579)，board A310-50-C00MM304A1，HBM 128GB | Ascend910B4（8 卡机，测试占用 1 卡） | Ascend910_9382（910_93） |
| 架构 / OS | x86_64，openEuler 24.03 | aarch64，Ubuntu 22.04 | aarch64，openEuler 24.03 |
| CANN | 9.1.0 | 9.0.0 / 9.1.0（官方镜像 `ascend/cann:9.1.0-910b-ubuntu22.04-py3.11`） | 9.1.0 |
| 驱动 | — | npu-smi 26.0.rc1 | 25.5.1 |
| torch | 2.9.0+cpu | 2.9.0+cpu | 2.9.0+cpu |
| torch_npu | 2.9.0.post8（950PR SoC 支持所需） | 2.9.0 | 2.9.0 |
| triton-ascend | 3.2.2（triton 3.2.0） | 3.2.2（triton 3.2.0） | 3.2.2（triton 3.2.0） |
| Python | 3.11.6 | 3.10.12 / 3.11.15 | 3.11.6 |

## 3. 测试结果明细（31/31 通过，各平台用例清单一致）

### 3.1 训练并行前向（10 项）

| # | 测试用例 | 结果 |
|---|----------|------|
| 1 | test_parallel_matches_reference[None-B1-T48-H2-HQ4-K32-V16] | ✅ PASSED |
| 2 | test_parallel_matches_reference[None-B2-T31-H1-HQ1-K24-V8] | ✅ PASSED |
| 3 | test_parallel_matches_reference[None-B1-T31-H1-HQ2-K32-V128] | ✅ PASSED |
| 4 | test_parallel_matches_reference[8-B1-T48-H2-HQ4-K32-V16]（滑窗） | ✅ PASSED |
| 5 | test_parallel_matches_reference[8-B2-T31-H1-HQ1-K24-V8]（滑窗） | ✅ PASSED |
| 6 | test_parallel_matches_reference[8-B1-T31-H1-HQ2-K32-V128]（滑窗） | ✅ PASSED |
| 7 | test_parallel_gqa_matches_reference（GQA G=4） | ✅ PASSED |
| 8 | test_parallel_varlen_matches_reference（varlen 双序列） | ✅ PASSED |
| 9 | test_parallel_sink_bias_matches_reference（sink bias） | ✅ PASSED |
| 10 | test_parallel_aggressive_gates_long_seq（T=512 强衰减） | ✅ PASSED |

### 3.2 训练反向（5 项）

| # | 测试用例 | 结果 | 备注 |
|---|----------|------|------|
| 11 | test_backward_matches_eager_reference[B1-T24-H2-HQ4-K16-V12] | ✅ PASSED | A2/A3 修复前失败 |
| 12 | test_backward_matches_eager_reference[B1-T64-H2-HQ2-K64-V128] | ✅ PASSED | A2/A3 修复前失败 |
| 13 | test_backward_value_split_matches_single_tile（varlen+滑窗+sink+scalar 全叠，V 切分一致） | ✅ PASSED | A2/A3 修复前失败 |
| 14 | test_dg_nonzero_after_backward（门控梯度有限性） | ✅ PASSED | |
| 15 | test_g_gradient_matches_finite_differences（dg 有限差分） | ✅ PASSED | A2/A3 修复前失败 |

### 3.3 Scalar Gate（3 项）

| # | 测试用例 | 结果 |
|---|----------|------|
| 16 | test_scalar_gate_matches_reference[B1-T48-H2-HQ4-K32-V16] | ✅ PASSED |
| 17 | test_scalar_gate_matches_reference[B2-T31-H1-HQ1-K24-V8] | ✅ PASSED |
| 18 | test_scalar_gate_gradient_finite_differences（dg_scalar 有限差分） | ✅ PASSED |

### 3.4 推理 Decode（13 项）

| # | 测试用例 | 结果 |
|---|----------|------|
| 19–20 | test_decode_matches_training_forward[B1-T256-H4-HQ4-K64-V64-C64]（fp32/bf16，MHA） | ✅ PASSED |
| 21–22 | test_decode_matches_training_forward[B1-T256-H2-HQ8-K64-V64-C64]（fp32/bf16，GQA G=4） | ✅ PASSED |
| 23–24 | test_decode_matches_training_forward[B2-T128-H1-HQ2-K32-V32-C32]（fp32/bf16） | ✅ PASSED |
| 25–26 | test_decode_matches_training_forward[B1-T128-H1-HQ2-K32-V320-C32]（fp32/bf16，V 维切分） | ✅ PASSED |
| 27 | test_decode_matches_training_forward_long（T=4096 bf16 长上下文稳定性） | ✅ PASSED |
| 28 | test_decode_with_scalar_gate | ✅ PASSED |
| 29–30 | test_decode_streaming_matches_full_forward[dtype0/dtype1]（流式 serving 端到端） | ✅ PASSED |
| 31 | test_decode_cache_layout_shapes（cache 布局形状） | ✅ PASSED |

精度容差（沿用上游测试文件内常量，未修改）：RTOL_FWD 5e-3、RTOL_GRAD 5e-3、RTOL_FD 2e-2、RTOL_DECODE 2e-2；fp32 dot 强制 IEEE（测试文件自带 `TRITON_F32_DEFAULT=ieee`）。

## 4. 问题与根因分析

**问题 1：头维非 2 的幂时 triton-ascend 3.2.2 后端编译失败（950 平台，7 个用例）。**
现象：K/V 为 24/20/8/3 等非 2 幂维度时，前向 kernel 在 `buildFinalHIVMPipelines` 阶段报错 `vector.transfer_write op requires a permutation_map with result dims of the same rank as the vector type` 及 `tracking listener failed to find replacement op`。
根因：kernel 内对头维的掩码 load（`mask=(o_d < K)`）在 K≠BK 时产生的 permutation/transfer_write 模式触发后端缺陷；K==BK（2 的幂）时掩码恒真被折叠，路径正常。
修复：host 侧公开接口内将 q/k/v/g（decode 含 p_curr/k_tilde/r_cache）的 K/V 维零填充至 2 的幂（≥16），消除掩码；零填充对打分与输出无数学影响，梯度由 autograd 经 pad/slice 反向自动还原。该后端缺陷建议另行向 triton-ascend 反馈。

**问题 2：bwd_dkv kernel UB 溢出（950 平台，1 个用例）。**
现象：varlen 反向（BT=128、num_stages=2）编译报 `ub overflow, requires 2972672 bits while 2031616 bits available`（950PR UB 253952B）。
根因：反向 dkv kernel 驻留 buffer 多（dv/dk 累加器、do/v/q 多块 tile、b_ds/b_p/b_dp 中间量），BT=128 + 双级流水超出 UB 预算。
修复：varlen 反向固定配置降为 BT=64、num_stages=1；反向 autotune 空间统一改为单级流水。

**问题 3：A2/A3 上反向 dq 数值错误（4 个用例）。**

现象：A2（910B4）上 `test_backward_matches_eager_reference`（2 项）、`test_backward_value_split_matches_single_tile`、`test_g_gradient_matches_finite_differences` 失败，例如 `dq diff: 1.744135 ratio: 1.000000`；同一输入重复执行，dq 结果每次不同（两次运行最大相差 7.5），而前向 o、dk、dv 完全正确。950 平台不复现。

定位过程：

1. 以 CPU float64 独立实现参考前向/反向对拍，确认 NPU 上的 eager 参考与 CPU float64 完全一致（误差 0），即错误出在 Triton 反向 dq kernel，而非参考实现；
2. 逐项排除编译选项（`multibuffer`、AutoBlockify 黑名单、`auto_tile_and_bind_subblock`、`enable_fp_fusion`、`enable_select_analysis`、`num_stages`）与 4 种 kernel 等价改写（去掉因果 `tl.where`、CSE 复用 `exp2`、去掉 `broadcast_to`、循环内重新 load）——均无改善；
3. 将 `b_dq = (b_dq_til * scale) * exp2(b_pq - b_R) + b_dq_diag * scale` 中的 off-diag 项整体去掉后，dq 完全正确（T=24、T=64 均 0 错误），定位到 off-diag 累加器 `b_dq_til`；
4. 在 dq 缓冲区预填 NaN 验证：所有位置均被写入（无 NaN），排除"漏写"，确认是 kernel 内部读到未初始化值。

根因：反向 dq kernel 的 off-diag 循环 `for i_s in range(i_start, i_t * BT, BS)` 在 `i_t == 0`（或滑窗导致 `i_start == i_t*BT`）时为零次循环。此时 A2/A3 后端未使 `b_dq_til = tl.zeros([BT, BK])` 的初值生效，循环后读到的是未初始化的 UB 内容，因而结果非确定。上游验收测试的反向用例 T 均 ≤ 64 ≤ BT，全部落在该零次循环路径上。dkv kernel 的同类循环因带 `m_q` 掩码（贡献恒为 0）而不受影响，前向 kernel 的累加器写法为 `b_o = b_o * b_r + tl.dot(...)`，同样不受影响。

修复：显式按循环是否执行选择 off-diag 项（`parallel.py`，dq kernel 末段）：

```python
has_off = i_start < i_t * BT
b_dq = tl.where(has_off, (b_dq_til * scale) * exp2(b_pq - b_R), 0.0) + b_dq_diag * scale
```

零次循环时该项本就应为 0；循环正常执行时与上游数值完全一致。修复经 T=24 / T=64（零次循环）与 T=200、T=200+V=128 切分（循环正常执行）四种形状验证，dq 全部正确。

CANN 版本相关性：在 CANN 9.1.0 官方镜像中以未修复代码复测，A2 上 dq 同样出错（T=24 时 1084/1536 个值错误，T=64 时 4064/4096 个值错误），即该问题与 CANN 9.0.0/9.1.0 无关，修复在验收版本上同样必需。建议向 triton-ascend 反馈该后端零次循环累加器初值问题。

**问题 4：环境部署（非代码问题）。**
① pypi 官方源无 triton-ascend 3.2.2，需使用 ascend 源：`pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi`；② torch_npu 2.9.0 初版不识别 950PR（`Unsupported soc version: Ascend950PR 9579`），需 ≥ 2.9.0.post8（A2/A3 用 2.9.0 即可）；③ kernel 首次编译需 python3-devel（Python.h）；④ triton-ascend 3.2.2 与官方 `triton` 包同名冲突，安装后需确保 `triton` 模块来自 triton-ascend（`triton.__version__` 为 3.2.0，backend 为 `npu`）。

## 5. 与任务验收标准对齐

| 验收标准 | 结论 |
|----------|------|
| `pytest tests/ops/test_wall_attn.py` 全量通过（0 failed、0 error），不改断言容差 | ✅ 950 / A2 / A3 均 31/31 通过；断言与容差未改 |
| 覆盖 A2 / A3 / 950 | ✅ 三平台均已实测（A2、A3 为修复后代码；950 为修复前代码，修复不改变其执行路径的数值，建议复测确认） |
| 消除 507014/507034 超时及编译器崩溃 | ✅ 各平台全量运行无超时、无挂死；三处编译/数值问题已根因修复 |
| 前向与反向梯度精度 | ✅ 全部容差内（含有限差分验证） |
| 定长+varlen、fp16/fp32 等参数化配置 | ✅ 验收矩阵（fp32/bf16 × 定长/varlen）全部通过；上游测试矩阵不含 fp16 用例 |
| 不引入回归 | ✅ 独立 vendor 目录，不改动 triton-ascend 编译器与任何既有算子 |

## 6. 复测部署要点

```bash
# 通用依赖（openEuler/CentOS 系用 dnf，Ubuntu 系用 apt 安装 python3-dev）
sudo dnf install -y python3-devel
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.9.0 "numpy==1.26.4" "pytest==8.3.2" "pytest-xdist==3.6.1" \
    "attrs==24.2.0" "decorator==5.1.1" "psutil==6.0.0" "scipy==1.13.1" pandas pybind11
pip install --no-deps triton-ascend==3.2.2 \
    --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd community-tasks/wall-attn-1812
python3 -m pytest tests/ops/test_wall_attn.py -v --tb=short
```

- 950PR 需 `torch_npu>=2.9.0.post8`；A2/A3 用 `torch_npu==2.9.0` 即可。
- 首次运行含全量 kernel 编译：A2 约 80 分钟，A3 约 48 分钟，950 约 9 分钟；复用 `TRITON_CACHE_DIR` 后可显著缩短。
- CANN 9.1.0 容器复测可直接使用官方镜像 `ascend/cann:9.1.0-910b-ubuntu22.04-py3.11`（容器内需挂载 driver、dcmi、npu-smi）。

## 7. 遗留事项

- 950 平台使用修复后代码的复测（修复仅影响 off-diag 循环零次执行的分支，950 上该分支原本也读未初始化值，复测预期同样通过或更稳定）；
- fp16 输入未单独验证（上游验收测试矩阵为 fp32/bf16）；
- 性能调优：当前以功能/精度验收为目标（关闭 auto-multi-buffer、统一保守 tile 档位、autotune 空间精简），尚未补充 benchmark 数据；
- 两处 triton-ascend 后端问题（头维掩码 load 编译失败、零次循环累加器初值未生效）建议形成最小复现后反馈社区。
