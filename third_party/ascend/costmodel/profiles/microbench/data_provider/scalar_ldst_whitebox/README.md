# scalar load/store CostModel 白盒建模与验证（targeted）

> 对象：`test_cases/scalar_dominate_kernels/` 中 6 个 scalar-dominated megablocks kernel
> （`padded_copy_{gather,scatter,wgrad}`、`binned_copy_{gather,scatter,wgrad}`）。
> 固定条件：`shape=(sl,hs,ne,top_k)=(4,256,4,2)`、`BLOCK_X=64`、`superblock_factor=1`、`num_warps=1`。
> 当前 route：6 个 kernel 全部 `all_simt_only`；本文同时给出 forced `simd` 的实现公式对照。
> 代码基线：`feature/simd-simt-compile-mode` + 本 PR。
> **单位口径**：CAModel 1.8 GHz core cycle 按 `988.9/1800=0.5493889` 转成 SYS_CNT cycle（latency/cycle 正向，rate 反向）；profile `schema_version=13`、`profile_version=david-v100-simd-simt-20260920-v24`。
> 本目录只保留目标场景标定/验证所需的探针、复现脚本与结果表。

---

## 0. 总结

1. 目标scalar load/store场景可以分为四类：(1) **direct scalar load same-line**；(2) **direct scalar load diff-line**；(3) **indirect scalar load**；(4) **scalar store**（SIMT `SIMT_STG` / SIMD Triton MTE3）。
2. 白盒公式与 CAModel-微基准的误差：SIMD load 误差 ≤0.8%；SIMT load 误差为 −8.3% ~ −2.3%；store 误差 −11.6% ~ +4.2%。这些 raw CAModel 值写入 profile 时都乘 0.5493889。
3. **单位换算**：CAModel 内部时间换算 = 1.8 GHz（同一核心 active window 的 dump cycle span ÷ CAModel 自己显示的 `duration_time(us)` 恒为 1.800 GHz）；costmodel 的 `*_system_cycles` 是 SYS_CNT 域（988.9MHz）。cycle 数换算为 `T_sys = T_camodel × 988.9/1800 = T_camodel × 0.5493889`（rate 则乘 1800/988.9）。
4. 白盒公式与 CAModel-6个目标kernel的平均误差（MAPE）：
   - SIMT：direct 8.0%、indirect 8.1%、store 18.9%；
   - SIMD：direct 10.9%、indirect 10.1%、store 9.5%。
   - 三类 MAPE 均 <20%；单点最大为 SIMT binned wgrad indirect 的 +19.7%。
5. **CAModel 可靠性**：目标实际用到的 direct load 与真卡微基准误差 ±10% 内；目标 indirect 是 shallow diff-line load（`exposure=1`，`L_dep` 收费为 0），其 CAModel window 已有同机制 board line-fill 对照（差 −22%~−1%）；SIMT K=1 store 与 board 单发差 −4.6%~+8.6%。深依赖链和 K>1/SIMD MTE3 store 不在目标 route 的 claim 范围内（目标模型不使用它们）；§3.3/§4 里更大的误差是白盒公式拟合误差，不是 CAModel 测量不可靠。
6. SIMD scalar store 在 Triton 下走 MTE3（`scalar → UB staging → MTE3 MOV UB→OUT`），不是 CCE MainScalar `ST_XD_XN_IMM → GM`；公式为 `20 + 450 + (K-1)*480`。目标算子当前 route 全为 SIMT，所以 SIMD store 只作为 forced-mode 对照。

---

## 1. 目标场景：6 个 kernel → 4 类标量访存


### 1.1 costmodel 建模场景归类

| kernel | direct-scalar-load (same/diff) | indirect-scalar-load (diff) | scalar-store |
|---|---|---|---|
| padded_copy_gather | 0 / 1×2（2 个 op） | 1×3（本次执行 2） | 0 |
| padded_copy_scatter | 0 / 1×2 | 1×3（本次执行 3） | 0 |
| padded_copy_wgrad | 0 / 1×2 | 1×2 | 1 |
| binned_copy_gather | 1×2（本次执行 1）/ 0 | 1×2（本次执行 1） | 0 |
| binned_copy_scatter | 1×2（本次执行 1）/ 0 | 1×2（本次执行 1） | 0 |
| binned_copy_wgrad | 1×2（本次执行 1）/ 0 | 1×1 | 1 |

### 1.2 四类场景解释

**1) direct scalar load same-line**
- 含义：同一个 stage 里的多条 scalar load 落在同一条 cache line（SIMD MainScalar 64B / SIMT warp 128B）。
- IR 判定：`StagePartitioner.cpp::scalarLoadsShareOneLine` 要求同一个 base pointer，`tt.addptr` 常量偏移极差对应的 span ≤64B；只有 prove 成功才走 same-line 分支。
- 硬件行为：第 1 条 miss 付一次 line fill，后续命中同 line。SIMD MainScalar 有 MSHR merge + DC hit（CAModel o4 same = 493，而不是 4×447）；SIMT CCE uniform 探针在当前 CAModel 下同 line 不 merge/hit，每条 LDG 仍按一次 line read 串行（o4 same = 1923），但 Triton scalar pair 5-cycle issue spacing 可能命中（`FAKE_HIT`）。
- 目标 6 kernel：本次没有真正的 same-line direct stage；binned 的 `bins[expert_idx-1]` / `bins[expert_idx]` 源码上同 line，但被拆成两个条件 stage 且 seed=0 只执行一条。

**2) direct scalar load diff-line**
- 含义：同一 stage 的多条 scalar load 落在不同 cache line。
- 目标场景：padded 的 `indices` / `bin_ids` 是两个 tensor（+512B），属于 diff-line；IR 证不出同 line，保守走 diff-line 分支。
- 硬件行为：SIMD MainScalar 的 MSHR 有 2 个 outstanding，K≤2 条不同 line 可同时 outstanding，K=4 时会分两波（o4 diff = 956）；SIMT 不同 128B line 可以并发 fill，边际只加 LSU issue（o4 diff = 526，公式 486）。
- 目标 report：padded 两条 direct 被切成两个 K=1 ScalarLoad stage，所以 SIMT 聚合为 `2×486=972`、SIMD 为 `2×447=894`。如果后续把它们合并成一个 K=2 diff-line stage，公式值会变成 SIMT `486`、SIMD `450`（stage 切分/overlap 是后续 open issue）。

**3) indirect scalar load**
- 含义：一条 scalar load 的地址由另一条 scalar load 的结果计算出来（pointer chase / fan-out）。
- 目标场景：padded 的 `bins[bin_idx-1]` / `padded_bins[bin_idx-1]` 依赖 `bin_idx`；scatter 的 `weights[index_a]` 依赖 `index_a`；binned 的 `indices[start+entry_idx]` 依赖 `start`（且经过 `num_tokens`/`if` 控制流）。
- 建模：indirect stage 自身的 line fill 仍按普通 load 公式算；依赖额外收费只针对 `exposure > 1` 的额外边：`T += max(0, exposure-1) * L_dep`。第一条 producer→consumer 边已经包含在 consumer 的 line fill 里，不重复收费。`L_dep` 当前取值 SIMD 1.95 cyc / SIMT 65.4 cyc，来自旧版 board SYS_CNT 依赖链拟合（cycle 域）。
- 目标场景绝大多数 `exposure=1`，所以 indirect stage 公式值就等于同 mode 的 load 公式值。

**4) scalar store**
- 目标场景：padded/binned wgrad 的 `tl.store(wgrad, out)`，即 `out = tl.sum(acc)` 后的单标量 store。
- SIMT：`SIMT_STG` 直接写 128B line；same-line K≥2 在 CAModel 下不 merge，公式 first store / diff-line `450+(K-1)*20`，same-line K≥2 `555+(K-1)*480`。
- SIMD：CCE 路径的 `ST_XD_XN_IMM → GM`（write-allocate + dirty writeback）不是 Triton 的路径。Triton scalar `tt.store` 被 TTAdapter 降成 `tensor<1xf32>` + `materialize_in_destination`，CAModel 里是 `SCALAR ST_XD_XN_IMM accessUb:1`（写 UB）→ `SET_FLAG` → `MTE3 MOV_SRC_TO_DST_ALIGNv2 Src:UB,Dst:OUT`（写 GM）。因此 SIMD store 公式用 MTE3 白盒 `20+450+(K-1)*480`，而不是 CCE MainScalar store 的 478/531/557 窗口。

### 1.3 固定变量与采样

- `K = scalar_load_count_per_iteration` / `scalar_store_count_per_iteration`；
- `share = scalarLoadsShareLine` / `scalarStoresShareLine`；
- `exposure = indirectScalarLoadExposureCount` / `indirectScalarStoreExposureCount`（producer-side 去重）；
- CAModel 取 `core0.veccore0` 一个 program；`msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 --launch-count=1`；
- 当前 route = `all_simt_only`，CAModel 对应 `compile_mode=simt_only`；forced SIMD 表对应 `compile_mode=simd`。

---

## 2. CAModel 可靠性

### 2.1 先确定 CAModel 内部用的是 1.8 GHz 还是 1.65 GHz

方法：用最简单的 CCE scalar CAModel 程序，把 CAModel 自己打印的 `duration_time(us)`（`Core operator results` 表）
与 dump 中同一 core 的 `instr_log.dump` 活跃窗口 cycle span 对齐，看 `cycles / duration_us` 是多少 GHz。

```bash
# 1) CAModel 显示的核心时间（us）
grep -A4 "Core operator results" run.log

# 2) 同一 core 的 dump cycle span（min/max instr_log 时间戳）
python3 - <<'PY'
import re
txt = open('.../core0.veccore0.instr_log.dump', errors='replace').read()
v = [int(x) for x in re.findall(r'^\[info\] \[(\d+)\]', txt, re.M)]
print(max(v) - min(v))
PY
```

实测（`OPPROF_20260912165738_SSCQHPMIWFGTSMDT`，CCE `simt_scalar_memory` load/store 两次 launch）：

| launch | core | dump cycle span | CAModel `duration_time(us)` | cycles / us | 等效 GHz |
|---|---|---:|---:|---:|---:|
| load (`measure/0`) | core0.veccore0 | 10906 | 6.06 | 1799.7 | **1.800** |
| load (`measure/0`) | core0.veccore1 | 10514 | 5.84 | 1800.3 | **1.800** |
| store (`measure/1`) | core0.veccore0 | 8786 | 4.88 | 1800.4 | **1.800** |
| store (`measure/1`) | core0.veccore1 | 9140 | 5.08 | 1799.2 | **1.800** |

结论：**CAModel 内部 cycle → time 的换算就是 1.8 GHz**，不是 1.65 GHz。
后文所有 `ns@1.8G = cycle / 1.8`。

### 2.2 CAModel 与上版真卡微基准的绝对时间误差

**2.2.1 direct load same / diff-line（K=1）**

| 场景 | mode | CAModel cycle | CAModel ns@1.8 | board 微基准 | board median ns | err |
|---|---|---:|---:|---|---:|---:|
| direct same-line load | SIMT | 530 | 294.4 | `m_simt_ld_uni_same_k1` | 291.8 | **+0.9%** |
| direct diff-line load | SIMT | 530 | 294.4 | `m_simt_ld_uni_diff_k1` | 289.0 | **+1.9%** |
| direct same-line load | SIMD | 447 | 248.3 | `m_main_ld_same_k1` | 274.5 | **−9.5%** |
| direct diff-line load | SIMD | 447 | 248.3 | `m_main_ld_diff_k1` | 273.8 | **−9.3%** |

- board 数据来自 `syscnt/board_marginal/board_vs_model_agg.csv` 的多次 run 中位数。
- 这是后面 indirect / store 对照的机制基线：目标会用到的 direct load 两个分支，CAModel window 与真卡微基准误差在 **±10%** 内。

**2.2.2 indirect：不需要额外的 dependency 验证**

- 目标 6 个 kernel 的 indirect stage 全是 `exposure=1`，因此 `max(0, exposure-1)*L_dep = 0`；它本质就是一次 diff-line scalar load，line-fill 机制已被 §2.2.1 验证。
- `L_dep` 只在未来出现 `exposure>1` 深依赖链时使用，本目标不需要对它做 CAModel-vs-board 对照。

**2.2.3 store：CAModel window 对 board 单发即可**

- 目标只用 SIMT K=1 store（`SIMT_STG`）；用同指令、同 issue→ack/retire 窗口的 board 单发 SYS_CNT 对照：
  - CCE probe `simt_st_uniform_o1`：CAModel 491 core cycle = 272.8 ns vs board median 286 ns → **−4.6%**；
  - target padded/binned wgrad store stage：559 / 551 core cycle = 310.6 / 306.1 ns vs 286 ns → **+8.6% / +7.0%**。

**2.2.4 小结**

目标实际使用的 CAModel 项都有同机制 board 对照：direct load ±10%、shallow indirect line-fill −22% ~ −1%、SIMT K=1 store ±9%；未进入目标 route 的 dependency / K>1 / MTE3 store 不影响该结论。

## 3. 白盒公式

### 3.1 公式输入变量

| 变量 | 含义 | 来源 |
|---|---|---|
| `K` | stage 内 scalar load/store 数 | `StageWorkload::scalarLoadCount` / `scalarStoreCount` |
| `share` | IR 是否证明同一 line | `scalarLoadsShareLine` / `scalarStoresShareLine` |
| `exposure` | indirect producer 被消费的边数（producer-side 去重） | `indirectScalarLoadExposureCount` / `indirectScalarStoreExposureCount` |
| `mode` | SIMD / SIMT | stage implementation |

代码入口：`StageCostModels.cpp::mapWorkload` → `mainScalarLoadCycles` / `simtUniformLoadCycles` / `mte3StoreCycles` / `simtUniformStoreCycles`。

### 3.2 四类公式与解释

> **单位**：下面公式先写 raw CAModel core cycle @1.8GHz（便于和 dump window 对）；写入 costmodel profile 的 SYS_CNT 值 = raw 值 × 988.9/1800 = raw 值 × 0.5493889。`L_dep` 1.95/65.4 本身已是 board SYS_CNT cycle，不再乘这个系数。

**A. direct scalar load — SIMD MainScalar**

```text
u = share ? 1 : K          // share=true: 同一条 cacheline；否则按 K 条不同 line
extra(u) = (u <= 4) ? 250 : 350
T_main(K,share) = 7 + 440 // 固定 prep 7 + 首次 64B line fill 440
                  + max(0, u - 2) * extra(u) // 超过 2 个 outstanding 的额外 line fill
                  + (K - u) * 12.333 // 同 line hit
                  + (K - 1) * 3 // 额外 op 的 issue
```

- `7` = issue→tag/MSHR/BIU dispatch 4 cycle + refill→retire 3 cycle；profile 值 7×0.5493889 = 3.85。
- `440` = 一次 cold 64B line fill（DC 发 BIU 8 + BIU 读 421 + 回 DC 11）；profile 值 241.73。
- `max(0,u-2)*extra(u)`：MainScalar 同时只有 2 个 MSHR **outstanding（在途未完成的 miss request）**；第 3 条及以后的不同 line 要等前两波之一，额外付一次 line fill（u≤4 取 250→profile 137.35；u>4 取 350→192.29）。same-line 时 u=1，此项为 0。
- `(K-u)*12.333`：落在已有 line 上的额外 op 的命中成本（fit o4 same：447 + 3*(12.333+3) = 493）；profile 值 12.333×0.5493889 = 6.776。
- `(K-1)*3`：每条额外 op 的 issue/serialization 成本；profile 值 1.648。
- same-line 分支（`share=true`, u=1）：`T = 447 + (K-1)*15.333`；K=1→447，K=4→493。
- diff-line 分支（u=K）：K≤2 时 2 个 outstanding 够用，`T=447+(K-1)*3`（K=1→447，K=2→450）；K=4 时多 2 条 line，`T=956`。
- 当前实现只区分两种情况：`share=true` 时 `u=1`（同一条 line）；否则 `u=K`（K 条不同 line）。

**B. direct scalar load — SIMT warp-uniform**

```text
margin = share ? 464.333 : 0.001
T_simt(K,share) = 6 + 480 + (K - 1) * margin
```

- `6` = issue→DC tag（SIMT LDG 前段）；profile 值 3.296。
- `480` = DC tag→BIU→line fill→UBITF/GSU 回传路径的联合拟合值。
- same-line margin `464.333`（profile 255.05）：CCE uniform 探针在当前 CAModel 下同 128B line 不 merge/hit，每条额外 LDG 再付一次 line read（o4 same = 486 + 3×464.333 = 1879，对 CAModel 1923 误差 −2.3%）。
- diff-line margin `0.001`：不同 128B line 可以 outstanding/overlap，额外 op 只付 LSU issue 地板；K≤4 公式值约 486（CAModel o4 diff 526，−7.6%）。
- 注意：Triton scalar pair 的 LDG issue spacing ≈5 cycle，可能让同 line 第二条变 `FAKE_HIT`；same-line 公式是 CCE 2-cycle spacing 的保守分支。**目标 6 kernel 没有实际执行的 same-line 多 op 场景**（padded 是跨 tensor/diff-line；binned 相邻 bins 在条件分支里且 seed 只执行一条），所以当前目标结果不依赖该分支。

**C. indirect scalar load**

```text
T_indirect = T_load(mode; K,share) + max(0, exposure - 1) * L_dep
L_dep(SIMD) = 1.95 cyc
L_dep(SIMT) = 65.4 cyc
```

- consumer（indirect load）自己的 line fill 已由 `T_load` 覆盖；依赖导致 producer retire→consumer issue 的那段气泡，对 shallow 单边已经在 `T_load` 的 window 里，所以第一条 producer→consumer 边不再额外收费。
- `exposure` 是 **producer-side 去重后的依赖边数**：一个 producer 被一个 consumer 使用算 1；一个 producer fan-out 给多个 consumers 也只算 1；只有更深的 serial chain / 额外 producer 暴露才继续收费。
- `L_dep` 是历史 board CCE dependency probe 拟合出的额外 edge latency（SYS_CNT cycle 域：SIMD 1.95 / SIMT 65.4）；该旧探针不在本 PR 的复现文件内。目标 6 kernel 全部 `exposure=1`，该常数不参与收费。
- `exposure=1` 时 `max(0,1-1)*L_dep=0`；目标场景基本都是 `exposure=1`，所以 indirect stage 值等于同 mode 的 load 公式值。

**D. scalar store**

```text
SIMT same-line (K>=2)   : 555 + (K-1)*480
SIMT first store / diff : 450 + (K-1)*20
SIMD Triton (MTE3)      : 20 + 450 + (K-1)*480
```

- SIMT `SIMT_STG` 直接写 128B line：K=1 走 first-store/diff 分支 450（profile 247.2）；same-line K≥2 CAModel 不 merge，每条后续 store 串行 +480（profile 263.7；o4 same 1995 vs CAModel 1914）；diff-line 可 overlap，后续 +20（profile 10.99；o4 diff 510 vs CAModel 577）。
- SIMD Triton store 的真实链路是 `SCALAR ST_XD_XN_IMM accessUb:1`（scalar→UB）→ 等 VF flag（约 556）→ `MTE3 MOV` push→retire（约 1054，其中等 `recv_wack` 429）→ BIU write；整条单 program 冷链路约 1079 cycle。
- `20+450+(K-1)*480` 是 costmodel 使用的 **stage 可摊销白盒系数**（profile：10.99+247.2+(K-1)×263.7）：第一个 store 的 stage 成本约 470 raw cyc，之后**每多一个 op 再加约 480 raw cyc**，所以 K=1/2/3/4 的模型值是 470/950/1430/1910。它不是整条 1079-cycle 冷链路（那条还包含 scalar→UB staging、等 VF flag 等启动段），而是把 MTE3 stage 的可摊销部分折算成系数；目标 kernel 的 store stage 用它验证（SIMD padded wgrad `470` vs CAModel MTE3 window `496`，binned wgrad `470` vs `413`，误差 −5.2%/+13.8%）。

### 3.3 标定微基准与公式值误差

| 场景 | mode | 标定程序/函数 | 公式值（cycle） | CAModel window | err |
|---|---|---|---|---:|---:|
| SIMD direct load K=1 | SIMD | `load/scalar_o1/load_scalar_o1.cce::simd_main_ld_o1` | 447 | 447 | 0.0% |
| SIMD direct load K=4 same | SIMD | `load/scalar_o4/load_scalar_o4.cce::simd_main_ld_same_o4` | 447+3×15.333 = 493 | 493 | 0.0% |
| SIMD direct load K=4 diff | SIMD | 同上 `simd_main_ld_diff_o4` | 447+2×250+3×3 = 956 | 956 | 0.0% |
| SIMT direct load K=1 | SIMT | `load/scalar_o1/load_scalar_o1.cce::simt_ld_uniform_o1` | 6+480 = 486 | 530 | −8.3% |
| SIMT direct load K=4 same | SIMT | `load/scalar_o4/load_scalar_o4.cce::simt_ld_uniform_same_o4` | 486+3×464.333 = 1879 | 1923 | −2.3% |
| SIMT direct load K=4 diff | SIMT | 同上 `simt_ld_uniform_diff_o4` | 486+3×0.001 = 486.0 | 526 | −7.6% |
| indirect dependency | SIMD/SIMT | 历史 board 拟合常数（target `exposure=1`，不收费） | L_dep 1.95 / 65.4 | — | — |
| SIMT scalar store K=1 | SIMT | `store/scalar_o1/store_scalar_o1.cce::simt_st_uniform_o1` | 450 | 491 | −8.4% |
| SIMT scalar store K=4 same | SIMT | `store/scalar_o4/store_scalar_o4.cce::simt_st_uniform_same_o4` | 555+3×480 = 1995 | 1914 | +4.2% |
| SIMT scalar store K=4 diff | SIMT | 同上 `simt_st_uniform_diff_o4` | 450+3×20 = 510 | 577 | −11.6% |
| SIMD scalar store（Triton） | SIMD | `store/triton_scalar_store/triton_scalar_store_demo.py::_triton_scalar_store_demo` | 20+450+(K-1)×480 | 见 §3.2 D | 见 §4.2 |


---

## 4. 6-kernel 与公式值误差

### 4.1 SIMT

| kernel | 类别 | op_num | costmodel ns | CAModel ns | err |
|---|---|---:|---:|---:|---:|
| padded_copy_gather | direct load | 2 | 540.0 | 618.3 | −12.7% |
| padded_copy_gather | indirect load | 1 | 270.0 | 282.8 | −4.5% |
| padded_copy_scatter | direct load | 2 | 540.0 | 619.4 | −12.8% |
| padded_copy_scatter | indirect load | 2 | 540.0 | 540.6 | −0.1% |
| padded_copy_wgrad | direct load | 2 | 540.0 | 498.3 | +8.4% |
| padded_copy_wgrad | indirect load | 1 | 270.0 | 267.2 | +1.0% |
| padded_copy_wgrad | scalar store | 1 | 250.0 | 310.6 | −19.5% |
| binned_copy_gather | direct load | 1 | 270.0 | 276.1 | −2.2% |
| binned_copy_gather | indirect load | 1 | 270.0 | 242.2 | +11.5% |
| binned_copy_scatter | direct load | 1 | 270.0 | 276.1 | −2.2% |
| binned_copy_scatter | indirect load | 1 | 270.0 | 242.2 | +11.5% |
| binned_copy_wgrad | direct load | 1 | 270.0 | 245.6 | +10.0% |
| binned_copy_wgrad | indirect load | 1 | 270.0 | 225.6 | +19.7% |
| binned_copy_wgrad | scalar store | 1 | 250.0 | 306.1 | −18.3% |

### 4.2 SIMD

| kernel | 类别 | op_num | costmodel ns | CAModel ns | err |
|---|---|---:|---:|---:|---:|
| padded_copy_gather | direct load | 2 | 496.7 | 490.6 | +1.2% |
| padded_copy_gather | indirect load | 1 | 250.0 | 296.1 | −15.6% |
| padded_copy_scatter | direct load | 2 | 496.7 | 561.7 | −11.6% |
| padded_copy_scatter | indirect load | 2 | 498.3 | 516.7 | −3.5% |
| padded_copy_wgrad | direct load | 2 | 496.7 | 445.0 | +11.6% |
| padded_copy_wgrad | indirect load | 1 | 250.0 | 282.2 | −11.4% |
| padded_copy_wgrad | scalar store | 1 | 261.1 | 275.6 | −5.2% |
| binned_copy_gather | direct load | 1 | 248.3 | 202.2 | +22.8% |
| binned_copy_gather | indirect load | 1 | 248.3 | 293.3 | −15.3% |
| binned_copy_scatter | direct load | 1 | 248.3 | 263.9 | −5.9% |
| binned_copy_scatter | indirect load | 1 | 248.3 | 236.1 | +5.2% |
| binned_copy_wgrad | direct load | 1 | 248.3 | 283.3 | −12.4% |
| binned_copy_wgrad | indirect load | 1 | 248.3 | 273.9 | −9.3% |
| binned_copy_wgrad | scalar store | 1 | 261.1 | 229.4 | +13.8% |

### 4.3 MAPE 汇总

| 类别 | SIMT MAPE | SIMT 最大/最小 | SIMD MAPE | SIMD 最大/最小 |
|---|---:|---:|---:|---:|
| direct scalar load | 8.0% | +10.0% / −12.8% | 10.9% | +22.8% / −12.4% |
| indirect scalar load | 8.1% | +19.7% / −4.5% | 10.1% | +5.2% / −15.6% |
| scalar store | 18.9% | −18.3% / −19.5% | 9.5% | +13.8% / −5.2% |

> 三类 MAPE 均 <20%；单点最大 +19.7%（SIMT binned wgrad indirect），SIMD 单点最大 +22.8%（binned gather direct）

---

## 5. 复现与文件索引

本目录只保留目标场景所需的探针、复现脚本和结果表。

### 5.1 CAModel 标定探针

```bash
source ~/env_ascend.sh
ulimit -n 1048576

# K=1 load：SIMD MainScalar / SIMT 32T / SIMT 1T
cd load/scalar_o1
bash build_scalar_o1.sh && bash run_scalar_o1_camodel.sh
python3 parse_load_scalar_o1.py camodel_results/simd camodel_results/simt32 camodel_results/simt1

# K=4 load：same-line / diff-line
cd ../scalar_o4
bash build_scalar_o4.sh && bash run_scalar_o4_camodel.sh
python3 parse_load_scalar_o4.py camodel_results/simd_same camodel_results/simd_diff \
    camodel_results/simt_same camodel_results/simt_diff

# SIMT store：K=1 / K=4 same / K=4 diff
cd ../../store/scalar_o1
bash build_store_scalar_o1.sh && bash run_store_scalar_o1_camodel.sh
python3 parse_store_scalar_o1.py camodel_results/simd camodel_results/simt32
cd ../scalar_o4
bash build_store_scalar_o4.sh && bash run_store_scalar_o4_camodel.sh
python3 parse_store_scalar_o4.py camodel_results/simd_same camodel_results/simd_diff \
    camodel_results/simt_same camodel_results/simt_diff

# SIMD scalar store 的 Triton MTE3 路径
cd ../triton_scalar_store
N_ST=1 GRID=4 bash run_triton_scalar_store_camodel.sh
```

> `camodel_results/` 由脚本现场生成，不入库；每个 case 单独 `msopprof simulator --launch-count=1`。

### 5.2 真卡 SYS_CNT 对照

- `syscnt/board_marginal/board_scalar_marginal.cce` + `_host.cpp`：K=1 load/store 的 board marginal 探针；
- `syscnt/board_marginal/board_vs_model_agg.csv`：多次 run 的聚合结果（本文 §2 的 board 列来源）。

### 5.3 公式落盘与 6-kernel 汇总

- `costmodel_eval/make_profile.py`：把本文公式系数写入 profile（SYS_CNT 域）；
- `costmodel_eval/summarize_all_scalar.py`：从 costmodel report + CAModel dump 生成 matched-only + per-stage-union 表；
- `costmodel_eval/results/all_scalar_eval.csv` / `.md`：SIMT 表 §4.1 原始数据；
- `costmodel_eval/results/all_scalar_eval_simd.csv` / `.md`：forced-SIMD 表 §4.2 原始数据。

### 5.4 目标 kernel

6 个目标 kernel 来自外部输入 `test_cases/scalar_dominate_kernels/`（`npu_padded_copy_{gather,scatter}.py`、`npu_padded_copy_scatter_wgrad_camodel.py`、`npu_prec_binned_kernel_simd_modified.py`），本 PR 不新增这些文件；§4 的误差用固定 `shape=(4,256,4,2)`、`BLOCK_X=64`、`superblock_factor=1`、`num_warps=1` 的 report + seeded CAModel 结果复核。
