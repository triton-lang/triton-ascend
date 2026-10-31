# Ask Oracle：如何优化**本机系统**与 **triton-ascend 项目构建**

> 文档用途：把一份自包含的问题上下文交给更强的模型，请给出**系统层面**与**项目层面**的可落地优化方案。
> 标注约定：`[事实]` = 现场取证或实测数据；`[推断]` = pi agent 的假设；`[待答]` = 请你回答。
> 回答格式：按 `Q1..Q18` 逐条回答；每条给「**结论 → 具体配置/命令 → 风险与回退**」。不要复述我已知的根因。
> 本文档**不涉及** `ScalarClosure` / `Closure` 接口设计（用户已裁定不再提问）；旧草稿内容保留在本文档折回 `localhack`（`ukxrlwzt/0`）之前的版本历史里。

---

## 0. 环境信息（完整，判断用）

### 0.1 硬件 / 操作系统

| 项目 | 值 |
|---|---|
| 发行版 | openSUSE Tumbleweed-Slowroll, `VERSION_ID="20260901"` |
| 内核 | `7.2.8-1-default` |
| CPU | x86_64, **`nproc = 24`**（逻辑核），调频驱动 `intel_pstate`，governor = **`powersave`** |
| 内存 | **MemTotal 32619360 kB ≈ 31.1 GiB** |
| Swap | **`SwapTotal = 0 kB`**；无 zram（`lsmod` 无 zram，`/dev/zram*` 不存在） |
| 磁盘 | `nvme0n1` YMTC PC411-1TB-B（953.9G），分区 `nvme0n1p1` vfat `/boot/efi`、`nvme0n1p2` **`crypto_LUKS`** |
| 根文件系统 | `/dev/mapper/cr_root`，**btrfs（LUKS 全盘加密之上）**，953G 总 / **184G 已用 / 769G 可用（20%）** |
| btrfs 子卷 | `/` = `subvol=/@/.snapshots/163/snapshot`（**snapper 快照子卷**）；`/home` = `subvol=/@/home` |
| 挂载选项 | `rw,seclabel,relatime,ssd,space_cache=v2` —— **无 `noatime`、无 `compress=`** |
| NVMe 调度器 | `[none]`（其余 `mq-deadline kyber bfq`） |
| `fstrim.timer` | active |
| `/tmp` | **tmpfs, 16G**（当前 7.3M）→ 写大日志等于吃内存 |
| 桌面 | KDE/Plasma（Wayland），`baloo_file` 正在运行 |

### 0.2 内存相关 sysctl / 逃逸阀

| 键 | 值 | 说明 |
|---|---|---|
| `vm.overcommit_memory` | `0` | heuristic：**超量申请不会失败**，缺页时才真正要内存 |
| `vm.overcommit_ratio` | `50` | `CommitLimit = 16309680 kB ≈ 15.6 GiB`，`Committed_AS = 14089700 kB ≈ 13.4 GiB`（空载即 86%） |
| `vm.swappiness` | `60` | 无 swap，无意义 |
| `vm.min_free_kbytes` | `67584`（66 MB） | 偏低 |
| `vm.dirty_ratio` / `dirty_background_ratio` | `20` / `10` | 脏页上限 ≈ 6.2 GiB |
| `vm.vfs_cache_pressure` | `100` | 默认 |
| `vm.max_map_count` | `1048576` | 正常 |
| `systemd-oomd` | **inactive** | 无"尽早杀"机制 |
| cgroup `memory.max` | unlimited | 无上限保护 |
| `ulimit -v` / `-u` | unlimited / 127107 | 无进程级限制 |
| `kernel.hung_task_timeout_secs` | **0（禁用）** | 卡死无 D-state 日志 |
| `kernel.hung_task_panic` / `softlockup_panic` | `0` / `0` | |
| `/sys/fs/pstore` | **空**（无 ramoops） | 无崩溃取证通道 |

### 0.3 编译工具链

| 组件 | 版本 / 路径 |
|---|---|
| C/C++ 编译器 | `clang++-22` / `clang-22`（`/usr/bin`） |
| LLVM | `~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/`（预编译，含 `FileCheck`） |
| 链接器 | **mold 3.0.0**（`mise`，链接经 `mold -run`） |
| ccache | **4.14.1**（`mise`），`CCACHE_DIR` 默认 `~/.cache/ccache` |
| sccache | 0.18.0 **已安装但未使用** |
| cmake | 4.4.4（`mise`） |
| ninja | **1.13.2**（`mise`） |
| 构建类型 | **Debug**（`--config Debug`；单 TU 只有 `-g`，无 `-O`、无 `-gz`、无 `-gsplit-dwarf`） |
| 其它编译 flag | `-DGTEST_HAS_RTTI=0`、`-D__LLVM_MAJOR_VERSION_21_COMPATIBLE__`、`-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__`；`safe_compile.cmake` 追加 `-fPIC -fstack-protector-strong`，非 Apple 平台 EXE 加 `-Wl,-z,now -pie -s`，`CMAKE_SKIP_RPATH TRUE` |
| CMake 关键选项 | `-DCMAKE_C_COMPILER=clang-22 -DCMAKE_CXX_COMPILER=clang++-22 -DTRITON_BUILD_UT=OFF ...` |

### 0.4 ccache 现状 `[事实]`

| 项 | 值 |
|---|---|
| `cache_dir` / `max_size` | 默认 `~/.cache/ccache` / **5.0 GiB**，实测占用 **5.0 GiB（99.51%）→ 已满** |
| 调用统计 | Cacheable 44948/45265（99.30%），**命中 61.47%**（direct 98.16%），miss 38.53%，Errors **1** |
| `compression` / `compression_level` | `true` / `0`（默认 zstd 级） |
| `compiler_check` | `mtime`（默认） |
| `depend_mode` / `hard_link` | `false` / `false` |
| `inode_cache` | `true` |
| `sloppiness` | **空** |
| 磁盘余量 | 769 GiB 空闲（缓存可放大两个数量级） |

### 0.5 项目与构建系统规模 `[事实]`

| 项 | 值 |
|---|---|
| 仓库 | `/home/olivercwy/workspace/triton-ascend`（jj 0.46.0 + git 双栈），`HEAD = 3a164e7f2` |
| 构建目录 | `build/cmake.linux-x86_64-cpython-3.11/`（在仓库内、在 btrfs+LUKS 上） |
| `build.ninja` | 5,629,142 字节 / **5123 条 `build` 边** |
| 已生成对象 | **539 个 `.o`，合计 1.5 GiB** |
| **官方构建入口** | `~/personal/build_ta.sh`：`export MAX_JOBS=16` → `mold -run uv run setup.py bdist_wheel`（内部 `cmake --build . --config Debug -j16`）。整跑 **≈465 s** |
| **危险用法** | 裸 `ninja -C build/... triton-opt`：`MAX_JOBS` **对 ninja 无效**，ninja 1.13.2 默认并行度 = `nproc + 2` = **26** |
| 单目标增量 | `ninja -C build/... triton-opt` 是唯一快速路径（无需重新 configure） |
| 测试 | `third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/{PlanComputeBlock,StandardizeOp,SplitDataflow,ComputeBlockOpt}`，共 **54 个用例 / 56 条 `// RUN`**，形如 `// RUN: triton-opt <pass> %s | FileCheck %s`；**系统 python 无 `lit` 包，LLVM bin 无 `llvm-lit`**（当前只能手写脚本模拟 `RUN` 行） |
| 工作树 | 采用 `setup_patch.py` 把 `third_party/ascend/patch/triton-ascend-3.6.0.patch` 应用到 in-tree Triton 源码；**当前已应用**（`git apply --reverse --check` = APPLIED）。`restore_sources.py` 可还原 |

### 0.6 构建期间的实际资源占用 `[事实]`

| 场景 | 内存 |
|---|---|
| 重启后空载（KDE + 浏览器 + pi） | used 3.6 GiB |
| 6 并发编译进行中 | used 5.5 GiB，cached 6.1 GiB，**available 25 GiB** |
| 构建结束 | used 3.7 GiB，cached 10 GiB，available 27 GiB |

后台常驻进程内存（RSS）：`baloo_file 733 MB`、`plasmashell 372`、`brave ×4 ≈ 857`、`v2rayN 292`、`pi 262`、`kwin_wayland 262`、`dolphin 150`、`konsole 149`（合计 ≈3.0 GiB）。
**`balooctl config list excludeFolders` 输出为空** → KDE 索引器正在索引整个 `$HOME`（含 `~/workspace/triton-ascend/build` 与 `third_party/`）。

---

## 1. 已发生的事故与已定位的根因（`[事实]`，不需要你复述）

一次**未加 `-j`** 的 `ninja` 导致整机卡死约 1 小时，用户强制断电。

| 时刻 | 事件 |
|---|---|
| 12:11–12:19:54 | `build_ta.sh`（`-j16`）跑完，报 4 个编译错误（我的代码问题），退出 1 |
| 12:20:18 | 我修完源码 |
| 12:20:32 | 我执行 `ninja -C build/cmake.linux-x86_64-cpython-3.11 triton-opt > /tmp/ninja_pn6462ac.log 2>&1`（**无 `-j`**） |
| 12:24:44 | 该次启动 journal 的**最后一条日志**（此后 journald 也无法写入） |
| ~13:27 | 用户强制断电重启 |

**根因（已确认）**：`ninja` 默认 `-j26`，而本机 24 核 / **31 GiB / 完全无 swap**；单 TU 峰值 RSS 实测 0.8–2.2 GiB ⇒ 26 并发需 44–57 GiB。内存耗尽后内核长时间直接回收，全系统（含 journald）无进展；因为 `overcommit_memory=0` + 无 swap + `systemd-oomd` inactive，没有任何 fail-fast 路径。

**直接物证**：崩溃前一刻存在 **26 个 `*.o.tmp`，mtime 全为 12:20** —— 与 `nproc+2` 精确吻合，证明当时 26 个 clang 并发；崩溃后无任何 `.o` 的 mtime ≥ 12:20:32（即 26 个编译全部未产出，无半成品混入下游）。

**已采取的止血**：改用 `ninja -j6`（最坏 ≈13 GiB）+ `nice -n 5` + 日志写 `build/`（不再写 tmpfs）→ 本次编译 69 个目标成功、0 error、峰值 used 仅 5.5 GiB。

---

## 2. 实测数据（对照实验，请以此为优化依据）

### 2.1 单 TU 墙钟 / 峰值 RSS（`/usr/bin/time -v`，直接调用 clang，绕过 ccache）

| TU | 墙钟 | 峰值 RSS | 生成对象 |
|---|---|---|---|
| `third_party/ascend/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/HIVMDialect.cpp` | 19.0 s | **2216 MB** | 66.2 MB |
| `python/src/ir.cc` | 13.4 s | 1698 MB | 26.4 MB |
| `lib/DynamicCVPipeline/.../PlanComputeBlock/OpClassifier.cpp` | 8.2 s | 1118 MB | 2.6 MB |
| `third_party/ascend/lib/TritonToHIVM/TritonToHIVM.cpp` | 5.5 s | 820 MB | — |

Top-8 对象合计 ≈207 MB（`HIVMDialect 66.2`、`ir.cc 26.4`、`TritonToLinalgPass 21.8`、`HFusionDialect 20.5`、`HFusionOps 19.5`、`ascend_ir.cc 19.0`、`TritonToTritonGPUPass 16.9`、`TritonGPU Dialect 16.8`）。

### 2.2 `-g` 级别对照实验（同一命令，仅换 debug flag）

| TU | `-g`（现状） | `-g1` | `-gline-tables-only` | `-gsplit-dwarf` | `-g0` |
|---|---|---|---|---|---|
| `OpClassifier.cpp` | 7.90 s / 1092 MB | 7.75 / 1052 | 7.77 / 1052 | 7.65 / 1047 | 7.68 / 1048 |
| `HIVMDialect.cpp` | 18.99 s / **2164 MB** | 17.40 / 1886 (−13%) | 17.42 / 1885 | **15.80 / 1829 (−15% 内存, −17% 时间)** | 15.87 / 1829 |

`[推断]` 结论：debug info 只占 ≤15%，内存/时间主要由**头文件与模板实例化**（MLIR + TableGen `.inc`）决定。`-gsplit-dwarf` 是唯一"既不损失 DWARF 又拿回全部收益"的选项。

### 2.3 并行度 → 内存需求（按实测 RSS 推算，含桌面 ~3 GiB 基线）

| `-j` | 最坏需求（×2.2 GiB） | 中位需求（×1.7 GiB） | 31 GiB 是否安全 |
|---|---|---|---|
| 26（ninja 默认） | **57 GiB** | 44 GiB | ❌ 必然溢出 |
| 16（`build_ta.sh`） | 35 GiB | 27 GiB | ⚠️ 临界，风险高 |
| 10 | 22 GiB | 17 GiB | ⚠️ 偏紧 |
| 8 | 17.6 GiB | 14 GiB | ✅ |
| 6（当前采用） | 13.2 GiB | 10 GiB | ✅ 富余 |

`[推断]` 即"24 核只能用到 6–8 路"，**25%–33% 的 CPU 被内存上限浪费**。这正是希望你优化的核心矛盾。

---

## 3. 待优化问题

### 3.1 系统层面

**Q1 [并行度策略]** 如何把 `-j` 从"人肉常数"变成"由内存自动决定"？
ninja 只有 `-l/--load-average`（1 分钟负载，**不是内存感知**）。可选路径：(a) `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` 包裹构建（超限即杀，快速失败）；(b) 自写 wrapper 读 `/proc/meminfo` 动态 `-j`；(c) `ninja -j8` 固定。请评估哪条最稳，并给出对 btrfs+LUKS+cgroup-v2 环境的具体命令与副作用（例如 MemoryMax 被 OOM 杀时是否留下可用日志）。

**Q2 [无 swap 的取舍]** 本机 `SwapTotal=0`、`overcommit_memory=0`。
(a) 加 swapfile（8–16 GiB）能否真正避免这次的 livelock，还是只把它变成"极慢但活着"？btrfs 上建 swapfile 必须 `chattr +C` / `nodatacow` 且不能压缩，请给正确步骤与容量建议。
(b) 改用 zram（`zram-generator`）是否更合适（编译期大量匿名页、压缩比高）？会不会与 intel_pstate/桌面产生新问题？
(c) 哪个组合最符合"宁可 OOM-kill 也不要整机卡死"的目标？

**Q3 [失败可见性]** 现状：`hung_task_timeout_secs=0`、`hung_task_panic=0`、无 pstore/ramoops、`systemd-oomd` inactive、`/tmp` 是 tmpfs（日志随重启丢失）。
请给一套"下次卡死/内存耗尽能留下证据"的最小配置（hung_task 超时、`kernel.sysrq`、journald 落盘、earlyoom/systemd-oomd 的选型与阈值、必要时 netconsole/ramoops）。

**Q4 [btrfs + LUKS + NVMe 的构建 I/O]** 构建目录与 ccache 都在 btrfs（上面套 LUKS）上，挂载为 `relatime,ssd,space_cache=v2`、无压缩。
(a) 给 `build/`（及 `~/.cache/ccache`）做 `chattr +C`（nodatacow）是否值得？CoW 对 1.5 GiB 的 `.o` + 5.6 MB `build.ninja` + 百万级小文件的收益量级？
(b) 是否该加 `noatime`（甚至 `nodiratime`）？`compress=zstd` 对 `.o`/`.dwo` 是净收益还是 CPU 负担？
(c) LUKS 的额外开销在多大程度上要计入（本机 AES-NI 可用）？有无 `cryptsetup` 侧可调项（如 `--perf-*`、sector size、`no_read_workqueue`）？

**Q5 [snapper / 快照] ** `/` 挂载自 `subvol=/@/.snapshots/163/snapshot`，说明 snapper 在跑（184G 已用可能与旧快照钉住有关）。
(a) 如何确认 `build/` 是否被快照重复引用（`btrfs filesystem du`/`compsize`/snapper 配置）？
(b) 正确做法（snapper 排除路径 / 把 `build/` 放到独立子卷 / 定期 `snapper cleanup`）与清理收益估计？

**Q6 [KDE Baloo 索引器]** `baloo_file` 常驻 733 MB RSS，且 `excludeFolders` 为空 → 它在索引整个 `$HOME`（含构建树）。
请给：禁用 vs 排除 `~/workspace`（`balooctl6 disable` / `config add excludeFolders` / 需要 indexer 时再开）的具体命令，以及对增量编译 I/O/CPU 的预期收益。若必须保留索引，如何把 `build/`、`.jj`、`.git`、`third_party` 精确排除？

**Q7 [ccache 调优]** 缓存 5 GiB 已 99.5% 满、命中率 61.47%、有 1 个 error、磁盘空 769 GiB。
(a) `max_size` 建议值（考虑"巨型头文件 + 频繁改 DCV 源 + btrfs+LUKS"）？
(b) 是否该开 `sloppiness`（如 `time_macros,include_file_mtime,include_file_ctime`）来提命中率？风险？
(c) `compiler_check=mtime` 是否应换 `content`（对频繁切分支/打 patch 的工作流）？
(d) 是否该开 `hard_link`/关 `inode_cache`/开 `depend_mode`？
(e) 已装的 **sccache 0.18.0** 是否更优（尤其配合 `-gsplit-dwarf`、分布式缓存、内存上限）？迁移成本？
(f) 缓存目录是否该换到非加密卷（若有）或 tmpfs（RAM 受限，多半不行）？

**Q8 [CPU 调频]** governor 为 `powersave`（intel_pstate）。编译期间是否应改 `performance` / `EPP=performance`？给出**按需切换**的最省事方案（如 `powerprofilesctl`、`cpupower`、systemd 服务或构建脚本里临时切换），以及收益/温度/耗电的取舍。
**Q9 [日志与临时文件]** 我此前把构建日志写到 `/tmp`（tmpfs 16G）→ 吃内存且随重启丢失。请给本仓库的推荐落点与轮转方式（`build/*.log`、`journald`、`tmux`/`tee`）以及长日志的注意事项。
**Q10 [其它系统级隐患]** 除上述之外，你从 §0 数据里还能看出哪些会拖慢/卡死构建的点？（例如 `vm.min_free_kbytes` 偏小、`dirty_ratio=20` 造成的写抖动、`vm.max_map_count`、`inotify` 上限（watchman/jj/KDE 都在用）、`ulimit -n`、`nvme` 队列深度、`fstrim` 时段与构建重叠、ccache 与 sccache 并存、mise 多层 PATH 等。）请只给**有证据链**的项。

### 3.2 项目 / 构建层面

**Q11 [巨型 TU 拆分]** `HIVMDialect.cpp` 单 TU 19 s / 2.2 GB / 66 MB 对象（TableGen 生成），Top-8 对象 207 MB。这些都是 LLVM/MLIR dialect TU（`*Dialect.cpp`、`*Ops.cpp`、`ir.cc`）。
(a) 如何在 MLIR/TableGen 层面把这些 TU 拆成多个 TU（`-gen-op-defs` 按 op 分组、`ADDITIONAL_HEADER_DIRS`、分文件 `Ops1.cpp/Ops2.cpp`）？
(b) 收益、上游惯例（LLVM 自己怎么拆）、以及对本仓库 vendored `bishengir` 代码的可行性与维护成本？
(c) 若不拆，是否可用 unity build 反向合并以减少重复解析（内存会更高，需权衡）？

**Q12 [`-gsplit-dwarf` 是否作为项目默认]** 实测最重 TU：RSS −15%、时间 −17%、DWARF 完整保留；`OpClassifier.cpp` 基本无变化。
请评估：设为项目 Debug 构建默认（`-DLLVM_USE_SPLIT_DWARF=ON` 或直接加 flag）是否值得？需要配套什么（`.dwo` 生命周期、`--gdb-index`/`dwz`、mold 行为、ccache 命中率、`objcopy --only-keep-debug`、磁盘增长）？是否应"本机开、CI 关"？

**Q13 [头文件成本]** §2.2 证明 `-g` 只影响 ≤15%：真正的成本是 MLIR/TableGen 头解析与模板实例化（单 TU 1–2 GB 仅是**解析**）。
(a) 请给一套**可执行的诊断流程**（`-ftime-trace` + `chrome://tracing`/pprof 的解读要点、`-H`、`-fno-modules` 之类）来定位"本仓库最贵的 3 个头"。
(b) 优化手段排序与预期收益：PCH（`-include-pch`，注意 `-D` 组合爆炸）、C++20 modules、减少 `#include`/前置声明、`#pragma once` 覆盖、把 TableGen `.inc` 的 `#include` 粒度调细、`-fno-standalone-debug`。
(c) 在"改一个 DCV 头 → 78 个 TU 重编"（实测 `Common/{Utils,Analysis,ScalarClosure,ScopeOpUtils}.h` 各被 78/92 个 DCV `.cpp` 包含）这种耦合下，最现实的降耦合做法？

**Q14 [并行池 / 按 TU 限流]** 能否在 `build.ninja` 层面为"重 TU"单独限流（CMake `JOB_POOLS` + `JOB_POOL_COMPILE`，或 `CMAKE_BUILD_PARALLEL_LEVEL`），实现"重 TU 最多 6 个、轻 TU 放开"？给出 CMake 侧最小改动与验证方法。`ninja -l`（负载）在内存受限场景下是否比 `-j` 更有用？

**Q15 [CMake reconfigure 开销]** 任何 `CMakeLists.txt` / glob 变化都会触发整树 reconfigure（本仓库 `build.ninja` 5.6 MB），且 patch 机制每跑 `build_ta.sh` 会 `git checkout` + `git apply` in-tree 源码 → 触发 reconfigure + 大面积重编。
(a) 如何缩短/避免 reconfigure（`CONFIGURE_DEPENDS` 的代价、`--fresh`、显式 `ninja build.ninja` 预生成）？
(b) 能否让"改 DCV 源 + 打 patch"的工作流不产生全树失效？

**Q16 [日常增量工作流]** 我为 DCV 开发，日常只改 10–20 个 DCV 文件。请给一份**本机最优 fast loop**（命令级）：
(a) 单目标 `ninja` + `-j` 取值 + `nice`；(b) ccache 预热策略（例如把常用配置的 TU 先跑一遍）；(c) 只跑相关 UT（当前无 lit，只能脚本模拟 `RUN` 行，见 §0.5）——是否值得装 `lit`（`uv pip install lit`）并生成 `lit.site.cfg.py`，还是维护一个 30 行的 `.mlir` → `triton-opt | FileCheck` 驱动器？
(d) 如何避免"误触发全树重编"（哪些改动会触发热点，需不需要 pre-commit 检查）。

**Q17 [调试符号 vs 剥离]** `safe_compile.cmake` 对 EXE 强制 `-Wl,-z,now -pie -s`（strip），导致 gdb 反栈全是 `??`；要调试就得改构建（去掉 `-s`、加 `-g`）或另开 debug 目录。
(a) 请给"**同时**保留可用 gdb 与不牺牲日常增量构建"的推荐方案（分离 debug 目录 / `-gsplit-dwarf` + `.dwo` / `objcopy --only-keep-debug` + `.debug` 文件 / `build-id` 关联）。
(b) 对本仓库（pass 级问题主要靠 `triton-opt -debug` 定位，偶发崩溃才需要 gdb）的最省事配置？

**Q18 [其它项目级机会]** 从 §0/§2 数据看，还有哪些你能指出的机会？例如：`mold --threads`/内存上限、`ccache` × `-gsplit-dwarf` 的组合效应、`-fno-...` 系列、把 `triton-opt` 拆成多个可执行以降低链接内存（181 MB 二进制）、`RUSTC`/Python 侧构建（`uv run setup.py bdist_wheel`）的浪费、`TRITON_BUILD_UT=OFF` 的取舍、把 LLVM 预编译成静态+`-flto` 之外的选项等。

---

## 4. 回答时的约束与偏好

1. **不接受"换机器/加钱"类答案**；请给本机可落地的配置、命令、以及预期的定量收益（时间/内存/CPU）。
2. 本机是**个人开发机**：允许改系统配置、允许加脚本/服务；但请标注**风险与回退**。
3. 必须保住 **pass 级调试**能力（`triton-opt -debug`）与**偶发崩溃的 gdb**能力；不要求顶级源码级步进体验。
4. 磁盘 769 GB 空闲：接受"用空间换时间"（缓存更大、`.dwo` 更多、多份构建目录）。
5. 项目侧改动要能**与团队/CI 共存**（若只适合本机，请明确标注 "local only"）。
6. 构建入口现状：`~/personal/build_ta.sh`（唯一官方入口，`-j16`）＋ 单目标增量 `ninja`（当前我手动 `-j6`）。若你建议改 `-j16`，请说明在 31 GiB 无 swap 下为何安全。
7. 请**优先回答** Q1、Q2、Q3、Q7、Q11、Q12、Q13、Q16（这 8 条是我最关心的）。

---

## 5. 附录：路径与原始数据速查

```
仓库:            /home/olivercwy/workspace/triton-ascend            (jj + git, HEAD 3a164e7f2)
构建目录:        build/cmake.linux-x86_64-cpython-3.11/             (Debug)
triton-opt:      build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt   (181,836,640 B, 重建于 13:37:05)
FileCheck:       ~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/FileCheck
官方构建脚本:    ~/personal/build_ta.sh                          (MAX_JOBS=16)
in-tree patch:   third_party/ascend/patch/triton-ascend-3.6.0.patch  (当前 APPLIED)
patch applier:   third_party/ascend/build/setup_patch.py         (非幂等: 先 git checkout 再 git apply)
还原脚本:        restore_sources.py
UT(54 例):       third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/{PlanComputeBlock,StandardizeOp,SplitDataflow,ComputeBlockOpt}
```
