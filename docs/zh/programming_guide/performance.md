# Triton 算子性能调优指南

## 第一篇：基础篇 — 理解硬件差异，建立心智模型

### 第 1 章 GPU 与 NPU 微架构差异：Triton 算子开发迁移视角

- 1.1 执行模型与 Grid 映射：Warp 调度 → 物理核绑定
  - GPU：SIMT 执行模型，32 线程/Warp 锁步执行
    - Grid 与物理 SM **解耦**：可启动远超 SM 数量的 Block，硬件自动排队调度
    - `num_warps` 控制每个 Block 的并行度，`num_stages` 控制软件流水线深度
    - 开发者只需按逻辑任务量设置 Grid（如 `cdiv(N, BLOCK_SIZE)`），硬件负责调度
  - NPU：SIMD 执行模型（默认），一个 Program 绑定一个 AI Core，无线程概念
    - Grid 与物理核**强绑定**：Grid 规模超出物理核数时，Block需分批（轮转）调度，每次切换都会产生额外的设备侧开销（Block调度与参数装载、核内上下文初始化等）；且跨Block的存算流水线无法并行，前一个Block需排空流水线后，下一个Block才重新预热启动。
    - `num_warps` / `num_stages` **无效**（NPU 无 Warp、无 GPU 式软件流水线）
    - 开发者需主动查询核心数并将 Grid 对齐：`grid = (num_cores,)` + 核内 for 循环
  - **对 Triton 代码的影响**：
    - 迁移第一步：删除 `num_warps`、`num_stages`，加入 `multibuffer` 参数
    - Grid 设计从"逻辑分块"变为"物理核持久化"（Persistent Kernel 模式）
    - 纯 Vector 算子用 `num_vectorcore`（910\_95 为 aicore 的 2 倍），CV 融合算子用 `num_aicore`
- 1.2 片上存储管理：Shared Memory 手动管理 → UB 编译器管理
  - GPU：Shared Memory（48-164 KB/SM）由程序员手动控制
    - 通过 `num_stages` 隐式控制 Shared Memory 多缓冲深度
    - Tensor Core 输出直接写入寄存器，后续运算立即可用
    - 开发者通过 `__shared__` 或 Triton 编译器间接管理 Shared Memory 布局
  - NPU：UB（248KB 可用）由编译器全权管理，开发者**不能手动分配**
    - UB 是 Vector 的唯一工作区：所有 `tl.load` 的数据、中间计算结果、`tl.store` 前的数据都在 UB
    - Cube 输出进入 L0C（256KB），需经 fixpipe 搬到 UB 后 Vector 才能处理
    - 开发者通过 `BLOCK_SIZE` **间接控制** UB 占用（而非直接分配 Shared Memory）
    - `multibuffer=True` 启用双缓冲，但 UB 可用空间减半（248KB → 124KB）
  - **对 Triton 代码的影响**：
    - `BLOCK_SIZE` 不仅是 Tiling 参数，还直接决定 UB 占用——选错即 `ub overflow` 编译错误
    - UB 预算公式：`Σ(每个 tensor 的 BLOCK_SIZE × dtype_size) ≤ 248KB`（开启 multibuffer 则 ≤ 124KB）
    - 大 BLOCK + 核内二级分块（`BLOCK_SIZE_SUB`）是 NPU 常用模式：核间大块控并行度，核内小块控 UB

***

## 第二篇：分析篇 — 性能剖析与瓶颈定位

### 第 2 章 性能分析方法论（调优的前提）

- 2.1 性能数据采集（数据从哪来）
  - torch\_npu.profiler（Python API）
    - `_ExperimentalConfig` 配置（AiCMetrics.PipeUtilization）
    - 采集 schedule 设计（wait / warmup / active / repeat）
  - msprof op 命令行：单算子级性能剖析
  - 仿真流水图（离线采集）
  - 辅助分析环境变量：`TRITON_PRINT_AUTOTUNING` / `TRITON_ALL_BLOCKS_PARALLEL` / `TRITON_DEBUG` / `TRITON_BENCH_METHOD`
  - **仓库文档参考**（部分覆盖，按内容相关度排序）：
    - 上板采集与 PipeUtilization 数据解读（走 msprof 命令行路径，非 torch\_npu.profiler）：[profiling.md](../debug_guide/profiling.md)
    - `torch_npu.profiler` 计时路径的实现与"设置预热和运行次数"（warmup/repeat，do\_bench\_npu 内部机制）：[do\_bench\_npu\_guide.md](../debug_guide/do_bench_npu_guide.md)
    - `TRITON_BENCH_METHOD="npu"` 下用 `torch_npu.profiler.profile` 采集每个 config 的片上时间：[06\_autotune\_example.md](../examples/06_autotune_example.md)
  - **覆盖缺口**：`_ExperimentalConfig` 配置、schedule 的 wait/active/repeat 在当前仓库文档中尚无直接说明，建议结合 CANN《性能调优工具用户指南》与 PyTorch profiler 官方文档补充
- 2.2 关键性能指标解读（数据怎么看）
  - 指标来源：采集阶段的 op\_summary / PipeUtilization.csv（见 2.1）
  - `aiv_vec_ratio`（Vector 利用率，理想 > 80%）
  - `aiv_mte2_ratio`（搬入占比，理想 < 50%）
  - `aiv_scalar_ratio`（标量利用率，警惕 > 30% 的标量退化）
  - `aic_cube_ratio`（Cube 利用率，理想 > 80%）
- 2.3 结果可视化与流水图解读（怎么直观呈现）
  - TensorBoard 可视化与结果解读（当前仓库无直接覆盖，参考 PyTorch profiler 官方文档）
  - Chrome Tracing / MindStudio Insight 流水图分析
  - 指令时间线诊断规则（断流检测、空白段识别）
- 2.4 瓶颈分类与优化方向决策树（看完怎么办）
  - 计算密集型 → 增大分块 / CV 融合
  - 访存密集型 → 存算并行 / care\_padding / 对齐优化
  - 标量退化 → 数据类型优化（避免 int64）
  - 同步开销 → 减少同步 / 调整编译参数

***

## 第三篇：调优篇 — 系统化性能优化方法论

### 第 3 章 Grid 映射与分核策略：让所有核心都饱和

> **核心理念**：NPU 的 Grid 与物理核强绑定，分核策略直接决定硬件利用率。优化的第一步不是改算法，而是确保每个核心都有足够的工作量。
>
> **（1）Grid 与物理核的强绑定关系。** GPU 的 Grid 与物理 SM 解耦——开发者只需按逻辑任务量设置 Grid（如 `cdiv(N, BLOCK_SIZE)`），硬件自动排队调度，可以启动远超 SM 数量的 Block（如 A100 有 108 个 SM，但 Grid 可以是数万个 Block），硬件的 Block Scheduler 负责将 Block 分配到空闲 SM。NPU 则完全不同：一个 Program 绑定一个 AI Core，Grid 与物理核**强绑定**。当 Grid 超出物理核数时，硬件分批调度（如 910\_95 有 \~40 个 AICore，Grid=120 时分 3 批执行），每批切换有额外的设备侧初始化开销（寄存器清零、流水线冲刷），核启动开销急剧增加。因此 NPU 上的最佳实践是 Grid = 物理核数，让每个核通过核内循环持久化处理多个任务块。
>
> **（2）核心数查询 API 与核型选择。** NPU 的 AI Core 包含 1 个 Cube（矩阵乘单元）+ 2 个 Vector（向量计算单元）。因此有两种核心数：`num_aicore`（AI Core 总数，含 Cube+Vector）和 `num_vectorcore`（Vector 核数，910\_95 上为 aicore 的 2 倍）。查询方式：`import triton.runtime.driver as driver; properties = driver.active.utils.get_device_properties(device); aicore_num = properties["num_aicore"]; vector_num = properties["num_vectorcore"]`。选择规则：**纯 Vector 算子**（无 `tl.dot`，如 RMSNorm、RoPE、Softcap）使用 `num_vectorcore` 分核，充分利用 2 倍的 Vector 核数；**CV 融合算子**（含 `tl.dot`，如 Flash Attention、Fused MatMul）使用 `num_aicore` 分核，因为 Cube 单元是计算瓶颈。选错核型会导致 Vector 算子只用一半 Vector 核，或 CV 算子无法启用 Cube。
>
> **（3）coreDim 硬件约束。** NPU 的 Grid 各维度乘积受 `coreDim ≤ 65535 (UINT16_MAX)` 硬件约束——这是硬件调度器用 16 位整数索引 program\_id 的限制。多维 Grid（如 `(blocks_m, q_head, batch)` = `(128, 32, 8)` = 32768）在 head 数和 batch 数较大时容易触及上限，编译报错 `coreDim exceeds UINT16_MAX`。展平为一维 `(NUM_CORES,)` 可彻底规避此约束。

- 3.1 Persistent Kernel：从逻辑 Grid 到物理核的持久化映射
  - **问题**：多维逻辑 Grid（如 Flash Attention 的 `(blocks_m, q_head, batch)`）使每个 program 只处理一个逻辑块，Grid 规模 `blocks_m × q_head × batch` （如 128×32×8=32768）远超物理核数，造成频繁的核间调度与切换开销。且多维 Grid 可能触及 `coreDim ≤ 65535` 的硬件上限
  - **方案**：将 Grid 展平为一维 `(NUM_CORES,)`，核内通过 `for` 循环以步长 `NUM_CORES` 遍历全部任务块（interleaved 分配），并从 `block_idx` 反推原始的多维索引。
  - **协同优化**：
    - 核心数动态获取（`device_properties.get("num_vectorcore")`），不硬编码——910B 与 950 核心数不同
    - 纯 Vector 算子使用 `num_vectorcore`，CV 融合算子用 `num_aicore`
    - 持久化循环天然形成软件流水线——上一轮 Vector 计算与下一轮 MTE2 加载重叠
  ```python
  # ── Before: 多维逻辑 Grid，每 program 处理一个 (M块, head, batch) 组合 ──
  @triton.jit
  def fwd_kernel(q_ptr, k_ptr, v_ptr, ...,
                 BLOCK_M: tl.constexpr, ...):
      start_m = tl.program_id(0)    # M 维度 block 索引
      start_qh = tl.program_id(1)   # head 维度
      start_b = tl.program_id(2)    # batch 维度
      # ... 单块计算逻辑 ...

  # 调用处：Grid 远超物理核数
  grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]),
                       q_head, batch_size)
  fwd_kernel[grid](q, k, v, ...)

  # ── After: 一维物理 Grid，每 program 循环处理多个块（interleaved 分配）──
  @triton.jit
  def fwd_kernel(q_ptr, k_ptr, v_ptr, ...,
                 AICORE_NUM: tl.constexpr, MAX_Q_LEN: tl.constexpr,
                 BATCH_SIZE: tl.constexpr, q_head: tl.constexpr,
                 kv_head: tl.constexpr, BLOCK_M: tl.constexpr, ...):
      pid = tl.program_id(0)
      NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)
      NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head
      # interleaved 遍历：步长 = AICORE_NUM，各核交替领取任务
      for block_idx in range(pid, NUM_BLOCKS, AICORE_NUM):
          # 从一维 block_idx 反推多维索引
          task_hz_idx = block_idx // NUM_BLOCKS_M
          start_m = block_idx % NUM_BLOCKS_M
          start_b = task_hz_idx // q_head
          start_qh = task_hz_idx % q_head
          start_kvh = start_qh * kv_head // q_head  # GQA head 映射
          # ... 单块计算逻辑（与 Before 相同）...

  # 调用处：Grid = 物理核数，核心数动态获取
  import triton.runtime.driver as driver
  properties = driver.active.utils.get_device_properties(device)
  NUM_CORES = properties["num_aicore"]  # CV 融合算子用 aicore
  grid = (NUM_CORES,)
  fwd_kernel[grid](q, k, v, ...,
                   AICORE_NUM=NUM_CORES,
                   MAX_Q_LEN=max_seqlen_q, ...)
  ```
  - **案例**：FA v8 fwd/bwd — 3D Grid `(blocks_m, q_head, batch)` → 1D Grid `(AICORE_NUM,)` + interleaved 循环
  - **案例**：RoPE — 3D Grid `(seq_blocks, bs, head)` → 1D Grid `(bs,)` + 核内遍历 seq\_len 和 head
- 3.2 自适应分核：按任务量缩减核数
  - **问题**：当任务总量小于总核数时（如 batch\_size=4 而 num\_vectorcore=80），部分核心分不到任务，产生空转——`range(pid, total_tasks, num_cores)` 中 `pid >= total_tasks` 的核心循环体为空，属于无效启动、造成资源空转。
  - **方案**：根据实际任务量动态缩减 Grid 核心数，确保每个核心至少分到一个完整任务块，避免无效核心参与调度
  ```python
  # ── Before: 固定核心数，部分核心空转 ──
  NUM_CORES = VECTOR_NUM  # 80 核
  grid = (NUM_CORES,)
  # 当 total_tasks=4 时，76 个核心空转

  # ── After: 根据任务量动态缩减 ──
  # 计算每核至少分到一个 BLOCK_SIZE 大小的任务块
  total_tasks = triton.cdiv(numel, BLOCK_SIZE)
  if total_tasks < VECTOR_NUM:
      NUM_CORES = total_tasks  # 缩减到实际任务数
  else:
      NUM_CORES = VECTOR_NUM
  grid = (NUM_CORES,)
  ```
  - **案例**：RMSNorm — `n_rows < num_cores` 时缩减 Grid 到 `n_rows`
- 3.3 SPLIT\_K 拆分：K 维度并行化
  - **问题**：当 M×N 维度不足以饱和所有核心时（如 decode 场景的小 batch，M=1、N=head\_dim=128，只有 1×8=8 个输出块），大量核心闲置。此时 M×N 的并行度远小于物理核数
  - **方案**：在 K 维度上拆分 matmul，多个 program 并行计算部分和，通过原子操作或锁机制累加到最终输出
  - **关键细节**：SPLIT\_K 个 program 写同一输出地址，必须用 `tl.atomic_add` 累加（无锁）或 `LOCK` 数组 + `tl.store`（有锁）；autotune 时需 `reset_to_zero=["output_ptr"]` 避免多次评估累加
  ```python
  # ── Before: 单 program 处理完整 K 维度，M×N 并行度不足 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
      # 单 program 遍历完整 K 维度
      for k in range(0, K, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)
      tl.store(c_ptr + ..., acc)  # 直接写入，无冲突

  grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))  # M×N 并行度

  # ── After: K 维度拆分，多 program 并行 + 原子累加 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      pid_k = tl.program_id(2)  # K 维度拆分索引
      acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
      # 每个 program 只处理 K/SPLIT_K 部分
      k_start = pid_k * K // SPLIT_K
      k_end = (pid_k + 1) * K // SPLIT_K
      for k in range(k_start, k_end, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)
      # 原子累加：SPLIT_K 个 program 写同一地址
      tl.atomic_add(c_ptr + ..., acc)

  # Grid 增加 K 维度：M×N×SPLIT_K 并行度
  grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), SPLIT_K)
  ```
  - **案例**：Fused MatMul bwd\_w — `SPLIT_K` 参数（8/16）拆分 K 维度，配合 `LOCK_W` 数组处理写冲突
  - **案例**：FA bwd\_qkv — `dq` 用 `tl.atomic_add` 累加多核结果，autotune 配 `reset_to_zero=["dq_ptr"]`
- 3.4 Sorted Round-Robin 负载均衡：稀疏注意力的任务不均衡问题
  - **问题**：稀疏注意力（Flex Attention / SWA）与 GQA 场景中，不同任务块的工作量差异极大：causal mask 使上三角全零，前几个 query block 只需处理少量 key block，而最后一个 query block 要处理全部 key block。此时若按stride 方式遍历任务（`range(pid, total, num_cores)`），各核任务大小混杂不均，导致各核总工作量严重失衡：分配到靠后 query block 的核满载，分配到靠前的核大量空转。
  - **方案**：主机端先枚举所有任务 → 计算 workload（如每个 query block 需处理的 key block 数）→ 按 workload 降序排序 → 以轮询方式分配到各核（round-robin）→ kernel 端通过任务张量读取本核的任务列表
  - **原理**：降序排序保证大任务优先被分配，轮询则让每个核在排序序列上等距采样（核 0 取第1、第 C+1 、第 2C+1...大的任务），因此各核拿到的任务集合在大小分布上高度同构，总工作量趋于均衡；同时避免"最后才遇到一个超大任务，已无处摊平"的情况。
  ```python
  # ── Before: kernel 内 stride 遍历，各核工作量严重不均 ──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      pid = tl.program_id(0)
      # interleaved 遍历：靠前的 query block 工作量小，靠后的大
      for block_idx in range(pid, total_tasks, num_cores):
          start_m = block_idx % NUM_BLOCKS_M
          # causal mask：start_m 越大，需处理的 key block 越多
          num_kv_blocks = start_m + 1  # 工作量随 start_m 线性增长
          for kv in range(num_kv_blocks):
              ...  # 靠前的核只执行 1 轮，靠后的执行 NUM_BLOCKS_M 轮

  # ── After: 主机端预排序 + 轮询分配，各核均衡 ──
  # 主机端：计算每个任务的 workload 并降序排序
  def assign_tasks(total_tasks, num_cores):
      workloads = [compute_workload(t) for t in range(total_tasks)]
      tasks_sorted = sorted(range(total_tasks), key=lambda t: workloads[t],
                            reverse=True)  # 降序：重任务在前
      # 轮询分配：核 0 拿 tasks_sorted[0], tasks_sorted[num_cores], ...
      core_tasks = [[] for _ in range(num_cores)]
      for i, task in enumerate(tasks_sorted):
          core_tasks[i % num_cores].append(task)  # round-robin
      # 展平为连续张量供 kernel 读取
      core_tasks_ranges = [...]  # 每核的任务起止偏移
      core_tasks_flat = tensor([task for core in core_tasks for task in core])
      return core_tasks_ranges, core_tasks_flat

  # kernel 端：通过任务张量读取分配到本核的任务
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, core_tasks_ptr, core_tasks_ranges_ptr, ...):
      pid = tl.program_id(0)
      start_offset = tl.load(core_tasks_ranges_ptr + pid * 2)
      end_offset = tl.load(core_tasks_ranges_ptr + pid * 2 + 1)
      # 只遍历分配到本核的任务（已均衡）
      for i in range(start_offset, end_offset):
          block_idx = tl.load(core_tasks_ptr + i)
          start_m = block_idx % NUM_BLOCKS_M
          ...  # 各核总工作量趋于一致
  ```
  - **案例**：FlexAttention — 主机端 sorted round-robin 分配，消除 causal mask 导致的工作量不均
  - **案例**：SWA prefill — 滑动窗口场景的 workload 排序分配

### 第 4 章 Tiling 与分块策略：在 UB 约束下找到最优分块

> **核心理念**：BLOCK\_SIZE 不仅是 Tiling 参数，还直接决定 UB 占用——选错即 `ub overflow`。Tiling 的本质是在 UB 容量、对齐约束和计算效率之间找到最优平衡点。
>
> **（1）UB 容量与预算公式。** NPU 的 UB（Unified Buffer）是 Vector 的唯一工作区——所有 `tl.load` 的数据、中间计算结果、`tl.store` 前的数据都在 UB。910\_95（A5）的 UB 硬件为 256KB，编译器预留 8KB 后实际可用 **248KB**；910B（A2）的 UB 为 **192KB**。与 GPU 的 Shared Memory（开发者手动管理、通过 `__shared__` 或 `num_stages` 间接控制）不同，NPU 的 UB 由编译器全权管理，开发者通过 `BLOCK_SIZE` **间接控制** UB 占用。UB 预算公式：`所需UB = Σ(每个 tensor 的 BLOCK_SIZE × dtype_bytes) × multibuffer副本数 ≤ UB可用`。例如 FP16 的 RMSNorm kernel，输入 `x` (BLOCK\_SIZE×2B) + weight `w` (BLOCK\_SIZE×2B) + 输出 `y` (BLOCK\_SIZE×2B) + 中间 `mean/var` (32B) ≈ BLOCK\_SIZE×6B + 32B。开启 `multibuffer=True` 后 UB 可用空间减半（910\_95 为 124KB，910B 为 96KB），因为双缓冲需要两份 UB 空间。超出 UB 容量时编译报错 `ub overflow`，必须减小 BLOCK\_SIZE 或减少核内同时存活的 tensor 数量。
>
> **（2）对齐约束。** NPU 有两层对齐要求：**Vector 尾轴 32B 连续对齐**是硬约束——FP16（2B）尾轴元素须为 16 的倍数，FP32（4B）须为 8 的倍数，否则 DMA 引擎无法发起连续搬运；**CV 融合算子（含** **`tl.dot`）要求 512B 对齐**——FP16 尾轴元素须为 256 的倍数，因为 Cube 单元的矩阵分块粒度更大。非对齐场景需要额外的 Mask 处理（加载时用 mask 截断多余元素），但会引入性能开销。此外，单值缓冲区（如归约结果 mean/var/max/sum）逻辑上只需 4B，但硬件要求 32B 对齐，实际分配 32B。
>
> **（3）计算效率与分块大小的关系。** 大 BLOCK 提升 Cube 利用率和数据复用（矩阵乘的 BLOCK\_M×BLOCK\_N 越大，每个元素被复用的次数越多），但受 UB 容量限制；小 BLOCK 增加 Grid 分核数（更多并行度），但 Cube 利用率低。NPU 上 Cube 单元的最优分块通常是 128（`BLOCK_M=BLOCK_N=BLOCK_K=128` 是 Fused MatMul 的收敛值），因为 Cube 的 `mmad` 指令以 128×128 矩阵为基本计算单元。对于纯 Vector 算子（无 `tl.dot`），BLOCK\_SIZE 可以更大（如 Softcap 的 BLOCK\_SIZE=2048），因为 Vector 的 UB 压力小于 CV 融合算子。

- 4.1 核内多块循环：提升 MTE2/MTE3/计算流水并行度
  - **问题**：标准 Grid 模式下每个 program 只处理一个数据块，Grid 规模大（如 `cdiv(numel, BLOCK_SIZE)=1000`）但核内无流水线深度——每个 program 只做一次 load→compute→store，无法利用存算并行
  - **方案**：新增 `BLOCK_NUM` 参数，单 program 内循环处理多个数据块，Grid 相应缩小 `BLOCK_NUM` 倍。核内多块循环减少 Grid 规模（降低核切换开销），同时实现存算并行（上一块 compute 与下一块 load 重叠）
  ```python
  # ── Before: 每 program 一个块，Grid 大但无核内流水线 ──
  @triton.jit
  def softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap,
                         BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask)
      y = softcap * libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Grid 可能很大

  # ── After: 每 program 处理 BLOCK_NUM 个块，Grid 缩小 ──
  @triton.jit
  def softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap,
                         BLOCK_SIZE: tl.constexpr,
                         BLOCK_NUM: tl.constexpr):   # 新增多块参数
      pid = tl.program_id(0)
      for i in range(BLOCK_NUM):
          # 每 i 轮处理一个 BLOCK_SIZE 大小的块
          block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
          offsets = block_start + tl.arange(0, BLOCK_SIZE)
          mask = offsets < n_elements
          x = tl.load(x_ptr + offsets, mask=mask)
          y = softcap * libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
          tl.store(y_ptr + offsets, y, mask=mask)

  # Grid 缩小 BLOCK_NUM 倍，核内循环实现存算并行
  grid = (triton.cdiv(n_elements, BLOCK_SIZE * BLOCK_NUM),)
  ```
  - **案例**：Softcap — 新增 `BLOCK_NUM` 参数（1/2/4/8/16/32/64），核内循环处理多块，BLOCK\_SIZE×BLOCK\_NUM 决定每 program 总数据量
- 4.2 1D 展平替代 2D：小 shape 场景的索引开销消除
  - **问题**：小 shape 下 2D row-blocked kernel 的 2D 索引开销占比高——`offsets = row[:, None] * stride + col[None, :]` 需要二维广播乘加，当 `n_cols` 较小时（如 head\_dim=128），索引计算的时间接近甚至超过实际计算时间
  - **方案**：改为 1D flatten kernel，将二维 `(rows, cols)` 展平为一维 `rows*cols`，消除 2D 索引，提升内存合并（连续地址访问）
  ```python
  # ── Before: 2D kernel，小 shape 下索引开销高 ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_rows, n_cols, ...,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
      col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
      # 2D 索引：行偏移 × stride + 列偏移
      offsets = row[:, None] * n_cols + col[None, :]
      mask = (row[:, None] < n_rows) & (col[None, :] < n_cols)
      x = tl.load(x_ptr + offsets, mask=mask)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N))

  # ── After: 1D flatten kernel，消除 2D 索引 ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_elements, ...,
                      BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      # 1D 连续偏移，无二维广播
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_rows * n_cols, BLOCK_SIZE),)
  ```
  - **案例**：GELU / SiLU — `n_cols <= 10240` 或 `n_rows < num_cores` 时切换到 1D flatten kernel
- 4.3 移除 next\_power\_of\_2：避免不必要的 padding
  - **问题**：`next_power_of_2(d)` 将实际维度向上取整到 2 的幂（如 d=300 → 512，d=192 → 256），导致 padding 浪费 UB 空间（多分配 70% 的 UB）和多余循环迭代。更关键的是，当 `BLOCK_DIN = next_power_of_2(Din)` 使 `BLOCK_DIN > Din` 时，归约循环需要执行多次（无法单 Pass 完成），阻碍算子融合
  - **方案**：当实际维度满足 32B 对齐要求时（FP16 的 d 须为 16 的倍数），直接使用实际维度 `D_ROUNDED = d`，消除 padding 浪费并使循环只执行一次
  ```python
  # ── Before: 向上取整到 2 的幂，padding 浪费 + 多次循环 ──
  @triton.jit
  def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                         D_ROUNDED: tl.constexpr):
      # D_ROUNDED = triton.next_power_of_2(d)
      # 当 d=300 时 D_ROUNDED=512，多分配 212 个元素的 UB 空间
      offs_d = tl.arange(0, D_ROUNDED)
      mask_d = offs_d < d  # 需要 mask 截断 padding 区域
      x = tl.load(x_ptr + row * d + offs_d, mask=mask_d, other=0.0)
      # 当 D_ROUNDED > d 时，循环可能执行多次
      mean = tl.sum(x * x, axis=0) / d
      ...

  # 调用处
  D_ROUNDED = triton.next_power_of_2(d)  # 300 → 512

  # ── After: 直接使用实际维度，无 padding + 单次循环 ──
  @triton.jit
  def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                         D_ROUNDED: tl.constexpr):
      # D_ROUNDED = d（当 d 满足 32B 对齐时）
      offs_d = tl.arange(0, D_ROUNDED)
      # D_ROUNDED == d，无需 mask（已对齐）
      x = tl.load(x_ptr + row * d + offs_d)
      # D_ROUNDED == d，循环只执行一次 → 单 Pass 融合
      mean = tl.sum(x * x, axis=0) / d
      ...

  # 调用处
  D_ROUNDED = d  # 直接用 300（假设 300 是 16 的倍数，或已对齐）
  ```
  - **案例**：RMSNorm — `D_ROUNDED=triton.next_power_of_2(d)` → `D_ROUNDED=d`，使归约循环只执行一次
  - **案例**：Gate RMSNorm — `BLOCK_DIN=min(3264, next_power_of_2(Din))` → `BLOCK_DIN=Din`，使循环只执行一次，实现单 Pass 融合
- 4.4 SWA 循环结构优化：合并 vs 拆分
  - **问题**：滑动窗口注意力（SWA）中，全局窗口块（始终满密度，无需 local mask）和非全局窗口块（需 causal+local mask）的循环结构影响性能。两种循环结构在不同全局窗口占比下性能差异显著
  - **两种策略**（根据全局窗口占比选择）：
    - **合并**（全局窗口占比低时更优）：将两个独立循环合并为一个，用 `tl.where` 统一 mask，减少循环数量和 Grid 开销
    - **拆分**（全局窗口占比高时更优）：将统一循环拆分为两段——全局段无 mask 开销（满密度直接计算），非全局段只执行 causal+local mask
  ```python
  # ── 合并策略（全局窗口占比低时更优）──
  # Before: 两个独立循环，循环开销翻倍
  for kv in range(global_window_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = global_mask  # 全局窗口 mask
      qk = tl.where(mask, qk, -inf)
  for kv in range(local_window_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = causal_local_mask  # causal + local mask
      qk = tl.where(mask, qk, -inf)

  # After: 合并为一个循环，用 tl.where 统一 mask
  for kv in range(all_kv_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      is_global = (kv < global_threshold)
      # 统一 mask：全局段用 global_mask，非全局段用 causal_local_mask
      mask = tl.where(is_global, global_mask, causal_local_mask)
      qk = tl.where(mask, qk, -inf)

  # ── 拆分策略（全局窗口占比高时更优）──
  # Before: 统一循环，全局段也承受 mask 检查开销
  for kv in range(all_kv_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      # 每轮都检查 global_window_check，即使全局段不需要
      mask = causal_mask & (local_window_check | global_window_check)
      qk = tl.where(mask, qk, -inf)

  # After: 拆分为两段，全局段零 mask 开销
  for kv in range(global_blocks):       # 段 1：全局窗口，无 mask 开销
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      # 全局窗口满密度，无需 mask
  for kv in range(local_blocks):        # 段 2：非全局窗口，causal + local mask
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = causal_mask & local_window_mask
      qk = tl.where(mask, qk, -inf)
  ```
  - **案例**：SWA prefill — 根据全局窗口占比选择合并或拆分策略

### 第 5 章 Autotune 调优策略：让调优器替你找到最优配置

> **核心理念**：NPU 的最优 BLOCK 尺寸取决于 UB 容量、数据类型、矩阵形状和硬件架构的复杂交互，人工试错效率低下。Autotune 让调优器自动搜索最优配置，是 NPU 算子调优的标准武器。
>
> **（1）Config 内容差异。** GPU 的 `triton.Config` 包含 Tiling 参数 + `num_stages`（软件流水线深度，1\~4）+ `num_warps`（线程束数量，4/8/16/32）。NPU 的 Config **仅含 Tiling 参数**——`num_stages` 和 `num_warps` 在 NPU 上无意义（NPU 无 GPU 式 warp 和软件流水线 stage），必须移除。NPU 的流水线并行通过 `multibuffer`（UB 双缓冲）和核内 `for` 循环实现，而非 `num_stages`。此外，NPU 的 Config 可以将编译参数（如 `multibuffer`、`enable_vf_fusion`）作为搜索维度。
>
> **（2）候选值约束。** NPU 的 BLOCK 候选值受 UB 容量硬约束——大 BLOCK 提升 Cube 利用率和数据复用，但超出 UB 容量则编译失败（`ub overflow`）。经验公式：`最大Tiling元素数 ≈ UB_SIZE / (dtype_bytes × buffer数量)`，如 FP16（2B）3 buffers 时 248KB/(2×3) ≈ 40960 元素。此外，NPU 的 BLOCK 值应满足 32B 对齐（FP16 须为 16 的倍数），使用 2 的幂次值（32/64/128/256/512/1024/2048）天然满足对齐要求。非 2 的幂的值（如 100、86）也可使用，但需确保对齐。
>
> **（3）key 参数化与缓存。** `key` 参数指定调优触发维度——当 key 值变化时重新调优，相同值复用缓存。NPU 上需要特别注意：不同 dtype（fp16 vs bf16）的 UB 占用不同（bf16 和 fp16 都是 2B，但中间计算可能不同），应将 dtype 加入 key；不同算子属性（如 FA 的 QK\_DIM/V\_DIM/MASK\_FN/SPARSE\_OPT）影响最优 Tiling，也应加入 key。Autotune 的首次运行会尝试所有候选 Config，配置数过多会显著增加首次运行时间——建议总配置数控制在 20 个以内，使用 `prune_configs_by` 提前剪枝不可行配置。

- 5.1 BLOCK 尺寸搜索：从小 BLOCK 起步逐步增大
  - **问题**：固定小 BLOCK 导致 Cube 利用率低和数据复用差，但盲目增大可能溢出 UB。开发者难以人工判断特定 shape 下的最优 BLOCK 值
  - **方案**：从小 BLOCK 起步，通过 Autotune 逐步增大搜索最优值。候选值可包含非 2 的幂以精确匹配实际维度（如序列长度 100）
  ```python
  # ── Before: 固定小值，无法适配不同 shape ──
  @triton.jit
  def fwd_kernel(..., BLOCK_M: tl.constexpr):
      # BLOCK_M 固定为 32，大 shape 下 Cube 利用率低
      ...

  # ── After: Autotune 搜索，自动选择最优 BLOCK ──
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_M": 32}),
          triton.Config({"BLOCK_M": 64}),
          triton.Config({"BLOCK_M": 128}),
          triton.Config({"BLOCK_M": 256}),
          # 场景：非 2 的幂精确匹配实际维度
          triton.Config({"BLOCK_M": 100}),  # 序列长度恰好 100
      ],
      key=["QK_DIM", "V_DIM"],  # 按 head_dim 缓存最优 config
  )
  @triton.jit
  def fwd_kernel(..., BLOCK_M: tl.constexpr):
      ...
  ```
  - **案例**：FA bwd\_q blkM 32→64→128，bwd\_kv blkN 32→64→128，bwd\_preprocess 32→128/256
  - **案例**：Fused MatMul fwd 从搜索空间收敛到 BM=BN=BK=128（Cube 单元最优分块）
  - **案例**：RoPE TOKEN\_BLOCK\_SIZE=3（非 2 的幂精确匹配）
- 5.2 多维组合搜索：从单 config 到全空间
  - **问题**：单维度搜索无法捕捉 BLOCK\_M × BLOCK\_N × BLOCK\_K 的交互效应——如 BLOCK\_M=128 在 BLOCK\_N=32 时最优，但在 BLOCK\_N=128 时可能 BLOCK\_M=64 更优（UB 约束）
  - **方案**：从单 config 扩展到多维组合搜索，使用列表推导生成笛卡尔积
  ```python
  # ── Before: 单 config，无法捕捉维度交互 ──
  @triton.autotune(
      configs=[triton.Config({"BLOCK_M": 128})],  # 只有一个配置
      key=["M", "N"],
  )

  # ── After: 多维组合搜索，覆盖 BM×BN×BK 全空间 ──
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK})
          for BM in [128, 256, 512]
          for BN in [64, 128, 256, 512]
          for BK in [32, 64, 128, 512]
      ],  # 3×4×4 = 48 个配置
      key=["M", "N", "K"],
  )
  @triton.jit
  def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
      ...
  ```
  - **案例**：SwiGLU — 从单 config 扩展到 BM×BN×BK 三维搜索
- 5.3 Key 参数化：按形状/dtype 缓存最优 config
  - **问题**：不同输入形状/属性的最优 config 不同，无 key 参数化时跨形状重复调优（每次 shape 变化都重新搜索全部 Config）。更隐蔽的问题：不同 dtype（fp16 vs bf16）的 UB 占用和计算路径不同，但若 dtype 不在 key 中，fp16 选出的 config 可能对 bf16 不是最优
  - **方案**：按输入形状/属性缓存最优 config；将 dtype 也加入 key（通过 `X_ptr.dtype` 引用指针的 dtype）
  ```python
  # ── Before: 基础 key，跨 shape/dtype 重复调优 ──
  @triton.autotune(
      configs=[...],
      key=["n_rows", "n_cols"],  # 缺少 dtype，fp16/bf16 共用 config
  )

  # ── After: 扩展 key（含 dtype + 算子属性）──
  @triton.autotune(
      configs=[...],
      # 将 dtype 加入 key：fp16 和 bf16 各自缓存最优 config
      key=["n_rows", "n_cols", "X_ptr.dtype"],
  )

  # 或按算子属性缓存（Flash Attention 场景）
  @triton.autotune(
      configs=get_fwd_configs(),
      # QK_DIM/V_DIM 影响 UB 占用，MASK_FN/SPARSE_OPT 影响计算量
      key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
  )
  ```
  - **案例**：FA v5 — 按 QK\_DIM/V\_DIM/MASK\_FN/SPARSE\_OPT 缓存
  - **案例**：RMSNorm — 将 dtype 加入 key（fp16 vs bf16 UB 占用不同）
- 5.4 搜索空间剪枝：提前过滤无效 config
  - **问题**：大搜索空间中包含两类无效 Config：(1) 会溢出 UB 的 Config（BLOCK\_M×BLOCK\_N 过大），首次运行时编译失败浪费时间；(2) 特定架构上永远不会胜出的 Config（如 910B 上大 BLOCK 因 UB 只有 192KB 而总是不如小 BLOCK）。这些无效 Config 浪费首次运行编译时间
  - **方案一（UB 剪枝）**：通过 `prune_configs_by` 的 `early_config_prune` 回调函数，在编译前过滤会溢出 UB 的 Config
  - **方案二（架构剪枝）**：按架构裁剪搜索空间，只保留该架构上可能胜出的 Config
  ```python
  # ── UB 剪枝：过滤会溢出 UB 的 Config ──
  def _prune_oversized_tiles(configs, named_args, **kwargs):
      # 过滤 BLOCK_M × BLOCK_N 超过 UB 容量的 Config
      return [c for c in configs
              if c.kwargs['BLOCK_M'] * c.kwargs['BLOCK_N'] <= 1048576]

  @triton.autotune(
      configs=[...],  # 原始搜索空间
      prune_configs_by={"early_config_prune": _prune_oversized_tiles},
      key=["M", "N"],
  )

  # ── 架构剪枝：按架构裁剪搜索空间 ──
  def get_configs():
      all_configs = [
          triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
          for BM in [32, 64, 128, 256]
          for BN in [32, 64, 128, 256]
      ]
      if is_910():  # 910B UB 只有 192KB，大 BLOCK 容易溢出
          # 只保留 BLOCK_M ≤ 128 的 Config（910B 胜出 config）
          return [c for c in all_configs if c.kwargs['BLOCK_M'] <= 128]
      return all_configs  # 910_95 UB 248KB，全空间搜索

  @triton.autotune(configs=get_configs(), key=["M", "N"])
  ```
  - **案例**：RMSNorm / Int8GEMM — UB 溢出 config 提前过滤
  - **案例**：SWA a2 — 8 configs → 1 config（910B 架构剪枝）
- 5.5 编译参数加入搜索
  - **问题**：编译参数（如 `enable_vf_fusion`、`multibuffer`）的最优值取决于具体 shape，人工选择容易遗漏。特别是 `multibuffer`：开启后 UB 减半但实现存算并行，某些 shape 下开启更优（计算密集），某些 shape 下关闭更优（UB 受限需要大 BLOCK）
  - **方案**：将编译参数作为 Autotune 搜索维度，让调优器自动选择
  ```python
  # ── 编译参数加入搜索 ──
  @triton.autotune(
      configs=[
          # 同一 BLOCK 下，multibuffer 和 enable_vf_fusion 各试一次
          triton.Config({"BLOCK_M": 8}, multibuffer=True, enable_vf_fusion=True),
          triton.Config({"BLOCK_M": 8}, multibuffer=True, enable_vf_fusion=False),
          triton.Config({"BLOCK_M": 8}, multibuffer=False, enable_vf_fusion=True),
          triton.Config({"BLOCK_M": 8}, multibuffer=False, enable_vf_fusion=False),
      ],
      key=["M", "N"],
  )
  ```
  - **案例**：RMSNorm — `enable_vf_fusion` 作为搜索维度加入 autotune configs
  - **案例**：RoPE — `multibuffer` 作为搜索维度加入 autotune configs
- 5.6 autotune 结果磁盘缓存：避免重复调优
  - **问题**：每次运行都重新 autotune 浪费时间，特别适用于固定 shape 的生产环境（如推理场景的 seq\_len 固定）
  - **方案**：开启 autotune 的**磁盘缓存**——`cache_results=True`，或环境变量 `TRITON_CACHE_AUTOTUNING=1` 全局开启。首次调用正常编译并 benchmark 全部 config，把每个 config 的耗时写入 `TRITON_CACHE_DIR/<hash>/<函数名>.autotune.json`；后续相同 `key` 的调用直接加载缓存、取耗时最短的 config，**跳过整个搜索阶段**。注意这是**按 key 缓存**而非永久固定：`key` 参数取值、dtype、编译选项或 config 列表任一变化都会使缓存失效并重新调优
  ```python
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": 1024}, multibuffer=True),
          triton.Config({"BLOCK_SIZE": 2048}, multibuffer=True),
          triton.Config({"BLOCK_SIZE": 1024}, multibuffer=False),
      ],
      key=["n"],            # n 不变 → 命中磁盘缓存，跳过 benchmark
      cache_results=True,   # 关键：开启磁盘缓存
  )
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
      offs = tl.arange(0, BLOCK_SIZE)
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)
  ```
  - **缓存文件内容**：`<函数名>.autotune.json` 保存 `key` 与每个 config 的实测耗时（中位数/下界/上界），命中后按耗时取 `min`
  ```json
  {
    "key": [8192],
    "configs_timings": [
      [{"kwargs": {"BLOCK_SIZE": 1024}, "multibuffer": true}, [0.023, 0.021, 0.030]],
      [{"kwargs": {"BLOCK_SIZE": 2048}, "multibuffer": true}, [0.031, 0.029, 0.035]],
      [{"kwargs": {"BLOCK_SIZE": 1024}, "multibuffer": false}, [0.040, 0.038, 0.045]]
    ]
  }
  ```
  - **彻底固定配置**：若 shape 长期不变，可调优一次后把最优 config 直接写死、去掉 `@triton.autotune`，连首次调优开销也省掉
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
      offs = tl.arange(0, BLOCK_SIZE)
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)

  # 直接指定最优 config（BLOCK_SIZE=1024 + multibuffer=True），跳过搜索
  kernel[(grid,)](x, y, n, BLOCK_SIZE=1024, multibuffer=True)
  ```
  - **案例**：RoPE / 推理场景 — seq\_len 固定使 `key` 不变，后续推理直接命中缓存
- 5.7 动态序列长度：避免每个新长度都触发 Autotune
  - **问题**：在自回归解码、变长输入等整网场景中，切分轴 `seq_len` 可能连续取 `1001、1002、1003……`。如果使用 `key=["seq_len"]`，每个未见过的长度都会产生新的调优缓存项。相同的完整 key 可以复用已有结果，但开启 5.6 节的磁盘缓存只能解决相同 key 的跨进程复用，不能自动让新长度沿用附近长度的配置。
  - **选择原则**：将大量精确长度归并为有限的性能区间，并让 kernel 在运行时处理真实长度。
    **方案1：长度分桶：用桶值做调优 key，用真实长度执行。**
  将长度向上归并到 2 的幂，例如 `1001、1002、1024` 的桶值均为 `1024`，`1025` 的桶值为 `2048`。也可以根据性能拐点设置不等宽区间。桶值只控制配置复用，不要求将输入 padding 到桶上界；grid、循环边界和 mask 仍使用真实长度。
  ```python
  def length_bucket(seq_len):
      return triton.next_power_of_2(max(128, seq_len))

  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": 256}),
          triton.Config({"BLOCK_SIZE": 512}),
          triton.Config({"BLOCK_SIZE": 1024}),
      ],
      key=["seq_bucket"],  # seq_len 的精确值不进入调优 key
      cache_results=True,
  )
  @triton.jit(do_not_specialize=["seq_len", "seq_bucket"])
  def add_one_bucket_kernel(X, Y, seq_len, seq_bucket,
                            BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      for block_id in range(pid, tl.cdiv(seq_len, BLOCK_SIZE),
                            tl.num_programs(0)):
          offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
          mask = offsets < seq_len
          x = tl.load(X + offsets, mask=mask, other=0.0)
          tl.store(Y + offsets, x + 1.0, mask=mask)

  def add_one_bucket(x, num_cores):
      y = torch.empty_like(x)
      seq_len = x.numel()
      if seq_len == 0:
          return y
      seq_bucket = length_bucket(seq_len)
      grid = lambda meta: (
          min(num_cores, triton.cdiv(seq_len, meta["BLOCK_SIZE"])),
      )
      add_one_bucket_kernel[grid](x, y, seq_len, seq_bucket)
      return y
  ```
  同桶的第一次调用会在当时的真实长度上 benchmark，后续直接复用，**不会对桶内每个长度继续搜索**。应选择有代表性的长度预热，并验证桶下界、上界及常见长度的正确性和性能；若桶内性能差异明显，拆分桶，或离线在多个代表点上综合选择配置。所有候选配置都必须对桶内全部长度正确，不能依赖首次调优长度的整除性，也不能省略必要的尾部 mask。

  **方案2：启发式分派：离线调优形成规则，线上直接选择配置。**

  在目标硬件、dtype、布局和典型长度上离线测量，找到不同配置的性能交叉点，再将结果写成区间表或条件分支。线上只选择配置并调用普通 `@triton.jit` kernel，不执行 `@triton.autotune` 的搜索；也可以用 `@triton.heuristics` 把选配置的规则直接挂到 kernel 上，在 launch 时动态算出 `BLOCK_SIZE`，效果与手动分派等价：
  ```python
  def select_block_size(seq_len):
      if seq_len <= 512:
          return 256
      if seq_len <= 4096:
          return 512
      return 1024  # 为更长输入提供覆盖路径

  @triton.heuristics({"BLOCK_SIZE": lambda args: select_block_size(args["seq_len"])})
  @triton.jit(do_not_specialize=["seq_len"])
  def add_one_fixed_kernel(X, Y, seq_len, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      for block_id in range(pid, tl.cdiv(seq_len, BLOCK_SIZE),
                            tl.num_programs(0)):
          offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
          mask = offsets < seq_len
          x = tl.load(X + offsets, mask=mask, other=0.0)
          tl.store(Y + offsets, x + 1.0, mask=mask)

  def add_one_dispatch(x, num_cores):
      y = torch.empty_like(x)
      seq_len = x.numel()
      if seq_len == 0:
          return y
      grid = lambda meta: (
          min(num_cores, triton.cdiv(seq_len, meta["BLOCK_SIZE"])),
      )
      add_one_fixed_kernel[grid](x, y, seq_len)  # BLOCK_SIZE 由 heuristics 计算
      return y
  ```
  **调优缓存与编译缓存需要分别处理**：缩小 autotune key 只减少 benchmark 次数。如果将每步变化的真实长度标为 `tl.constexpr`，即使命中调优缓存，JIT 仍可能为每个长度编译新版本。上述示例将真实长度保留为运行时参数，并用 `do_not_specialize` 避免按该标量的值或对齐特征进一步特化；它不禁止 `seq_bucket` 作为 autotune key。有限的 `BLOCK_SIZE: tl.constexpr` 仍会对应不同编译版本，可在上线前预热各桶或分派分支。`prune_configs_by` 只能减少每次搜索的候选数，无法消除大量精确长度带来的新 key。

### 第 6 章 CV 流水线优化：让 Cube 和 Vector 并行工作

> **核心理念**：NPU 的 Cube 和 Vector 是物理分离的两个执行单元——不优化则串行执行，流水线气泡严重。CV 流水线优化的核心是让 Cube 和 Vector 并行工作，消除气泡。
>
> **（1）Cube/Vector 分离架构与数据通路。** 910\_95 的每个 AI Core 由 1 Cube（矩阵乘单元）+ 2 Vector（向量计算单元）+ Scalar 组成。`tl.dot` 映射到 Cube 的 `mmad` 指令，数据路径是 `GM →(MTE2)→ L1 →(MTE1)→ L0A/L0B → Cube → L0C →(fixpipe)→ UB`。关键点：Cube 输出先写入 **L0C**（256KB，NZ 格式——一种分块压缩格式），需 **fixpipe** 硬件流水线搬运到 UB 并完成格式转换（NZ→ND，即分块压缩→常规行优先）后 Vector 才能处理。fixpipe 还支持**随路操作**（在搬运同时完成）：bias 累加、精度转换（fp32→fp16）、转置——可替代后续 Vector 操作，是融合后处理的关键硬件特性。A5（910\_95）的 fixpipe L0C→UB **直通**（零 GM 中转，单步完成）；A2（910B）的 L0C→**GM**→UB（额外两次 GM 访问，性能损失显著）。
>
> **（2）PIPE 模型与串行执行问题。** NPU 的硬件流水线用 PIPE 枚举标识：`PIPE_M`（Cube 计算流水线）、`PIPE_V`（Vector 计算流水线）、`PIPE_FIX`（fixpipe 流水线）、`PIPE_MTE2`（GM→UB 搬入）、`PIPE_MTE3`（UB→GM 搬出）。不优化时，Cube 计算（PIPE\_M）和 Vector 计算（PIPE\_V）串行执行——Vector 必须等 Cube 的 fixpipe（PIPE\_FIX）完成后才能开始，形成流水线气泡。
>
> **（3）CV 流水线优化的核心思想。** 通过**指令重排序**，将没有数据依赖的 Cube 操作（`tl.dot`）提前到 Vector 操作之前，使 Cube 和 Vector 并行执行：当 Cube 执行矩阵计算时，Vector 同时执行向量计算。优化后的执行时序：Cube 先启动 `tl.dot`，Vector 随后并行执行独立计算，最后 Vector 处理依赖 Cube 结果的操作。

- 6.1 指令重排序 + VF 融合控制：让 Cube 提前启动
  - **问题一**：无数据依赖的 `tl.dot` 排在 Vector 计算之后（原始代码顺序），Cube 等待 Vector 完成后才启动，整个 Vector 执行期间 Cube 空闲
  - **方案一**：将无数据依赖的 `tl.dot` 提前到 Vector 计算之前，使 Cube 与 Vector 并行。注意：依赖 Cube 结果的操作（如 `acc += pv`）必须延后
  - **问题二**：前驱 Vector 计算与 `tl.dot` 的 VF（Vector Function）融合链过长，导致 UB 溢出或阻止 dot 调度到 Cube（编译器将 `ds` 的计算融合到 dot 的 VF 链中，使整个链在 Vector 执行，Cube 无法独立调度）
  - **方案二**：在 `tl.dot` 之前插入 `compile_hint("break_vf")`，打断 VF 融合链，确保 `ds` 独立计算后 `tl.dot` 单独调度到 Cube
  ```python
  # ── 指令重排序：让 Cube 提前启动 ──
  # Before: Vector 先执行，Cube 等待（Flash Attention softmax 循环体）
  m_new = tl.maximum(m, tl.max(s, 1))
  alpha = tl.math.exp2((m - m_new) * qk_scale)  # Vector
  p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
  p_sum = tl.sum(p, 1)       # Vector
  acc *= alpha[:, None]      # Vector
  v = load_if(v_block_ptr, False, True)
  acc += tl.dot(p.to(dtype), v)  # Cube（等以上全部 Vector 完成后启动）

  # After: Cube 提前启动，与 Vector 并行
  m_new = tl.maximum(m, tl.max(s, 1))
  p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
  v = load_if(v_block_ptr, False, True)
  pv = tl.dot(p.to(dtype), v)          # Cube 提前启动（无依赖 acc）
  p_sum = tl.sum(p, 1)                  # Vector（与 Cube 并行）
  alpha = tl.math.exp2((m - m_new) * qk_scale)  # Vector（与 Cube 并行）
  acc = acc * alpha[:, None] + pv       # 合并：依赖 Cube 结果，放最后
  l = l * alpha + p_sum
  m = m_new

  # ── break_vf：打断 VF 融合链（Flash Attention 反向）──
  import triton.language.extra.cann.extension as extension

  # Before: ds 可能被融合到 dot 的 VF 链，阻止 Cube 调度
  ds = p * (dp - d[:, None])
  ds = tl.where(mask, ds, 0.0)
  ds = ds.to(dtype)
  dq += tl.dot(ds, k)  # ds 的 VF 链可能过长，UB 溢出或 Cube 无法独立调度

  # After: 打断 VF 链，ds 独立计算后 tl.dot 单独调度到 Cube
  ds = p * (dp - d[:, None])
  ds = tl.where(mask, ds, 0.0)
  ds = ds.to(dtype)
  extension.compile_hint(ds, "break_vf")  # 在此处打断 VF 融合
  dq += tl.dot(ds, k)  # tl.dot 独立调度到 Cube
  ```
  - **案例**：FA v4/v8 fwd — `tl.dot(p, v)` 提前到 `p_sum` / `alpha` 之前
  - **案例**：FA bwd\_q/bwd\_qkv — `ds` 计算后插入 `break_vf`，确保 `tl.dot` 调度到 Cube
- 6.2 CV 协同配置：子块绑定与完整七参数
  - **问题**：CV 融合算子需要 Cube 和 Vector 以 1:2 比例协同（匹配 1 Cube + 2 Vector 架构），但默认配置不启用协同——Cube 和 Vector 各自独立调度，无法实现流水线并行。此外，CV 融合算子需要一组完整的编译参数来控制 CV 协同行为
  - **方案一**：CV 融合算子启用 `enable_auto_bind_sub_block=True`（自动绑定子块，实现 CV 1:2 协同），纯 Vector 算子禁用（无 Cube 需要协同）
  - **方案二**：CV 融合算子配置完整的七参数集，精确控制 CV 协同、多缓冲、同步策略等
  ```python
  # ── 子块绑定决策 ──
  # CV 融合算子（含 tl.dot）：启用子块绑定，实现 CV 1:2 协同
  fwd_kernel[grid](..., enable_auto_bind_sub_block=True)
  # 纯 Vector 算子（无 tl.dot）：禁用子块绑定
  rmsnorm_kernel[grid](..., enable_auto_bind_sub_block=False)

  # ── CV 完整七参数配置（CV 融合算子专用）──
  fwd_kernel[grid](
      ...,
      enable_mixed_cv=True,                              # 启用 Cube/Vector 混合调度
      sync_solver=True,                                  # 自动求解 CV 间同步点
      enable_auto_bind_sub_block=True,                   # 自动绑定子块（CV 1:2 协同）
      set_workspace_multibuffer=2,                       # workspace 双缓冲
      limit_auto_multi_buffer_of_local_buffer="no-limit", # 不限制 UB 多缓冲分配
      enable_flatten=False,                              # CV 融合必须 False（不能展平）
      multibuffer=True,                                  # UB 双缓冲，存算并行
  )

  # ── 纯 Vector 算子配置（只需 2 个参数）──
  rmsnorm_kernel[grid](
      ...,
      multibuffer=True,
      limit_auto_multi_buffer_of_local_buffer="no-limit",
  )
  ```
  - **案例**：FA v8 全系列 — 完整七参数配置
  - **案例**：FA bwd\_q — 从 `enable_auto_bind_sub_block=False` 改为启用
- 6.3 Scale 融合：减少 FixPipe 后处理
  - **问题**：`tl.dot` 之后的 scale 乘法、累加等操作在 fixpipe 阶段或 Vector 单元执行，增加 Vector 单元压力。特别是 online softmax 中的 `alpha` 缩放 + PV matmul + 累加三步操作，每步都占用 Vector 周期
  - **方案一（Scale 融合到输入）**：将 dot 之后的 scale 乘法移到 dot 输入中，减少一次 Vector 乘法
  - **方案二（Accumulator-in-dot）**：将 alpha 缩放 + PV matmul + 累加融合为单条 `tl.dot` 三参数调用 `tl.dot(p, v, acc)`，利用 Cube 的硬件累加器
  - **方案三（k\_scale post-dot）**：将 k\_scale 从 pre-dot（dot 前对 k 乘 scale）移到 post-dot（dot 后与结果融合），消除额外 Vector 乘法
  ```python
  # ── Scale 融合到输入（SDPA 反向）──
  # Before: scale 在 dot 之后，额外一次 Vector 乘法
  dq += tl.dot(ds, k) * scale
  # After: scale 融入 ds（dot 输入），消除 dot 后乘法
  ds = ds * scale
  dq += tl.dot(ds, k)

  # ── Accumulator-in-dot（SDPA 前向 online softmax）──
  # Before: 缩放 + matmul + 累加（三步，三次 Vector/Cube 操作）
  block_o = alpha[:, None] * block_o + tl.dot(block_p, block_v)
  # After: 融合为单条 dot 三参数形式（Cube 硬件累加器完成缩放+累加）
  block_o = tl.dot(block_p, block_v, block_o * alpha[:, None])

  # ── k_scale post-dot（LightningIndexer）──
  # Before: pre-dot Vector 乘法（对 k 整个矩阵乘 scale）
  k = k * k_scale[:, None]
  qk = tl.dot(q, tl.trans(k))
  # After: scale 融合到后续 ReLU（只对 qk 结果乘，元素更少）
  qk = tl.dot(q, tl.trans(k))
  relu_qk = tl.maximum(qk * k_scale[None, :], 0.0)
  ```
  - **案例**：SDPA fwd — accumulator-in-dot 融合
  - **案例**：SDPA bwd — scale 融合到 matmul 输入
  - **案例**：LightningIndexer — k\_scale post-dot 融合

### 第 7 章 存算并行与软件流水线：让数据搬运与计算重叠

> **核心理念**：NPU 的数据搬运由专用 DMA 引擎执行，与计算单元物理分离。存算并行的核心是利用这一物理分离，让 MTE2（搬入）与 Vector/Cube 计算**自动重叠**。
>
> **（1）专用 DMA 引擎与物理分离。** GPU 上 `tl.load` / `tl.store` 通过 SM 内部的 LD/ST 队列执行，与计算共享 Warp 调度资源——需要 `cp.async` + `num_stages` 显式管理异步拷贝才能实现存算重叠，否则 load 和 compute 串行执行。NPU 则有**物理独立的专用 DMA 引擎**：MTE2（GM↔L1/UB，数据搬入引擎）和 MTE3（UB→GM/L1，数据搬出引擎），与 Vector/Cube 计算单元是不同的硬件 Pipeline（PIPE\_MTE2/PIPE\_MTE3 vs PIPE\_V/PIPE\_M）。这意味着 DMA 搬运和计算**天然并行**——当 MTE2 搬入下一块数据时，Vector 可以同时处理当前块数据，无需开发者显式管理异步操作。
>
> **（2）multibuffer 双缓冲机制。** `multibuffer=True` 启用 UB 双缓冲（Ping-Pong）：将 UB 分为两份，一份供当前块计算（Vector 读取），另一份供下一块搬入（MTE2 写入），两份交替使用。这使得下一块的 MTE2 搬入与当前块的 Vector 计算**自动重叠**——无需修改代码逻辑，只需开启参数。权衡：开启后 UB 可用空间减半（910\_95 从 248KB → 124KB，910B 从 192KB → 96KB），可能需要减小 BLOCK\_SIZE。910\_95 的 multibuffer **默认关闭**（需显式 `multibuffer=True`），910B 默认开启。
>
> **（3）核内 for 循环——存算并行的标准模式。** 持久化 kernel（Grid = 物理核数）的核内 `for` 循环天然形成软件流水线——上一轮 compute 与下一轮 load 自动重叠。这是因为循环内 `tl.load`（MTE2）和 `tl.dot`/计算（Cube/Vector）分属不同 PIPE，编译器自动调度它们并行执行。循环不变量外提（hoisting）和指针基址+offset 重算（避免 `+=`）直接影响 DMA 并行效率——循环内重复加载不变量或使用 `+=` 引入 RAW 依赖都会阻止 MTE2 并行发射。

- 7.1 核内循环 + 循环不变量外提：存算并行的基础模式
  - **问题**：标准 Grid 模式下每 program 一个块，无核内流水线深度（只做一次 load→compute→store，无法重叠）；循环内重复加载不变量（如 weight、col\_offsets）是纯浪费——每轮重新发起 MTE2 搬运相同数据
  - **方案一**：用核内 for 循环替代大 Grid，使上一轮 compute 与下一轮 load 重叠（持久化 kernel 模式）
  - **方案二**：将循环内不变的计算/加载提到循环外（hoisting），只加载一次，循环内直接复用
  ```python
  # ── 核内循环实现存算并行（RMSNorm 持久化 kernel）──
  # Before: 每 program 一个块，无核内流水线
  @triton.jit
  def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...):
      pid = tl.program_id(0)
      # 单行处理，无循环
      row = pid
      x = tl.load(x_ptr + row * stride + offs_d)
      ...  # compute
      tl.store(y_ptr + row * stride + offs_d, y)

  grid = (n_rows,)  # Grid 可能远超物理核数

  # After: 持久化 kernel，核内循环处理多行
  @triton.jit
  def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                     NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      # 核内循环：上一轮 compute 与下一轮 load 自动重叠
      for row in range(pid, n_rows, NUM_CORES):
          x = tl.load(x_ptr + row * stride + offs_d)  # 本轮 load (MTE2)
          # 上轮 compute (Vector) 与本轮 load 并行
          mean = tl.sum(x * x, axis=0) / D
          rstd = 1.0 / tl.sqrt(mean + eps)
          y = (x * rstd[:, None]) * w + b
          tl.store(y_ptr + row * stride + offs_d, y)

  grid = (NUM_CORES,)  # Grid = 物理核数

  # ── 循环不变量外提（RMSNorm weight 加载）──
  # Before: 每轮重复加载 weight（循环不变量）
  for row in range(pid, n_rows, NUM_CORES):
      w = tl.load(w_ptr + offs_d)  # weight 每轮重新加载（浪费）
      x = tl.load(x_ptr + row * stride + offs_d)
      y = (x * rstd[:, None]) * w + b
      tl.store(y_ptr + ..., y)

  # After: weight 循环外加载一次，循环内直接复用
  w = tl.load(w_ptr + offs_d)  # 循环外加载一次
  col_offsets = tl.arange(0, D_ROUNDED)  # 循环外计算一次
  for row in range(pid, n_rows, NUM_CORES):
      x = tl.load(x_ptr + row * stride + col_offsets)
      y = (x * rstd[:, None]) * w + b
      tl.store(y_ptr + ..., y)
  ```
  - **案例**：RMSNorm — 持久化 kernel + weight/col\_offsets 循环外加载
- 7.2 循环手动展开：双路并行增加流水线深度
  - **问题**：单路循环的流水线深度有限——每轮只有一个 load 和一个 compute，load/compute 重叠不充分（MTE2 搬入一块后 Vector 计算一块，MTE2 空闲等待 Vector 完成）
  - **方案**：每轮处理 2 个（或更多）数据块，双路 load/compute/store——路 0 的 compute 与路 1 的 load 重叠，MTE2 和 Vector 都满载
  ```python
  # ── Before: 单路循环，流水线深度有限 ──
  @triton.jit
  def rmsnorm_kernel(x_ptr, ..., NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      for row in range(pid, n_rows, NUM_CORES):
          x = tl.load(x_ptr + row * stride + offs_d)  # 单路 load
          mean = tl.sum(x * x, axis=0) / D             # 单路 compute
          y = (x * rstd[:, None]) * w
          tl.store(y_ptr + row * stride + offs_d, y)   # 单路 store

  # ── After: 双路展开，路 0 compute 与路 1 load 重叠 ──
  @triton.jit
  def rmsnorm_kernel(x_ptr, ..., NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      # 每轮处理 2 行，步长翻倍
      for row in range(pid * 2, n_rows, NUM_CORES * 2):
          # 路 0 和路 1 的 load 可并行发射（无依赖）
          x_0 = tl.load(x_ptr + (row) * stride + offs_d)      # 路 0 load
          x_1 = tl.load(x_ptr + (row + 1) * stride + offs_d)  # 路 1 load
          # 路 0 compute（与路 1 load 重叠）
          mean_0 = tl.sum(x_0 * x_0, axis=0) / D
          rstd_0 = 1.0 / tl.sqrt(mean_0 + eps)
          y_0 = (x_0 * rstd_0[:, None]) * w
          # 路 1 compute
          mean_1 = tl.sum(x_1 * x_1, axis=0) / D
          rstd_1 = 1.0 / tl.sqrt(mean_1 + eps)
          y_1 = (x_1 * rstd_1[:, None]) * w
          # 双路 store
          tl.store(y_ptr + (row) * stride + offs_d, y_0)
          tl.store(y_ptr + (row + 1) * stride + offs_d, y_1)
  ```
  - **案例**：RMSNorm — 每轮处理 2 行（`pid*2` 步长），双路 load/compute/store
- 7.3 multibuffer 启用：Ping-Pong 双缓冲
  - **问题**：当不启用 Ping-Pong 双缓冲时， load 和 compute 串行执行——MTE2 搬入完成后 Vector 才能开始计算，MTE2 在 Vector 计算期间空闲
  - **方案**：当前 multibuffer 已默认开启，或者通过设置 `multibuffer=True` 的方式手动启用，UB 双缓冲使下一 tile 的内存加载与当前 tile 的计算重叠。Ping-Pong 模式：buffer A 供当前 tile 计算，buffer B 供下一 tile 搬入，交替使用
  - **权衡**：开启后 UB 可用空间减半（248KB → 124KB），需评估对 BLOCK\_SIZE 的影响——可能需要减小 BLOCK\_SIZE 或减少核内同时存活的 tensor
  ```python
  # ── Before: multibuffer 默认关闭（910_95），load 和 compute 串行 ──
  rope_kernel[grid](
      in_ptr, pos_ptr, cu_seqlens, out_ptr,
      head, base, DIM, max_seq_len, REVERSE,
      # 无 multibuffer 参数，使用默认值 False
  )

  # ── After: 显式开启 multibuffer ──
  rope_kernel[grid](
      in_ptr, pos_ptr, cu_seqlens, out_ptr,
      head, base, DIM, max_seq_len, REVERSE,
      multibuffer=True,  # 显式开启 Ping-Pong 双缓冲
  )
  ```
  - **案例**：RoPE / Softcap / FA / RMSNorm — 显式开启 multibuffer
- 7.4 指针依赖消除：避免循环内 `+=` 引入 RAW 依赖
  - **问题**：循环内 `ptrs += BLOCK_SIZE` 引入 Read-After-Write（RAW）依赖——当前迭代的指针值依赖前一次迭代的写入结果，导致编译器无法重排指令，MTE2 load 指令必须等上一轮 `+=` 完成后才能发射，阻塞流水线
  - **方案**：改为每轮独立计算 offset（`base + k * BLOCK_SIZE`），消除迭代间 RAW 依赖——每轮的指针地址只依赖循环变量 `k` 和循环外常量 `base`，编译器可自由重排 load 指令
  ```python
  # ── Before: += 引入 RAW 依赖，阻塞 MTE2 并行发射 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, ...):
      # 循环外计算基址
      a_ptrs = a_ptr + (offset_m[:, None] * K + offset_k[None, :])
      b_ptrs = b_ptr + (offset_k[:, None] * N + offset_n[None, :])
      for k in range(0, tl.cdiv(K, BLOCK_K)):
          a = tl.load(a_ptrs)  # load 必须等上一轮 a_ptrs += 完成后才能发射
          b = tl.load(b_ptrs)
          acc += tl.dot(a, b)
          a_ptrs += BLOCK_K       # RAW：下轮 load 等上轮完成
          b_ptrs += BLOCK_K * N   # RAW：下轮 load 等上轮完成

  # ── After: 每轮独立计算 offset，无 RAW 依赖 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, ...):
      # 循环外计算基址（不变量）
      a_ptrs_base = a_ptr + (offset_m[:, None] * K + offset_k[None, :])
      b_ptrs_base = b_ptr + (offset_k[:, None] * N + offset_n[None, :])
      for k in range(0, tl.cdiv(K, BLOCK_K)):
          # 每轮独立计算地址：base + k * stride（无迭代间依赖）
          a_ptrs = a_ptrs_base + k * BLOCK_K
          b_ptrs = b_ptrs_base + k * BLOCK_K * N
          a = tl.load(a_ptrs)  # 无 RAW，可提前发射
          b = tl.load(b_ptrs)
          acc += tl.dot(a, b)
  ```
  - **案例**：Fused MatMul fwd — `a_ptrs_base + k * BLOCK_SIZE_K` 替代循环内 `+=`
  - **案例**：RMSNorm — 行方向 offset 每轮独立计算
- 7.5 高级流水线：parallel 循环 + Preload/Cool-down
  - **问题一**：无携带依赖的循环（循环内各轮独立）未被编译器识别为可并行，多个 GM load 串行发射——编译器保守地按顺序发射各轮 load，即使它们之间无数据依赖
  - **方案一**：用 CANN extension 的 `parallel` 标记无携带依赖的循环，提示编译器并行发射各轮 load
  - **问题二**：逐块串行发射 Cube 任务，流水线深度不足——每轮只发射一个 `tl.dot`，Cube 单元在 fixpipe 期间空闲
  - **方案二**：外层循环多发射 `preload` 个 Cube 任务（预加载多块数据），最后 `preload` 次循环处理 cool-down（等待 Cube 结果回写）
  ```python
  # ── parallel 循环：并行化多个无依赖 GM load ──
  import triton.language.extra.cann.extension as extension

  # Before: 普通循环，编译器串行发射各轮 load
  for page_iter in range(PAGE_AGGREGATION_NUM):
      k_slice = tl.load(k_ptr + page_iter * stride)  # 串行 load

  # After: 标记为可并行，编译器并行发射各轮 load
  for page_iter in extension.parallel(0, PAGE_AGGREGATION_NUM):
      k_slice = tl.load(k_ptr + page_iter * stride)  # 并行 load

  # ── Preload + Cool-down：多发射 Cube 任务 ──
  preload = 3  # 预加载 3 个块
  # 外层循环多发射 preload 个 Cube 任务
  end_block = end_block + preload  # 扩展循环范围，多发射 3 个块
  for sq_loop in range(start, end_block, step):
      if sq_loop < last_loop:
          # 正常路径：发射 Cube 任务 + preload 下一个块的数据
          qk = tl.dot(q, k)  # Cube 任务
          k_next = tl.load(...)  # 预加载下一块
      else:
          # cool-down：最后 preload 次循环，等待 Cube 结果回写
          # 不发射新 Cube 任务，只处理已发射的结果
          acc = acc + qk_result
  ```
  - **案例**：FA/SWA page aggregation — parallel 循环并行化多个 GM load
  - **案例**：fa-opt/fast\_fa\_fwd — preload=3 多发射 + cool-down

### 第 8 章 算子融合：消除不必要的内存往返

> **核心理念**：Memory-Bound 算子的性能瓶颈在于 GM 访问带宽。算子融合通过将多个操作合并到单个 kernel 中，让中间数据保留在寄存器/UB 中，消除 GM 写回与重新加载。
>
> **（1）GM 往返的代价。** 未融合的算子链中，每个操作的输出都要写回 GM（Global Memory，HBM），下一个操作再从 GM 读回——每次 GM 往返消耗数百到数千个时钟周期（GM 带宽约 1-2 TB/s，而 UB 带宽约 30-50 TB/s，差距 20-30 倍）。以 `RMSNorm(x) * gated` 为例，未融合时：kernel 1 计算 `gated = silu(x @ w_gate)` → 写回 GM → kernel 2 读回 `gated` → 计算 `rmsnorm(gated)` → 写回 GM。两次 GM 往返消耗大量带宽。融合后，`gated` 全程保留在寄存器/UB 中，GM 访问次数从 4 次（2 读 + 2 写）降为 2 次（1 读 + 1 写）。
>
> **（2）融合的层次。** NPU 上的算子融合从低到高有四个层次：**逐元素融合**（最简单，如 `x * sigmoid(x)` 融合为 silu，在同一 Vector 指令流中完成）→ **多 kernel 融合**（消除 kernel 启动开销和中间 GM 访问，如反向 QKV 三合一）→ **Pass 融合**（消除 scratch 中间存储，如 Two-Pass → Single-Pass，中间数据全程在寄存器）→ **计算单元升级融合**（Vector → Cube，如逐 Q head 的 Vector 乘改为按 KV head 分组的 Cube 矩阵乘）。
>
> **（3）NPU 特有的融合手段。** 除了通用融合手段，NPU 还有两种独特融合方式：**3D Block Pointer 融合**——用 3D block\_ptr `(BLOCK, Q_HEAD_NUM, V_DIM)` 一次加载所有 head 的数据，利用硬件 DMA 描述符的多维搬运能力，消除逐 head 串行加载；**布局一致性融合**——统一数据布局（如全程 D-T 布局），消除 kernel 内的 `tl.trans` 转置操作（每次转置需要一次 UB 内数据搬运 + 格式转换，开销显著）。

- 8.1 多 Kernel 融合：消除启动开销与中间 GM 访问
  - **问题**：多个独立 kernel 串行执行，每个 kernel 的输出写回 GM，下一个 kernel 再读回；更严重的是，前端计算（如反向传播中的 Q\@K^T 和 softmax 概率矩阵 P）在多个 kernel 中重复执行——bwd\_q kernel 计算 P，bwd\_kv kernel 又计算一遍 P
  - **方案**：将多个独立 kernel 融合为单 kernel，前端计算只执行一遍，中间数据全程在寄存器/UB
  - **三种典型场景**：
    - 反向 QKV 三合一：bwd\_q + bwd\_kv → bwd\_qkv（dq 用 atomic\_add 或 per-core workspace 处理多核写冲突）
    - 逐元素操作融合：GELU(x) + GELU(x)\*y → 单 kernel（消除中间 GELU(x) 的 GM 往返）
    - softmax + 累积采样：softmax + multinomial → 单 kernel（消除概率矩阵的 GM 往返）
  ```python
  # ── 反向 QKV 三合一（Flash Attention 反向）──
  # Before: 两个独立 kernel，前端计算（P 矩阵）执行两遍
  # kernel 1: bwd_q → 计算 P = softmax(Q@K^T) → dq = P @ (do - sum) @ K
  bwd_q_kernel[grid](q, k, v, do, dq, ...)  # P 在此计算
  # kernel 2: bwd_kv → 重新计算 P → dk = P^T @ do @ Q, dv = P^T @ do @ V
  bwd_kv_kernel[grid](q, k, v, do, dk, dv, ...)  # P 又计算一遍

  # After: 单 kernel，前端计算只执行一遍
  # bwd_qkv → dq, dk, dv 同时输出
  # P 只计算一次，dq/dk/dv 共享 P 矩阵
  bwd_qkv_kernel[grid](
      q, k, v, do,
      dq, dk, dv,  # 三合一输出
      ...,
      # dq 用 atomic_add 处理多核写冲突（多个 program 写同一 dq 地址）
  )
  # autotune 需配 reset_to_zero=["dq_ptr"] 避免多次评估累加
  ```
  - **案例**：FA v8 bwd — `fused bwd_qkv_kernel`，dq 用 atomic\_add
  - **案例**：GELU+Mul — 两个逐元素操作融合为单 kernel
  - **案例**：Sample — `_top_k_sample_kernel` 融合 softmax + multinomial
- 8.2 Two-Pass → Single-Pass 融合：消除 scratch 中间存储
  - **问题**：两趟计算中，Pass 1 的中间结果写回 GM（scratch buffer），Pass 2 再读回继续计算——两次 GM 访问（一次写 + 一次读），且 scratch buffer 占用额外显存。以 Gate RMSNorm 为例：Pass 1 计算 `gated = silu(x @ w_gate)` → 写回 GM scratch → Pass 2 读回 scratch → 计算 `rmsnorm(gated)`
  - **方案**：融合为单趟，中间数据（gated 值）全程在寄存器/UB，消除 scratch GM 往返和额外显存占用
  ```python
  # ── Before: 两趟，gated 写回 GM scratch ──
  # Pass 1: compute gated → write to scratch (GM)
  gated = silu(x @ w_gate)  # 前端计算
  torch.empty_like(gated).copy_(gated)  # 写回 GM scratch

  # Pass 2: read scratch → compute RMSNorm
  gated_from_gm = load_from_scratch()  # 从 GM 读回
  mean = tl.sum(gated_from_gm ** 2, axis=0) / D
  y = gated_from_gm * rsqrt(mean + eps) * weight

  # ── After: 单趟，gated 全程在寄存器/UB ──
  # gated 值不写回 GM，直接在寄存器中传给 RMSNorm 计算
  gated = silu(x @ w_gate)  # 前端计算，结果在寄存器
  # 直接对寄存器中的 gated 做 RMSNorm（无需 GM 往返）
  mean = tl.sum(gated ** 2, axis=0) / D  # gated 在寄存器，零 GM 访问
  y = gated * rsqrt(mean + eps) * weight
  ```
  - **案例**：Gate RMSNorm — 两趟融合为单趟，gated 值全程保留在寄存器
  - **案例**：RMSNorm 单 pass — 配合 `BLOCK_DIN=Din`（移除 next\_power\_of\_2）使归约循环只执行一次
- 8.3 维度融合：多 Head 一次加载 + 头循环展平
  - **问题一**：逐 head 加载数据，核内 head 循环串行执行——每个 head 独立发起 GM load，MTE2 搬运串行，且 head 循环的循环开销（索引计算、边界检查）重复执行
  - **方案一（3D Block Pointer）**：用 3D block\_ptr `(BLOCK, Q_HEAD_NUM, V_DIM)` 一次加载所有 head，利用硬件 DMA 描述符的多维搬运能力，单次 MTE2 完成所有 head 数据搬入
  - **问题二**：核内头循环串行处理各 KV head，无法利用核间并行
  - **方案二（头展平到 Grid）**：将核内头循环展平到 Grid 维度，`h = pid % num_kv_heads`，让不同核并行处理不同 head
  ```python
  # ── 3D Block Pointer：多 Head 一次加载（RoPE）──
  # Before: 逐 head 加载，核内 head 循环串行
  for head in range(num_heads):
      x = tl.load(make_block_ptr(
          base=in_ptr + begin * head * DIM + head * DIM,
          shape=(len, DIM),          # 2D：单 head
          strides=(head * DIM, 1),
          block_shape=(BLOCK_M, DIM // 2),
      ))
      # ... 旋转计算 ...

  # After: 3D block_ptr 一次加载所有 head
  x = tl.load(make_block_ptr(
      base=in_ptr + begin * head * DIM,
      shape=(len, head, DIM),        # 3D：所有 head
      strides=(head * DIM, DIM, 1),  # 三维 stride
      block_shape=(BLOCK_M, head, DIM // 2),  # 一次加载所有 head
      order=(2, 1, 0),               # 最内维 DIM 连续
  ))
  # ... 旋转计算（3D 广播）...

  # ── 头展平到 Grid：利用核间并行（KV Cache）──
  # Before: 核内串行处理各 head
  grid = (num_chunks,)
  for h in range(num_kv_heads):  # 核内串行
      process(h)

  # After: Grid 维度展开，各核并行处理不同 head
  grid = (num_chunks * num_kv_heads,)
  h = pid % num_kv_heads  # 从 pid 反推 head 索引
  process(h)
  ```
  - **案例**：FA v8 bwd\_preprocess — 3D block\_ptr 融合所有 head 的归约
  - **案例**：KV Cache — 头维度展平到 Grid
- 8.4 计算单元升级融合：Vector → Cube
  - **问题**：GQA decode 场景下（GROUP\_SIZE 个 Q head 共享一组 KV），逐 Q head 的 Vector 逐元素乘 + 归约效率低——每个 Q head 独立做 `qk = tl.sum(q[None, :] * k, axis=...)`，Vector 单元逐元素乘 + 归约，Cube 单元空闲
  - **方案**：将逐 Q head 的 Vector 乘法改为按 KV head 分组的 Cube 矩阵乘，GROUP\_SIZE 个 Q head 融合为一个矩阵乘的 M 维度——Cube 单元的 `mmad` 指令一次完成 GROUP\_SIZE 个 Q head 与 KV 的点积
  ```python
  # ── Before: Vector 逐元素乘 + 归约（GQA decode）──
  # 逐 Q head 处理，Vector 单元做逐元素乘 + sum 归约
  for q_head in range(GROUP_SIZE):
      q = tl.load(q_ptr + q_head * stride)  # 单 Q head
      # Vector 逐元素乘 + 归约（Cube 空闲）
      qk = tl.sum(q[None, :] * k, axis=head_dim)  # Vector 乘 + sum
      # ... 后续 softmax ...

  # ── After: Cube 矩阵乘，GROUP_SIZE 个 Q head 融合 ──
  # 按 KV head 分组，GROUP_SIZE 个 Q head 融合为矩阵乘的 M 维
  q_group = tl.load(q_ptr + kv_head * GROUP_SIZE * stride)  # GROUP_SIZE 个 Q head
  # Cube 矩阵乘：一次完成 GROUP_SIZE 个 Q head 与 K 的点积
  qk = tl.dot(q_group, tl.trans(k))  # Cube mmad，任务数减少 GROUP_SIZE 倍
  # ... 后续 softmax ...
  ```
  - **案例**：FA/SWA decode — GQA group-fusion + Cube grid
- 8.5 Mask 预计算融合 + Group 融合 + 布局一致性
  - **问题一**：运行时 mask\_fn 逐元素计算 mask 开销大——每个 program 在 kernel 内动态计算 causal mask、local mask、attention arg mask 的组合，占用大量 Vector 周期
  - **方案一**：预计算 mask\_tensor（主机端 CPU 计算），kernel 内直接 load 预计算结果，将 Vector 计算转为 MTE2 搬入
  - **问题二**：逐 group 循环调用 RMSNorm 有 kernel 启动开销——每个 group 独立启动一个 kernel，N 个 group 有 N 次启动开销
  - **方案二**：单 kernel 融合所有 group，通过 `head_to_group` 查找表 + `tl.where` 权重选择，在核内循环中处理不同 group 的不同 weight
  - **问题三**：kernel 内 `tl.trans` 转置操作昂贵——每次转置需要一次 UB 内数据搬运 + 格式转换
  - **方案三**：统一数据布局，全程 D-T 布局一致（输入、权重、输出都用同一布局），消除所有转置
  ```python
  # ── Mask 预计算（Flash Attention）──
  # Before: 运行时逐元素计算 mask（kernel 内 Vector 计算）
  @triton.jit
  def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset):
      tril_causal = q_offset[:, None] >= k_offset[None, :]
      attn_args_mask = (q_attn_arg == k_attn_arg) | (k_attn_arg == 0)
      return (tril_causal & attn_args_mask) | (q_offset == k_offset)

  mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n)  # Vector 计算

  # After: 直接加载预计算 mask（kernel 内 MTE2 搬入）
  # 主机端预计算 mask_tensor [batch, max_q_len, max_k_len]
  mask = tl.load(mask_block_ptr)  # MTE2 搬入，零 Vector 计算

  # ── Group RMSNorm 融合 ──
  # Before: 逐 group 循环，N 个 kernel 启动
  for group in range(num_groups):
      rmsnorm_kernel[grid](x[group], weight[group])  # 每次启动一个 kernel

  # After: 单 kernel 融合所有 group
  @triton.jit
  def group_rmsnorm_kernel(x_ptr, weight_ptr, head_to_group_ptr, ...):
      pid = tl.program_id(0)
      head = pid  # 每个 program 处理一个 head
      group = tl.load(head_to_group_ptr + head)  # 查找 head 属于哪个 group
      # 用 tl.where 选择对应 group 的 weight
      w = tl.load(weight_ptr + group * stride + offs_d)
      ...  # RMSNorm 计算

  # ── Conv1D 布局一致性 ──
  # Before: 加载后转置 × 3 次（输入、权重、输出各转置一次）
  x = tl.trans(tl.load(x_ptr + ...))   # 转置 1：输入
  w = tl.trans(tl.load(w_ptr + ...))   # 转置 2：权重
  result = tl.dot(x, w)
  out = tl.trans(result)                # 转置 3：输出
  tl.store(out_ptr + ..., out)

  # After: 全程 D-T 布局一致，无转置
  x = tl.load(x_ptr + ...)   # D-T 布局加载，不转置
  w = tl.load(w_ptr + ...)   # D-T 布局加载，不转置
  result = tl.dot(x, w)      # D-T 布局矩阵乘
  tl.store(out_ptr + ..., result)  # D-T 布局存储，不转置
  ```
  - **案例**：FA v5/v8 bwd — mask\_tensor 替代 mask\_fn
  - **案例**：GroupRMSNorm — `_group_rmsnorm_interleaved_kernel`
  - **案例**：Conv1D — 4 次 trans → 全程布局一致

### 第 9 章 Mask 计算优化：最小化边界处理开销

> **核心理念**：Mask 操作是纯开销——不产生有用的计算结果，只为保证边界正确性。Mask 优化的目标是系统性地消除不必要的 mask 计算，将必要的 mask 开销降到最低。
>
> **（1）Mask 的 NPU 代价。** 每条 mask 指令占用 Vector 执行周期——mask 的布尔比较（如 `offs < N`）、布尔运算（如 `mask1 & mask2`）、类型转换（如 `mask.to(float)`）都在 Vector 单元执行，与有用计算竞争 Vector 资源。更隐蔽的代价：乘法 mask（`(1 - mask) * -inf`）需要类型转换 + 乘法 + 减法三条指令，而 `tl.where(mask, value, -inf)` 只需一条选择指令。此外，mask 的布尔运算结果在 NPU 上是 int1 类型，参与浮点运算时需要显式类型转换，否则可能触发隐式转换的不确定行为。
>
> **（2）三个优化层次。** Mask 优化有三个层次，效率从低到高：**实现方式优化**——将乘法 mask 改为 `tl.where`，将多 mask 组合改为预计算单 mask，减少指令数量；**预计算与缓存**——将运行时 mask 计算移到主机端 CPU（kernel 外），kernel 内只需一次 MTE2 load 搬入预计算结果，将 Vector 计算转为 DMA 搬入；**冗余消除**——利用对齐信息和 constexpr 编译期常量，完全消除 mask 代码（死代码消除），这是最高效的——完全消除 mask 代码而非优化其执行。
>
> **（3）冗余 mask 的常见场景。** 当 `BLOCK_SIZE = HEAD_DIM` 时，尾维 mask（`offs_k < HEAD_DIM`）恒为 True，是死代码；当 Grid 已保证 pid 不越界时，`% M` 取模运算（防止 pid 越界）是冗余的；当 mask 已覆盖边界或 BLOCK 整除维度时，额外的边界检查 mask 是重复的。这些冗余 mask 可以通过 `tl.constexpr` 编译期常量控制，让编译器直接消除整段代码。

- 9.1 Mask 实现方式优化：tl.where 替代乘法 mask
  - **问题**：`(1 - mask) * -inf` 的乘法 mask 需要三条指令：`mask.to(float)`（bool→float 类型转换）+ `1.0 - mask_float`（减法）+ `result * 1e6`（乘法）。三条 Vector 指令只为实现一个条件选择
  - **方案**：`tl.where(mask, value, -inf)` 只需一条 Vector 选择指令——硬件直接按 mask 条件选择 value 或 -inf，无需类型转换和乘法
  ```python
  # ── Before: 乘法 mask（3 条指令）──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      # ... 计算 qk 分数 ...
      mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < K_LEN)
      # 乘法 mask：类型转换 + 减法 + 乘法（3 条 Vector 指令）
      block_s = qk - (1.0 - mask.to(tl.float32)) * 1e6  # 越界位置减去大数

  # ── After: where mask（1 条指令）──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      # ... 计算 qk 分数 ...
      mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < K_LEN)
      # where mask：硬件条件选择（1 条 Vector 指令）
      block_s = tl.where(mask, qk, -1e6)  # 越界位置直接置为 -inf
  ```
- 9.2 预计算 Mask + 全局缓存：从运行时计算到直接加载
  - **问题**：多个 mask load + 布尔运算在 kernel 内逐元素计算，开销大。以 SWA 为例：causal mask + local window mask + attention arg mask + query mask，4 次 mask load + 3 次布尔运算（`&` 和 `|` 组合），每次都占用 Vector 周期。更严重的是，这些 mask 在每次 forward 时都重新计算，即使 mask 参数（window\_size 等）没有变化
  - **方案一**：预计算单 mask 替代多 mask 组合——主机端将所有 mask 组合为一个预计算 mask\_tensor，kernel 内只需 1 次 MTE2 load
  - **方案二**：mask 参数不变时缓存预计算 mask 张量——按 mask 参数（如 window\_size）作为缓存键，参数不变时直接返回缓存的 mask\_tensor，避免重复预计算
  ```python
  # ── 预计算单 mask（SWA prefill）──
  # Before: 4 次 load + 3 次布尔运算（kernel 内逐元素计算）
  @triton.jit
  def swa_kernel(q_ptr, k_ptr, ...):
      # 4 次 mask load
      tril_mask = gen_mask_tril(offs_m, offs_n)       # causal 下三角
      triu_mask = gen_mask_triu(offs_m, offs_n)       # attention arg 上三角
      bound_mask = gen_mask_bound(offs_n, K_LEN)      # 边界 mask
      q_mask = gen_mask_q(offs_m, Q_LEN)              # query 边界
      # 3 次布尔运算组合
      mask = tril_mask & (triu_mask | bound_mask) & q_mask  # Vector 计算

  # After: 1 次 load（主机端预计算单 mask）
  @triton.jit
  def swa_kernel(q_ptr, k_ptr, mask_ptr, ...):
      # 1 次 MTE2 load，零 Vector 计算
      mask = tl.load(mask_ptr + offsets)  # 预计算 mask 直接加载

  # ── 全局缓存（SWA prefill）──
  # Before: 每次调用重算预计算 mask
  def swa_forward(q, k, window_size, ...):
      mask = precompute_mask(window_size, ...)  # 每次 forward 都重算
      swa_kernel[grid](q, k, mask, ...)

  # After: mask 参数不变时缓存命中
  _GLOBAL_WINDOW_SIZE = None
  _CACHED_MASK = None

  def swa_forward(q, k, window_size, ...):
      global _GLOBAL_WINDOW_SIZE, _CACHED_MASK
      if _GLOBAL_WINDOW_SIZE == window_size:
          mask = _CACHED_MASK  # 缓存命中，零计算
      else:
          mask = precompute_mask(window_size, ...)
          _CACHED_MASK = mask
          _GLOBAL_WINDOW_SIZE = window_size
      swa_kernel[grid](q, k, mask, ...)
  ```
  - **案例**：SWA prefill — `gen_mask_causal_with_window` JIT + `_GLOBAL_WINDOW_SIZE` 缓存
- 9.3 冗余 Mask 消除：利用对齐信息删除死代码
  - **问题**：当 BLOCK\_SIZE = HEAD\_DIM、BLOCK 整除维度、或 mask 已覆盖边界时，额外 mask 检查是死代码——条件恒为 True，但运行时仍会生成 mask 指令并执行
  - **四种消除场景**：
    - 尾维 mask：当 `BLOCK_SIZE = HEAD_DIM` 时，`offs_k < HEAD_DIM` 恒为 True（因为 `offs_k = arange(0, HEAD_DIM)`），移除 `& (offs_k < HEAD_DIM)`
    - masked\_out\_rows 安全保护：非首块路径中 m\_i 已有有效值（上一轮的 m\_new），不存在全 -inf 行，`masked_out_rows` 检查是冗余的
    - KV\_LEN / IS\_FULL\_BLOCKS 条件 mask：当 mask 已覆盖边界或 BLOCK 整除维度时，额外的 KV\_LEN 边界检查重复
    - `% M / % N` 取模运算：Grid 已保证 pid 不越界（`pid < NUM_BLOCKS_M`），取模 `% M` 是冗余的，且 `%` 运算在 NPU 上导致标量退化（性能下降 5-10 倍）
  ```python
  # ── 尾维 mask 消除（FlexAttention）──
  # Before: 双维检查（即使 BLOCK_K == HEAD_DIM，尾维检查仍执行）
  mask = (offs_m[:, None] < Q_LEN) & (offs_k[None, :] < HEAD_DIM)
  # 当 BLOCK_K == HEAD_DIM 时，offs_k = arange(0, HEAD_DIM) < HEAD_DIM 恒为 True

  # After: HEAD_DIM = BLOCK_SIZE，尾维检查冗余，移除
  mask = (offs_m[:, None] < Q_LEN)  # 只保留 M 维边界检查

  # ── masked_out_rows 消除（FlexAttention online softmax）──
  # Before: 安全保护（3 条指令），防止全 -inf 行导致 NaN
  masked_out_rows = (m_ij == float('-inf'))  # 检查是否全 -inf
  m_ij_masked = tl.where(masked_out_rows, 0, m_ij)  # 全 -inf 时置 0
  alpha = tl.exp(m_i - m_ij_masked)  # 安全的 exp

  # After: 非首块路径中 m_i 已有有效值，直接使用（消除 3 条指令）
  alpha = tl.exp(m_i - m_ij)  # 无需安全保护

  # ── 取模运算消除（SwiGLU fwd）──
  # Before: 取模防止越界（且 % 导致标量退化）
  offset = (pid * BLOCK + tl.arange(0, BLOCK)) % M  # % 运算，标量退化

  # After: Grid 已保证 pid 不越界，取模冗余
  offset = pid * BLOCK + tl.arange(0, BLOCK)  # 连续地址，向量化
  ```
  - **案例**：FlexAttention — 移除尾维 mask、masked\_out\_rows、KV\_LEN 边界保护（6 处）
  - **案例**：SwiGLU fwd — offset 移除取模
- 9.4 编译期 Mask 消除：constexpr 控制代码生成
  - **问题**：当列数/token 数对齐时（如 `n_cols % BLOCK_SIZE == 0`），mask 代码是死代码（条件恒为 True），但运行时条件判断仍会生成 mask 指令——编译器无法在运行时判断条件是否恒为 True，保守地生成 mask 代码
  - **方案**：用编译期常量（`tl.constexpr`）控制是否生成 mask 代码——将"是否对齐"的判断从运行时提升到编译时，编译器在编译期直接消除整段死代码
  ```python
  # ── Before: 总是生成 mask（即使列已对齐）──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_cols, ...,
                      BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      # 总是生成 mask，即使 n_cols % BLOCK_SIZE == 0（mask 恒为 True）
      mask = (offs < n_cols)
      x = tl.load(x_ptr + offs, mask=mask, other=0.0)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offs, y, mask=mask)

  # ── After: constexpr 控制，编译期消除 mask ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_cols, ...,
                      BLOCK_SIZE: tl.constexpr,
                      COL_ALIGNED: tl.constexpr):  # 编译期常量
      pid = tl.program_id(0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      if not COL_ALIGNED:
          # 仅不对齐时生成 mask 代码（编译期决定）
          mask = (offs < n_cols)
          x = tl.load(x_ptr + offs, mask=mask, other=0.0)
          y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
          tl.store(y_ptr + offs, y, mask=mask)
      else:
          # 对齐时无 mask（编译器消除 if not 分支）
          x = tl.load(x_ptr + offs)  # 无 mask
          y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
          tl.store(y_ptr + offs, y)  # 无 mask

  # 调用处：编译期传入 COL_ALIGNED
  COL_ALIGNED = (n_cols % BLOCK_SIZE == 0)  # 编译期计算
  gelu_fwd_kernel[grid](x, y, n_cols, COL_ALIGNED=COL_ALIGNED, ...)
  ```
  - **案例**：GELU — `COL_ALIGNED` constexpr 消除 mask
  - **案例**：VisionRoPE — `TOKEN_ALIGNED` constexpr 消除 mask

### 第 10 章 内存访问与数据通路：优化数据搬运

> **核心理念**：NPU 的数据搬运由专用 DMA 引擎执行，与计算单元物理分离。内存访问优化的核心是：减少搬运次数、提升搬运效率、利用硬件数据通路特性。
>
> **（1）连续地址流依赖。** NPU 的 DMA 引擎（MTE2/MTE3）依赖**连续地址流**——DMA 描述符以固定 stride 发起连续搬运，离散访存（stride 最低维 ≠ 1，即非连续地址）性能急剧下降。这与 GPU 的 Warp 合并访问（coalesced access）模式不同：GPU 上 32 线程的 Warp 可以合并访问分散地址（只要地址在同一 cache line），NPU 的 DMA 引擎则需要严格的连续地址。离散访存时 DMA 引擎退化为逐元素搬运，带宽利用率可能降低数倍。
>
> **（2）Block Pointer 的硬件优势。** `tl.make_block_ptr` 利用硬件 DMA 描述符自动处理边界——DMA 引擎根据 block\_ptr 的 shape/strides/block\_shape 自动计算搬运计划，越界位置自动填充 0（无需手动 mask）。这比手动指针算术 + mask 更高效：手动方式需要 Vector 单元计算 offset 和 mask（占用 Vector 周期），block\_ptr 方式由 DMA 描述符硬件完成（零 Vector 开销）。但注意：NPU 上 block\_ptr 的 **stride 必须反映真实内存布局**，不能通过交换 stride 实现转置（GPU 上可以），转置语义只能通过 **order 参数**表达。
>
> **（3）`%`** **取模的标量退化。** NPU 上 `%` 取模运算通过标量 ALU 执行，导致向量运算退化为逐元素标量计算——地址不连续，DMA 引擎无法发起连续搬运，性能可能下降 **5-10 倍**。应尽量用 mask 替代取模：mask 是向量化的比较操作（保持连续地址），取模是标量运算（破坏连续地址）。
>
> **（4）CANN extension 切片指令。** CANN extension 的 `extract_slice` / `insert_slice` 直接映射 NPU DMA 切片指令，无通用版本的范围检查开销——通用 Triton 的 slice 操作在运行时检查 offset 和 shape 是否越界（占用 Vector 周期），CANN 扩展版本跳过检查直接发起 DMA 切片。
>
> **（5）care\_padding 依赖跳过。** `care_padding=False` 跳过 padding 区域的依赖检查——默认情况下，编译器检查 load 的 padding 区域（BLOCK\_SIZE > 实际维度的部分）是否与其他操作有依赖，阻止 load 指令并行发射。设置 `care_padding=False` 后，编译器跳过此检查（通过 mask 保证正确性），允许 load 指令提前并行发射。

- 10.1 预计算缓存：从运行时计算到直接加载
  - **问题**：kernel 内的 sin/cos 等超越函数计算、每次 forward 的权重转置都是昂贵的重复计算。以 RoPE 为例：每个 program 在 kernel 内计算 `freqs = pos * inv_freq; sin = tl.sin(freqs); cos = tl.cos(freqs)`，`tl.sin`/`tl.cos` 是昂贵的超越函数（多条 Vector 指令），且每次 forward 都重复计算相同的 sin/cos 值
  - **方案一（Sin/Cos 预计算）**：CPU 预计算 sin/cos 表后传入 kernel 直接 load，将超越函数计算转为 MTE2 搬入
  - **方案二（权重转置缓存）**：按 `data_ptr()` 键缓存转置结果，消除重复 O(N·K) 转置——每次 forward 都做 `weight.t().contiguous()` 是 O(N·K) 的数据搬运，但权重内容不变时转置结果可缓存复用
  ```python
  # ── Sin/Cos 预计算（RoPE）──
  # Before: kernel 内计算超越函数（多条 Vector 指令）
  @triton.jit
  def rope_kernel(in_ptr, pos_ptr, out_ptr, ...):
      pos = tl.load(pos_ptr + offsets)
      freqs = pos[:, None] * inv_freq[None, :]  # 计算频率
      sin = tl.sin(freqs)  # 昂贵的超越函数（多条 Vector 指令）
      cos = tl.cos(freqs)  # 昂贵的超越函数
      x1 = x0 * cos - y0 * sin  # 旋转
      ...

  # After: 直接 load 预计算结果（1 次 MTE2 load）
  @triton.jit
  def rope_kernel(in_ptr, pos_ptr, sin_cache_ptr, cos_cache_ptr, out_ptr, ...):
      # 直接加载预计算 sin/cos（零超越函数计算）
      sin = tl.load(sin_cache_ptr + offsets)  # MTE2 搬入
      cos = tl.load(cos_cache_ptr + offsets)  # MTE2 搬入
      x1 = x0 * cos - y0 * sin  # 旋转
      ...

  # 主机端预计算
  def rope_forward(x, position, ...):
      sin_cache, cos_cache = precompute_sin_cos(position, base, DIM)  # CPU 计算
      rope_kernel[grid](x, position, sin_cache, cos_cache, out, ...)

  # ── 权重转置缓存（GEMM）──
  # Before: 每次 forward 重算转置（O(N·K) 数据搬运）
  def gemm_forward(x, weight, ...):
      weight_t = weight.t().contiguous()  # 每次 forward 都转置
      gemm_kernel[grid](x, weight_t, ...)

  # After: 按 data_ptr() 缓存，权重不变时复用
  _bt_cache = {}  # 全局缓存字典

  def gemm_forward(x, weight, ...):
      key = weight.data_ptr()  # 用张量内存地址作为缓存键
      if key not in _bt_cache:
          _bt_cache[key] = weight.t().contiguous()  # 首次计算并缓存
      weight_t = _bt_cache[key]  # 后续直接复用
      gemm_kernel[grid](x, weight_t, ...)
  ```
  - **案例**：RoPE — sin/cos 预计算缓存
  - **案例**：GEMM — `_bt_cache` 消除重复转置
- 10.2 数据聚合与 CANN 切片：提升 Cube 利用率与切片效率
  - **问题一**：小 page 场景下（PagedAttention 的 page\_size 较小，如 16/32），逐页 matmul 的 Cube 利用率低——每个 page 独立做 `tl.dot(q, k_page)`，Cube 单元的矩阵乘规模太小（M=BLOCK\_M, N=page\_size），Cube 利用率不足
  - **方案一（L1 Cache 聚合）**：多个页用 `insert_slice` 聚合为单个大 tile 后再做 matmul——PAGE\_AGG\_NUM 个 page 聚合为 `(PAGE_AGG_NUM * BLOCK_N, D)` 的大矩阵，Cube 一次处理大矩阵，利用率大幅提升
  - **问题二**：通用 Triton 的 slice 操作有范围检查开销——`tl.extract_slice` 在运行时检查 offset 和 shape 是否越界，占用 Vector 周期
  - **方案二（CANN extension）**：用 CANN 扩展版本替代通用 slice，直接映射 DMA 切片指令，跳过范围检查
  ```python
  # ── L1 Cache 聚合（FA/SWA page aggregation）──
  # Before: 逐页加载 + 小 matmul，Cube 利用率低
  @triton.jit
  def fa_kernel(q_ptr, k_ptr, ...):
      for kv_block in range(num_kv_blocks):
          k = tl.load(k_ptr + kv_block * page_stride)  # 单页加载
          qk = tl.dot(q, tl.trans(k))  # 小 matmul，Cube 利用率低
          # ... softmax ...

  # After: 聚合为大 tile + 大 matmul，Cube 利用率高
  @triton.jit
  def fa_kernel(q_ptr, k_ptr, ..., PAGE_AGG_NUM: tl.constexpr):
      PAGE_AGG = PAGE_AGG_NUM  # 聚合页数
      for kv_block in range(0, num_kv_blocks, PAGE_AGG):
          # 预分配大 tile
          k = tl.zeros((PAGE_AGG * BLOCK_N, D), dtype=tl.float16)
          for i in range(PAGE_AGG):
              # 加载单页并插入大 tile 的对应位置
              k_slice = tl.load(k_ptr + (kv_block + i) * page_stride)
              k = insert_slice(k, k_slice, offset=i * BLOCK_N, axis=0)
          # 大 matmul，Cube 利用率高
          qk = tl.dot(q, tl.trans(k))  # 矩阵规模增大 PAGE_AGG 倍
          # ... softmax ...

  # ── CANN extension slice（RoPE / Conv1D / GroupGEMM / VisionRoPE）──
  # Before: 通用版本（运行时范围检查）
  slice_result = tl.extract_slice(tensor, offsets, shape, strides)
  # 运行时检查 offsets + shape 是否越界（Vector 开销）

  # After: CANN 扩展（直接映射 DMA 指令，跳过检查）
  import triton.language.extra.cann.extension as extension
  slice_result = extension.extract_slice(tensor, offsets, shape, strides)
  # 直接发起 DMA 切片指令，零范围检查开销
  ```
  - **案例**：FA/SWA prefill — 小 page 聚合提升 Cube 利用率
  - **案例**：RoPE / Conv1D / GroupGEMM / VisionRoPE — CANN extension slice
- 10.3 访存并行优化：care\_padding + Load 重排序
  - **问题一**：padding 区域的依赖检查阻止 load 指令并行发射——当 BLOCK\_SIZE > 实际维度时，load 的 padding 部分（越界位置）被编译器检查是否与其他操作有依赖，此检查阻止 load 提前发射
  - **方案一**：`care_padding=False` 跳过 padding 区域依赖检查（通过 mask 保证正确性），允许 load 指令提前并行发射
  - **问题二**：无依赖 load 排在依赖 load 之后，串行执行——编译器不重排 load 顺序，如果排在前的 load 因等待上一轮 store 而阻塞，排在后的无依赖 load 也无法提前发射
  - **方案二**：将无依赖 load 提前到有依赖 load 之前，使无依赖 load 与上一轮 store 并行执行
  ```python
  # ── care_padding（KV Cache）──
  # Before: 默认检查 padding，load 被阻塞
  @triton.jit
  def kv_cache_kernel(k_ptr, ...):
      # padding 区域依赖检查阻止 load 并行发射
      k = tl.load(k_ptr + offsets, mask=mask)  # 默认 care_padding=True

  # After: 跳过 padding 检查，load 提前并行发射
  @triton.jit
  def kv_cache_kernel(k_ptr, ...):
      # 跳过 padding 依赖检查（mask 保证正确性）
      k = tl.load(k_ptr + offsets, mask=mask, care_padding=False)  # 允许并行发射

  # ── Load 重排序（Quant/RoPE）──
  # Before: 依赖 load 先执行，无依赖 load 后执行（串行）
  @triton.jit
  def kernel(x_ptr, scale_ptr, ...):
      for i in range(num_iters):
          # x 的 load 依赖上一轮 store（同一地址），阻塞
          x = tl.load(x_ptr + offsets)
          # scale 的 load 无依赖，但必须等 x 的 load 完成后才能发射
          scale = tl.load(scale_ptr + offsets)
          # ... compute ...
          tl.store(x_ptr + offsets, result)  # 下一轮 x load 依赖此 store

  # After: 无依赖 load 先发射，两个 load 并行
  @triton.jit
  def kernel(x_ptr, scale_ptr, ...):
      for i in range(num_iters):
          # scale 的 load 无依赖，先发射（与上一轮 store 并行）
          scale = tl.load(scale_ptr + offsets)
          # x 的 load 依赖上一轮 store，后发射
          x = tl.load(x_ptr + offsets)
          # ... compute ...
          tl.store(x_ptr + offsets, result)
  ```
  - **案例**：KV Cache — `care_padding=False` 提升数据加载并行度
  - **案例**：Quant/RoPE — Load 重排序隐藏内存延迟
- 10.4 Block Pointer + 内存分配优化
  - **问题一**：手动指针算术 + mask 比 Block Pointer 效率低——手动方式需要 Vector 单元计算 `offsets = row[:, None] * stride + col[None, :]` 和 `mask = (row < M) & (col < N)`，占用 Vector 周期；Block Pointer 由 DMA 描述符硬件自动处理边界，零 Vector 开销
  - **方案一**：用 `tl.make_block_ptr` 替代手动指针算术，利用硬件 DMA 描述符自动处理边界
  - **问题二**：`torch.clone()` 拷贝整个张量数据，配合原地 kernel 引入 RAW 冒险——`clone` 分配新内存并拷贝数据（O(N) 数据搬运），原地 kernel 读写同一地址引入 Read-After-Write 冒险
  - **方案二**：用 `torch.empty_like()`（仅分配内存，无拷贝）替代 `clone`（分配 + 拷贝），配合非原地 kernel（读写不同地址）
  ```python
  # ── Block Pointer 替代手动指针算术 ──
  # Before: 手动指针算术 + mask（Vector 计算 offset 和 mask）
  @triton.jit
  def kernel(ptr, M, N, stride, ...,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
      col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
      # 手动计算 offset（Vector 计算）
      offsets = row[:, None] * stride + col[None, :]
      # 手动计算 mask（Vector 计算）
      mask = (row[:, None] < M) & (col[None, :] < N)
      x = tl.load(ptr + offsets, mask=mask)  # 手动 load

  # After: block_ptr 自动处理边界（DMA 描述符硬件完成）
  @triton.jit
  def kernel(ptr, M, N, stride, ...,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      # block_ptr：DMA 描述符自动处理边界，零 Vector 开销
      block_ptr = tl.make_block_ptr(
          base=ptr,
          shape=(M, N),
          strides=(stride, 1),
          offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
          block_shape=(BLOCK_M, BLOCK_N),
          order=(1, 0),
      )
      x = tl.load(block_ptr)  # 自动 load，越界位置填充 0

  # ── empty_like 替代 clone（RoPE）──
  # Before: 分配 + 拷贝（O(N) 数据搬运 + 原地 RAW 冒险）
  def rope_forward(q, ...):
      q_out = torch.clone(q)  # 分配新内存 + 拷贝 q 的所有数据
      rope_kernel[grid](q_out, ...)  # 原地修改 q_out（读写同一地址）

  # After: 仅分配（无拷贝，非原地无 RAW）
  def rope_forward(q, ...):
      q_out = torch.empty_like(q)  # 仅分配内存，不拷贝数据
      rope_kernel[grid](q, q_out, ...)  # 非原地：读 q，写 q_out（不同地址）
  ```
  - **案例**：VisionRoPE — 3D block\_ptr for cos/sin/q/k
  - **案例**：RoPE — `empty_like` 节省一个 q 张量大小的显存峰值
- 10.5 循环不变量外提：Bias 提前加载 + Cast 位置提前
  - **问题一**：bias 在 K 循环内每轮重复加载——bias 是循环不变量（不随 K 变化），但在循环内每轮 `tl.load(bias_ptr + offsets)`，重复发起 MTE2 搬运相同数据
  - **方案一**：bias 在循环外提前加载一次 + `broadcast_to` 显式广播，循环内直接复用
  - **问题二**：`.to(dtype)` 在 `tl.trans` 之后，cast 结果无法被多处复用——`p.to(dtype)` 在 `tl.trans(p)` 之后，每次需要转置后的 p 时都要重新 cast
  - **方案二**：将 cast 提前到 transpose 之前，cast 结果可被 dv 和 dk 两处复用（`tl.trans` 只是布局变换，不改变数据）
  ```python
  # ── Bias 提前加载（Fused MatMul fwd）──
  # Before: 每轮重复加载 bias（循环不变量在循环内）
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, ...):
      bias = tl.load(bias_ptr + offsets)  # 循环外？不，在循环内
      for k in range(0, K, BLOCK_K):
          bias = tl.load(bias_ptr + offsets)  # 每轮重复加载（浪费）
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b) + bias[None, :]  # bias 每轮加一次

  # After: 循环外加载一次 + 显式广播
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, ...):
      # 循环外加载一次
      bias = tl.load(bias_ptr + offsets)
      bias = tl.broadcast_to(bias[None, :])  # 显式广播到矩阵维度
      for k in range(0, K, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)  # 循环内不加 bias
      acc += bias  # 循环外加一次

  # ── Cast Hoisting（FA v5 bwd_kv / FA v8 bwd_q）──
  # Before: trans 后 cast，重复转换
  # p 需要转置后用于 dv 和 dk 两处，每处都重新 cast
  dv = tl.dot(tl.trans(p).to(dtype), do)       # trans 后 cast（第 1 次）
  dk = tl.dot(tl.trans(p).to(dtype), do_other)  # trans 后 cast（第 2 次，重复）

  # After: trans 前 cast 一次，复用结果
  p = p.to(dtype)                       # trans 前 cast（只 1 次）
  dv = tl.dot(tl.trans(p), do)          # 复用 cast 后的 p
  dk = tl.dot(tl.trans(p), do_other)    # 复用 cast 后的 p
  ```
  - **案例**：Fused MatMul fwd — bias 循环外提前加载
  - **案例**：FA v5 bwd\_kv / FA v8 bwd\_q — p 的 cast 提前复用
- 10.6 指令简化：Mask 类型转换 + tl.full 初始化
  - **问题一**：布尔 mask 的隐式类型推断导致不确定行为——`mask = (q_offset <= k_offset)` 产生 bool 类型，参与浮点运算时（如 `block_s = tl.where(mask, block_s, -inf)`）可能触发隐式转换的不确定行为
  - **方案一**：对布尔 mask 进行显式类型转换 `.to(tl.int32)`，确保类型安全
  - **问题二**：`tl.zeros([M]) - float("inf")` 需要先创建零张量再减（2 条指令：`tl.zeros` 创建零张量 + 减法），用于初始化 online softmax 的最大值数组 `m_i`
  - **方案二**：`tl.full([M], float("-inf"))` 直接填充目标值（1 条指令），消除减法操作
  ```python
  # ── Mask 类型转换（FA v4 fwd）──
  # Before: 隐式 bool（可能触发不确定行为）
  @triton.jit
  def fa_kernel(...):
      q_offset = tl.arange(0, BLOCK_M) * stride
      k_offset = tl.arange(0, BLOCK_N) * stride
      # 隐式 bool 类型，参与 where 时可能不确定
      mask = (q_offset <= k_offset)
      block_s = tl.where(mask, block_s, -1e6)

  # After: 显式 int32（类型安全）
  @triton.jit
  def fa_kernel(...):
      q_offset = tl.arange(0, BLOCK_M) * stride
      k_offset = tl.arange(0, BLOCK_N) * stride
      # 显式转换为 int32，确保类型安全
      mask = (q_offset <= k_offset).to(tl.int32)
      block_s = tl.where(mask, block_s, -1e6)

  # ── tl.full 替代 zeros - inf（FlexAttention）──
  # Before: 2 条指令（创建零张量 + 减法）
  @triton.jit
  def fa_kernel(...):
      # 先创建零张量，再减去 inf（2 条 Vector 指令）
      m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

  # After: 1 条指令（直接填充 -inf）
  @triton.jit
  def fa_kernel(...):
      # 直接填充目标值（1 条 Vector 指令）
      m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
  ```
  - **案例**：FA v4 fwd — mask\_fn 显式类型转换
  - **案例**：FlexAttention — `m_i` 初始化优化

***

### 第 11 章 编译器图优化：固定模式匹配的自动优化

> **核心理念**：Triton-Ascend 的图优化 pass 在 IR 图上做固定模式匹配（pattern match），命中特定子图后自动重写为更优的等价形式。开发者无需（也无法）手写这些变换，但理解匹配的模式与触发条件，可以预判编译器行为、定位"为什么没被优化"，并知道何时需要显式干预。

> **背景知识**：图优化 pass 针对 IR 图中的固定子图形态做自动重写，主要覆盖五类模式——**融合类**（逐元素链、dot 后处理、向量函数打包）、**循环结构类**（多缓冲标记）、**布局类**（NZ↔ND 转换）、**访存地址与布局类**（取模线性化、读写置换、store 合并、跨步改写）、**程序映射与 Grid 变换类**（独立轴张量化、持久化任务挖掘、行合并）。每个 pass 只认自己关心的子图结构，未命中则原样保留。

- 11.1 匹配机制概述：IR 图 → pass 扫描固定子图 → 重写
  - **匹配对象**：pass 扫描特定子图结构，常见有三类触发条件——**结构模式**（op 类型与连接关系，如 `mulf + addf`、循环内 `load/store` 对）、**布局/属性约束**（如 `ascend.dot` 的 `fractal_a/b/c`、block 尺寸 16×16）、**循环结构约束**（候选必须位于 `scf.for`/`scf.while` 内且支持每迭代轮转槽位）
  - **重写结果**：命中后替换为更优等价形式（融合、布局转换、多缓冲标记等），未命中则原样保留
  - **与手写代码的关系**：绝大多数自动生效；少数场景需要提供 hint 或开关配合（见 11.8）
- 11.2 自动融合类 pattern：逐元素链合并、dot 后处理与 fixpipe 随路、VF 打包
  - **逐元素链合并**：`x * sigmoid(x)` 等连续逐元素链被合并为单指令（silu），`mul + add` 折叠为 fma，连续 cast 折叠
  - **dot 后处理随路**：`a*b + c` 或带 bias/精度转换的结果不再单独做一次 Vector 运算——编译器把 bias 累加、fp32→fp16、转置融进 **fixpipe** 从 L0C 搬出时的随路操作（按下游消费方式选择 NZ2ND / NZ2DN / NZ2NZ 模式），这是 Cube 结果喂给 Vector 前最常见的自动融合
  - **VF 打包**：连续向量运算链被 outline 成独立向量函数（`--vf-fusion-mode`，默认 max-parallel），减少指令间同步；代价是链过长时 UB 溢出或吞掉 dot 的独立调度（回指第 6 章）
- 11.3 循环结构 pattern：多缓冲标记
  - **多缓冲标记**：循环内 `load/store` 对命中多缓冲模式，在分配点打 `annotation.mark {hivm.multi_buffer = N}` 标记，后续经内存规划分配 N 份地址、按迭代计数取模选择槽位（回指第 7 章）。本地 Buffer 的 N 固定为 2，GM Workspace 的 N 由 `--set-workspace-multibuffer` 指定
  - **硬约束**：候选 Buffer 从分配点到最外层祖先循环必须都是 `scf.for`/`scf.while`，位于 `scf.parallel`/`scf.forall` 内的 Buffer 不会被标记
- 11.4 布局 pattern：NZ↔ND 转换与消除
  - **分形布局推断**：`ascend.dot` 的 `fractal_a/b/c` 属性触发布局推断（如 DOTA\_ND\_TO\_zN），编译器在 GM 的 ND 与 Cube 的 NZ（`[N1, M1, 16, 16]` 分形块）之间自动插入 `convert_layout`
  - **fixpipe 收尾转换**：Cube 结果在 L0C 为 NZ，需 fixpipe 搬运到 UB 并转回 ND，Vector 才能逐元素处理（回指第 10 章）
  - **转换消除**：连续的 `convert_layout` 被合并/消除；布局一致的数据流可省掉整段转换
- 11.5 访存地址与布局优化 pattern：把离散、低效的内存访问改写为整块连续访问
  - **共同目标**：针对**地址表达式本身**（取模、跨步、维度顺序）做改写，不动数据布局（与 11.4 的 convert\_layout 类 pattern 互补），让后端把离散访问识别为整块连续内存、一次 DMA 搬完，同时消除标量地址计算。下文每个 pattern 均给出触发模式、改写、收益与约束
- 11.5.1 ConvertModuloToMask：取模线性化
  - **触发模式**：kernel 写 `offs % n` 让越界下标绕回数组开头以避免非法访问，但写回处已有 `offs < n` 掩码丢弃越界数据——读进来的绕回数据本就无用
  - **DSL 写法（触发示例）**：load 用取模绕回地址，store 带 `offs < n` 掩码
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      wrap = offs % n                  # 越界下标绕回数组开头
      mask = offs < n                  # store 的守卫掩码
      v = tl.load(x_ptr + wrap)        # load 无掩码（读绕回数据）
      tl.store(y_ptr + offs, v, mask=mask)
  ```
  - **改写**：去掉取模让地址恢复连续，同时给 load 补上同样的掩码（越界位置读 0）
  ```ttir
  # Before：offs % n 把连续地址打乱成跳跃的，load 无掩码
  %wrap = arith.remsi %offs, %n       # offs % n
  %addr = arith.addi %kb, %wrap
  %v    = tt.load %wa                 # load 无掩码
  tt.store %oa, %v, %gb               # store 已带 offs < n 掩码

  # After：地址恢复线性，load 补掩码 + 零填充
  %cmp  = arith.cmpi slt, %offs, %n   # offs < n
  %addr = arith.addi %kb, %offs       # 地址线性化
  %v    = tt.load %wa, %cmp, %cst     # 补掩码，越界读 0
  tt.store %oa, %v, %cmp
  ```
  - **收益**：地址线性化后后端可识别为整块连续内存，用一次 DMA 搬完，而非因取模被迫逐元素离散访问
  - **约束**：除数必须是运行时变量（编译期常量交由 TritonToStructured 处理更优，它保留绕回语义并改写为跨步访问，完全等价）；下标必须是一维 i32、无 layout，长度等于 make\_range 长度；必须存在 store 上的 offs < n 守卫作为依据，因此归约轴上的取模无法处理（归约轴无输出坐标，不会有这种掩码）；取模结果只能流向 load 地址，不能流向 other、tt.trans、tt.reshape 或其他计算；expand\_dims 的轴必须唯一；不支持跨嵌套 scf.for 的多条循环携带地址
- 11.5.2 LoadStoreTranspose：读写维度置换
  - **触发模式**：分析 load/store 的地址表达式，判断张量各维的实际内存步长。若最内维（代码中变化最快的那维）在内存里是跳着走的，则触发
  - **DSL 写法（触发示例）**：tile 最内维（cols）在内存中跳着走——列主序访问
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr):
      rows = tl.arange(0, M)           # 32 行
      cols = tl.arange(0, N)           # 16 列
      # 元素 (r, c) 的地址 = r + c * M：内维步长 M，读一个元素跳 M 个元素
      offs = rows[:, None] + cols[None, :] * M
      v = tl.load(x_ptr + offs)        # [32, 16]
      tl.store(y_ptr + offs, v + 1.0)
  ```
  - **改写**：把整个"读→算→写"闭包的维度顺序调换，使最内维变成内存中真正连续的那维；同步改写掩码、assert、循环携带指针，以及循环之后 reduce 的轴号
  ```ttir
  # Before：最内维 16 列每列相隔 32 个元素，读一个元素跳 128 字节
  %off = arith.addi %mb, %nb          # 32×16 偏移
  %sa  = tt.addptr %sp, %off
  %x   = tt.load %sa : tensor<32x16x!tt.ptr>
  %y   = arith.addf %x, %one
  tt.store %da, %y

  # After：调换维度顺序，最内维 32 个元素紧邻，一次连续搬 128 字节
  %off = arith.addi %mb, %nb          # 16×32 偏移
  %sa  = tt.addptr %sp, %off
  %x   = tt.load %sa : tensor<16x32x!tt.ptr>
  %y   = arith.addf %x, %cst
  tt.store %da, %y
  ```
  - **收益**：改写前每读一个元素跳 128 字节、16 个元素是 16 次离散访问；改写后最内维 32 个元素紧邻，一次连续搬 128 字节
  - **约束**：指针必须是 addptr(splat(标量指针), 偏移) 形式，维度 ≥2、静态 shape、无 layout，不支持 block pointer；每一维须恰好由一个 make\_range 驱动，且各维步长大小关系能在结构上证明（运行时步长需掩码提供取值范围）；必须同时有 load 和 store 且存在 load→store 数据通路，所有访存点置换一致；不支持 volatile、padding、boundary\_check；load 的 other 必须是均匀常量；闭包内只允许逐元素算子、splat、broadcast、expand\_dims、addptr、trans、常量，不支持矩阵乘和 reshape；循环结果只能被单输入 tt.reduce 消费；不跨 scf.if/scf.while 区域边界；遇到用户自己写的张量 assert 直接放弃
- 11.5.3 StoreCoalescing：相邻 store 合并
  - **触发模式**：函数内有几次 store 写入同一块内存的相邻区段（前一次结尾正好接下一次开头）
  - **DSL 写法（触发示例）**：对同一基址的相邻区段分多次 store，首尾严格相接
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      offs0 = tl.arange(0, BLOCK)
      offs1 = tl.arange(0, BLOCK) + BLOCK
      v0 = tl.load(x_ptr + offs0)
      v1 = tl.load(x_ptr + offs1)
      tl.store(y_ptr + offs0, v0)      # [0, BLOCK)
      tl.store(y_ptr + offs1, v1)      # [BLOCK, 2*BLOCK)，紧邻上一段
  ```
  - **改写**：先在寄存器中把数据拼成一个大张量，最后一次性写出
  ```ttir
  # Before：两段相邻 store 分两次写
  tt.store %base, %v0                 # [0, 64)
  tt.store %base + 64, %v1            # [64, 128)

  # After：寄存器拼成 128 元素大张量，一次写出
  %t = tensor.insert_slice %v0 into %empty[0] [64] [1]
  %t = tensor.insert_slice %v1 into %t[64] [64] [1]
  tt.store %base, %t                  # 一次 128 元素连续写
  ```
  - **收益**：两次 256 字节的写变成一次 512 字节的写，搬运次数减半、单次更长，GM 写带宽利用率更高
  - **约束**：只处理函数最外层 store（循环或分支内的不处理）；store 不能带掩码或 boundary\_check，且须能静态证明按行主序连续排列；基地址必须是同一 SSA 值（纯语法比较，不做别名推断）；元素类型、cache/evict 修饰及其他属性须完全一致；至少两段，段间不重叠且首尾严格相接；合并后总字节数不超过 UB 预算；被推迟的 store 与中间夹着的其他访存必须能证明不冲突；不支持 i1 和不足一字节的类型；偏移量须在 i32 范围内；simt\_only 下关闭
- 11.5.4 MergeConcatLoadBuffer：concat 缓冲合并
  - **触发模式**：在降级到 memref 之后生效。带掩码的 load 降级后会变成"分配整块缓冲 → 用填充值填满 → 把有效部分拷入"，若紧接着做 tl.cat 拼接，就会出现两块缓冲各写一半
  - **DSL 写法（触发示例）**：两个带掩码 load 后紧跟 tl.cat 拼接
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      mask = offs < n
      a = tl.load(x_ptr + offs, mask=mask, other=0.0)     # 带掩码 load
      b = tl.load(x_ptr + BLOCK + offs, mask=mask, other=0.0)
      c = tl.cat(a, b)                  # 拼接 → 降级后两块缓冲各写一半
      tl.store(y_ptr + tl.arange(0, 2 * BLOCK), c)
  ```
  - **改写**：把两块缓冲合并为一块；若能证明两次拷贝覆盖了后续全部读取范围，两个填充操作都删除
  ```ttir
  # Before：两块 32×512 缓冲各写一半（各含一次 fill + copy + to_tensor）
  %a0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst) outs(%a0)
  memref.copy %ss0, %sd0              # 写 [0, 256)
  %t0 = bufferization.to_tensor %a0
  %a1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst) outs(%a1)
  memref.copy %ss1, %sd1              # 写 [256, 512)
  %t1 = bufferization.to_tensor %a1
  %ex  = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1]
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1]

  # After：合并为一块缓冲，拷贝写不同 subview，两次填充删除
  %a = memref.alloc() : memref<32x512xf32>
  memref.copy %ss0, %sv0              # 写 [0, 256)
  memref.copy %ss1, %sv1              # 写 [256, 512)
  %t = bufferization.to_tensor %a
  ```
  - **收益**：UB 占用从 2 块 tile 降到 1 块，并省去两次 32×512 元素的填充与一次切片数据搬移
  - **约束**：只在 910\_95/950 上启用。910\_9362 的代码生成只保证紧邻 to\_tensor 的那次拷贝可见，合并后另一半 tile 会读到填充值，故该芯片上禁用；extract 与 insert 的偏移、大小必须完全一致；填充必须发生在拷贝之前，否则放弃；只保留一个填充时，要求两个填充都是单纯 fill 且填充值相同
- 11.5.5 StridedAxisCoalescing：跨步轴折回
  - **触发模式**：kernel 实现中把本来连续的 H 轴（如 head 数）摊到 grid 上，用 `pid % S` 取出自己负责的 head，导致每个 program 读的数据在内存中每隔 S 个元素取一个
  - **DSL 写法（触发示例）**：H 轴摊到 grid、`pid % S` 取 head，用步长 S 的一维 block pointer 读
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, S: tl.constexpr, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      i_h = pid % S                        # 取自己负责的 head
      tp = tl.make_block_ptr(x_ptr, (n,), (S,), (i_h,), (BLOCK,), (0,))
      v = tl.load(tp)                      # 每隔 S 个元素取一个
      tl.store(tp, v + 1.0)
  ```
  - **改写**：把 H 轴重新折回张量内层，组成二维 tile 一次读完；逐 head 的标量 load 相应提升为向量 load
  ```ttir
  # Before：一维、步长 4，每 program 处理一个 head（pid % 4）
  %i_h = arith.remsi %pid, %c4        # i_h = pid % 4
  %tp  = tt.make_tensor_ptr %base, [16], [4], [i_h]
  %v   = tt.load %tp : tensor<16xf32>

  # After：二维、内维步长 1，取模消失，整块连续内存一次 DMA 搬完
  %tp  = tt.make_tensor_ptr %base, [16, 4], [4, 1], [0, 0]
  %v   = tt.load %tp : tensor<16x4xf32>
  ```
  - **收益**：改写前 16 个元素分散在 16×4 范围内、需 16 次离散访问；改写后是一整块连续内存一次 DMA 搬完；program 数同时降为 1/4
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行，不受 rule-mask 控制；种子 load 必须是一维 block pointer，步长为大于 1 的常量，且所有种子步长相同；kernel 不能读对应轴的 num\_programs（grid 被除以 S 后该值会变，读了就会算错）；子图只允许逐元素算子、类型转换、splat、沿 T 轴的 scan/reduce；不支持转置、矩阵乘、reshape，也不支持沿 lane 方向的归约；输出 store 也必须是同样步长的 block pointer；i\_h 不能参与间接寻址；按整个 module 改写，任一条件不满足即整体放弃
- 11.5.6 ChunkCoalescing：小块合并
  - **触发模式**：对 `offs = pid * T + arange(T)` 这种标准分块写法，若每个 tile 太小导致连续字节数不足、DMA 效率低，就把 H 个相邻 tile 合并到一个 program，使单次连续访问至少达到 512 字节
  - **DSL 写法（触发示例）**：`offs = pid * T + arange(T)` 的小 tile 分块 + 尾部掩码
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, T: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * T + tl.arange(0, T)     # 每 tile 仅 T 个元素（连续字节数不足）
      mask = offs < n
      v = tl.load(x_ptr + offs, mask=mask, other=0.0)
      tl.store(y_ptr + offs, v, mask=mask)
  ```
  ```ttir
  # Before：一维 16 元素 = 64 字节、带掩码（64 字节远低于 DMA 高效传输长度）
  %4 = arith.addi %3, %2 : tensor<16xi32>            # offs = pid*16 + arange(16)
  %5 = arith.cmpi slt, %4, %cst_0 : tensor<16xi32>   # offs < 256
  %8 = tt.load %7, %5, %cst : tensor<16x!tt.ptr>
  tt.store %10, %8, %5 : tensor<16x!tt.ptr>

  # After：二维 16×16 = 1024 字节，掩码删除
  %7  = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %8  = tt.broadcast %7 : tensor<16x1xi32> -> tensor<16x16xi32>
  %11 = arith.addi %8, %10 : tensor<16x16xi32>
  %16 = tt.load %15 : tensor<16x16x!tt.ptr>          # 无掩码
  tt.store %20, %16 : tensor<16x16x!tt.ptr>
  ```
  - **收益**：单次连续搬运从 64 字节增至 1024 字节（64 字节远低于 DMA 高效传输长度，大部分时间消耗在启动开销上），program 数降为 1/16
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行；对应 grid 轴只有一个 program\_id 且不读 num\_programs；pid 派生链上不能出现 min/max/取余/除法；派生出的比较结果除作为那一个掩码外，只能流向 device\_assert（溢出检查默认开启，会给几乎每个 pid 派生下标插桩，必须明确放行）；不支持 boundary\_check；H 取值范围为 \[max(2, ⌈512B ÷ tile字节⌉), min(16, 128KB ÷ 单tile占用)] 且必须整除总 tile 数，算不出合法 H 就不改；若 StridedAxisCoalescing 已设过 coalesce\_factor 则直接跳过
- 11.5.7 StridedLoadStoreRewrite：非 2 幂跨步改写
  - **触发模式**：硬件跨步 DMA 引擎只支持 2 的幂步长，遇到步长 3、5、7 这类会退化为极慢的逐元素标量访问
  - **DSL 写法（触发示例）**：地址步长为非 2 的幂常量（3、5、7…）
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK) * 3       # 步长 3：硬件跨步 DMA 不支持
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)
  ```
  - **改写**：把步长为非 2 的幂常量的 load/store 改写为专用的 ascend.stride\_load/stride\_store 或间接访问算子
  ```ttir
  # Before：步长 3 的地址链，标量循环逐个取
  %1 = arith.muli %arange, %c3        # 步长 3
  %3 = tt.addptr %splat, %1
  %4 = tt.load %3 : tensor<256x!tt.ptr>

  # After：专用跨步访问算子，交由 SIMT 并行访问单元处理
  %1 = ascend.stride_load %arg0, %c0_i64, %cst, [%c3_i64], [%c256_i64]
  ```
  - **收益**：原本 256 个元素需靠标量循环逐个取，改写后交由 SIMT 并行访问单元处理，整个指针算术链一并消失
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行；以下情况有意不改写：步长 1（本就连续）、步长 2（走 deinterleave 路径）、步长为 ≥4 的 2 的幂（走跨步 DMA 更快）、步长为运行时变量（可能运行时就是 1 或 2 的幂，保留 SIMD 路径更稳妥）、做过维度置换的布局（交由 ImplicitPermute）；stride\_load 只支持 1\~3 维静态 shape，快速路径最多 5 维；非浮点类型要求 NaN 填充时放弃；addptr 路径会调用指针分析，可能留下无用辅助 IR，需后续 CSE/canonicalize 清理（T2L pipeline 已安排）
- 11.6 程序映射与 Grid 变换 pattern（AutoBlockifyV2）：合并 program、提升单 program 吞吐
  - **共同目标**：把多个独立 program 合并成一个，用单 program 内的向量化/张量化计算换掉重复的标量地址计算——program 数减少、Vector 单次处理数据量增大、UB 利用率提升，本质是"以 program 数换单 program 吞吐"。均不受 rule-mask 控制
- 11.6.1 IndependentAxisTensorize：独立轴张量化（IAT）
  - **触发模式**：grid 为二维（token 轴、head 轴）且不同 head 间计算完全独立时，把 head 轴的 16 个 program 合并成一个，让一个 program 一次处理 16 个 head
  - **DSL 写法（触发示例）**：二维 grid（token 轴、head 轴），head 间计算完全独立
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      pid_t = tl.program_id(0)             # token 轴
      pid_h = tl.program_id(1)             # head 轴
      offs = tl.arange(0, BLOCK)
      # MergeSplit 形态：4×BLOCK tile 沿 T 轴归约 → BLOCK（每 program 只处理一个 head）
      v = tl.load(x_ptr + pid_h * 4 * BLOCK + tl.arange(0, 4)[:, None] * BLOCK + offs[None, :])
      s = tl.sum(v, axis=0)                # 归约
      tl.store(y_ptr + pid_h * BLOCK + offs, s)
  ```
  - **改写**：原 program\_id(1) 变为长度 16 的向量，相关张量各增加一维，尾部补掩码。支持 MergeSplit 与 NormRope 两种归约形态
  ```ttir
  # Before：二维 grid，每 program 处理一个 head（标量 head 索引）
  %head = tt.get_program_id y
  %th   = arith.addi %th0, %head
  %v    = tt.load %xa : tensor<4x128x!tt.ptr>
  %sum  = tt.reduce(%v) {axis = 0 : i32} : (tensor<4x128xf32>) -> tensor<128xf32>
  tt.store %ya, %sum

  # After：head 轴张量化，一个 program 处理 16 个 head
  %lanes = arith.addi %pid_y*16, %arange(16)          # 16 条 head lane
  %mask  = arith.cmpi ult, %lanes, %extent            # 尾部掩码
  %v     = tt.load %xa : tensor<4x16x128x!tt.ptr>
  %sum   = tt.reduce(%v) {axis = 0 : i32} : (tensor<4x16x128xf32>) -> tensor<16x128xf32>
  tt.store %ya, %sum, %mask
  ```
  - **收益**：program 数降为 1/16；原本每 program 都要重算的标量地址计算现由 16 个 head 分摊一次；Vector 指令单次处理数据量增加 16 倍，UB 利用率显著提升
  - **约束**：必须是非 simt\_only 模式；函数只能有一个基本块，不能有 scf/cf 控制流、函数调用、原子操作、barrier；axis0 与 axis1 的 program\_id 各只能出现一次；axis1 依赖闭包内不能有 make\_tensor\_ptr，也不能读 num\_programs；store 与 load 指针必须来自不同入参；UB 峰值×16 后不能超预算；动态 shape 直接拒绝；合并因子固定为 16，无自动调优；module 上已有其他 grid 契约时不再叠加；NormRope 形态要求 PTSM 同时启用
- 11.6.2 PersistentTaskStripMining：持久化任务条带挖掘（PTSM）
  - **触发模式**：针对逐 token 做行归约的 kernel——LayerNorm 类同时算 sum(x) 与 sum(x²)，RMSNorm 类只算 sum(x²)
  - **DSL 写法（触发示例）**：逐 token 做行归约的 norm 类 kernel（一个 token 一个 program）
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, D: tl.constexpr):
      pid = tl.program_id(0)             # 一个 token 一个 program
      offs = pid * D + tl.arange(0, D)
      v = tl.load(x_ptr + offs)          # 一行 D 个元素
      s2 = tl.sum(v * v)                 # RMSNorm 类：sum(x²)
      # LayerNorm 类再加：s1 = tl.sum(v)（同时算 sum(x) 与 sum(x²)）
      tl.store(y_ptr + pid, s2)
  ```
  - **改写**：原本有多少 token 就启动多少 program，改为固定数量的 program，每个用跨步循环反复领取任务，单次处理 BLOCK\_T 个 token（LayerNorm 类 64，RMSNorm 类配合 IAT 时 4），尾部按原始 token 数加掩码
  ```ttir
  # Before：逐 token 一个 program，标量行归约
  %row = arith.muli %pid, %cD
  %v   = tt.load %xa : tensor<128x!tt.ptr>
  %s1  = tt.reduce(%v) {axis = 0 : i32}       # sum(x)
  %sq  = arith.mulf %v, %v
  %s2  = tt.reduce(%sq) {axis = 0 : i32}      # sum(x²)
  tt.store %ya, %o

  # After：固定数量 program + 跨步循环，向量化行归约
  %groups = arith.divui %total, %c64          # ceil(T / 64) 组
  scf.for %iv = %pid to %groups step %num_programs {
    %v   = tt.load %xa : tensor<64x128x!tt.ptr>    # 单次 32KB 连续数据
    %s1  = tt.reduce(%v) {axis = 1 : i32}     # → 64
    %sq  = arith.mulf %v, %v
    %s2  = tt.reduce(%sq) {axis = 1 : i32}
    tt.store %ya, %o, %mask                   # 尾部掩码
  }
  ```
  - **收益**：program 数从 T 降为固定值；单次 load 从 512 字节变为 32KB 连续数据；标量归约变为向量化行归约
  - **约束**：只能有一个 public 入口函数、返回值为空、不能有函数调用；函数内不能出现 scf 控制流、原子操作、barrier、随机数、print；必须匹配上述两种归约形态之一（判定严格：需检查平方项是否同一值自乘、两个归约结果下游是否汇合等）；token 轴 program\_id 唯一且不读 num\_programs；store/load 来自不同入参；改写后 UB 峰值须通过资源评估；已有 grid 契约只能是 1 个非 token 轴、非持久化的变换（即 IAT）；BLOCK\_T 固定，不做调优；simt\_only 下关闭
- 11.6.3 RowCoalescing：行合并
  - **VV 形态（逐行 Vector kernel）**：匹配"取 pid → 越界直接返回 → 否则处理一行"这种写法的 kernel，改为一个 program 处理 R 行。R 依据工作区内最大张量元素数自动选取：≤16 选 8，17\~1023 选 4，1024 选 2，超过 1024 不做
  - **DSL 写法（触发示例）**：取 pid → 越界直接返回 → 否则处理一行
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n_rows, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      if pid >= n_rows:                  # 越界直接返回
          return
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      v = tl.load(x_ptr + offs)
      o = tl.maximum(v, 0.0)
      tl.store(y_ptr + offs, o)
  ```
  ```ttir
  # Before：取 pid → 越界直接返回 → 否则处理一行
  %done = arith.cmpi sge, %pid, %n_rows
  cf.cond_br %done, ^exit, ^work
  ^work:
    %v = tt.load %xa : tensor<16x!tt.ptr>
    %o = arith.maxnumf %v, %z
    tt.store %ya, %o

  # After：一个 program 处理 8 行，分支消失
  %rows = arith.addi %pid*8, %arange(8)      # 8 个行号
  %mask = arith.cmpi slt, %rows, %n_rows     # 尾部掩码
  %v    = tt.load %xa : tensor<8x16x!tt.ptr>
  %o    = arith.maxnumf %v, %cst
  tt.store %ya, %o, %mask
  ```
  - **VV 收益**：program 数降为 1/8；cf.cond\_br 分支开销消失；单次处理 8×16 元素，SIMT 线程利用更充分
  - **VV 约束**：只在 compile\_mode=simt\_only 下生效；module 只能有一个 public 入口且无函数调用；只有一个 program\_id，且不读该轴 num\_programs；pid 派生值不能在工作区外使用；工作区只允许 arith/math 运算、load/store、指针与形状类算子、scan/reduce、scf.for，不支持矩阵乘；工作区计算结果不能流出；与 IAT、PTSM 或已有 coalesce 属性互斥；在所有阶段之后单独执行一次，不占用 64 次改写配额
  - **CV 形态（decode attention 行合并）**：针对"每个 program 处理一行 query、在 K 维循环里做矩阵乘"的 decode attention 模式：把 R 个 program 合成一个，随行变化的值加一个前导行维，矩阵乘时 reshape 为 \[R×M, K] 以撑大 M 维；K/V 这类与行无关的数据保持二维、R 行共享。R 从 32/16/8/4/2 中选取，要求 M×R ≤ 128 且最大张量字节数×R ≤ 128KB
  - **DSL 写法（触发示例）**：每 program 处理一行 query，在 K 维循环里做矩阵乘
  ```python
  @triton.jit
  def decode_attn(q_ptr, k_ptr, o_ptr, K, M: tl.constexpr, N: tl.constexpr):
      pid = tl.program_id(0)             # 一行 query 一个 program
      offs_m = tl.arange(0, M)
      offs_n = tl.arange(0, N)
      a = tl.load(q_ptr + pid * M + offs_m)[:, None]     # [M, 1]，随行变化
      acc = tl.zeros((M, N), dtype=tl.float32)
      for i in range(0, K, N):           # K 维循环做矩阵乘
          b = tl.load(k_ptr + i + offs_n)[None, :]       # [1, N]，K/V 与行无关
          acc += tl.dot(a, b)            # [M,1] × [1,N] → [M,N]
      tl.store(o_ptr + pid * M + offs_m, tl.sum(acc, axis=1))
  ```
  ```ttir
  # Before：每 program 一行 query，K 维循环矩阵乘（M=16 过小浪费 Cube）
  %a = tt.load %pa : tensor<16x16x!tt.ptr>
  %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) {
    %b   = tt.load %pb : tensor<16x16x!tt.ptr>
    %dot = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
  }
  tt.store %pc, %loop

  # After：R 行合成一个 program，M 撑到 128 填满 Cube；K/V 二维共享
  %a   = tt.load %pa : tensor<8x16x16x!tt.ptr>      # A 增加前导行维
  %rhs = tt.load %pb : tensor<16x16x!tt.ptr>        # B 保持二维，8 行共享
  %dot = tt.dot %a2d, %rhs, %acc : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
  tt.store %pc, %dot, %mask
  ```
  - **CV 收益**：矩阵乘 M 维从 16 撑到 128，填满 Cube 单元（M 维过小会严重浪费算力）；K/V 从 GM 读取 8 次降为 1 次
  - **CV 约束**：只支持 Ascend950；只能有一个基本块和一个 public 入口；必须恰好有 1 个 axis0 program\_id、1 个下界为 0 的顶层 scf.for、1 个顶层 store，且至少 1 个矩阵乘；矩阵乘必须是二维 f16/bf16，且 A 随行变化、B 不随行变化；不支持 tt.reduce、循环内依赖行的 load、volatile load、用于变长/间接索引的标量 load；循环上界依赖 pid 时必须能证明单调递增；循环体内含不可推测执行的算子（如可能除零的整数除法）时拒绝；R 的选取是保守经验策略，非实测最优值
- 11.7 确认图优化 pass 是否命中
  - **IR 检索**：在生成的 IR 中搜索特征标记——`hivm.multi_buffer` + `arith.select` 槽位结构（多缓冲）、`convert_layout`（布局转换）、`annotation.mark {break_vf}`（VF 断点）、fixpipe（NZ2ND）——缺失说明对应模式未命中
  - **常用图优化开关**：
    | 开关                            | 作用                                           | 默认           |
    | ----------------------------- | -------------------------------------------- | ------------ |
    | `--enable-auto-multi-buffer`  | 自动多缓冲总开关                                     | true         |
    | `--vf-fusion-mode`            | VF 融合模式（max-parallel / ub-aware-op / all-op） | max-parallel |
    | `--set-workspace-multibuffer` | GM Workspace 多缓冲数量（950 默认 2，A3/A2 默认 4）      | 随硬件          |
- 11.8 干预手段
  - **hint 断点**：`compile_hint("break_vf")` 在 VF 融合链上放置不可跨越的断点，拆开过长的融合域（回指第 6 章）
  - **关闭开关**：`--enable-auto-multi-buffer=false`、`--limit-auto-multi-buffer-*` 系列，用于性能对比或问题定位
  - **溢出回退**：多缓冲溢出时编译器先退让溢出层次再关总开关，回退成功但对应收益丢失，此时更合适的是减小 tiling 分块
  - **调优建议**：先保持默认验证命中，再逐项关闭对比，避免一步到位导致误判

***
