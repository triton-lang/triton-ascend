# Triton算子性能调优指南

## 第一篇：基础篇 — 理解硬件差异，建立心智模型

### 第1章GPU与NPU微架构差异：Triton算子开发迁移视角

- 1.1执行模型与Grid映射：Warp调度 → 物理核绑定
  - GPU：SIMT执行模型，32线程/Warp 锁步执行
    - Grid与物理SM **解耦**：可启动远超SM数量的Block，硬件自动排队调度
    - `num_warps` 控制每个Block的并行度，`num_stages` 控制软件流水线深度
    - 开发者只需按逻辑任务量设置Grid（如 `cdiv(N, BLOCK_SIZE)`），硬件负责调度
  - NPU：SIMD执行模型（默认），一个Program绑定一个AI Core，无线程概念
    - Grid与物理核**强绑定**：Grid规模超出物理核数时，Block需分批（轮转）调度，每次切换都会产生额外的设备侧开销（Block调度与参数装载、核内上下文初始化等）；且跨Block的存算流水线无法并行，前一个Block需排空流水线后，下一个Block才重新预热启动。
    - `num_warps` / `num_stages` **无效**（NPU无Warp、无GPU式软件流水线）
    - 开发者需主动查询核心数并将Grid对齐：`grid = (num_cores,)` + 核内for循环
  - **对Triton代码的影响**：
    - 迁移第一步：删除 `num_warps`、`num_stages`，加入 `multibuffer` 参数
    - Grid设计从"逻辑分块"变为"物理核持久化"（Persistent Kernel模式）
    - 纯Vector算子用 `num_vectorcore`（910\_95 为aicore的2倍），CV融合算子用 `num_aicore`
- 1.2片上存储管理：Shared Memory手动管理 → UB编译器管理
  - GPU：Shared Memory（48-164 KB/SM）由程序员手动控制
    - 通过 `num_stages` 隐式控制Shared Memory多缓冲深度
    - Tensor Core输出直接写入寄存器，后续运算立即可用
    - 开发者通过 `__shared__` 或Triton编译器间接管理Shared Memory布局
  - NPU：UB（248KB可用）由编译器全权管理，开发者**不能手动分配**
    - UB是Vector的唯一工作区：所有 `tl.load` 的数据、中间计算结果、`tl.store` 前的数据都在UB
    - Cube输出进入L0C（256KB），需经fixpipe搬到UB后Vector才能处理
    - 开发者通过 `BLOCK_SIZE` **间接控制** UB占用（而非直接分配Shared Memory）
    - `multibuffer=True` 启用双缓冲，但UB可用空间减半（248KB → 124KB）
  - **对Triton代码的影响**：
    - `BLOCK_SIZE` 不仅是Tiling参数，还直接决定UB占用——选错即 `ub overflow` 编译错误
    - UB预算公式：`Σ(每个 tensor 的 BLOCK_SIZE × dtype_size) ≤ 248KB`（开启multibuffer则 ≤ 124KB）
    - 大BLOCK + 核内二级分块（`BLOCK_SIZE_SUB`）是NPU常用模式：核间大块控并行度，核内小块控UB

***

## 第二篇：分析篇 — 性能剖析与瓶颈定位

### 第2章 性能分析方法论（调优的前提）

- 2.1性能数据采集（数据从哪来）
  - torch\_npu.profiler（Python API）
    - `_ExperimentalConfig` 配置（AiCMetrics.PipeUtilization）
    - 采集schedule设计（wait / warmup / active / repeat）
  - msprof op命令行：单算子级性能剖析
  - 仿真流水图（离线采集）
  - 辅助分析环境变量：`TRITON_PRINT_AUTOTUNING` / `TRITON_ALL_BLOCKS_PARALLEL` / `TRITON_DEBUG` / `TRITON_BENCH_METHOD`
  - **仓库文档参考**（部分覆盖，按内容相关度排序）：
    - 上板采集与PipeUtilization数据解读（走msprof命令行路径，非 torch\_npu.profiler）：[profiling.md](../debug_guide/profiling.md)
    - `torch_npu.profiler` 计时路径的实现与"设置预热和运行次数"（warmup/repeat，do\_bench\_npu 内部机制）：[do\_bench\_npu\_guide.md](../debug_guide/do_bench_npu_guide.md)
    - `TRITON_BENCH_METHOD="npu"` 下用 `torch_npu.profiler.profile` 采集每个config的片上时间：[06\_autotune\_example.md](../examples/06_autotune_example.md)
  - **覆盖缺口**：`_ExperimentalConfig` 配置、schedule的 wait/active/repeat 在当前仓库文档中尚无直接说明，建议结合CANN《性能调优工具用户指南》与PyTorch profiler官方文档补充
- 2.2关键性能指标解读（数据怎么看）
  - 指标来源：采集阶段的 op\_summary / PipeUtilization.csv（见2.1）
  - `aiv_vec_ratio`（Vector利用率，理想 > 80%）
  - `aiv_mte2_ratio`（搬入占比，理想 < 50%）
  - `aiv_scalar_ratio`（标量利用率，警惕 > 30% 的标量退化）
  - `aic_cube_ratio`（Cube利用率，理想 > 80%）
- 2.3结果可视化与流水图解读（怎么直观呈现）
  - TensorBoard可视化与结果解读（当前仓库无直接覆盖，参考PyTorch profiler官方文档）
  - Chrome Tracing / MindStudio Insight流水图分析
  - 指令时间线诊断规则（断流检测、空白段识别）
- 2.4瓶颈分类与优化方向决策树（看完怎么办）
  - 计算密集型 → 增大分块 / CV融合
  - 访存密集型 → 存算并行 / Load重排序 / 对齐优化
  - 标量退化 → 数据类型优化（避免int64）
  - 同步开销 → 减少同步 / 调整编译参数

***

## 第三篇：调优篇 — 系统化性能优化方法论

### 第3章Grid映射与分核策略：让所有核心都饱和

> **核心理念**：NPU的Grid与物理核强绑定，分核策略直接决定硬件利用率。优化的第一步不是改算法，而是确保每个核心都有足够的工作量。
>
> **（1）Grid与物理核的强绑定关系。** GPU的Grid与物理SM解耦——开发者只需按逻辑任务量设置Grid（如 `cdiv(N, BLOCK_SIZE)`），硬件自动排队调度，可以启动远超SM数量的Block（如A100有108个SM，但Grid可以是数万个Block），硬件的Block Scheduler负责将Block分配到空闲SM。NPU则完全不同：一个Program绑定一个AI Core，Grid与物理核**强绑定**。当Grid超出物理核数时，硬件分批调度（如 910\_95 有 \~40 个AICore，Grid=120时分3批执行），每批切换有额外的设备侧初始化开销（寄存器清零、流水线冲刷），核启动开销急剧增加。因此NPU上的最佳实践是Grid = 物理核数，让每个核通过核内循环持久化处理多个任务块。
>
> **（2）核心数查询API与核型选择。** NPU的AI Core包含1个Cube（矩阵乘单元）+ 2个Vector（向量计算单元）。因此有两种核心数：`num_aicore`（AI Core总数，含Cube+Vector）和 `num_vectorcore`（Vector核数，910\_95 上为aicore的2倍）。查询方式：`import triton.runtime.driver as driver; properties = driver.active.utils.get_device_properties(device); aicore_num = properties["num_aicore"]; vector_num = properties["num_vectorcore"]`。选择规则：**纯Vector算子**（无 `tl.dot`，如RMSNorm、RoPE、Softcap）使用 `num_vectorcore` 分核，充分利用2倍的Vector核数；**CV融合算子**（含 `tl.dot`，如Flash Attention、Fused MatMul）使用 `num_aicore` 分核，因为Cube单元是计算瓶颈。选错核型会导致Vector算子只用一半Vector核，或CV算子无法启用Cube。
>
> **（3）coreDim硬件约束。** NPU的Grid各维度乘积受 `coreDim ≤ 65535 (UINT16_MAX)` 硬件约束——这是硬件调度器用16位整数索引 program\_id 的限制。多维Grid（如 `(blocks_m, q_head, batch)` = `(128, 32, 8)` = 32768）在head数和batch数较大时容易触及上限，编译报错 `coreDim exceeds UINT16_MAX`。展平为一维 `(NUM_CORES,)` 可彻底规避此约束。

- 3.1 Persistent Kernel：从逻辑Grid到物理核的持久化映射
  - **问题**：多维逻辑Grid（如Flash Attention的 `(blocks_m, q_head, batch)`）使每个program只处理一个逻辑块，Grid规模 `blocks_m × q_head × batch` （如128×32×8=32768）远超物理核数，造成频繁的核间调度与切换开销。且多维Grid可能触及 `coreDim ≤ 65535` 的硬件上限
  - **方案**：将Grid展平为一维 `(NUM_CORES,)`，核内通过 `for` 循环以步长 `NUM_CORES` 遍历全部任务块（interleaved分配），并从 `block_idx` 反推原始的多维索引。
  - **协同优化**：
    - 核心数动态获取（`device_properties.get("num_vectorcore")`），不硬编码——910B与950核心数不同
    - 纯Vector算子使用 `num_vectorcore`，CV融合算子用 `num_aicore`
    - 持久化循环天然形成软件流水线——上一轮Vector计算与下一轮MTE2加载重叠

  ```python
  # ── Before: multi-dimensional logical Grid, each program handles one (M-block, head, batch) ──
  @triton.jit
  def fwd_kernel(q_ptr, k_ptr, v_ptr, ...,
                 BLOCK_M: tl.constexpr, ...):
      start_m = tl.program_id(0)    # block index along the M dimension
      start_qh = tl.program_id(1)   # head dimension
      start_b = tl.program_id(2)    # batch dimension
      # ... per-block compute logic ...

  # Call site: Grid far exceeds the number of physical cores
  grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]),
                       q_head, batch_size)
  fwd_kernel[grid](q, k, v, ...)

  # ── After: 1-D physical Grid, each program loops over multiple blocks (interleaved assignment) ──
  @triton.jit
  def fwd_kernel(q_ptr, k_ptr, v_ptr, ...,
                 AICORE_NUM: tl.constexpr, MAX_Q_LEN: tl.constexpr,
                 BATCH_SIZE: tl.constexpr, q_head: tl.constexpr,
                 kv_head: tl.constexpr, BLOCK_M: tl.constexpr, ...):
      pid = tl.program_id(0)
      NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)
      NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head
      # interleaved traversal: stride = AICORE_NUM, each core alternately claims tasks
      for block_idx in range(pid, NUM_BLOCKS, AICORE_NUM):
          # reconstruct multi-dimensional indices from the 1-D block_idx
          task_hz_idx = block_idx // NUM_BLOCKS_M
          start_m = block_idx % NUM_BLOCKS_M
          start_b = task_hz_idx // q_head
          start_qh = task_hz_idx % q_head
          start_kvh = start_qh * kv_head // q_head  # GQA head mapping
          # ... per-block compute logic (same as Before) ...

  # Call site: Grid = number of physical cores, core count obtained dynamically
  import triton.runtime.driver as driver
  properties = driver.active.utils.get_device_properties(device)
  NUM_CORES = properties["num_aicore"]  # CV-fused ops use aicore
  grid = (NUM_CORES,)
  fwd_kernel[grid](q, k, v, ...,
                   AICORE_NUM=NUM_CORES,
                   MAX_Q_LEN=max_seqlen_q, ...)
  ```

  - **案例**：FA v8 fwd/bwd — 3D Grid `(blocks_m, q_head, batch)` → 1D Grid `(AICORE_NUM,)` + interleaved循环
  - **案例**：RoPE — 3D Grid `(seq_blocks, bs, head)` → 1D Grid `(bs,)` + 核内遍历 seq\_len 和head
- 3.2自适应分核：按任务量缩减核数
  - **问题**：当任务总量小于总核数时（如 batch\_size=4而 num\_vectorcore=80），部分核心分不到任务，产生空转——`range(pid, total_tasks, num_cores)` 中 `pid >= total_tasks` 的核心循环体为空，属于无效启动、造成资源空转。
  - **方案**：根据实际任务量动态缩减Grid核心数，确保每个核心至少分到一个完整任务块，避免无效核心参与调度

  ```python
  # ── Before: fixed core count, some cores idle ──
  NUM_CORES = VECTOR_NUM  # 80 cores
  grid = (NUM_CORES,)
  # when total_tasks=4, 76 cores stay idle

  # ── After: shrink dynamically based on the task count ──
  # ensure every core gets at least one task block of size BLOCK_SIZE
  total_tasks = triton.cdiv(numel, BLOCK_SIZE)
  if total_tasks < VECTOR_NUM:
      NUM_CORES = total_tasks  # shrink to the actual number of tasks
  else:
      NUM_CORES = VECTOR_NUM
  grid = (NUM_CORES,)
  ```

  - **案例**：RMSNorm — `n_rows < num_cores` 时缩减Grid到 `n_rows`
- 3.3 SPLIT\_K 拆分：K维度并行化
  - **问题**：当M×N维度不足以饱和所有核心时（如decode场景的小batch，M=1、N=head\_dim=128，只有1×8=8个输出块），大量核心闲置。此时M×N的并行度远小于物理核数
  - **方案**：在K维度上拆分matmul，多个program并行计算部分和，通过原子操作或锁机制累加到最终输出
  - **关键细节**：SPLIT\_K 个program写同一输出地址，必须用 `tl.atomic_add` 累加（无锁）或 `LOCK` 数组 + `tl.store`（有锁）；autotune时需 `reset_to_zero=["output_ptr"]` 避免多次评估累加

  ```python
  # ── Before: a single program processes the full K dimension, M×N parallelism is insufficient ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
      # one program iterates over the full K dimension
      for k in range(0, K, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)
      tl.store(c_ptr + ..., acc)  # direct write, no conflicts

  grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))  # M×N parallelism

  # ── After: split the K dimension, parallel programs + atomic accumulation ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      pid_k = tl.program_id(2)  # index of the K-dimension split
      acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
      # each program only processes its K/SPLIT_K slice
      k_start = pid_k * K // SPLIT_K
      k_end = (pid_k + 1) * K // SPLIT_K
      for k in range(k_start, k_end, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)
      # atomic accumulation: SPLIT_K programs write to the same address
      tl.atomic_add(c_ptr + ..., acc)

  # Grid gains a K dimension: M×N×SPLIT_K parallelism
  grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), SPLIT_K)
  ```

  - **案例**：Fused MatMul bwd\_w — `SPLIT_K` 参数（8/16）拆分K维度，配合 `LOCK_W` 数组处理写冲突
  - **案例**：FA bwd\_qkv — `dq` 用 `tl.atomic_add` 累加多核结果，autotune配 `reset_to_zero=["dq_ptr"]`
- 3.4 Sorted Round-Robin负载均衡：稀疏注意力的任务不均衡问题
  - **问题**：稀疏注意力（Flex Attention / SWA）与GQA场景中，不同任务块的工作量差异极大：causal mask使上三角全零，前几个query block只需处理少量key block，而最后一个query block要处理全部key block。此时若按stride方式遍历任务（`range(pid, total, num_cores)`），各核任务大小混杂不均，导致各核总工作量严重失衡：分配到靠后query block的核满载，分配到靠前的核大量空转。
  - **方案**：主机端先枚举所有任务 → 计算workload（如每个query block需处理的key block数）→ 按workload降序排序 → 以轮询方式分配到各核（round-robin）→ kernel端通过任务张量读取本核的任务列表
  - **原理**：降序排序保证大任务优先被分配，轮询则让每个核在排序序列上等距采样（核0取第1、第C+1、第2C+1...大的任务），因此各核拿到的任务集合在大小分布上高度同构，总工作量趋于均衡；同时避免"最后才遇到一个超大任务，已无处摊平"的情况。

  ```python
  # ── Before: stride traversal inside the kernel, workload heavily imbalanced across cores ──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      pid = tl.program_id(0)
      # interleaved traversal: earlier query blocks have less work, later ones more
      for block_idx in range(pid, total_tasks, num_cores):
          start_m = block_idx % NUM_BLOCKS_M
          # causal mask: larger start_m means more key blocks to process
          num_kv_blocks = start_m + 1  # workload grows linearly with start_m
          for kv in range(num_kv_blocks):
              ...  # early cores run only 1 iteration, later cores run NUM_BLOCKS_M

  # ── After: host-side pre-sorting + round-robin assignment, cores balanced ──
  # Host side: compute each task workload and sort in descending order
  def assign_tasks(total_tasks, num_cores):
      workloads = [compute_workload(t) for t in range(total_tasks)]
      tasks_sorted = sorted(range(total_tasks), key=lambda t: workloads[t],
                            reverse=True)  # descending: heavy tasks first
      # round-robin: core 0 gets tasks_sorted[0], tasks_sorted[num_cores], ...
      core_tasks = [[] for _ in range(num_cores)]
      for i, task in enumerate(tasks_sorted):
          core_tasks[i % num_cores].append(task)  # round-robin
      # flatten into a contiguous tensor for the kernel to read
      core_tasks_ranges = [...]  # per-core start/end offsets of tasks
      core_tasks_flat = tensor([task for core in core_tasks for task in core])
      return core_tasks_ranges, core_tasks_flat

  # Kernel side: read this core assigned tasks through the task tensors
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, core_tasks_ptr, core_tasks_ranges_ptr, ...):
      pid = tl.program_id(0)
      start_offset = tl.load(core_tasks_ranges_ptr + pid * 2)
      end_offset = tl.load(core_tasks_ranges_ptr + pid * 2 + 1)
      # only iterate over this core assigned tasks (already balanced)
      for i in range(start_offset, end_offset):
          block_idx = tl.load(core_tasks_ptr + i)
          start_m = block_idx % NUM_BLOCKS_M
          ...  # total workload across cores converges to balance
  ```

  - **案例**：FlexAttention — 主机端sorted round-robin分配，消除causal mask导致的工作量不均
  - **案例**：SWA prefill — 滑动窗口场景的workload排序分配

### 第4章Tiling与分块策略：在UB约束下找到最优分块

> **核心理念**：BLOCK\_SIZE 不仅是Tiling参数，还直接决定UB占用——选错即 `ub overflow`。Tiling的本质是在UB容量、对齐约束和计算效率之间找到最优平衡点。
>
> **（1）UB容量与预算公式。** NPU的UB（Unified Buffer）是Vector的唯一工作区——所有 `tl.load` 的数据、中间计算结果、`tl.store` 前的数据都在UB。910\_95（A5）的UB硬件为256KB，编译器预留8KB后实际可用 **248KB**；910B（A2）的UB为 **192KB**。与GPU的Shared Memory（开发者手动管理、通过 `__shared__` 或 `num_stages` 间接控制）不同，NPU的UB由编译器全权管理，开发者通过 `BLOCK_SIZE` **间接控制** UB占用。UB预算公式：`所需UB = Σ(每个 tensor 的 BLOCK_SIZE × dtype_bytes) × multibuffer副本数 ≤ UB可用`。例如FP16的RMSNorm kernel，输入 `x` (BLOCK\_SIZE×2B) + weight `w` (BLOCK\_SIZE×2B) + 输出 `y` (BLOCK\_SIZE×2B) + 中间 `mean/var` (32B) ≈ BLOCK\_SIZE×6B + 32B。开启 `multibuffer=True` 后UB可用空间减半（910\_95 为124KB，910B为96KB），因为双缓冲需要两份UB空间。超出UB容量时编译报错 `ub overflow`，必须减小 BLOCK\_SIZE 或减少核内同时存活的tensor数量。
>
> **（2）对齐约束。** NPU有两层对齐要求：**Vector尾轴32B连续对齐**是硬约束——FP16（2B）尾轴元素须为16的倍数，FP32（4B）须为8的倍数，否则DMA引擎无法发起连续搬运；**CV融合算子（含** **`tl.dot`）要求512B对齐**——FP16尾轴元素须为256的倍数，因为Cube单元的矩阵分块粒度更大。非对齐场景需要额外的Mask处理（加载时用mask截断多余元素），但会引入性能开销。此外，单值缓冲区（如归约结果 mean/var/max/sum）逻辑上只需4B，但硬件要求32B对齐，实际分配32B。
>
> **（3）计算效率与分块大小的关系。** 大BLOCK提升Cube利用率和数据复用（矩阵乘的 BLOCK\_M×BLOCK\_N 越大，每个元素被复用的次数越多），但受UB容量限制；小BLOCK增加Grid分核数（更多并行度），但Cube利用率低。NPU上Cube单元的最优分块通常是128（`BLOCK_M=BLOCK_N=BLOCK_K=128` 是Fused MatMul的收敛值），因为Cube的 `mmad` 指令以128×128矩阵为基本计算单元。对于纯Vector算子（无 `tl.dot`），BLOCK\_SIZE 可以更大（如Softcap的 BLOCK\_SIZE=2048），因为Vector的UB压力小于CV融合算子。

- 4.1核内多块循环：提升 MTE2/MTE3/计算流水并行度
  - **问题**：标准Grid模式下每个program只处理一个数据块，Grid规模大（如 `cdiv(numel, BLOCK_SIZE)=1000`）但核内无流水线深度——每个program只做一次load→compute→store，无法利用存算并行
  - **方案**：新增 `BLOCK_NUM` 参数，单program内循环处理多个数据块，Grid相应缩小 `BLOCK_NUM` 倍。核内多块循环减少Grid规模（降低核切换开销），同时实现存算并行（上一块compute与下一块load重叠）

  ```python
  # ── Before: one block per program, large Grid but no intra-core pipeline ──
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

  grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Grid may be very large

  # ── After: each program processes BLOCK_NUM blocks, Grid shrinks ──
  @triton.jit
  def softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap,
                         BLOCK_SIZE: tl.constexpr,
                         BLOCK_NUM: tl.constexpr):   # new multi-block parameter
      pid = tl.program_id(0)
      for i in range(BLOCK_NUM):
          # each i-th iteration processes a block of size BLOCK_SIZE
          block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
          offsets = block_start + tl.arange(0, BLOCK_SIZE)
          mask = offsets < n_elements
          x = tl.load(x_ptr + offsets, mask=mask)
          y = softcap * libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
          tl.store(y_ptr + offsets, y, mask=mask)

  # Grid shrinks by BLOCK_NUM×; the intra-core loop overlaps load and compute
  grid = (triton.cdiv(n_elements, BLOCK_SIZE * BLOCK_NUM),)
  ```

  - **案例**：Softcap — 新增 `BLOCK_NUM` 参数（1/2/4/8/16/32/64），核内循环处理多块，BLOCK\_SIZE×BLOCK\_NUM 决定每program总数据量
- 4.2 1D展平替代2D：小shape场景的索引开销消除
  - **问题**：小shape下2D row-blocked kernel的2D索引开销占比高——`offsets = row[:, None] * stride + col[None, :]` 需要二维广播乘加，当 `n_cols` 较小时（如 head\_dim=128），索引计算的时间接近甚至超过实际计算时间
  - **方案**：改为1D flatten kernel，将二维 `(rows, cols)` 展平为一维 `rows*cols`，消除2D索引，提升内存合并（连续地址访问）

  ```python
  # ── Before: 2D kernel, high indexing overhead for small shapes ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_rows, n_cols, ...,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
      col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
      # 2D indexing: row offset × stride + column offset
      offsets = row[:, None] * n_cols + col[None, :]
      mask = (row[:, None] < n_rows) & (col[None, :] < n_cols)
      x = tl.load(x_ptr + offsets, mask=mask)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N))

  # ── After: 1-D flattened kernel, eliminates 2D indexing ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_elements, ...,
                      BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      # 1-D contiguous offsets, no 2D broadcasting
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_rows * n_cols, BLOCK_SIZE),)
  ```

  - **案例**：GELU / SiLU — `n_cols <= 10240` 或 `n_rows < num_cores` 时切换到1D flatten kernel
- 4.3移除 next\_power\_of\_2：避免不必要的padding
  - **问题**：`next_power_of_2(d)` 将实际维度向上取整到2的幂（如d=300 → 512，d=192 → 256），导致padding浪费UB空间（多分配70% 的UB）和多余循环迭代。更关键的是，当 `BLOCK_DIN = next_power_of_2(Din)` 使 `BLOCK_DIN > Din` 时，归约循环需要执行多次（无法单Pass完成），阻碍算子融合
  - **方案**：当实际维度满足32B对齐要求时（FP16的d须为16的倍数），直接使用实际维度 `D_ROUNDED = d`，消除padding浪费并使循环只执行一次

  ```python
  # ── Before: round up to a power of 2, padding waste + multiple loop iterations ──
  @triton.jit
  def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                         D_ROUNDED: tl.constexpr):
      # D_ROUNDED = triton.next_power_of_2(d)
      # when d=300, D_ROUNDED=512, wasting UB space for 212 extra elements
      offs_d = tl.arange(0, D_ROUNDED)
      mask_d = offs_d < d  # mask needed to clip the padding region
      x = tl.load(x_ptr + row * d + offs_d, mask=mask_d, other=0.0)
      # when D_ROUNDED > d, the reduction loop may run multiple times
      mean = tl.sum(x * x, axis=0) / d
      ...

  # Call site
  D_ROUNDED = triton.next_power_of_2(d)  # 300 → 512

  # ── After: use the real dimension directly, no padding + single loop pass ──
  @triton.jit
  def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                         D_ROUNDED: tl.constexpr):
      # D_ROUNDED = d (when d satisfies 32B alignment)
      offs_d = tl.arange(0, D_ROUNDED)
      # D_ROUNDED == d, no mask needed (already aligned)
      x = tl.load(x_ptr + row * d + offs_d)
      # D_ROUNDED == d, loop runs once → single-pass fusion
      mean = tl.sum(x * x, axis=0) / d
      ...

  # Call site
  D_ROUNDED = d  # use 300 directly (assuming 300 is a multiple of 16, or already aligned)
  ```

  - **案例**：RMSNorm — `D_ROUNDED=triton.next_power_of_2(d)` → `D_ROUNDED=d`，使归约循环只执行一次
  - **案例**：Gate RMSNorm — `BLOCK_DIN=min(3264, next_power_of_2(Din))` → `BLOCK_DIN=Din`，使循环只执行一次，实现单Pass融合
- 4.4 SWA循环结构优化：合并vs拆分
  - **问题**：滑动窗口注意力（SWA）中，全局窗口块（始终满密度，无需local mask）和非全局窗口块（需causal+local mask）的循环结构影响性能。两种循环结构在不同全局窗口占比下性能差异显著
  - **两种策略**（根据全局窗口占比选择）：
    - **合并**（全局窗口占比低时更优）：将两个独立循环合并为一个，用 `tl.where` 统一mask，减少循环数量和Grid开销
    - **拆分**（全局窗口占比高时更优）：将统一循环拆分为两段——全局段无mask开销（满密度直接计算），非全局段只执行causal+local mask

  ```python
  # ── Merge strategy (better when the global-window fraction is low) ──
  # Before: two separate loops, doubled loop overhead
  for kv in range(global_window_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = global_mask  # global-window mask
      qk = tl.where(mask, qk, -inf)
  for kv in range(local_window_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = causal_local_mask  # causal + local mask
      qk = tl.where(mask, qk, -inf)

  # After: merge into one loop, unify the mask with tl.where
  for kv in range(all_kv_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      is_global = (kv < global_threshold)
      # unified mask: global_mask for global segments, causal_local_mask otherwise
      mask = tl.where(is_global, global_mask, causal_local_mask)
      qk = tl.where(mask, qk, -inf)

  # ── Split strategy (better when the global-window fraction is high) ──
  # Before: unified loop, global segments also pay mask-check overhead
  for kv in range(all_kv_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      # checks global_window_check every iteration, even for global segments
      mask = causal_mask & (local_window_check | global_window_check)
      qk = tl.where(mask, qk, -inf)

  # After: split into two segments, global segment has zero mask overhead
  for kv in range(global_blocks):       # segment 1: global window, no mask overhead
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      # global window is fully dense, no mask needed
  for kv in range(local_blocks):        # segment 2: non-global window, causal + local mask
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = causal_mask & local_window_mask
      qk = tl.where(mask, qk, -inf)
  ```

  - **案例**：SWA prefill — 根据全局窗口占比选择合并或拆分策略

### 第5章Autotune调优策略：让调优器替你找到最优配置

> **核心理念**：NPU的最优BLOCK尺寸取决于UB容量、数据类型、矩阵形状和硬件架构的复杂交互，人工试错效率低下。Autotune让调优器自动搜索最优配置，是NPU算子调优的标准武器。
>
> **（1）Config内容差异。** GPU的 `triton.Config` 包含Tiling参数 + `num_stages`（软件流水线深度，1\~4）+ `num_warps`（线程束数量，4/8/16/32）。NPU的Config **仅含Tiling参数**——`num_stages` 和 `num_warps` 在NPU上无意义（NPU无GPU式warp和软件流水线stage），必须移除。NPU的流水线并行通过 `multibuffer`（UB双缓冲）和核内 `for` 循环实现，而非 `num_stages`。此外，NPU的Config可以将编译参数（如 `multibuffer`、`vf_fusion_mode`）作为搜索维度。
>
> **（2）候选值约束。** NPU的BLOCK候选值受UB容量硬约束——大BLOCK提升Cube利用率和数据复用，但超出UB容量则编译失败（`ub overflow`）。经验公式：`最大Tiling元素数 ≈ UB_SIZE / (dtype_bytes × buffer数量)`，如FP16（2B）3 buffers时 248KB/(2×3) ≈ 40960元素。此外，NPU的BLOCK值应满足32B对齐（FP16须为16的倍数），使用2的幂次值（32/64/128/256/512/1024/2048）天然满足对齐要求。非2的幂的值（如100、86）也可使用，但需确保对齐。
>
> **（3）key参数化与缓存。** `key` 参数指定调优触发维度——当key值变化时重新调优，相同值复用缓存。NPU上需要特别注意：不同dtype（fp16 vs bf16）的UB占用不同（bf16和fp16都是2B，但中间计算可能不同），应将dtype加入key；不同算子属性（如FA的 QK\_DIM/V\_DIM/MASK\_FN/SPARSE\_OPT）影响最优Tiling，也应加入key。Autotune的首次运行会尝试所有候选Config，配置数过多会显著增加首次运行时间——建议总配置数控制在20个以内，使用 `prune_configs_by` 提前剪枝不可行配置。

- 5.1 BLOCK尺寸搜索：从小BLOCK起步逐步增大
  - **问题**：固定小BLOCK导致Cube利用率低和数据复用差，但盲目增大可能溢出UB。开发者难以人工判断特定shape下的最优BLOCK值
  - **方案**：从小BLOCK起步，通过Autotune逐步增大搜索最优值。候选值可包含非2的幂以精确匹配实际维度（如序列长度100）

  ```python
  # ── Before: fixed small value, cannot adapt to different shapes ──
  @triton.jit
  def fwd_kernel(..., BLOCK_M: tl.constexpr):
      # BLOCK_M fixed at 32, low Cube utilization for large shapes
      ...

  # ── After: Autotune searches and picks the optimal BLOCK automatically ──
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_M": 32}),
          triton.Config({"BLOCK_M": 64}),
          triton.Config({"BLOCK_M": 128}),
          triton.Config({"BLOCK_M": 256}),
          # use case: non-power-of-2 to exactly match the real dimension
          triton.Config({"BLOCK_M": 100}),  # sequence length is exactly 100
      ],
      key=["QK_DIM", "V_DIM"],  # cache the best config by head_dim
  )
  @triton.jit
  def fwd_kernel(..., BLOCK_M: tl.constexpr):
      ...
  ```

  - **案例**：FA bwd\_q blkM 32→64→128，bwd\_kv blkN 32→64→128，bwd\_preprocess 32→128/256
  - **案例**：Fused MatMul fwd从搜索空间收敛到BM=BN=BK=128（Cube单元最优分块）
  - **案例**：RoPE TOKEN\_BLOCK\_SIZE=3（非2的幂精确匹配）
- 5.2多维组合搜索：从单config到全空间
  - **问题**：单维度搜索无法捕捉 BLOCK\_M × BLOCK\_N × BLOCK\_K 的交互效应——如 BLOCK\_M=128在 BLOCK\_N=32时最优，但在 BLOCK\_N=128时可能 BLOCK\_M=64更优（UB约束）
  - **方案**：从单config扩展到多维组合搜索，使用列表推导生成笛卡尔积

  ```python
  # ── Before: single config, cannot capture dimension interactions ──
  @triton.autotune(
      configs=[triton.Config({"BLOCK_M": 128})],  # only one config
      key=["M", "N"],
  )

  # ── After: multi-dimensional combinatorial search covering the full BM×BN×BK space ──
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK})
          for BM in [128, 256, 512]
          for BN in [64, 128, 256, 512]
          for BK in [32, 64, 128, 512]
      ],  # 3×4×4 = 48 configs
      key=["M", "N", "K"],
  )
  @triton.jit
  def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
      ...
  ```

  - **案例**：SwiGLU — 从单config扩展到BM×BN×BK三维搜索
- 5.3 Key参数化：按形状/dtype 缓存最优config
  - **问题**：不同输入形状/属性的最优config不同，无key参数化时跨形状重复调优（每次shape变化都重新搜索全部Config）。更隐蔽的问题：不同dtype（fp16 vs bf16）的UB占用和计算路径不同，但若dtype不在key中，fp16选出的config可能对bf16不是最优
  - **方案**：按输入形状/属性缓存最优config；将dtype也加入key（通过 `X_ptr.dtype` 引用指针的dtype）

  ```python
  # ── Before: basic key, redundant tuning across shapes/dtypes ──
  @triton.autotune(
      configs=[...],
      key=["n_rows", "n_cols"],  # dtype missing, fp16/bf16 share one config
  )

  # ── After: extended key (dtype + operator attributes) ──
  @triton.autotune(
      configs=[...],
      # add dtype to the key: fp16 and bf16 each cache their own best config
      key=["n_rows", "n_cols", "X_ptr.dtype"],
  )

  # or cache by operator attributes (Flash Attention scenario)
  @triton.autotune(
      configs=get_fwd_configs(),
      # QK_DIM/V_DIM affect UB usage, MASK_FN/SPARSE_OPT affect the compute amount
      key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
  )
  ```

  - **案例**：FA v5 — 按 QK\_DIM/V\_DIM/MASK\_FN/SPARSE\_OPT 缓存
  - **案例**：RMSNorm — 将dtype加入key（fp16 vs bf16 UB占用不同）
- 5.4搜索空间剪枝：提前过滤无效config
  - **问题**：大搜索空间中包含两类无效Config：(1) 会溢出UB的Config（BLOCK\_M×BLOCK\_N 过大），首次运行时编译失败浪费时间；(2) 特定架构上永远不会胜出的Config（如910B上大BLOCK因UB只有192KB而总是不如小BLOCK）。这些无效Config浪费首次运行编译时间
  - **方案一（UB剪枝）**：通过 `prune_configs_by` 的 `early_config_prune` 回调函数，在编译前过滤会溢出UB的Config
  - **方案二（架构剪枝）**：按架构裁剪搜索空间，只保留该架构上可能胜出的Config

  ```python
  # ── UB pruning: filter out Configs that would overflow UB ──
  def _prune_oversized_tiles(configs, named_args, **kwargs):
      # filter Configs whose BLOCK_M × BLOCK_N exceeds UB capacity
      return [c for c in configs
              if c.kwargs['BLOCK_M'] * c.kwargs['BLOCK_N'] <= 1048576]

  @triton.autotune(
      configs=[...],  # original search space
      prune_configs_by={"early_config_prune": _prune_oversized_tiles},
      key=["M", "N"],
  )

  # ── Architecture pruning: trim the search space per architecture ──
  def get_configs():
      all_configs = [
          triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
          for BM in [32, 64, 128, 256]
          for BN in [32, 64, 128, 256]
      ]
      if is_910():  # 910B has only 192KB UB, large BLOCKs easily overflow
          # keep only Configs with BLOCK_M ≤ 128 (winning configs on 910B)
          return [c for c in all_configs if c.kwargs['BLOCK_M'] <= 128]
      return all_configs  # 910_95 has 248KB UB, search the full space

  @triton.autotune(configs=get_configs(), key=["M", "N"])
  ```

  - **案例**：RMSNorm / Int8GEMM — UB溢出config提前过滤
  - **案例**：SWA a2 — 8 configs → 1 config（910B架构剪枝）
- 5.5编译参数加入搜索
  - **问题**：编译参数（如 `vf_fusion_mode`、`multibuffer`）的最优值取决于具体shape，人工选择容易遗漏。特别是 `multibuffer`：开启后UB减半但实现存算并行，某些shape下开启更优（计算密集），某些shape下关闭更优（UB受限需要大BLOCK）
  - **方案**：将编译参数作为Autotune搜索维度，让调优器自动选择

  ```python
  # ── include compile parameters in the search ──
  @triton.autotune(
      configs=[
          # try multibuffer and vf_fusion_mode once each for the same BLOCK
          triton.Config({"BLOCK_M": 8}, multibuffer=True, vf_fusion_mode="max-parallel"),
          triton.Config({"BLOCK_M": 8}, multibuffer=True, vf_fusion_mode="ub-aware-op"),
          triton.Config({"BLOCK_M": 8}, multibuffer=True, vf_fusion_mode="all-op"),
          triton.Config({"BLOCK_M": 8}, multibuffer=False, vf_fusion_mode="max-parallel"),
          triton.Config({"BLOCK_M": 8}, multibuffer=False, vf_fusion_mode="ub-aware-op"),
          triton.Config({"BLOCK_M": 8}, multibuffer=False, vf_fusion_mode="all-op"),
      ],
      key=["M", "N"],
  )
  ```

  - **案例**：RMSNorm — `enable_vf_fusion` 作为搜索维度加入autotune configs（注意：`enable_vf_fusion` 已废弃，不再透传给NPU编译器，请改用 `vf_fusion_mode`）
  - **案例**：RoPE — `multibuffer` 作为搜索维度加入autotune configs
- 5.6 autotune结果磁盘缓存：避免重复调优
  - **问题**：每次运行都重新autotune浪费时间，特别适用于固定shape的生产环境（如推理场景的 seq\_len 固定）
  - **方案**：开启autotune的**磁盘缓存**——`cache_results=True`，或环境变量 `TRITON_CACHE_AUTOTUNING=1` 全局开启。首次调用正常编译并benchmark全部config，把每个config的耗时写入 `TRITON_CACHE_DIR/<hash>/<函数名>.autotune.json`；后续相同 `key` 的调用直接加载缓存、取耗时最短的config，**跳过整个搜索阶段**。注意这是**按key缓存**而非永久固定：`key` 参数取值、dtype、编译选项或config列表任一变化都会使缓存失效并重新调优

  ```python
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": 1024}, multibuffer=True),
          triton.Config({"BLOCK_SIZE": 2048}, multibuffer=True),
          triton.Config({"BLOCK_SIZE": 1024}, multibuffer=False),
      ],
      key=["n"],            # n unchanged → disk cache hit, skip benchmark
      cache_results=True,   # key: enable disk cache
  )
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
      offs = tl.arange(0, BLOCK_SIZE)
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)
  ```

  - **缓存文件内容**：`<函数名>.autotune.json` 保存 `key` 与每个config的实测耗时（中位数/下界/上界），命中后按耗时取 `min`

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

  - **彻底固定配置**：若shape长期不变，可调优一次后把最优config直接写死、去掉 `@triton.autotune`，连首次调优开销也省掉

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
      offs = tl.arange(0, BLOCK_SIZE)
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)

  # pin the optimal config directly (BLOCK_SIZE=1024 + multibuffer=True), skipping the search
  kernel[(grid,)](x, y, n, BLOCK_SIZE=1024, multibuffer=True)
  ```

  - **案例**：RoPE / 推理场景 — seq\_len 固定使 `key` 不变，后续推理直接命中缓存
- 5.7动态序列长度：避免每个新长度都触发Autotune
  - **问题**：在自回归解码、变长输入等整网场景中，切分轴 `seq_len` 可能连续取 `1001、1002、1003……`。如果使用 `key=["seq_len"]`，每个未见过的长度都会产生新的调优缓存项。相同的完整key可以复用已有结果，但开启5.6节的磁盘缓存只能解决相同key的跨进程复用，不能自动让新长度沿用附近长度的配置。
  - **选择原则**：将大量精确长度归并为有限的性能区间，并让kernel在运行时处理真实长度。
    **方案1：长度分桶：用桶值做调优key，用真实长度执行。**
  将长度向上归并到2的幂，例如 `1001、1002、1024` 的桶值均为 `1024`，`1025` 的桶值为 `2048`。也可以根据性能拐点设置不等宽区间。桶值只控制配置复用，不要求将输入padding到桶上界；grid、循环边界和mask仍使用真实长度。

  ```python
  def length_bucket(seq_len):
      return triton.next_power_of_2(max(128, seq_len))

  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": 256}),
          triton.Config({"BLOCK_SIZE": 512}),
          triton.Config({"BLOCK_SIZE": 1024}),
      ],
      key=["seq_bucket"],  # the exact seq_len value does not enter the tuning key
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

  同桶的第一次调用会在当时的真实长度上benchmark，后续直接复用，**不会对桶内每个长度继续搜索**。应选择有代表性的长度预热，并验证桶下界、上界及常见长度的正确性和性能；若桶内性能差异明显，拆分桶，或离线在多个代表点上综合选择配置。所有候选配置都必须对桶内全部长度正确，不能依赖首次调优长度的整除性，也不能省略必要的尾部mask。

  **方案2：启发式分派：离线调优形成规则，线上直接选择配置。**

  在目标硬件、dtype、布局和典型长度上离线测量，找到不同配置的性能交叉点，再将结果写成区间表或条件分支。线上只选择配置并调用普通 `@triton.jit` kernel，不执行 `@triton.autotune` 的搜索；也可以用 `@triton.heuristics` 把选配置的规则直接挂到kernel上，在launch时动态算出 `BLOCK_SIZE`，效果与手动分派等价：

  ```python
  def select_block_size(seq_len):
      if seq_len <= 512:
          return 256
      if seq_len <= 4096:
          return 512
      return 1024  # coverage path for longer inputs

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
      add_one_fixed_kernel[grid](x, y, seq_len)  # BLOCK_SIZE computed by heuristics
      return y
  ```

  **调优缓存与编译缓存需要分别处理**：缩小autotune key只减少benchmark次数。如果将每步变化的真实长度标为 `tl.constexpr`，即使命中调优缓存，JIT仍可能为每个长度编译新版本。上述示例将真实长度保留为运行时参数，并用 `do_not_specialize` 避免按该标量的值或对齐特征进一步特化；它不禁止 `seq_bucket` 作为autotune key。有限的 `BLOCK_SIZE: tl.constexpr` 仍会对应不同编译版本，可在上线前预热各桶或分派分支。`prune_configs_by` 只能减少每次搜索的候选数，无法消除大量精确长度带来的新key。

### 第6章CV流水线优化：让Cube和Vector并行工作

> **核心理念**：NPU的Cube和Vector是物理分离的两个执行单元——不优化则串行执行，流水线气泡严重。CV流水线优化的核心是让Cube和Vector并行工作，消除气泡。
>
> **（1）Cube/Vector 分离架构与数据通路。** 910\_95 的每个AI Core由1 Cube（矩阵乘单元）+ 2 Vector（向量计算单元）+ Scalar组成。`tl.dot` 映射到Cube的 `mmad` 指令，数据路径是 `GM →(MTE2)→ L1 →(MTE1)→ L0A/L0B → Cube → L0C →(fixpipe)→ UB`。关键点：Cube输出先写入 **L0C**（256KB，NZ格式——一种分块压缩格式），需 **fixpipe** 硬件流水线搬运到UB并完成格式转换（NZ→ND，即分块压缩→常规行优先）后Vector才能处理。fixpipe还支持**随路操作**（在搬运同时完成）：bias累加、精度转换（fp32→fp16）、转置——可替代后续Vector操作，是融合后处理的关键硬件特性。A5（910\_95）的fixpipe L0C→UB **直通**（零GM中转，单步完成）；A2（910B）的L0C→**GM**→UB（额外两次GM访问，性能损失显著）。
>
> **（2）PIPE模型与串行执行问题。** NPU的硬件流水线用PIPE枚举标识：`PIPE_M`（Cube计算流水线）、`PIPE_V`（Vector计算流水线）、`PIPE_FIX`（fixpipe流水线）、`PIPE_MTE2`（GM→UB搬入）、`PIPE_MTE3`（UB→GM搬出）。不优化时，Cube计算（PIPE\_M）和Vector计算（PIPE\_V）串行执行——Vector必须等Cube的fixpipe（PIPE\_FIX）完成后才能开始，形成流水线气泡。
>
> **（3）CV流水线优化的核心思想。** 通过**指令重排序**，将没有数据依赖的Cube操作（`tl.dot`）提前到Vector操作之前，使Cube和Vector并行执行：当Cube执行矩阵计算时，Vector同时执行向量计算。优化后的执行时序：Cube先启动 `tl.dot`，Vector随后并行执行独立计算，最后Vector处理依赖Cube结果的操作。

- 6.1指令重排序 + VF融合控制：让Cube提前启动
  - **问题一**：无数据依赖的 `tl.dot` 排在Vector计算之后（原始代码顺序），Cube等待Vector完成后才启动，整个Vector执行期间Cube空闲
  - **方案一**：将无数据依赖的 `tl.dot` 提前到Vector计算之前，使Cube与Vector并行。注意：依赖Cube结果的操作（如 `acc += pv`）必须延后
  - **问题二**：前驱Vector计算与 `tl.dot` 的VF（Vector Function）融合链过长，导致UB溢出或阻止dot调度到Cube（编译器将 `ds` 的计算融合到dot的VF链中，使整个链在Vector执行，Cube无法独立调度）
  - **方案二**：在 `tl.dot` 之前插入 `compile_hint("break_vf")`，打断VF融合链，确保 `ds` 独立计算后 `tl.dot` 单独调度到Cube

  ```python
  # ── Instruction reordering: start Cube earlier ──
  # Before: Vector runs first while Cube waits (Flash Attention softmax loop body)
  m_new = tl.maximum(m, tl.max(s, 1))
  alpha = tl.math.exp2((m - m_new) * qk_scale)  # Vector
  p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
  p_sum = tl.sum(p, 1)       # Vector
  acc *= alpha[:, None]      # Vector
  v = load_if(v_block_ptr, False, True)
  acc += tl.dot(p.to(dtype), v)  # Cube (starts only after all the Vector ops above finish)

  # After: Cube starts early, in parallel with Vector
  m_new = tl.maximum(m, tl.max(s, 1))
  p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
  v = load_if(v_block_ptr, False, True)
  pv = tl.dot(p.to(dtype), v)          # Cube starts early (no dependency on acc)
  p_sum = tl.sum(p, 1)                  # Vector (runs in parallel with Cube)
  alpha = tl.math.exp2((m - m_new) * qk_scale)  # Vector (runs in parallel with Cube)
  acc = acc * alpha[:, None] + pv       # merge: depends on the Cube result, placed last
  l = l * alpha + p_sum
  m = m_new

  # ── break_vf: break the VF fusion chain (Flash Attention backward) ──
  import triton.language.extra.cann.extension as extension

  # Before: ds may get fused into dot VF chain, blocking Cube scheduling
  ds = p * (dp - d[:, None])
  ds = tl.where(mask, ds, 0.0)
  ds = ds.to(dtype)
  dq += tl.dot(ds, k)  # ds VF chain may be too long, causing UB overflow or blocking Cube scheduling

  # After: break the VF chain so ds computes independently and tl.dot dispatches to Cube alone
  ds = p * (dp - d[:, None])
  ds = tl.where(mask, ds, 0.0)
  ds = ds.to(dtype)
  extension.compile_hint(ds, "break_vf")  # break the VF fusion here
  dq += tl.dot(ds, k)  # tl.dot dispatches independently to Cube
  ```

  - **案例**：FA v4/v8 fwd — `tl.dot(p, v)` 提前到 `p_sum` / `alpha` 之前
  - **案例**：FA bwd\_q/bwd\_qkv — `ds` 计算后插入 `break_vf`，确保 `tl.dot` 调度到Cube
- 6.2 CV协同配置：子块绑定与完整七参数
  - **问题**：CV融合算子需要Cube和Vector以1:2比例协同（匹配1 Cube + 2 Vector架构），但默认配置不启用协同——Cube和Vector各自独立调度，无法实现流水线并行。此外，CV融合算子需要一组完整的编译参数来控制CV协同行为
  - **方案一**：CV融合算子启用 `enable_auto_bind_sub_block=True`（自动绑定子块，实现CV 1:2协同），纯Vector算子禁用（无Cube需要协同）
  - **方案二**：CV融合算子配置完整的七参数集，精确控制CV协同、多缓冲、同步策略等

  ```python
  # ── sub-block binding decision ──
  # CV-fused ops (with tl.dot): enable sub-block binding for CV 1:2 cooperation
  fwd_kernel[grid](..., enable_auto_bind_sub_block=True)
  # pure Vector ops (no tl.dot): disable sub-block binding
  rmsnorm_kernel[grid](..., enable_auto_bind_sub_block=False)

  # ── full seven-parameter CV config (for CV-fused ops only) ──
  fwd_kernel[grid](
      ...,
      enable_mixed_cv=True,                              # enable mixed Cube/Vector scheduling
      sync_solver=True,                                  # auto-solve CV sync points
      enable_auto_bind_sub_block=True,                   # auto-bind sub-blocks (CV 1:2 cooperation)
      set_workspace_multibuffer=2,                       # workspace double buffering
      limit_auto_multi_buffer_of_local_buffer="no-limit", # do not limit UB multi-buffer allocation
      enable_flatten=False,                              # must be False for CV fusion (cannot flatten)
      multibuffer=True,                                  # UB double buffering, load-compute overlap
  )

  # ── pure Vector op config (only 2 parameters needed) ──
  rmsnorm_kernel[grid](
      ...,
      multibuffer=True,
      limit_auto_multi_buffer_of_local_buffer="no-limit",
  )
  ```

  - **案例**：FA v8全系列 — 完整七参数配置
  - **案例**：FA bwd\_q — 从 `enable_auto_bind_sub_block=False` 改为启用
- 6.3 Scale融合：减少FixPipe后处理
  - **问题**：`tl.dot` 之后的scale乘法、累加等操作在fixpipe阶段或Vector单元执行，增加Vector单元压力。特别是online softmax中的 `alpha` 缩放 + PV matmul + 累加三步操作，每步都占用Vector周期
  - **方案一（Scale融合到输入）**：将dot之后的scale乘法移到dot输入中，减少一次Vector乘法
  - **方案二（Accumulator-in-dot）**：将alpha缩放 + PV matmul + 累加融合为单条 `tl.dot` 三参数调用 `tl.dot(p, v, acc)`，利用Cube的硬件累加器
  - **方案三（k\_scale post-dot）**：将 k\_scale 从pre-dot（dot前对k乘scale）移到post-dot（dot后与结果融合），消除额外Vector乘法

  ```python
  # ── fuse Scale into the input (SDPA backward) ──
  # Before: scale after the dot, one extra Vector multiply
  dq += tl.dot(ds, k) * scale
  # After: fold scale into ds (the dot input), removing the post-dot multiply
  ds = ds * scale
  dq += tl.dot(ds, k)

  # ── Accumulator-in-dot (SDPA forward online softmax) ──
  # Before: scale + matmul + accumulate (three steps, three Vector/Cube ops)
  block_o = alpha[:, None] * block_o + tl.dot(block_p, block_v)
  # After: fused into a single three-argument dot (Cube hardware accumulator does scale + add)
  block_o = tl.dot(block_p, block_v, block_o * alpha[:, None])

  # ── k_scale post-dot（LightningIndexer）──
  # Before: pre-dot Vector multiply (scale the whole k matrix)
  k = k * k_scale[:, None]
  qk = tl.dot(q, tl.trans(k))
  # After: fold scale into the following ReLU (only scales the qk result, fewer elements)
  qk = tl.dot(q, tl.trans(k))
  relu_qk = tl.maximum(qk * k_scale[None, :], 0.0)
  ```

  - **案例**：SDPA fwd — accumulator-in-dot融合
  - **案例**：SDPA bwd — scale融合到matmul输入
  - **案例**：LightningIndexer — k\_scale post-dot融合

### 第7章 存算并行与软件流水线：让数据搬运与计算重叠

> **核心理念**：NPU的数据搬运由专用DMA引擎执行，与计算单元物理分离。存算并行的核心是利用这一物理分离，让MTE2（搬入）与 Vector/Cube 计算**自动重叠**。
>
> **（1）专用DMA引擎与物理分离。** GPU上 `tl.load` / `tl.store` 通过SM内部的 LD/ST 队列执行，与计算共享Warp调度资源——需要 `cp.async` + `num_stages` 显式管理异步拷贝才能实现存算重叠，否则load和compute串行执行。NPU则有**物理独立的专用DMA引擎**：MTE2（GM↔L1/UB，数据搬入引擎）和MTE3（UB→GM/L1，数据搬出引擎），与 Vector/Cube 计算单元是不同的硬件Pipeline（PIPE\_MTE2/PIPE\_MTE3 vs PIPE\_V/PIPE\_M）。这意味着DMA搬运和计算**天然并行**——当MTE2搬入下一块数据时，Vector可以同时处理当前块数据，无需开发者显式管理异步操作。
>
> **（2）multibuffer双缓冲机制。** `multibuffer=True` 启用UB双缓冲（Ping-Pong）：将UB分为两份，一份供当前块计算（Vector读取），另一份供下一块搬入（MTE2写入），两份交替使用。这使得下一块的MTE2搬入与当前块的Vector计算**自动重叠**——无需修改代码逻辑，只需开启参数。权衡：开启后UB可用空间减半（910\_95 从248KB → 124KB，910B从192KB → 96KB），可能需要减小 BLOCK\_SIZE。910\_95 的multibuffer **默认关闭**（需显式 `multibuffer=True`），910B默认开启。
>
> **（3）核内for循环——存算并行的标准模式。** 持久化kernel（Grid = 物理核数）的核内 `for` 循环天然形成软件流水线——上一轮compute与下一轮load自动重叠。这是因为循环内 `tl.load`（MTE2）和 `tl.dot`/计算（Cube/Vector）分属不同PIPE，编译器自动调度它们并行执行。循环不变量外提（hoisting）和指针基址+offset重算（避免 `+=`）直接影响DMA并行效率——循环内重复加载不变量或使用 `+=` 引入RAW依赖都会阻止MTE2并行发射。

- 7.1核内循环 + 循环不变量外提：存算并行的基础模式
  - **问题**：标准Grid模式下每program一个块，无核内流水线深度（只做一次load→compute→store，无法重叠）；循环内重复加载不变量（如weight、col\_offsets）是纯浪费——每轮重新发起MTE2搬运相同数据
  - **方案一**：用核内for循环替代大Grid，使上一轮compute与下一轮load重叠（持久化kernel模式）
  - **方案二**：将循环内不变的计算/加载提到循环外（hoisting），只加载一次，循环内直接复用

  ```python
  # ── intra-core loop for load-compute overlap (RMSNorm persistent kernel) ──
  # Before: one block per program, no intra-core pipeline
  @triton.jit
  def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...):
      pid = tl.program_id(0)
      # processes a single row, no loop
      row = pid
      x = tl.load(x_ptr + row * stride + offs_d)
      ...  # compute
      tl.store(y_ptr + row * stride + offs_d, y)

  grid = (n_rows,)  # Grid may far exceed the number of physical cores

  # After: persistent kernel, intra-core loop handles multiple rows
  @triton.jit
  def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                     NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      # intra-core loop: previous compute overlaps with next load automatically
      for row in range(pid, n_rows, NUM_CORES):
          x = tl.load(x_ptr + row * stride + offs_d)  # this iteration load (MTE2)
          # previous compute (Vector) runs in parallel with this load
          mean = tl.sum(x * x, axis=0) / D
          rstd = 1.0 / tl.sqrt(mean + eps)
          y = (x * rstd[:, None]) * w + b
          tl.store(y_ptr + row * stride + offs_d, y)

  grid = (NUM_CORES,)  # Grid = number of physical cores

  # ── loop-invariant hoisting (RMSNorm weight load) ──
  # Before: reloads weight every iteration (loop invariant)
  for row in range(pid, n_rows, NUM_CORES):
      w = tl.load(w_ptr + offs_d)  # weight reloaded every iteration (wasteful)
      x = tl.load(x_ptr + row * stride + offs_d)
      y = (x * rstd[:, None]) * w + b
      tl.store(y_ptr + ..., y)

  # After: load weight once outside the loop, reuse it inside
  w = tl.load(w_ptr + offs_d)  # loaded once outside the loop
  col_offsets = tl.arange(0, D_ROUNDED)  # computed once outside the loop
  for row in range(pid, n_rows, NUM_CORES):
      x = tl.load(x_ptr + row * stride + col_offsets)
      y = (x * rstd[:, None]) * w + b
      tl.store(y_ptr + ..., y)
  ```

  - **案例**：RMSNorm — 持久化kernel + weight/col\_offsets 循环外加载
- 7.2循环手动展开：双路并行增加流水线深度
  - **问题**：单路循环的流水线深度有限——每轮只有一个load和一个compute，load/compute 重叠不充分（MTE2搬入一块后Vector计算一块，MTE2空闲等待Vector完成）
  - **方案**：每轮处理2个（或更多）数据块，双路 load/compute/store——路0的compute与路1的load重叠，MTE2和Vector都满载

  ```python
  # ── Before: single-lane loop, limited pipeline depth ──
  @triton.jit
  def rmsnorm_kernel(x_ptr, ..., NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      for row in range(pid, n_rows, NUM_CORES):
          x = tl.load(x_ptr + row * stride + offs_d)  # single-lane load
          mean = tl.sum(x * x, axis=0) / D             # single-lane compute
          y = (x * rstd[:, None]) * w
          tl.store(y_ptr + row * stride + offs_d, y)   # single-lane store

  # ── After: two-lane unrolling, lane 0 compute overlaps lane 1 load ──
  @triton.jit
  def rmsnorm_kernel(x_ptr, ..., NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      # processes 2 rows per iteration, stride doubled
      for row in range(pid * 2, n_rows, NUM_CORES * 2):
          # lane 0 and lane 1 loads can issue in parallel (no dependency)
          x_0 = tl.load(x_ptr + (row) * stride + offs_d)      # lane 0 load
          x_1 = tl.load(x_ptr + (row + 1) * stride + offs_d)  # lane 1 load
          # lane 0 compute (overlaps with lane 1 load)
          mean_0 = tl.sum(x_0 * x_0, axis=0) / D
          rstd_0 = 1.0 / tl.sqrt(mean_0 + eps)
          y_0 = (x_0 * rstd_0[:, None]) * w
          # lane 1 compute
          mean_1 = tl.sum(x_1 * x_1, axis=0) / D
          rstd_1 = 1.0 / tl.sqrt(mean_1 + eps)
          y_1 = (x_1 * rstd_1[:, None]) * w
          # two-lane store
          tl.store(y_ptr + (row) * stride + offs_d, y_0)
          tl.store(y_ptr + (row + 1) * stride + offs_d, y_1)
  ```

  - **案例**：RMSNorm — 每轮处理2行（`pid*2` 步长），双路 load/compute/store
- 7.3 multibuffer启用：Ping-Pong双缓冲
  - **问题**：当不启用Ping-Pong双缓冲时，load和compute串行执行——MTE2搬入完成后Vector才能开始计算，MTE2在Vector计算期间空闲
  - **方案**：当前multibuffer已默认开启，或者通过设置 `multibuffer=True` 的方式手动启用，UB双缓冲使下一tile的内存加载与当前tile的计算重叠。Ping-Pong模式：buffer A供当前tile计算，buffer B供下一tile搬入，交替使用
  - **权衡**：开启后UB可用空间减半（248KB → 124KB），需评估对 BLOCK\_SIZE 的影响——可能需要减小 BLOCK\_SIZE 或减少核内同时存活的tensor

  ```python
  # ── Before: multibuffer explicitly disabled (multibuffer=False), load and compute are serial ──
  rope_kernel[grid](
      in_ptr, pos_ptr, cu_seqlens, out_ptr,
      head, base, DIM, max_seq_len, REVERSE,
      multibuffer=False,  # explicitly disabled; default is True (on), disable manually for serial execution
  )

  # ── After: explicitly enable multibuffer ──
  rope_kernel[grid](
      in_ptr, pos_ptr, cu_seqlens, out_ptr,
      head, base, DIM, max_seq_len, REVERSE,
      multibuffer=True,  # explicitly enable Ping-Pong double buffering
  )
  ```

  - **案例**：RoPE / Softcap / FA / RMSNorm — 显式开启multibuffer
- 7.4指针依赖消除：避免循环内 `+=` 引入RAW依赖
  - **问题**：循环内 `ptrs += BLOCK_SIZE` 引入Read-After-Write（RAW）依赖——当前迭代的指针值依赖前一次迭代的写入结果，导致编译器无法重排指令，MTE2 load指令必须等上一轮 `+=` 完成后才能发射，阻塞流水线
  - **方案**：改为每轮独立计算offset（`base + k * BLOCK_SIZE`），消除迭代间RAW依赖——每轮的指针地址只依赖循环变量 `k` 和循环外常量 `base`，编译器可自由重排load指令

  ```python
  # ── Before: += introduces a RAW dependency, blocking parallel MTE2 issue ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, ...):
      # compute base addresses outside the loop
      a_ptrs = a_ptr + (offset_m[:, None] * K + offset_k[None, :])
      b_ptrs = b_ptr + (offset_k[:, None] * N + offset_n[None, :])
      for k in range(0, tl.cdiv(K, BLOCK_K)):
          a = tl.load(a_ptrs)  # load must wait for the previous a_ptrs += to finish before issuing
          b = tl.load(b_ptrs)
          acc += tl.dot(a, b)
          a_ptrs += BLOCK_K       # RAW: next load waits for this one to finish
          b_ptrs += BLOCK_K * N   # RAW: next load waits for this one to finish

  # ── After: recompute the offset every iteration, no RAW dependency ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, ...):
      # base addresses computed outside the loop (invariants)
      a_ptrs_base = a_ptr + (offset_m[:, None] * K + offset_k[None, :])
      b_ptrs_base = b_ptr + (offset_k[:, None] * N + offset_n[None, :])
      for k in range(0, tl.cdiv(K, BLOCK_K)):
          # recompute the address per iteration: base + k * stride (no cross-iteration dependency)
          a_ptrs = a_ptrs_base + k * BLOCK_K
          b_ptrs = b_ptrs_base + k * BLOCK_K * N
          a = tl.load(a_ptrs)  # no RAW, can issue early
          b = tl.load(b_ptrs)
          acc += tl.dot(a, b)
  ```

  - **案例**：Fused MatMul fwd — `a_ptrs_base + k * BLOCK_SIZE_K` 替代循环内 `+=`
  - **案例**：RMSNorm — 行方向offset每轮独立计算
- 7.5高级流水线：parallel循环 + Preload/Cool-down
  - **问题一**：无携带依赖的循环（循环内各轮独立）未被编译器识别为可并行，多个GM load串行发射——编译器保守地按顺序发射各轮load，即使它们之间无数据依赖
  - **方案一**：用CANN extension的 `parallel` 标记无携带依赖的循环，提示编译器并行发射各轮load
  - **问题二**：逐块串行发射Cube任务，流水线深度不足——每轮只发射一个 `tl.dot`，Cube单元在fixpipe期间空闲
  - **方案二**：外层循环多发射 `preload` 个Cube任务（预加载多块数据），最后 `preload` 次循环处理cool-down（等待Cube结果回写）

  ```python
  # ── parallel loop: parallelize multiple dependency-free GM loads ──
  import triton.language.extra.cann.extension as extension

  # Before: ordinary loop, compiler issues each load serially
  for page_iter in range(PAGE_AGGREGATION_NUM):
      k_slice = tl.load(k_ptr + page_iter * stride)  # serial load

  # After: marked as parallel, compiler issues each load in parallel
  for page_iter in extension.parallel(0, PAGE_AGGREGATION_NUM):
      k_slice = tl.load(k_ptr + page_iter * stride)  # parallel load

  # ── Preload + Cool-down: issue extra Cube tasks ──
  preload = 3  # preload 3 blocks
  # the outer loop issues preload extra Cube tasks
  end_block = end_block + preload  # extend the loop range to issue 3 extra blocks
  for sq_loop in range(start, end_block, step):
      if sq_loop < last_loop:
          # normal path: issue a Cube task + preload the next block
          qk = tl.dot(q, k)  # Cube task
          k_next = tl.load(...)  # preload the next block
      else:
          # cool-down: the last preload iterations wait for Cube results to drain
          # issue no new Cube tasks, only consume the already-issued results
          acc = acc + qk_result
  ```

  - **案例**：FA/SWA page aggregation — parallel循环并行化多个GM load
  - **案例**：fa-opt/fast\_fa\_fwd — preload=3多发射 + cool-down

### 第8章 算子融合：消除不必要的内存往返

> **核心理念**：Memory-Bound算子的性能瓶颈在于GM访问带宽。算子融合通过将多个操作合并到单个kernel中，让中间数据保留在寄存器/UB 中，消除GM写回与重新加载。
>
> **（1）GM往返的代价。** 未融合的算子链中，每个操作的输出都要写回GM（Global Memory，HBM），下一个操作再从GM读回——每次GM往返消耗数百到数千个时钟周期（GM带宽约1-2 TB/s，而UB带宽约30-50 TB/s，差距20-30倍）。以 `RMSNorm(x) * gated` 为例，未融合时：kernel 1计算 `gated = silu(x @ w_gate)` → 写回GM → kernel 2读回 `gated` → 计算 `rmsnorm(gated)` → 写回GM。两次GM往返消耗大量带宽。融合后，`gated` 全程保留在寄存器/UB 中，GM访问次数从4次（2读 + 2写）降为2次（1读 + 1写）。
>
> **（2）融合的层次。** NPU上的算子融合从低到高有四个层次：**逐元素融合**（最简单，如 `x * sigmoid(x)` 融合为silu，在同一Vector指令流中完成）→ **多kernel融合**（消除kernel启动开销和中间GM访问，如反向QKV三合一）→ **Pass融合**（消除scratch中间存储，如Two-Pass → Single-Pass，中间数据全程在寄存器）→ **计算单元升级融合**（Vector → Cube，如逐Q head的Vector乘改为按KV head分组的Cube矩阵乘）。
>
> **（3）NPU特有的融合手段。** 除了通用融合手段，NPU还有两种独特融合方式：**3D Block Pointer融合**——用3D block\_ptr `(BLOCK, Q_HEAD_NUM, V_DIM)` 一次加载所有head的数据，利用硬件DMA描述符的多维搬运能力，消除逐head串行加载；**布局一致性融合**——统一数据布局（如全程D-T布局），消除kernel内的 `tl.trans` 转置操作（每次转置需要一次UB内数据搬运 + 格式转换，开销显著）。

- 8.1多Kernel融合：消除启动开销与中间GM访问
  - **问题**：多个独立kernel串行执行，每个kernel的输出写回GM，下一个kernel再读回；更严重的是，前端计算（如反向传播中的 Q\@K^T和softmax概率矩阵P）在多个kernel中重复执行——bwd\_q kernel计算P，bwd\_kv kernel又计算一遍P
  - **方案**：将多个独立kernel融合为单kernel，前端计算只执行一遍，中间数据全程在寄存器/UB
  - **三种典型场景**：
    - 反向QKV三合一：bwd\_q + bwd\_kv → bwd\_qkv（dq用 atomic\_add 或per-core workspace处理多核写冲突）
    - 逐元素操作融合：GELU(x) + GELU(x)\*y → 单kernel（消除中间GELU(x) 的GM往返）
    - softmax + 累积采样：softmax + multinomial → 单kernel（消除概率矩阵的GM往返）

  ```python
  # ── fused backward QKV (Flash Attention backward) ──
  # Before: two separate kernels, the front-end compute (P matrix) runs twice
  # kernel 1: bwd_q → computes P = softmax(Q@K^T) → dq = P @ (do - sum) @ K
  bwd_q_kernel[grid](q, k, v, do, dq, ...)  # P is computed here
  # kernel 2: bwd_kv → recomputes P → dk = P^T @ do @ Q, dv = P^T @ do @ V
  bwd_kv_kernel[grid](q, k, v, do, dk, dv, ...)  # P is computed again

  # After: single kernel, the front-end compute runs only once
  # bwd_qkv → dq, dk, dv are produced together
  # P is computed once; dq/dk/dv share the P matrix
  bwd_qkv_kernel[grid](
      q, k, v, do,
      dq, dk, dv,  # fused three-in-one outputs
      ...,
      # dq uses atomic_add to handle multi-core write conflicts (multiple programs write the same dq address)
  )
  # autotune needs reset_to_zero=["dq_ptr"] to avoid accumulation across evaluations
  ```

  - **案例**：FA v8 bwd — `fused bwd_qkv_kernel`，dq用 atomic\_add
  - **案例**：GELU+Mul — 两个逐元素操作融合为单kernel
  - **案例**：Sample — `_top_k_sample_kernel` 融合softmax + multinomial
- 8.2 Two-Pass → Single-Pass融合：消除scratch中间存储
  - **问题**：两趟计算中，Pass 1的中间结果写回GM（scratch buffer），Pass 2再读回继续计算——两次GM访问（一次写 + 一次读），且scratch buffer占用额外显存。以Gate RMSNorm为例：Pass 1计算 `gated = silu(x @ w_gate)` → 写回GM scratch → Pass 2读回scratch → 计算 `rmsnorm(gated)`
  - **方案**：融合为单趟，中间数据（gated值）全程在寄存器/UB，消除scratch GM往返和额外显存占用

  ```python
  # ── Before: two passes, gated written back to a GM scratch ──
  # Pass 1: compute gated → write to scratch (GM)
  gated = silu(x @ w_gate)  # front-end compute
  torch.empty_like(gated).copy_(gated)  # write back to GM scratch

  # Pass 2: read scratch → compute RMSNorm
  gated_from_gm = load_from_scratch()  # read back from GM
  mean = tl.sum(gated_from_gm ** 2, axis=0) / D
  y = gated_from_gm * rsqrt(mean + eps) * weight

  # ── After: single pass, gated stays in registers/UB throughout ──
  # gated is not written back to GM; passed to RMSNorm directly from registers
  gated = silu(x @ w_gate)  # front-end compute, result lives in registers
  # apply RMSNorm directly on the in-register gated (no GM round trip)
  mean = tl.sum(gated ** 2, axis=0) / D  # gated is in registers, zero GM access
  y = gated * rsqrt(mean + eps) * weight
  ```

  - **案例**：Gate RMSNorm — 两趟融合为单趟，gated值全程保留在寄存器
  - **案例**：RMSNorm单pass — 配合 `BLOCK_DIN=Din`（移除 next\_power\_of\_2）使归约循环只执行一次
- 8.3维度融合：多Head一次加载 + 头循环展平
  - **问题一**：逐head加载数据，核内head循环串行执行——每个head独立发起GM load，MTE2搬运串行，且head循环的循环开销（索引计算、边界检查）重复执行
  - **方案一（3D Block Pointer）**：用3D block\_ptr `(BLOCK, Q_HEAD_NUM, V_DIM)` 一次加载所有head，利用硬件DMA描述符的多维搬运能力，单次MTE2完成所有head数据搬入
  - **问题二**：核内头循环串行处理各KV head，无法利用核间并行
  - **方案二（头展平到Grid）**：将核内头循环展平到Grid维度，`h = pid % num_kv_heads`，让不同核并行处理不同head

  ```python
  # ── 3D Block Pointer: load all heads at once (RoPE) ──
  # Before: per-head loads, serial head loop inside the kernel
  for head in range(num_heads):
      x = tl.load(make_block_ptr(
          base=in_ptr + begin * head * DIM + head * DIM,
          shape=(len, DIM),          # 2D: single head
          strides=(head * DIM, 1),
          block_shape=(BLOCK_M, DIM // 2),
      ))
      # ... rotation compute ...

  # After: 3D block_ptr loads all heads at once
  x = tl.load(make_block_ptr(
      base=in_ptr + begin * head * DIM,
      shape=(len, head, DIM),        # 3D: all heads
      strides=(head * DIM, DIM, 1),  # 3-D strides
      block_shape=(BLOCK_M, head, DIM // 2),  # load all heads at once
      order=(2, 1, 0),               # innermost dim DIM is contiguous
  ))
  # ... rotation compute (3D broadcast) ...

  # ── flatten heads onto the Grid: use inter-core parallelism (KV Cache) ──
  # Before: heads processed serially inside the kernel
  grid = (num_chunks,)
  for h in range(num_kv_heads):  # serial inside the kernel
      process(h)

  # After: Grid dimension expanded, each core handles a different head in parallel
  grid = (num_chunks * num_kv_heads,)
  h = pid % num_kv_heads  # derive the head index from pid
  process(h)
  ```

  - **案例**：FA v8 bwd\_preprocess — 3D block\_ptr 融合所有head的归约
  - **案例**：KV Cache — 头维度展平到Grid
- 8.4计算单元升级融合：Vector → Cube
  - **问题**：GQA decode场景下（GROUP\_SIZE 个Q head共享一组KV），逐Q head的Vector逐元素乘 + 归约效率低——每个Q head独立做 `qk = tl.sum(q[None, :] * k, axis=...)`，Vector单元逐元素乘 + 归约，Cube单元空闲
  - **方案**：将逐Q head的Vector乘法改为按KV head分组的Cube矩阵乘，GROUP\_SIZE 个Q head融合为一个矩阵乘的M维度——Cube单元的 `mmad` 指令一次完成 GROUP\_SIZE 个Q head与KV的点积

  ```python
  # ── Before: Vector elementwise multiply + reduction (GQA decode) ──
  # process Q heads one by one; the Vector unit does elementwise multiply + sum reduction
  for q_head in range(GROUP_SIZE):
      q = tl.load(q_ptr + q_head * stride)  # a single Q head
      # Vector elementwise multiply + reduction (Cube idle)
      qk = tl.sum(q[None, :] * k, axis=head_dim)  # Vector multiply + sum
      # ... subsequent softmax ...

  # ── After: Cube matmul, GROUP_SIZE Q heads fused ──
  # group by KV head; GROUP_SIZE Q heads fuse into the M dimension of the matmul
  q_group = tl.load(q_ptr + kv_head * GROUP_SIZE * stride)  # GROUP_SIZE Q heads
  # Cube matmul: dot products of GROUP_SIZE Q heads with K in one shot
  qk = tl.dot(q_group, tl.trans(k))  # Cube mmad, task count reduced by GROUP_SIZE×
  # ... subsequent softmax ...
  ```

  - **案例**：FA/SWA decode — GQA group-fusion + Cube grid
- 8.5 Mask预计算融合 + Group融合 + 布局一致性
  - **问题一**：运行时 mask\_fn 逐元素计算mask开销大——每个program在kernel内动态计算causal mask、local mask、attention arg mask的组合，占用大量Vector周期
  - **方案一**：预计算 mask\_tensor（主机端CPU计算），kernel内直接load预计算结果，将Vector计算转为MTE2搬入
  - **问题二**：逐group循环调用RMSNorm有kernel启动开销——每个group独立启动一个kernel，N个group有N次启动开销
  - **方案二**：单kernel融合所有group，通过 `head_to_group` 查找表 + `tl.where` 权重选择，在核内循环中处理不同group的不同weight
  - **问题三**：kernel内 `tl.trans` 转置操作昂贵——每次转置需要一次UB内数据搬运 + 格式转换
  - **方案三**：统一数据布局，全程D-T布局一致（输入、权重、输出都用同一布局），消除所有转置

  ```python
  # ── Mask precomputation (Flash Attention) ──
  # Before: elementwise mask computed at runtime (Vector compute inside the kernel)
  @triton.jit
  def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset):
      tril_causal = q_offset[:, None] >= k_offset[None, :]
      attn_args_mask = (q_attn_arg == k_attn_arg) | (k_attn_arg == 0)
      return (tril_causal & attn_args_mask) | (q_offset == k_offset)

  mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n)  # Vector compute

  # After: load the precomputed mask directly (MTE2 move inside the kernel)
  # precompute mask_tensor on the host [batch, max_q_len, max_k_len]
  mask = tl.load(mask_block_ptr)  # MTE2 move, zero Vector compute

  # ── Group RMSNorm fusion ──
  # Before: per-group loop, N kernel launches
  for group in range(num_groups):
      rmsnorm_kernel[grid](x[group], weight[group])  # one kernel launch each time

  # After: a single kernel fuses all groups
  @triton.jit
  def group_rmsnorm_kernel(x_ptr, weight_ptr, head_to_group_ptr, ...):
      pid = tl.program_id(0)
      head = pid  # each program handles one head
      group = tl.load(head_to_group_ptr + head)  # look up which group the head belongs to
      # use tl.where to select the weight of the corresponding group
      w = tl.load(weight_ptr + group * stride + offs_d)
      ...  # RMSNorm compute

  # ── Conv1D layout consistency ──
  # Before: 3 transposes after loads (input, weight, output each once)
  x = tl.trans(tl.load(x_ptr + ...))   # transpose 1: input
  w = tl.trans(tl.load(w_ptr + ...))   # transpose 2: weight
  result = tl.dot(x, w)
  out = tl.trans(result)                # transpose 3: output
  tl.store(out_ptr + ..., out)

  # After: D-T layout consistent throughout, no transposes
  x = tl.load(x_ptr + ...)   # D-T layout load, no transpose
  w = tl.load(w_ptr + ...)   # D-T layout load, no transpose
  result = tl.dot(x, w)      # D-T layout matmul
  tl.store(out_ptr + ..., result)  # D-T layout store, no transpose
  ```

  - **案例**：FA v5/v8 bwd — mask\_tensor 替代 mask\_fn
  - **案例**：GroupRMSNorm — `_group_rmsnorm_interleaved_kernel`
  - **案例**：Conv1D — 4次trans → 全程布局一致

### 第9章Mask计算优化：最小化边界处理开销

> **核心理念**：Mask操作是纯开销——不产生有用的计算结果，只为保证边界正确性。Mask优化的目标是系统性地消除不必要的mask计算，将必要的mask开销降到最低。
>
> **（1）Mask的NPU代价。** 每条mask指令占用Vector执行周期——mask的布尔比较（如 `offs < N`）、布尔运算（如 `mask1 & mask2`）、类型转换（如 `mask.to(float)`）都在Vector单元执行，与有用计算竞争Vector资源。更隐蔽的代价：乘法mask（`(1 - mask) * -inf`）需要类型转换 + 乘法 + 减法三条指令，而 `tl.where(mask, value, -inf)` 只需一条选择指令。此外，mask的布尔运算结果在NPU上是int1类型，参与浮点运算时需要显式类型转换，否则可能触发隐式转换的不确定行为。
>
> **（2）三个优化层次。** Mask优化有三个层次，效率从低到高：**实现方式优化**——将乘法mask改为 `tl.where`，将多mask组合改为预计算单mask，减少指令数量；**预计算与缓存**——将运行时mask计算移到主机端CPU（kernel外），kernel内只需一次MTE2 load搬入预计算结果，将Vector计算转为DMA搬入；**冗余消除**——利用对齐信息和constexpr编译期常量，完全消除mask代码（死代码消除），这是最高效的——完全消除mask代码而非优化其执行。
>
> **（3）冗余mask的常见场景。** 当 `BLOCK_SIZE = HEAD_DIM` 时，尾维mask（`offs_k < HEAD_DIM`）恒为True，是死代码；当Grid已保证pid不越界时，`% M` 取模运算（防止pid越界）是冗余的；当mask已覆盖边界或BLOCK整除维度时，额外的边界检查mask是重复的。这些冗余mask可以通过 `tl.constexpr` 编译期常量控制，让编译器直接消除整段代码。

- 9.1 Mask实现方式优化：tl.where替代乘法mask
  - **问题**：`(1 - mask) * -inf` 的乘法mask需要三条指令：`mask.to(float)`（bool→float类型转换）+ `1.0 - mask_float`（减法）+ `result * 1e6`（乘法）。三条Vector指令只为实现一个条件选择
  - **方案**：`tl.where(mask, value, -inf)` 只需一条Vector选择指令——硬件直接按mask条件选择value或 -inf，无需类型转换和乘法

  ```python
  # ── Before: multiply mask (3 instructions) ──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      # ... compute qk scores ...
      mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < K_LEN)
      # multiply mask: cast + subtract + multiply (3 Vector instructions)
      block_s = qk - (1.0 - mask.to(tl.float32)) * 1e6  # subtract a large number at out-of-bounds positions

  # ── After: where mask (1 instruction) ──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      # ... compute qk scores ...
      mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < K_LEN)
      # where mask: hardware conditional select (1 Vector instruction)
      block_s = tl.where(mask, qk, -1e6)  # set out-of-bounds positions directly to -inf
  ```

- 9.2预计算Mask + 全局缓存：从运行时计算到直接加载
  - **问题**：多个mask load + 布尔运算在kernel内逐元素计算，开销大。以SWA为例：causal mask + local window mask + attention arg mask + query mask，4次mask load + 3次布尔运算（`&` 和 `|` 组合），每次都占用Vector周期。更严重的是，这些mask在每次forward时都重新计算，即使mask参数（window\_size 等）没有变化
  - **方案一**：预计算单mask替代多mask组合——主机端将所有mask组合为一个预计算 mask\_tensor，kernel内只需1次MTE2 load
  - **方案二**：mask参数不变时缓存预计算mask张量——按mask参数（如 window\_size）作为缓存键，参数不变时直接返回缓存的 mask\_tensor，避免重复预计算

  ```python
  # ── precompute a single mask (SWA prefill) ──
  # Before: 4 loads + 3 boolean ops (elementwise compute inside the kernel)
  @triton.jit
  def swa_kernel(q_ptr, k_ptr, ...):
      # 4 mask loads
      tril_mask = gen_mask_tril(offs_m, offs_n)       # causal lower triangle
      triu_mask = gen_mask_triu(offs_m, offs_n)       # attention-arg upper triangle
      bound_mask = gen_mask_bound(offs_n, K_LEN)      # boundary mask
      q_mask = gen_mask_q(offs_m, Q_LEN)              # query boundary
      # combine with 3 boolean ops
      mask = tril_mask & (triu_mask | bound_mask) & q_mask  # Vector compute

  # After: 1 load (single mask precomputed on the host)
  @triton.jit
  def swa_kernel(q_ptr, k_ptr, mask_ptr, ...):
      # 1 MTE2 load, zero Vector compute
      mask = tl.load(mask_ptr + offsets)  # load the precomputed mask directly

  # ── global cache (SWA prefill) ──
  # Before: recomputes the precomputed mask on every call
  def swa_forward(q, k, window_size, ...):
      mask = precompute_mask(window_size, ...)  # recomputed every forward
      swa_kernel[grid](q, k, mask, ...)

  # After: cache hit when the mask parameters are unchanged
  _GLOBAL_WINDOW_SIZE = None
  _CACHED_MASK = None

  def swa_forward(q, k, window_size, ...):
      global _GLOBAL_WINDOW_SIZE, _CACHED_MASK
      if _GLOBAL_WINDOW_SIZE == window_size:
          mask = _CACHED_MASK  # cache hit, zero compute
      else:
          mask = precompute_mask(window_size, ...)
          _CACHED_MASK = mask
          _GLOBAL_WINDOW_SIZE = window_size
      swa_kernel[grid](q, k, mask, ...)
  ```

  - **案例**：SWA prefill — `gen_mask_causal_with_window` JIT + `_GLOBAL_WINDOW_SIZE` 缓存
- 9.3冗余Mask消除：利用对齐信息删除死代码
  - **问题**：当 BLOCK\_SIZE = HEAD\_DIM、BLOCK整除维度、或mask已覆盖边界时，额外mask检查是死代码——条件恒为True，但运行时仍会生成mask指令并执行
  - **四种消除场景**：
    - 尾维mask：当 `BLOCK_SIZE = HEAD_DIM` 时，`offs_k < HEAD_DIM` 恒为True（因为 `offs_k = arange(0, HEAD_DIM)`），移除 `& (offs_k < HEAD_DIM)`
    - masked\_out\_rows 安全保护：非首块路径中 m\_i 已有有效值（上一轮的 m\_new），不存在全 -inf行，`masked_out_rows` 检查是冗余的
    - KV\_LEN / IS\_FULL\_BLOCKS 条件mask：当mask已覆盖边界或BLOCK整除维度时，额外的 KV\_LEN 边界检查重复
    - `% M / % N` 取模运算：Grid已保证pid不越界（`pid < NUM_BLOCKS_M`），取模 `% M` 是冗余的，且 `%` 运算在NPU上导致标量退化（性能下降5-10倍）

  ```python
  # ── tail-dim mask elimination (FlexAttention) ──
  # Before: two-dimension check (the tail check still runs even when BLOCK_K == HEAD_DIM)
  mask = (offs_m[:, None] < Q_LEN) & (offs_k[None, :] < HEAD_DIM)
  # when BLOCK_K == HEAD_DIM, offs_k = arange(0, HEAD_DIM) < HEAD_DIM is always True

  # After: HEAD_DIM = BLOCK_SIZE, tail check is redundant and removed
  mask = (offs_m[:, None] < Q_LEN)  # keep only the M-dimension boundary check

  # ── masked_out_rows elimination (FlexAttention online softmax) ──
  # Before: safety guard (3 instructions) preventing NaN from all--inf rows
  masked_out_rows = (m_ij == float('-inf'))  # check whether the row is all -inf
  m_ij_masked = tl.where(masked_out_rows, 0, m_ij)  # set to 0 when all -inf
  alpha = tl.exp(m_i - m_ij_masked)  # safe exp

  # After: m_i already holds a valid value on non-first-block paths, use it directly (saves 3 instructions)
  alpha = tl.exp(m_i - m_ij)  # no safety guard needed

  # ── modulo elimination (SwiGLU fwd) ──
  # Before: modulo prevents out-of-bounds (and % causes scalar degradation)
  offset = (pid * BLOCK + tl.arange(0, BLOCK)) % M  # % op, scalar degradation

  # After: the Grid already guarantees pid stays in bounds, modulo is redundant
  offset = pid * BLOCK + tl.arange(0, BLOCK)  # contiguous addresses, vectorized
  ```

  - **案例**：FlexAttention — 移除尾维mask、masked\_out\_rows、KV\_LEN 边界保护（6处）
  - **案例**：SwiGLU fwd — offset移除取模
- 9.4编译期Mask消除：constexpr控制代码生成
  - **问题**：当列数/token 数对齐时（如 `n_cols % BLOCK_SIZE == 0`），mask代码是死代码（条件恒为True），但运行时条件判断仍会生成mask指令——编译器无法在运行时判断条件是否恒为True，保守地生成mask代码
  - **方案**：用编译期常量（`tl.constexpr`）控制是否生成mask代码——将"是否对齐"的判断从运行时提升到编译时，编译器在编译期直接消除整段死代码

  ```python
  # ── Before: always generates the mask (even when columns are aligned) ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_cols, ...,
                      BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      # always generates the mask, even when n_cols % BLOCK_SIZE == 0 (mask is always True)
      mask = (offs < n_cols)
      x = tl.load(x_ptr + offs, mask=mask, other=0.0)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offs, y, mask=mask)

  # ── After: constexpr controls it, mask eliminated at compile time ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_cols, ...,
                      BLOCK_SIZE: tl.constexpr,
                      COL_ALIGNED: tl.constexpr):  # compile-time constant
      pid = tl.program_id(0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      if not COL_ALIGNED:
          # generate mask code only when unaligned (decided at compile time)
          mask = (offs < n_cols)
          x = tl.load(x_ptr + offs, mask=mask, other=0.0)
          y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
          tl.store(y_ptr + offs, y, mask=mask)
      else:
          # no mask when aligned (the compiler removes the if-not branch)
          x = tl.load(x_ptr + offs)  # no mask
          y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
          tl.store(y_ptr + offs, y)  # no mask

  # Call site: COL_ALIGNED passed in at compile time
  COL_ALIGNED = (n_cols % BLOCK_SIZE == 0)  # computed at compile time
  gelu_fwd_kernel[grid](x, y, n_cols, COL_ALIGNED=COL_ALIGNED, ...)
  ```

  - **案例**：GELU — `COL_ALIGNED` constexpr消除mask
  - **案例**：VisionRoPE — `TOKEN_ALIGNED` constexpr消除mask

### 第10章 内存访问与数据通路：优化数据搬运

> **核心理念**：NPU的数据搬运由专用DMA引擎执行，与计算单元物理分离。内存访问优化的核心是：减少搬运次数、提升搬运效率、利用硬件数据通路特性。
>
> **（1）连续地址流依赖。** NPU的DMA引擎（MTE2/MTE3）依赖**连续地址流**——DMA描述符以固定stride发起连续搬运，离散访存（stride最低维 ≠ 1，即非连续地址）性能急剧下降。这与GPU的Warp合并访问（coalesced access）模式不同：GPU上32线程的Warp可以合并访问分散地址（只要地址在同一cache line），NPU的DMA引擎则需要严格的连续地址。离散访存时DMA引擎退化为逐元素搬运，带宽利用率可能降低数倍。
>
> **（2）Block Pointer的硬件优势。** `tl.make_block_ptr` 利用硬件DMA描述符自动处理边界——DMA引擎根据 block\_ptr 的 shape/strides/block\_shape 自动计算搬运计划，越界位置自动填充0（无需手动mask）。这比手动指针算术 + mask更高效：手动方式需要Vector单元计算offset和mask（占用Vector周期），block\_ptr 方式由DMA描述符硬件完成（零Vector开销）。但注意：NPU上 block\_ptr 的 **stride必须反映真实内存布局**，不能通过交换stride实现转置（GPU上可以），转置语义只能通过 **order参数**表达。
>
> **（3）`%`** **取模的标量退化。** NPU上 `%` 取模运算通过标量ALU执行，导致向量运算退化为逐元素标量计算——地址不连续，DMA引擎无法发起连续搬运，性能可能下降 **5-10倍**。应尽量用mask替代取模：mask是向量化的比较操作（保持连续地址），取模是标量运算（破坏连续地址）。
>
> **（4）CANN extension切片指令。** CANN extension的 `extract_slice` / `insert_slice` 直接映射NPU DMA切片指令，无通用版本的范围检查开销——通用Triton的slice操作在运行时检查offset和shape是否越界（占用Vector周期），CANN扩展版本跳过检查直接发起DMA切片。

- 10.1预计算缓存：从运行时计算到直接加载
  - **问题**：kernel内的 sin/cos 等超越函数计算、每次forward的权重转置都是昂贵的重复计算。以RoPE为例：每个program在kernel内计算 `freqs = pos * inv_freq; sin = tl.sin(freqs); cos = tl.cos(freqs)`，`tl.sin`/`tl.cos` 是昂贵的超越函数（多条Vector指令），且每次forward都重复计算相同的 sin/cos 值
  - **方案一（Sin/Cos 预计算）**：CPU预计算 sin/cos 表后传入kernel直接load，将超越函数计算转为MTE2搬入
  - **方案二（权重转置缓存）**：按 `data_ptr()` 键缓存转置结果，消除重复O(N·K) 转置——每次forward都做 `weight.t().contiguous()` 是O(N·K) 的数据搬运，但权重内容不变时转置结果可缓存复用

  ```python
  # ── Sin/Cos precomputation (RoPE) ──
  # Before: transcendentals computed inside the kernel (many Vector instructions)
  @triton.jit
  def rope_kernel(in_ptr, pos_ptr, out_ptr, ...):
      pos = tl.load(pos_ptr + offsets)
      freqs = pos[:, None] * inv_freq[None, :]  # compute frequencies
      sin = tl.sin(freqs)  # expensive transcendental (many Vector instructions)
      cos = tl.cos(freqs)  # expensive transcendental
      x1 = x0 * cos - y0 * sin  # rotation
      ...

  # After: load the precomputed results directly (1 MTE2 load)
  @triton.jit
  def rope_kernel(in_ptr, pos_ptr, sin_cache_ptr, cos_cache_ptr, out_ptr, ...):
      # load the precomputed sin/cos directly (zero transcendental compute)
      sin = tl.load(sin_cache_ptr + offsets)  # MTE2 move-in
      cos = tl.load(cos_cache_ptr + offsets)  # MTE2 move-in
      x1 = x0 * cos - y0 * sin  # rotation
      ...

  # host-side precomputation
  def rope_forward(x, position, ...):
      sin_cache, cos_cache = precompute_sin_cos(position, base, DIM)  # CPU compute
      rope_kernel[grid](x, position, sin_cache, cos_cache, out, ...)

  # ── weight-transpose cache (GEMM) ──
  # Before: transpose recomputed every forward (O(N·K) data movement)
  def gemm_forward(x, weight, ...):
      weight_t = weight.t().contiguous()  # transposed every forward
      gemm_kernel[grid](x, weight_t, ...)

  # After: cache by data_ptr(), reused while the weight is unchanged
  _bt_cache = {}  # global cache dict

  def gemm_forward(x, weight, ...):
      key = weight.data_ptr()  # use the tensor memory address as the cache key
      if key not in _bt_cache:
          _bt_cache[key] = weight.t().contiguous()  # compute and cache on first use
      weight_t = _bt_cache[key]  # reused directly afterwards
      gemm_kernel[grid](x, weight_t, ...)
  ```

  - **案例**：RoPE — sin/cos 预计算缓存
  - **案例**：GEMM — `_bt_cache` 消除重复转置
- 10.2数据聚合与CANN切片：提升Cube利用率与切片效率
  - **问题一**：小page场景下（PagedAttention的 page\_size 较小，如 16/32），逐页matmul的Cube利用率低——每个page独立做 `tl.dot(q, k_page)`，Cube单元的矩阵乘规模太小（M=BLOCK\_M, N=page\_size），Cube利用率不足
  - **方案一（L1 Cache聚合）**：多个页用 `insert_slice` 聚合为单个大tile后再做matmul——PAGE\_AGG\_NUM 个page聚合为 `(PAGE_AGG_NUM * BLOCK_N, D)` 的大矩阵，Cube一次处理大矩阵，利用率大幅提升
  - **问题二**：通用Triton的slice操作有范围检查开销——`tl.extract_slice` 在运行时检查offset和shape是否越界，占用Vector周期
  - **方案二（CANN extension）**：用CANN扩展版本替代通用slice，直接映射DMA切片指令，跳过范围检查

  ```python
  # ── L1 cache aggregation (FA/SWA page aggregation) ──
  # Before: per-page loads + small matmuls, low Cube utilization
  @triton.jit
  def fa_kernel(q_ptr, k_ptr, ...):
      for kv_block in range(num_kv_blocks):
          k = tl.load(k_ptr + kv_block * page_stride)  # single-page load
          qk = tl.dot(q, tl.trans(k))  # small matmul, low Cube utilization
          # ... softmax ...

  # After: aggregate into a large tile + big matmul, high Cube utilization
  @triton.jit
  def fa_kernel(q_ptr, k_ptr, ..., PAGE_AGG_NUM: tl.constexpr):
      PAGE_AGG = PAGE_AGG_NUM  # number of aggregated pages
      for kv_block in range(0, num_kv_blocks, PAGE_AGG):
          # pre-allocate the large tile
          k = tl.zeros((PAGE_AGG * BLOCK_N, D), dtype=tl.float16)
          for i in range(PAGE_AGG):
              # load one page and insert it at the right place in the large tile
              k_slice = tl.load(k_ptr + (kv_block + i) * page_stride)
              k = insert_slice(k, k_slice, offset=i * BLOCK_N, axis=0)
          # big matmul, high Cube utilization
          qk = tl.dot(q, tl.trans(k))  # matrix size grows by PAGE_AGG×
          # ... softmax ...

  # ── CANN extension slice（RoPE / Conv1D / GroupGEMM / VisionRoPE）──
  # Before: generic version (runtime range checks)
  slice_result = tl.extract_slice(tensor, offsets, shape, strides)
  # runtime check whether offsets + shape go out of bounds (Vector overhead)

  # After: CANN extension (maps directly to a DMA instruction, skipping the checks)
  import triton.language.extra.cann.extension as extension
  slice_result = extension.extract_slice(tensor, offsets, shape, strides)
  # issues a DMA slice instruction directly, zero range-check overhead
  ```

  - **案例**：FA/SWA prefill — 小page聚合提升Cube利用率
  - **案例**：RoPE / Conv1D / GroupGEMM / VisionRoPE — CANN extension slice
- 10.3访存并行优化：Load重排序
  - **问题一**：无依赖load排在依赖load之后，串行执行——编译器不重排load顺序，如果排在前的load因等待上一轮store而阻塞，排在后的无依赖load也无法提前发射
  - **方案一**：将无依赖load提前到有依赖load之前，使无依赖load与上一轮store并行执行

  ```python
  # ── Load reordering (Quant/RoPE) ──
  # Before: dependent load first, dependency-free load after (serial)
  @triton.jit
  def kernel(x_ptr, scale_ptr, ...):
      for i in range(num_iters):
          # x load depends on the previous store (same address), so it is blocked
          x = tl.load(x_ptr + offsets)
          # scale load has no dependency, but must wait for x load to finish before issuing
          scale = tl.load(scale_ptr + offsets)
          # ... compute ...
          tl.store(x_ptr + offsets, result)  # next iteration x load depends on this store

  # After: issue the dependency-free load first, both loads run in parallel
  @triton.jit
  def kernel(x_ptr, scale_ptr, ...):
      for i in range(num_iters):
          # scale load has no dependency, issued first (parallel with the previous store)
          scale = tl.load(scale_ptr + offsets)
          # x load depends on the previous store, issued later
          x = tl.load(x_ptr + offsets)
          # ... compute ...
          tl.store(x_ptr + offsets, result)
  ```

  - **案例**：Quant/RoPE — Load重排序隐藏内存延迟
- 10.4 Block Pointer + 内存分配优化
  - **问题一**：手动指针算术 + mask比Block Pointer效率低——手动方式需要Vector单元计算 `offsets = row[:, None] * stride + col[None, :]` 和 `mask = (row < M) & (col < N)`，占用Vector周期；Block Pointer由DMA描述符硬件自动处理边界，零Vector开销
  - **方案一**：用 `tl.make_block_ptr` 替代手动指针算术，利用硬件DMA描述符自动处理边界
  - **问题二**：`torch.clone()` 拷贝整个张量数据，配合原地kernel引入RAW冒险——`clone` 分配新内存并拷贝数据（O(N) 数据搬运），原地kernel读写同一地址引入Read-After-Write冒险
  - **方案二**：用 `torch.empty_like()`（仅分配内存，无拷贝）替代 `clone`（分配 + 拷贝），配合非原地kernel（读写不同地址）

  ```python
  # ── Block Pointer replaces manual pointer arithmetic ──
  # Before: manual pointer arithmetic + mask (Vector computes offsets and mask)
  @triton.jit
  def kernel(ptr, M, N, stride, ...,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
      col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
      # manually compute the offset (Vector compute)
      offsets = row[:, None] * stride + col[None, :]
      # manually compute the mask (Vector compute)
      mask = (row[:, None] < M) & (col[None, :] < N)
      x = tl.load(ptr + offsets, mask=mask)  # manual load

  # After: block_ptr handles boundaries automatically (done by the DMA descriptor hardware)
  @triton.jit
  def kernel(ptr, M, N, stride, ...,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      # block_ptr: the DMA descriptor handles boundaries automatically, zero Vector overhead
      block_ptr = tl.make_block_ptr(
          base=ptr,
          shape=(M, N),
          strides=(stride, 1),
          offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
          block_shape=(BLOCK_M, BLOCK_N),
          order=(1, 0),
      )
      x = tl.load(block_ptr)  # automatic load, out-of-bounds filled with 0

  # ── empty_like replaces clone (RoPE) ──
  # Before: allocate + copy (O(N) data movement + in-place RAW hazard)
  def rope_forward(q, ...):
      q_out = torch.clone(q)  # allocate new memory + copy all of q data
      rope_kernel[grid](q_out, ...)  # modifies q_out in place (reads and writes the same address)

  # After: allocate only (no copy, not in-place, no RAW)
  def rope_forward(q, ...):
      q_out = torch.empty_like(q)  # allocate memory only, no data copy
      rope_kernel[grid](q, q_out, ...)  # not in-place: read q, write q_out (different addresses)
  ```

  - **案例**：VisionRoPE — 3D block\_ptr for cos/sin/q/k
  - **案例**：RoPE — `empty_like` 节省一个q张量大小的显存峰值
- 10.5循环不变量外提：Bias提前加载 + Cast位置提前
  - **问题一**：bias在K循环内每轮重复加载——bias是循环不变量（不随K变化），但在循环内每轮 `tl.load(bias_ptr + offsets)`，重复发起MTE2搬运相同数据
  - **方案一**：bias在循环外提前加载一次 + `broadcast_to` 显式广播，循环内直接复用
  - **问题二**：`.to(dtype)` 在 `tl.trans` 之后，cast结果无法被多处复用——`p.to(dtype)` 在 `tl.trans(p)` 之后，每次需要转置后的p时都要重新cast
  - **方案二**：将cast提前到transpose之前，cast结果可被dv和dk两处复用（`tl.trans` 只是布局变换，不改变数据）

  ```python
  # ── Bias hoisting (Fused MatMul fwd) ──
  # Before: bias reloaded every iteration (loop invariant kept inside the loop)
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, ...):
      for k in range(0, K, BLOCK_K):
          bias = tl.load(bias_ptr + offsets)  # reloaded every iteration (wasteful)
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b) + bias[None, :]  # bias added every iteration

  # After: load once outside the loop + explicit broadcast
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, ...):
      # load once outside the loop
      bias = tl.load(bias_ptr + offsets)
      bias = tl.broadcast_to(bias[None, :])  # explicitly broadcast to the matrix dimensions
      for k in range(0, K, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)  # no bias added inside the loop
      acc += bias  # added once outside the loop

  # ── Cast Hoisting（FA v5 bwd_kv / FA v8 bwd_q）──
  # Before: cast after trans, repeated conversion
  # p is transposed for both dv and dk, and cast again at each use site
  dv = tl.dot(tl.trans(p).to(dtype), do)       # cast after trans (1st time)
  dk = tl.dot(tl.trans(p).to(dtype), do_other)  # cast after trans (2nd time, duplicated)

  # After: cast once before trans, reuse the result
  p = p.to(dtype)                       # cast before trans (only once)
  dv = tl.dot(tl.trans(p), do)          # reuse the cast p
  dk = tl.dot(tl.trans(p), do_other)    # reuse the cast p
  ```

  - **案例**：Fused MatMul fwd — bias循环外提前加载
  - **案例**：FA v5 bwd\_kv / FA v8 bwd\_q — p的cast提前复用
- 10.6指令简化：Mask类型转换 + tl.full初始化
  - **问题一**：布尔mask的隐式类型推断导致不确定行为——`mask = (q_offset <= k_offset)` 产生bool类型，参与浮点运算时（如 `block_s = tl.where(mask, block_s, -inf)`）可能触发隐式转换的不确定行为
  - **方案一**：对布尔mask进行显式类型转换 `.to(tl.int32)`，确保类型安全
  - **问题二**：`tl.zeros([M]) - float("inf")` 需要先创建零张量再减（2条指令：`tl.zeros` 创建零张量 + 减法），用于初始化online softmax的最大值数组 `m_i`
  - **方案二**：`tl.full([M], float("-inf"))` 直接填充目标值（1条指令），消除减法操作

  ```python
  # ── Mask type conversion (FA v4 fwd) ──
  # Before: implicit bool (may trigger undefined behavior)
  @triton.jit
  def fa_kernel(...):
      q_offset = tl.arange(0, BLOCK_M) * stride
      k_offset = tl.arange(0, BLOCK_N) * stride
      # implicit bool type, may be undefined when used with where
      mask = (q_offset <= k_offset)
      block_s = tl.where(mask, block_s, -1e6)

  # After: explicit int32 (type-safe)
  @triton.jit
  def fa_kernel(...):
      q_offset = tl.arange(0, BLOCK_M) * stride
      k_offset = tl.arange(0, BLOCK_N) * stride
      # explicitly cast to int32 to ensure type safety
      mask = (q_offset <= k_offset).to(tl.int32)
      block_s = tl.where(mask, block_s, -1e6)

  # ── tl.full replaces zeros - inf (FlexAttention) ──
  # Before: 2 instructions (create zero tensor + subtraction)
  @triton.jit
  def fa_kernel(...):
      # first create a zero tensor, then subtract inf (2 Vector instructions)
      m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

  # After: 1 instruction (fill -inf directly)
  @triton.jit
  def fa_kernel(...):
      # fill the target value directly (1 Vector instruction)
      m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
  ```

  - **案例**：FA v4 fwd — mask\_fn 显式类型转换
  - **案例**：FlexAttention — `m_i` 初始化优化

***

### 第11章 编译器图优化：固定模式匹配的自动优化

> **核心理念**：Triton Ascend的图优化pass在IR图上做固定模式匹配（pattern match），命中特定子图后自动重写为更优的等价形式。开发者无需（也无法）手写这些变换，但理解匹配的模式与触发条件，可以预判编译器行为、定位"为什么没被优化"，并知道何时需要显式干预。
>
> **背景知识**：图优化pass针对IR图中的固定子图形态做自动重写，主要覆盖五类模式——**融合类**（逐元素链、dot后处理、向量函数打包）、**循环结构类**（多缓冲标记）、**布局类**（NZ↔ND转换）、**访存地址与布局类**（取模线性化、读写置换、store合并、跨步改写）、**程序映射与Grid变换类**（独立轴张量化、持久化任务挖掘、行合并）。每个pass只认自己关心的子图结构，未命中则原样保留。

- 11.1匹配机制概述：IR图 → pass扫描固定子图 → 重写
  - **匹配对象**：pass扫描特定子图结构，常见有三类触发条件——**结构模式**（op类型与连接关系，如 `mulf + addf`、循环内 `load/store` 对）、**布局/属性约束**（如 `ascend.dot` 的 `fractal_a/b/c`、block尺寸16×16）、**循环结构约束**（候选必须位于 `scf.for`/`scf.while` 内且支持每迭代轮转槽位）
  - **重写结果**：命中后替换为更优等价形式（融合、布局转换、多缓冲标记等），未命中则原样保留
  - **与手写代码的关系**：绝大多数自动生效；少数场景需要提供hint或开关配合（见11.8）
- 11.2自动融合类pattern：逐元素链合并、dot后处理与fixpipe随路、VF打包
  - **逐元素链合并**：`x * sigmoid(x)` 等连续逐元素链被合并为单指令（silu），`mul + add` 折叠为fma，连续cast折叠
  - **dot后处理随路**：`a*b + c` 或带 bias/精度转换的结果不再单独做一次Vector运算——编译器把bias累加、fp32→fp16、转置融进 **fixpipe** 从L0C搬出时的随路操作（按下游消费方式选择NZ2ND / NZ2DN / NZ2NZ模式），这是Cube结果喂给Vector前最常见的自动融合
  - **VF打包**：连续向量运算链被outline成独立向量函数（`--vf-fusion-mode`，默认max-parallel），减少指令间同步；代价是链过长时UB溢出或吞掉dot的独立调度（回指第6章）
- 11.3循环结构pattern：多缓冲标记
  - **多缓冲标记**：循环内 `load/store` 对命中多缓冲模式，在分配点打 `annotation.mark {hivm.multi_buffer = N}` 标记，后续经内存规划分配N份地址、按迭代计数取模选择槽位（回指第7章）。本地Buffer的N固定为2，GM Workspace的N由 `--set-workspace-multibuffer` 指定
  - **硬约束**：候选Buffer从分配点到最外层祖先循环必须都是 `scf.for`/`scf.while`，位于 `scf.parallel`/`scf.forall` 内的Buffer不会被标记
- 11.4布局pattern：NZ↔ND转换与消除
  - **分形布局推断**：`ascend.dot` 的 `fractal_a/b/c` 属性触发布局推断（如 DOTA\_ND\_TO\_zN），编译器在GM的ND与Cube的NZ（`[N1, M1, 16, 16]` 分形块）之间自动插入 `convert_layout`
  - **fixpipe收尾转换**：Cube结果在L0C为NZ，需fixpipe搬运到UB并转回ND，Vector才能逐元素处理（回指第10章）
  - **转换消除**：连续的 `convert_layout` 被合并/消除；布局一致的数据流可省掉整段转换
- 11.5访存地址与布局优化pattern：把离散、低效的内存访问改写为整块连续访问
  - **共同目标**：针对**地址表达式本身**（取模、跨步、维度顺序）做改写，不动数据布局（与11.4的 convert\_layout 类pattern互补），让后端把离散访问识别为整块连续内存、一次DMA搬完，同时消除标量地址计算。下文每个pattern均给出触发模式、改写、收益与约束
- 11.5.1 ConvertModuloToMask：取模线性化
  - **触发模式**：kernel写 `offs % n` 让越界下标绕回数组开头以避免非法访问，但写回处已有 `offs < n` 掩码丢弃越界数据——读进来的绕回数据本就无用
  - **DSL写法（触发示例）**：load用取模绕回地址，store带 `offs < n` 掩码

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      wrap = offs % n                  # wrap out-of-bounds indices back to the start
      mask = offs < n                  # guard mask for the store
      v = tl.load(x_ptr + wrap)        # load without mask (reads the wrapped data)
      tl.store(y_ptr + offs, v, mask=mask)
  ```

  - **改写**：去掉取模让地址恢复连续，同时给load补上同样的掩码（越界位置读0）

  ```ttir
  # Before: offs % n scrambles contiguous addresses into jumps, load has no mask
  %wrap = arith.remsi %offs, %n       # offs % n
  %addr = arith.addi %kb, %wrap
  %v    = tt.load %wa                 # load without mask
  tt.store %oa, %v, %gb               # store already carries the offs < n mask

  # After: addresses become linear again, load gets a mask + zero fill
  %cmp  = arith.cmpi slt, %offs, %n   # offs < n
  %addr = arith.addi %kb, %offs       # linearized addresses
  %v    = tt.load %wa, %cmp, %cst     # mask added, out-of-bounds reads 0
  tt.store %oa, %v, %cmp
  ```

  - **收益**：地址线性化后后端可识别为整块连续内存，用一次DMA搬完，而非因取模被迫逐元素离散访问
  - **约束**：除数必须是运行时变量（编译期常量交由TritonToStructured处理更优，它保留绕回语义并改写为跨步访问，完全等价）；下标必须是一维i32、无layout，长度等于 make\_range 长度；必须存在store上的offs < n守卫作为依据，因此归约轴上的取模无法处理（归约轴无输出坐标，不会有这种掩码）；取模结果只能流向load地址，不能流向other、tt.trans、tt.reshape或其他计算；expand\_dims 的轴必须唯一；不支持跨嵌套scf.for的多条循环携带地址
- 11.5.2 LoadStoreTranspose：读写维度置换
  - **触发模式**：分析 load/store 的地址表达式，判断张量各维的实际内存步长。若最内维（代码中变化最快的那维）在内存里是跳着走的，则触发
  - **DSL写法（触发示例）**：tile最内维（cols）在内存中跳着走——列主序访问

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr):
      rows = tl.arange(0, M)           # 32 rows
      cols = tl.arange(0, N)           # 16 columns
      # address of element (r, c) = r + c * M: inner-dim stride M, each element jumps M elements
      offs = rows[:, None] + cols[None, :] * M
      v = tl.load(x_ptr + offs)        # [32, 16]
      tl.store(y_ptr + offs, v + 1.0)
  ```

  - **改写**：把整个"读→算→写"闭包的维度顺序调换，使最内维变成内存中真正连续的那维；同步改写掩码、assert、循环携带指针，以及循环之后reduce的轴号

  ```ttir
  # Before: the 16 innermost columns are 32 elements apart, each read jumps 128 bytes
  %off = arith.addi %mb, %nb          # 32×16 offsets
  %sa  = tt.addptr %sp, %off
  %x   = tt.load %sa : tensor<32x16x!tt.ptr>
  %y   = arith.addf %x, %one
  tt.store %da, %y

  # After: swap the dimension order, the 32 innermost elements are adjacent, moved in one 128-byte burst
  %off = arith.addi %mb, %nb          # 16×32 offsets
  %sa  = tt.addptr %sp, %off
  %x   = tt.load %sa : tensor<16x32x!tt.ptr>
  %y   = arith.addf %x, %cst
  tt.store %da, %y
  ```

  - **收益**：改写前每读一个元素跳128字节、16个元素是16次离散访问；改写后最内维32个元素紧邻，一次连续搬128字节
  - **约束**：指针必须是addptr(splat(标量指针), 偏移) 形式，维度 ≥2、静态shape、无layout，不支持block pointer；每一维须恰好由一个 make\_range 驱动，且各维步长大小关系能在结构上证明（运行时步长需掩码提供取值范围）；必须同时有load和store且存在load→store数据通路，所有访存点置换一致；不支持volatile、padding、boundary\_check；load的other必须是均匀常量；闭包内只允许逐元素算子、splat、broadcast、expand\_dims、addptr、trans、常量，不支持矩阵乘和reshape；循环结果只能被单输入tt.reduce消费；不跨 scf.if/scf.while 区域边界；遇到用户自己写的张量assert直接放弃
- 11.5.3 StoreCoalescing：相邻store合并
  - **触发模式**：函数内有几次store写入同一块内存的相邻区段（前一次结尾正好接下一次开头）
  - **DSL写法（触发示例）**：对同一基址的相邻区段分多次store，首尾严格相接

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      offs0 = tl.arange(0, BLOCK)
      offs1 = tl.arange(0, BLOCK) + BLOCK
      v0 = tl.load(x_ptr + offs0)
      v1 = tl.load(x_ptr + offs1)
      tl.store(y_ptr + offs0, v0)      # [0, BLOCK)
      tl.store(y_ptr + offs1, v1)      # [BLOCK, 2*BLOCK), contiguous with the previous segment
  ```

  - **改写**：先在寄存器中把数据拼成一个大张量，最后一次性写出

  ```ttir
  # Before: two adjacent stores write in two separate passes
  tt.store %base, %v0                 # [0, 64)
  tt.store %base + 64, %v1            # [64, 128)

  # After: stitch into a 128-element tensor in registers, write out in one pass
  %t = tensor.insert_slice %v0 into %empty[0] [64] [1]
  %t = tensor.insert_slice %v1 into %t[64] [64] [1]
  tt.store %base, %t                  # one contiguous 128-element write
  ```

  - **收益**：两次256字节的写变成一次512字节的写，搬运次数减半、单次更长，GM写带宽利用率更高
  - **约束**：只处理函数最外层store（循环或分支内的不处理）；store不能带掩码或 boundary\_check，且须能静态证明按行主序连续排列；基地址必须是同一SSA值（纯语法比较，不做别名推断）；元素类型、cache/evict 修饰及其他属性须完全一致；至少两段，段间不重叠且首尾严格相接；合并后总字节数不超过UB预算；被推迟的store与中间夹着的其他访存必须能证明不冲突；不支持i1和不足一字节的类型；偏移量须在i32范围内；simt\_only 下关闭
- 11.5.4 MergeConcatLoadBuffer：concat缓冲合并
  - **触发模式**：在降级到memref之后生效。带掩码的load降级后会变成"分配整块缓冲 → 用填充值填满 → 把有效部分拷入"，若紧接着做tl.cat拼接，就会出现两块缓冲各写一半
  - **DSL写法（触发示例）**：两个带掩码load后紧跟tl.cat拼接

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      mask = offs < n
      a = tl.load(x_ptr + offs, mask=mask, other=0.0)     # masked load
      b = tl.load(x_ptr + BLOCK + offs, mask=mask, other=0.0)
      c = tl.cat(a, b)                  # concat → after lowering, two buffers each get half the data
      tl.store(y_ptr + tl.arange(0, 2 * BLOCK), c)
  ```

  - **改写**：把两块缓冲合并为一块；若能证明两次拷贝覆盖了后续全部读取范围，两个填充操作都删除

  ```ttir
  # Before: two 32×512 buffers each written half (each with one fill + copy + to_tensor)
  %a0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst) outs(%a0)
  memref.copy %ss0, %sd0              # writes [0, 256)
  %t0 = bufferization.to_tensor %a0
  %a1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst) outs(%a1)
  memref.copy %ss1, %sd1              # writes [256, 512)
  %t1 = bufferization.to_tensor %a1
  %ex  = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1]
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1]

  # After: merge into one buffer, copies write different subviews, both fills removed
  %a = memref.alloc() : memref<32x512xf32>
  memref.copy %ss0, %sv0              # writes [0, 256)
  memref.copy %ss1, %sv1              # writes [256, 512)
  %t = bufferization.to_tensor %a
  ```

  - **收益**：UB占用从2块tile降到1块，并省去两次32×512元素的填充与一次切片数据搬移
  - **约束**：只在 910\_95/950 上启用。910\_9362 的代码生成只保证紧邻 to\_tensor 的那次拷贝可见，合并后另一半tile会读到填充值，故该芯片上禁用；extract与insert的偏移、大小必须完全一致；填充必须发生在拷贝之前，否则放弃；只保留一个填充时，要求两个填充都是单纯fill且填充值相同
- 11.5.5 StridedAxisCoalescing：跨步轴折回
  - **触发模式**：kernel实现中把本来连续的H轴（如head数）摊到grid上，用 `pid % S` 取出自己负责的head，导致每个program读的数据在内存中每隔S个元素取一个
  - **DSL写法（触发示例）**：H轴摊到grid、`pid % S` 取head，用步长S的一维block pointer读

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, S: tl.constexpr, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      i_h = pid % S                        # get the head this program handles
      tp = tl.make_block_ptr(x_ptr, (n,), (S,), (i_h,), (BLOCK,), (0,))
      v = tl.load(tp)                      # take one element every S elements
      tl.store(tp, v + 1.0)
  ```

  - **改写**：把H轴重新折回张量内层，组成二维tile一次读完；逐head的标量load相应提升为向量load

  ```ttir
  # Before: 1-D with stride 4, each program handles one head (pid % 4)
  %i_h = arith.remsi %pid, %c4        # i_h = pid % 4
  %tp  = tt.make_tensor_ptr %base, [16], [4], [i_h]
  %v   = tt.load %tp : tensor<16xf32>

  # After: 2-D with inner stride 1, modulo gone, one DMA covers the whole contiguous block
  %tp  = tt.make_tensor_ptr %base, [16, 4], [4, 1], [0, 0]
  %v   = tt.load %tp : tensor<16x4xf32>
  ```

  - **收益**：改写前16个元素分散在16×4范围内、需16次离散访问；改写后是一整块连续内存一次DMA搬完；program数同时降为 1/4
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行，不受rule-mask控制；种子load必须是一维block pointer，步长为大于1的常量，且所有种子步长相同；kernel不能读对应轴的 num\_programs（grid被除以S后该值会变，读了就会算错）；子图只允许逐元素算子、类型转换、splat、沿T轴的 scan/reduce；不支持转置、矩阵乘、reshape，也不支持沿lane方向的归约；输出store也必须是同样步长的block pointer；i\_h 不能参与间接寻址；按整个module改写，任一条件不满足即整体放弃
- 11.5.6 ChunkCoalescing：小块合并
  - **触发模式**：对 `offs = pid * T + arange(T)` 这种标准分块写法，若每个tile太小导致连续字节数不足、DMA效率低，就把H个相邻tile合并到一个program，使单次连续访问至少达到512字节
  - **DSL写法（触发示例）**：`offs = pid * T + arange(T)` 的小tile分块 + 尾部掩码

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, T: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * T + tl.arange(0, T)     # only T elements per tile (too few contiguous bytes)
      mask = offs < n
      v = tl.load(x_ptr + offs, mask=mask, other=0.0)
      tl.store(y_ptr + offs, v, mask=mask)
  ```

  ```ttir
  # Before: 1-D, 16 elements = 64 bytes, masked (64 bytes is far below the DMA-efficient length)
  %4 = arith.addi %3, %2 : tensor<16xi32>            # offs = pid*16 + arange(16)
  %5 = arith.cmpi slt, %4, %cst_0 : tensor<16xi32>   # offs < 256
  %8 = tt.load %7, %5, %cst : tensor<16x!tt.ptr>
  tt.store %10, %8, %5 : tensor<16x!tt.ptr>

  # After: 2-D, 16×16 = 1024 bytes, mask removed
  %7  = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %8  = tt.broadcast %7 : tensor<16x1xi32> -> tensor<16x16xi32>
  %11 = arith.addi %8, %10 : tensor<16x16xi32>
  %16 = tt.load %15 : tensor<16x16x!tt.ptr>          # no mask
  tt.store %20, %16 : tensor<16x16x!tt.ptr>
  ```

  - **收益**：单次连续搬运从64字节增至1024字节（64字节远低于DMA高效传输长度，大部分时间消耗在启动开销上），program数降为 1/16
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行；对应grid轴只有一个 program\_id 且不读 num\_programs；pid派生链上不能出现 min/max/取余/除法；派生出的比较结果除作为那一个掩码外，只能流向 device\_assert（溢出检查默认开启，会给几乎每个pid派生下标插桩，必须明确放行）；不支持 boundary\_check；H取值范围为 \[max(2, ⌈512B ÷ tile字节⌉), min(16, 128KB ÷ 单tile占用)] 且必须整除总tile数，算不出合法H就不改；若StridedAxisCoalescing已设过 coalesce\_factor 则直接跳过
- 11.5.7 StridedLoadStoreRewrite：非2幂跨步改写
  - **触发模式**：硬件跨步DMA引擎只支持2的幂步长，遇到步长3、5、7这类会退化为极慢的逐元素标量访问
  - **DSL写法（触发示例）**：地址步长为非2的幂常量（3、5、7…）

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK) * 3       # stride 3: not supported by hardware strided DMA
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)
  ```

  - **改写**：把步长为非2的幂常量的 load/store 改写为专用的 ascend.stride\_load/stride\_store 或间接访问算子

  ```ttir
  # Before: stride-3 address chain, scalar loop fetches one by one
  %1 = arith.muli %arange, %c3        # stride 3
  %3 = tt.addptr %splat, %1
  %4 = tt.load %3 : tensor<256x!tt.ptr>

  # After: dedicated strided-access op, handled by the SIMT parallel access unit
  %1 = ascend.stride_load %arg0, %c0_i64, %cst, [%c3_i64], [%c256_i64]
  ```

  - **收益**：原本256个元素需靠标量循环逐个取，改写后交由SIMT并行访问单元处理，整个指针算术链一并消失
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行；以下情况有意不改写：步长1（本就连续）、步长2（走deinterleave路径）、步长为 ≥4的2的幂（走跨步DMA更快）、步长为运行时变量（可能运行时就是1或2的幂，保留SIMD路径更稳妥）、做过维度置换的布局（交由ImplicitPermute）；stride\_load 只支持 1\~3 维静态shape，快速路径最多5维；非浮点类型要求NaN填充时放弃；addptr路径会调用指针分析，可能留下无用辅助IR，需后续 CSE/canonicalize 清理（T2L pipeline已安排）
- 11.6程序映射与Grid变换pattern（AutoBlockifyV2）：合并program、提升单program吞吐
  - **共同目标**：把多个独立program合并成一个，用单program内的向量化/张量化计算换掉重复的标量地址计算——program数减少、Vector单次处理数据量增大、UB利用率提升，本质是"以program数换单program吞吐"。11.6.1–11.6.3均不受rule-mask控制；11.6.4默认关闭（opt-in）
- 11.6.1 IndependentAxisTensorize：独立轴张量化（IAT）
  - **触发模式**：grid为二维（token轴、head轴）且不同head间计算完全独立时，把head轴的16个program合并成一个，让一个program一次处理16个head
  - **DSL写法（触发示例）**：二维grid（token轴、head轴），head间计算完全独立

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      pid_t = tl.program_id(0)             # token axis
      pid_h = tl.program_id(1)             # head axis
      offs = tl.arange(0, BLOCK)
      # MergeSplit form: 4×BLOCK tile reduced along the T axis → BLOCK (each program handles one head)
      v = tl.load(x_ptr + pid_h * 4 * BLOCK + tl.arange(0, 4)[:, None] * BLOCK + offs[None, :])
      s = tl.sum(v, axis=0)                # reduction
      tl.store(y_ptr + pid_h * BLOCK + offs, s)
  ```

  - **改写**：原 program\_id(1) 变为长度16的向量，相关张量各增加一维，尾部补掩码。支持MergeSplit与NormRope两种归约形态

  ```ttir
  # Before: 2-D grid, each program handles one head (scalar head index)
  %head = tt.get_program_id y
  %th   = arith.addi %th0, %head
  %v    = tt.load %xa : tensor<4x128x!tt.ptr>
  %sum  = tt.reduce(%v) {axis = 0 : i32} : (tensor<4x128xf32>) -> tensor<128xf32>
  tt.store %ya, %sum

  # After: head axis tensorized, one program handles 16 heads
  %lanes = arith.addi %pid_y*16, %arange(16)          # 16 head lanes
  %mask  = arith.cmpi ult, %lanes, %extent            # tail mask
  %v     = tt.load %xa : tensor<4x16x128x!tt.ptr>
  %sum   = tt.reduce(%v) {axis = 0 : i32} : (tensor<4x16x128xf32>) -> tensor<16x128xf32>
  tt.store %ya, %sum, %mask
  ```

  - **收益**：program数降为 1/16；原本每program都要重算的标量地址计算现由16个head分摊一次；Vector指令单次处理数据量增加16倍，UB利用率显著提升
  - **约束**：必须是非 simt\_only 模式；函数只能有一个基本块，不能有 scf/cf 控制流、函数调用、原子操作、barrier；axis0与axis1的 program\_id 各只能出现一次；axis1依赖闭包内不能有 make\_tensor\_ptr，也不能读 num\_programs；store与load指针必须来自不同入参；UB峰值×16后不能超预算；动态shape直接拒绝；合并因子固定为16，无自动调优；module上已有其他grid契约时不再叠加；NormRope形态要求PTSM同时启用
- 11.6.2 PersistentTaskStripMining：持久化任务条带挖掘（PTSM）
  - **触发模式**：针对逐token做行归约的kernel——LayerNorm类同时算sum(x) 与sum(x²)，RMSNorm类只算sum(x²)
  - **DSL写法（触发示例）**：逐token做行归约的norm类kernel（一个token一个program）

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, D: tl.constexpr):
      pid = tl.program_id(0)             # one token per program
      offs = pid * D + tl.arange(0, D)
      v = tl.load(x_ptr + offs)          # D elements in one row
      s2 = tl.sum(v * v)                 # RMSNorm-like: sum(x²)
      # LayerNorm-like additionally computes: s1 = tl.sum(v) (both sum(x) and sum(x²))
      tl.store(y_ptr + pid, s2)
  ```

  - **改写**：原本有多少token就启动多少program，改为固定数量的program，每个用跨步循环反复领取任务，单次处理 BLOCK\_T 个token（LayerNorm类64，RMSNorm类配合IAT时4），尾部按原始token数加掩码

  ```ttir
  # Before: one program per token, scalar row reduction
  %row = arith.muli %pid, %cD
  %v   = tt.load %xa : tensor<128x!tt.ptr>
  %s1  = tt.reduce(%v) {axis = 0 : i32}       # sum(x)
  %sq  = arith.mulf %v, %v
  %s2  = tt.reduce(%sq) {axis = 0 : i32}      # sum(x²)
  tt.store %ya, %o

  # After: fixed number of programs + strided loop, vectorized row reduction
  %groups = arith.divui %total, %c64          # ceil(T / 64) groups
  scf.for %iv = %pid to %groups step %num_programs {
    %v   = tt.load %xa : tensor<64x128x!tt.ptr>    # 32KB contiguous data in one shot
    %s1  = tt.reduce(%v) {axis = 1 : i32}     # → 64
    %sq  = arith.mulf %v, %v
    %s2  = tt.reduce(%sq) {axis = 1 : i32}
    tt.store %ya, %o, %mask                   # tail mask
  }
  ```

  - **收益**：program数从T降为固定值；单次load从512字节变为32KB连续数据；标量归约变为向量化行归约
  - **约束**：只能有一个public入口函数、返回值为空、不能有函数调用；函数内不能出现scf控制流、原子操作、barrier、随机数、print；必须匹配上述两种归约形态之一（判定严格：需检查平方项是否同一值自乘、两个归约结果下游是否汇合等）；token轴 program\_id 唯一且不读 num\_programs；store/load 来自不同入参；改写后UB峰值须通过资源评估；已有grid契约只能是1个非token轴、非持久化的变换（即IAT）；BLOCK\_T 固定，不做调优；simt\_only 下关闭
- 11.6.3 RowCoalescing：行合并
  - **VV形态（逐行Vector kernel）**：匹配"取pid → 越界直接返回 → 否则处理一行"这种写法的kernel，改为一个program处理R行。R依据工作区内最大张量元素数自动选取：≤16选8，17\~1023 选4，1024选2，超过1024不做
  - **DSL写法（触发示例）**：取pid → 越界直接返回 → 否则处理一行

  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n_rows, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      if pid >= n_rows:                  # out of bounds, return immediately
          return
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      v = tl.load(x_ptr + offs)
      o = tl.maximum(v, 0.0)
      tl.store(y_ptr + offs, o)
  ```

  ```ttir
  # Before: take pid → return immediately if out of bounds → otherwise process one row
  %done = arith.cmpi sge, %pid, %n_rows
  cf.cond_br %done, ^exit, ^work
  ^work:
    %v = tt.load %xa : tensor<16x!tt.ptr>
    %o = arith.maxnumf %v, %z
    tt.store %ya, %o

  # After: one program handles 8 rows, the branch disappears
  %rows = arith.addi %pid*8, %arange(8)      # 8 row indices
  %mask = arith.cmpi slt, %rows, %n_rows     # tail mask
  %v    = tt.load %xa : tensor<8x16x!tt.ptr>
  %o    = arith.maxnumf %v, %cst
  tt.store %ya, %o, %mask
  ```

  - **VV收益**：program数降为 1/8；cf.cond\_br 分支开销消失；单次处理8×16元素，SIMT线程利用更充分
  - **VV约束**：只在 compile\_mode=simt\_only 下生效；module只能有一个public入口且无函数调用；只有一个 program\_id，且不读该轴 num\_programs；pid派生值不能在工作区外使用；工作区只允许 arith/math 运算、load/store、指针与形状类算子、scan/reduce、scf.for，不支持矩阵乘；工作区计算结果不能流出；与IAT、PTSM或已有coalesce属性互斥；在所有阶段之后单独执行一次，不占用64次改写配额
  - **CV形态（decode attention行合并）**：针对"每个program处理一行query、在K维循环里做矩阵乘"的decode attention模式：把R个program合成一个，随行变化的值加一个前导行维，矩阵乘时reshape为 \[R×M, K] 以撑大M维；K/V 这类与行无关的数据保持二维、R行共享。R从 32/16/8/4/2 中选取，要求M×R ≤ 128且最大张量字节数×R ≤ 128KB
  - **DSL写法（触发示例）**：每program处理一行query，在K维循环里做矩阵乘

  ```python
  @triton.jit
  def decode_attn(q_ptr, k_ptr, o_ptr, K, M: tl.constexpr, N: tl.constexpr):
      pid = tl.program_id(0)             # one query row per program
      offs_m = tl.arange(0, M)
      offs_n = tl.arange(0, N)
      a = tl.load(q_ptr + pid * M + offs_m)[:, None]     # [M, 1], varies per row
      acc = tl.zeros((M, N), dtype=tl.float32)
      for i in range(0, K, N):           # K-dimension loop doing the matmul
          b = tl.load(k_ptr + i + offs_n)[None, :]       # [1, N], K/V independent of the row
          acc += tl.dot(a, b)            # [M,1] × [1,N] → [M,N]
      tl.store(o_ptr + pid * M + offs_m, tl.sum(acc, axis=1))
  ```

  ```ttir
  # Before: one query row per program, matmul loop over K (M=16 too small, wastes Cube)
  %a = tt.load %pa : tensor<16x16x!tt.ptr>
  %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) {
    %b   = tt.load %pb : tensor<16x16x!tt.ptr>
    %dot = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
  }
  tt.store %pc, %loop

  # After: R rows fused into one program, M padded to 128 to fill the Cube; K/V shared as 2-D
  %a   = tt.load %pa : tensor<8x16x16x!tt.ptr>      # A gains a leading row dimension
  %rhs = tt.load %pb : tensor<16x16x!tt.ptr>        # B stays 2-D, shared by 8 rows
  %dot = tt.dot %a2d, %rhs, %acc : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
  tt.store %pc, %dot, %mask
  ```

  - **CV收益**：矩阵乘M维从16撑到128，填满Cube单元（M维过小会严重浪费算力）；K/V 从GM读取8次降为1次
  - **CV约束**：只支持Ascend950；只能有一个基本块和一个public入口；必须恰好有1个axis0 program\_id、1个下界为0的顶层scf.for、1个顶层store，且至少1个矩阵乘；矩阵乘必须是二维 f16/bf16，且A随行变化、B不随行变化；不支持tt.reduce、循环内依赖行的load、volatile load、用于变长/间接索引的标量load；循环上界依赖pid时必须能证明单调递增；循环体内含不可推测执行的算子（如可能除零的整数除法）时拒绝；R的选取是保守经验策略，非实测最优值
- 11.6.4 QueryRowBatching：块稀疏attention的行合并
  - **触发模式**：NSA一类块稀疏attention——每个program一行query，循环按该行的块表跳块：每轮读一个标量块号，`scf.if` 判它是否被选中，中了才载入 `[K,BS]`/`[BS,V]` 做QK dot + online softmax。K/V 的token地址字面依赖数据，既过不了结构化地址分析，也无法证明可跨行共享，Cube的M轴只跑1行query。改写分两步：`SelectedBlockScan`（扫描）先把循环改成按token顺序的扫描，`QueryRowBatching`（行合并）再把R行query并进一个program；扫描只在这份沙箱里的行提升被接受时才提交，否则整次丢弃、IR逐字节不变
  - **DSL写法（触发示例）**：块表 + 块级 `if` 的稀疏attention循环（`_exp` 是fp32的exp辅助；bwd dQ是同一形态，carried state换成1个加法累加器）

  ```python
  for i in range(NS):
      i_s = tl.load(block_indices + i).to(tl.int32) * BS          # read one scalar block id per iteration
      if i_s <= Q_OFFSET + i_t and i_s >= 0:                     # only process selected blocks
          b_k = tl.load(tl.make_block_ptr(k, (K, TK), (1, H * K), (0, i_s), (BK, BS), (0, 1)), boundary_check=(0, 1))
          b_v = tl.load(tl.make_block_ptr(v, (TK, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)), boundary_check=(0, 1))
          b_s = tl.where((Q_OFFSET + i_t >= i_s + tl.arange(0, BS))[None, :], tl.dot(b_q, b_k), float("-inf"))
          b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m          # online softmax
          b_r = _exp(b_mp - b_m); b_p = _exp(b_s - b_m[:, None])
          b_acc = b_acc * b_r + tl.sum(b_p, 1)
          b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
  ```

  - **改写（扫描）**：整表在循环外一次向量读入；循环内无条件载入宽 `BN = max(64, BS)` 的tile，"该块是否被选中"变成每token权重，`scf.if` 折成零权重，K/V 地址于是只依赖循环变量

  ```ttir
  # Before: scalar block id read inside the loop + block-level scf.if
  %r:3 = scf.for %i = 0 to %NS step 1 iter_args(%o = 0, %m = -inf, %l = 0) {
    %id = tt.load (tt.addptr %rowBase, %i) : !tt.ptr<i64>     # one scalar block id per iteration
    %g  = andi(cmpi sle %bs, %row, cmpi sge %bs, 0)           # block-level predicate
    %t:3 = scf.if %g -> (o, m, l) { ... QK dot + online softmax ... }
           else { scf.yield %o, %m, %l }                      # not selected ≡ zero weight
    scf.yield %t#0, %t#1, %t#2
  }

  # After: the whole table is read early; the loop unconditionally loads a wide tile, and the block predicate becomes per-token weights
  %ids   = tt.load (tt.addptr (tt.splat %rowBase), %lane) : tensor<16xi64>  # read the whole table outside the loop
  %start = arith.muli (arith.trunci %ids), %c64                             # block start of each slot
  %scan:3 = scf.for %iv = 0 to %ub step 1 iter_args(%o = 0, %m = -inf, %l = 0) {
    %ts  = arith.muli %iv, %c64                                            # token start of the tile
    %k   = tt.load (tt.make_tensor_ptr %kBase ... [0, %ts])                 # no condition needed
    %cnt = "tt.reduce"(addi, axis = 1) (andi (cmpi eq %start, %ts), %slot)  # times this tile is selected
    %w   = arith.sitofp %cnt                                               # 0 = not selected
    %valid = andi (cmpf ogt %w, %zero), cmpi sle (%ts + range(64)), %row    # weight & causality
    ...                                                                    # scoring / masked softmax / PV
  }
  ```

  - **改写（行提升）**：R行query合进一个program，循环携带值压平成 `[R*M, N]`，dot的M轴从1行/块抬到R行/块；启动契约把网格换成 `ceil_div(grid, R)`，kernel用隐藏参数还原真实行数

  ```ttir
  module attributes { hacc.program_grid_transforms = {
    extent_source = "runtime_original_grid",
    hidden_argument_order = ["originalGridX", "originalGridY"],
    transforms = [{axis = 0, factor = 8, kind = "ceil_div"}], version = 2}}
  tt.func public @parallel_nsa_fwd_kernel(..., %TQ: i32, %TK: i32,
          %__hacc_original_grid_x: i32, %__hacc_original_grid_y: i32)
  ```

  - **收益**：program数降为 1/R；一个 K/V tile由R行共享，dot的M轴从1行撑到R行，Cube不再空转；每program的寻址与启动开销摊薄到R行
  - **约束**：单 `scf.for`、单axis-0 `program_id`；K/V load直供 `tt.make_tensor_ptr`（fwd）或 `tt.addptr` + 显式掩码（bwd，仅加法形态）；carried state只认两种——online softmax（`o/m/l`，`-inf` 中性元）与纯加法（1个累加器，0中性元，`BN = BS`）；`BS` 须为2的幂、表长 `S <= 4096`，表宽取 `TW = next_pow2(S)` 并把补齐槽位钳到 `S-1`；行相关 `scf.if`、循环内带掩码store、循环内多一个张量load、`grid` 不可证单调的循环上界一律拒绝；varlen能匹配但联合门不提交（无法证明一个tile内的R行同属一条序列）；R由 L0A/L0C/单值/网格规模与UB峰值预算选出，`R < 2` 不提升，预算不过只发 `warning: selected-block-scan: ... declining` 并放弃改写
- 11.7确认图优化pass是否命中
  - **IR检索**：在生成的IR中搜索特征标记——`hivm.multi_buffer` + `arith.select` 槽位结构（多缓冲）、`convert_layout`（布局转换）、`annotation.mark {break_vf}`（VF断点）、fixpipe（NZ2ND）——缺失说明对应模式未命中
  - **常用图优化开关**：

    | 开关                            | 作用                                           | 默认           |
    | ----------------------------- | -------------------------------------------- | ------------ |
    | `--enable-auto-multi-buffer`  | 自动多缓冲总开关                                     | true         |
    | `--vf-fusion-mode`            | VF融合模式（max-parallel / ub-aware-op / all-op） | max-parallel |
    | `--set-workspace-multibuffer` | GM Workspace多缓冲数量（950默认2，A3/A2 默认4）      | 随硬件          |

- 11.8干预手段
  - **hint断点**：`compile_hint("break_vf")` 在VF融合链上放置不可跨越的断点，拆开过长的融合域（回指第6章）
  - **关闭开关**：`--enable-auto-multi-buffer=false`、`--limit-auto-multi-buffer-*` 系列，用于性能对比或问题定位
  - **溢出回退**：多缓冲溢出时编译器先退让溢出层次再关总开关，回退成功但对应收益丢失，此时更合适的是减小tiling分块
  - **调优建议**：先保持默认验证命中，再逐项关闭对比，避免一步到位导致误判

***
