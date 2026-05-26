# Flag ID 复用策略与算法总结

> 本文总结 `FlagIdReuse` 当前的复用策略与算法实现，供算法逻辑审查。整体借鉴下层
> GraphSyncSolver（`EventIdSolver` / `CrossCoreGSS`）的事件 ID 分配思路（**干涉图 + 图着色**），
> 并针对 `InterCoreTransferAndSync` 这一层（尚未插入核内同步、循环未展开）的特点做了适配。
>
> 涉及文件：
> - `third_party/ascend/include/DynamicCVPipeline/SplitDataflow/FlagIdReuse.h`
> - `third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/FlagIdReuse.cpp`
> - `third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.cpp`
>   （`insertInterCoreSync` 在创建同步时登记 E2/E4 边；`insertAnalyzeFlagRelations` 构 E1/E3 边；
>   `remapInterCoreTransferFlagIds` 改写 flag 并清除内部标记）

---

## 1. 问题背景与触发条件

`InterCoreTransferAndSyncPass` 在 CUBE/VECTOR 跨核数据依赖处插入 `sync_block_set` /
`sync_block_wait`，每个 transfer 占用一个静态 flag id。硬件可用 flag 数量有限
（`FlagIdManager::MAX_FLAG_ID = 14`）。插入阶段先用 `acquireId` 单调分配；当 transfer 数超过预算时，
需要把**生命周期不重叠**的 transfer 复用到同一个 flag id 上，把所需 flag 数压回预算内。

**门控**：复用只在 `!flagManager.checkCurrentId()`（即已分配的最大 flag id > 14）时触发，避免对
本就够用的 kernel 做无谓重编号。

```
processDependencies 末尾：
    DenseMap<int,int> remapResult;                 // 默认空
    if (!flagManager.checkCurrentId())             // 仅当 flag 不够时
        remapResult = reuseInterCoreTransferFlagIds(insertAnalyzeFlagRelations(...));
    remapInterCoreTransferFlagIds(remapResult);    // 始终调用：清除内部标记 + 应用（可能为空的）remap
```

> `remapInterCoreTransferFlagIds` **始终**被调用：它负责清除内部标记 `ssbuffer.analyze_flag_id`
> （该标记仅用于复用分析、不能泄漏进输出 IR），并在 remap 非空时改写 `static_flag_id`。早期版本把它放在
> 门控内部，导致 ≤14 flag 的 kernel 标记泄漏，已修正。

---

## 2. 核心思想：干涉图 + 图着色

把“哪些 flag 能合并”反过来建模成“哪些 flag **不能**共用”，构建**干涉图（interference graph）**再着色：

- **节点**：每个原始 flagId（拥有一组 set/wait op，按 flagId 分桶于 `flagIdToOps`）。
- **边（干涉）**：两个 flag 的生命周期**可能同时活跃**时连边，二者必须保留不同 id。
- **着色**：贪心给每个节点分配“干涉邻居未占用的最小颜色”，颜色即复用后的 flag id；互不干涉的节点自动
  落到同一颜色——这就是复用。颜色总数 = 实际需要的物理 flag 数。

整体流程（`reuseInterCoreTransferFlagIds`）：

```
preworkForAnalyze(syncOps)   # 分桶 + 记录每个 op 的程序序 rank（opOrder）
buildInterferenceGraph()     # 两两 flagsInterfere -> 无向干涉图
return colorInterferenceGraph()   # 着色 + 紧凑重编号 -> origFlagId -> 新 flagId
```

---

## 3. 干涉判定（核心）：release-before-acquire

两个 flag 能共用 ⟺ 其中一个在程序序/同步语义上**“释放”早于另一个“获取”**。形式化：

```
flagsInterfere(A, B) = !flagReleasedBefore(A, B) && !flagReleasedBefore(B, A)

flagReleasedBefore(before, after) =
    opPrecedes( getLatestWait(before),   // before 组里程序序最靠后的 wait（释放点）
                getEarliestSet(after) )  // after  组里程序序最靠前的 set （获取点）
```

- `getEarliestSet(flagId)` / `getLatestWait(flagId)`：在该 flag 的 set/wait op 中，按 `opOrder`
  （程序序 rank）取最早的 set、最晚的 wait。即把一个 flag 组的生命周期收敛为 `[最早 set, 最晚 wait]`
  的“获取—释放”边界。
- `flagsInterfere`：两个方向都**不**满足 released-before 才判干涉。这是**保守**的——漏判（本可复用却判干涉）
  只会少复用，绝不产生竞争。

### 3.1 `opPrecedes`：程序序 + 可达性

`flagReleasedBefore` 的核心是判断“before 的释放点是否先于 after 的获取点”：

```
opPrecedes(p, q):
    if p == q:                                  return true
    if p、q 同属一个 MLIR block 且 p.isBeforeInBlock(q):  return true   # 同块直接用程序序
    return hasPath(p, q)                         # 否则在 relations 图上做 DFS 可达
```

- **同块**：直接用 MLIR 的 `isBeforeInBlock`（块内静态程序序），精确且无需建图。这正是**兄弟循环复用**
  成立的关键——loop3 的循环后 wait 与 loop4 的循环前 set 同处外层循环体块，`isBeforeInBlock` 直接成立。
- **跨块**：退化到 `hasPath`（relations happens-before 图上的 DFS 可达），需要真实的同步/数据边连接。
  这正是 **scf.if 互斥分支不复用**的原因——then/else 分属不同块、无可达路径 ⇒ 不可达 ⇒ 判干涉。

### 3.2 为什么这样既安全又能复用

- 不相交（一个释放点先于另一个获取点）⇒ 二者绝不同时活跃 ⇒ 共用安全。底层依据见
  [硬件语义](#7-安全性论证)：Ascend flag 是计数信号量 + pipe 内有序，程序序不相交即安全。
- “同块程序序”覆盖了直线链路与兄弟循环；“跨块可达”覆盖了需要同步/数据证明的情形；二者都不成立时
  保守判干涉。

---

## 4. happens-before 图（relations）的构建：E1–E4

`opPrecedes` 的跨块分支依赖 `relations` 这张图。它**只含物理上真实的顺序边**，所以“可达 ⟺ 顺序被保证
⟺ 复用安全”。四类边：

| 边 | 含义 | 何处登记 |
|---|---|---|
| **E1 同 (core, pipe) FIFO** | 单 pipe 按发射序执行；在**同一个 MLIR block 内**串联相邻的同 (core,pipe) op | `insertAnalyzeFlagRelations` |
| **E2 set→wait 同步边** | 跨核唯一的顺序来源 | `insertInterCoreSync`（创建时） |
| **E3 SSA 数据依赖** | 操作数定义先于使用 | `insertAnalyzeFlagRelations` |
| **E4 read-wait → 被传输数据** | 消费者读到有效数据前必先过该 wait | `insertInterCoreSync`（创建时） |

**刻意不建模**：跨核程序序、以及同核跨 pipe 的程序序——这些 op 除非有 E2/E3/E4 边相连，否则视为并发。
这是复用安全的根基。

### 4.1 E1：逐 block、逐 (core,pipe) 的 FIFO

```
module.walk(op):
    # 入图条件“宽松”：是 sync，或带 core（getAnalyzeCoreType）。
    # 宽松是为了让 E3 数据依赖能“穿过”像 math.exp 这种无具体 pipe 的计算 op。
    lastOpInBlock = lastOpOnPipe[op->getBlock()]      # 注意：按 block 隔离，不跨 region 线性化
    for (core, pipe) in getCorePipeInfos(op):         # set 用 tpipe，wait 用 pipe，宏 op 用 in/out
        if lastOpInBlock 有该 (core,pipe):  连边 last -> op
        lastOpInBlock[(core,pipe)] = op
```

- `getCorePipeInfos`：统一的核/管线分类。SyncSet 取 (tcore, tpipe)，SyncWait 取 (tcore, pipe)，
  CopyOp 经 `getCopyPipeForAnalyze`，宏 op 取 in/out 双 pipe，其余经 `OpPipeInterface::getPipe`。
- **按 block 隔离**（`lastOpOnPipe[block]`）：不把不同 region 的 op 全局线性化，避免引入跨 region 的
  虚假顺序；跨 region 的先后必须由显式 E2/E4 边或同块程序序证明。

### 4.2 E3：数据依赖

遍历入图 op 的每个操作数，若定义 op 也在图中，连边 `定义 -> 使用`（block 参数则取其 owner 的 parent op）。
入图条件“宽松”确保 `to_tensor → exp → … → 下一个 copy` 这条数据链不被无 pipe 的计算 op 截断。

### 4.3 E2 / E4：创建时登记（不改 set/wait 本身）

一个 transfer 的 6 个 sync op（read / write-back / loop start-end 三对）共享同一 flagId，且 write-back
对与 start/end 对的 `(core, tpipe, pipe, flagId)` 完全相同，**无法仅凭属性事后配对**。因此在
`insertInterCoreSync` 创建同步时即登记：

- **E2**：对**每一对** set→wait 调用 `insertRelationBetweenSetAndWait`（read / write-back / start-end 全登记）。
- **E4**：把 read-wait 连到它守护的、消费者实际读取的 `to_tensor`（该 op 由 transfer 插入函数经
  `consumedDataOp` 出参传出）。没有 E4，sync op 会与消费侧数据流脱节（详见踩坑记录）。

> 注意：sync op 的创建逻辑、属性、插入位置一字未改，这里只是**额外补登复用图的边**。

### 4.4 踩坑记录：程序序/数据流是骨架，纯 pipe 图会断链

最初只用 E1（per-pipe FIFO）+ E2，结果过于稀疏：`math.exp` 等无具体 pipe 的计算 op 掉出所有 pipe 链，
sync op 成孤岛，无法证明 transfer A 先于 B ⇒ 全员干涉 ⇒ 零复用（链式用例从 1 退化到 3）。
补上 E3（数据依赖，宽松入图）+ E4（read-wait→被传输数据）后，消费侧数据流被接通，链路恢复复用。

---

## 5. 着色与紧凑重编号（`colorInterferenceGraph`）

两步：

1. **Welsh–Powell 贪心着色**：按干涉度数降序（度数相同按 flagId 升序，保证确定性）逐个着色，每个节点取
   干涉邻居未用的最小正色 → `rawRemapResult`（flagId -> 原始色）。先着色最受约束的节点，使总色数尽量小。
2. **按首次出现顺序紧凑重编号**：按各 flag 第一个 op 的程序序排序，把原始色重映射为 `1,2,3,…`（首次用到
   即分配下一个号）→ `remapResult`。这样最终 flag id 既最小化数量，又按程序中首次出现的顺序稳定编号
   （便于阅读/比对，对应 `flag_reuse_compact_renumber` 用例）。

---

## 6. 与门控、标记清除的衔接

- `reuseInterCoreTransferFlagIds` 返回 `origFlagId -> 新 flagId`。
- `remapInterCoreTransferFlagIds(remapResult)`：遍历所有 sync op，**始终**移除 `ssbuffer.analyze_flag_id`
  标记；若该 op 被复用分析跟踪且 remap 非空，则把 `static_flag_id` 改写为新值。未被跟踪（无标记）的
  sync op 保持原 flag 不变。

---

## 7. 安全性论证

- **硬件依据**：Ascend sync flag 是计数信号量，且每条 pipe 内指令按序执行。因此只要两个 flag 的使用在
  程序序上不相交（一个的 wait 释放早于另一个的 set 获取），即便跨核并发，计数信号量也能保证不混淆——
  程序序不相交 ⇒ 复用安全。
- **算法保守性**：`flagReleasedBefore` 仅在“同块程序序”或“真实边可达”成立时返回 true；`relations` 只含
  E1–E4 这些真实顺序边，不含跨核/跨 pipe 虚假顺序。两个方向都不成立才判干涉 ⇒ 任何不确定都偏向“不复用”，
  绝不会把可能并发的两个 flag 错误合并。
- **同循环不复用**：同一 main loop 内的双缓冲 transfer，其 start/end 同步贯穿整个循环，生命周期天然重叠
  （获取点早于对方释放点不成立）⇒ 判干涉 ⇒ 不复用。这是正确语义。

---

## 8. 复杂度

设 flag 数 `F`、入图 op 数 `N`、relations 边数 `E`。
- 干涉图构建：`O(F²)` 对，每对最多对组内 op 做一次 `opPrecedes`（同块 `isBeforeInBlock` 为 O(1)~均摊，
  跨块为一次 DFS `O(N + E)`）。
- 着色 + 重编号：`O(F log F + F²)`。
- 对本场景 transfer 规模足够。

---

## 9. 验证结果

构建 `bash build.sh`；运行
`triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop`。

- **SplitDataflow UT 全量 19/19 通过**。
- `flag_reuse_chain`（16 直线 transfer）→ 全部复用到 1 个 flag。
- `flag_reuse_sibling_{loops,unserialized,dead_iterarg}`（2 个兄弟主循环 + 14 个 filler）→ 两个兄弟
  transfer 复用到同一个 flag（filler 占用另一个独立 flag）。
- `flag_reuse_compact_renumber`（含 scf.if 分支）→ then 两个 transfer 复用、else 与 after 各自独立，
  未标记对保持原值。
- `flag_reuse_over_limit`、`input.mlir`（均 >14 transfer，自然触发门控）→ 无 `flag = -1`，输出可
  round-trip，无 `analyze_flag_id` 泄漏；`input.mlir` 由 16 压到 12 个 flag。

> 小用例为触发门控（>14）做过相应放大：直线链路加长、兄弟循环用例追加 filler 链、compact 用例把未标记对
> 的 flag 提到 15。

---

## 10. 已知限制 / 后续

- **未移植参考实现的“部分重叠区间 + pipe 可达性 Dijkstra 精化”**：当前对生命周期重叠的一律保守判干涉
  （按 transfer 整体的“最早 set / 最晚 wait”边界判断），不会对“区间部分重叠但 set 侧、wait 侧各自被
  pipe 顺序分离”的情形回收复用。本场景的结构化 transfer 暂未触及该情形。
- **PIPE_S 内存依赖同步**仅参与 E1 同 (core,pipe) FIFO，未单独登记其 set→wait 边（保守，可能少复用，但安全）。
- **循环跨迭代**：依赖“外层循环串行执行”这一前提（main loop 仅指内层双缓冲循环；外层非软流水、迭代串行），
  从而同块程序序即可证明兄弟循环的释放-获取先后。若未来外层循环也被软流水，需要额外的迭代重叠（reverse-edge）
  建模。
