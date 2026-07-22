# 经典 Load-Compute-Store 场景：EnableMultiBuffer 输入输出 IR

来源：`bishengir/test/Dialect/HIVM/enable-multi-buffer.mlir` — `@multi_buffer_alloc_manual`

---

## 输入 IR（EnableMultiBuffer 之前）

中间形态：MarkMultiBuffer 已完成标记，pointer_cast 已携带两个物理地址（addr0=0, addr1=16 / addr0=128, addr1=144），但尚未拆分，也没有 round-robin 选槽逻辑。

```mlir
func.func @multi_buffer_alloc_manual(
    %arg0: memref<16xf16, #hivm.address_space<gm>>,
    %arg1: memref<16xf16, #hivm.address_space<gm>>
) {
  %c0_i64   = arith.constant 0   : i64
  %c16_i64  = arith.constant 16  : i64
  %c128_i64 = arith.constant 128 : i64
  %c144_i64 = arith.constant 144 : i64
  %c0  = arith.constant 0  : index
  %c4  = arith.constant 4  : index
  %c16 = arith.constant 16 : index

  // 单层循环，每次迭代 load → vadd → store
  scf.for %arg2 = %c0 to %c16 step %c4 {

    // === buffer %0: load 的目标 (GM→UB)，双地址 ===
    %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) []
           : memref<16xf16, #hivm.address_space<ub>>
    annotation.mark %0 {attr = 1 : i32} : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // === buffer %1: vadd 输出 / store 的来源 (UB→GM)，双地址 ===
    %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) []
           : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // ① DMA_load: 从 GM 搬运到 UB buffer %0
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                 outs(%0   : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // ② VEC 计算: 在 UB 上做 vadd，结果写入 %1
    hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>,
                             memref<16xf16, #hivm.address_space<ub>>)
                 outs(%1      : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // ③ DMA_store: 从 UB buffer %1 写回 GM
    hivm.hir.store ins(%1   : memref<16xf16, #hivm.address_space<ub>>)
                  outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
  }
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}
```

**关键特征**：
- 两个 `pointer_cast` 各带两个物理地址（addr0/addr1），代表两块不重叠的物理内存
- `annotation.mark {attr = 1}` 是 MarkMultiBuffer 阶段的标记残留
- `pipe_barrier[<PIPE_ALL>]` 是硬件同步屏障，multibuffer 不修改它们
- 循环体是经典的 **load → compute → store** 三步流水

---

## 输出 IR（EnableMultiBuffer 之后）

```mlir
func.func @multi_buffer_alloc_manual(
    %arg0: memref<16xf16, #hivm.address_space<gm>>,
    %arg1: memref<16xf16, #hivm.address_space<gm>>
) {
  %c0_i64   = arith.constant 0   : i64
  %c1_i64   = arith.constant 1   : i64
  %c2_i64   = arith.constant 2   : i64
  %c16_i64  = arith.constant 16  : i64
  %c128_i64 = arith.constant 128 : i64
  %c144_i64 = arith.constant 144 : i64
  %c0  = arith.constant 0  : index
  %c4  = arith.constant 4  : index
  %c16 = arith.constant 16 : index

  // ===================================================================
  // 新增 ①：迭代计数器 (alloca-based, 函数级, 统一 scf.for/scf.while)
  // ===================================================================
  %counter = memref.alloca() {hivm.multi_buffer_counter_for = 0 : i64}
               : memref<1xi64>
  memref.store %c0_i64, %counter[%c0] : memref<1xi64>

  // ===================================================================
  // 新增 ②：原 pointer_cast(%addr0, %addr1) 拆分为两个单地址 pointer_cast
  //          hoist 到函数入口，作为两个独立的物理 buffer
  // ===================================================================
  // buffer %0 的两个 slot (load 的目标):
  %0_slot0 = hivm.hir.pointer_cast(%c0_i64)  []
               : memref<16xf16, #hivm.address_space<ub>>
  %0_slot1 = hivm.hir.pointer_cast(%c16_i64) []
               : memref<16xf16, #hivm.address_space<ub>>
  // buffer %1 的两个 slot (vadd 输出 / store 的来源):
  %1_slot0 = hivm.hir.pointer_cast(%c128_i64) []
               : memref<16xf16, #hivm.address_space<ub>>
  %1_slot1 = hivm.hir.pointer_cast(%c144_i64) []
               : memref<16xf16, #hivm.address_space<ub>>

  // 循环体结构完全不变，只在外围插入选槽逻辑
  scf.for %arg2 = %c0 to %c16 step %c4
      {hivm.multi_buffer_loop_id = 0 : i64} {        // 新增：loop ID 属性

    // =================================================================
    // 新增 ③：读计数器 + Round-Robin 选槽 (buffer %0)
    //   idx = iter % 2
    //   buf = (idx == 1) ? slot1 : slot0
    //   偶数迭代选 slot0，奇数迭代选 slot1
    // =================================================================
    %iter_0    = memref.load %counter[%c0] : memref<1xi64>
    %idx_0     = arith.remui %iter_0, %c2_i64 : i64
    %cond_0    = arith.cmpi eq, %idx_0, %c1_i64 : i64
    %0         = arith.select %cond_0, %0_slot1, %0_slot0
                   : memref<16xf16, #hivm.address_space<ub>>

    // 新增 ③：Round-Robin 选槽 (buffer %1)
    %iter_1    = memref.load %counter[%c0] : memref<1xi64>
    %idx_1     = arith.remui %iter_1, %c2_i64 : i64
    %cond_1    = arith.cmpi eq, %idx_1, %c1_i64 : i64
    %1         = arith.select %cond_1, %1_slot1, %1_slot0
                   : memref<16xf16, #hivm.address_space<ub>>

    // =================================================================
    // 原有 load → compute → store（完全不变）
    // =================================================================
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                 outs(%0   : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>,
                             memref<16xf16, #hivm.address_space<ub>>)
                 outs(%1      : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.store ins(%1   : memref<16xf16, #hivm.address_space<ub>>)
                  outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)

    // =================================================================
    // 新增 ④：计数器 +1，写回
    // =================================================================
    %next_iter = arith.addi %iter_1, %c1_i64 : i64
    memref.store %next_iter, %counter[%c0] : memref<1xi64>
  }
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}
```

---

## 变换总结

```
输入                                         输出
────────────────────────────────────────    ────────────────────────────────────────
pointer_cast(addr0, addr1)  (循环内)  →    pointer_cast(addr0)  (函数入口，slot0)
                                            pointer_cast(addr1)  (函数入口，slot1)
                                            
无计数器                                →    memref.alloca<1xi64> 计数器 (函数入口)
                                            memref.store 初始化
                                            
无选槽逻辑                              →    循环头: load counter → remui %2 → cmpi eq → select
                                            循环尾: addi counter → store counter
                                            
无 loop_id 属性                         →    scf.for {hivm.multi_buffer_loop_id = 0}
```

**变换本质**：
1. `pointer_cast` 从"打包多地址"拆成"多个单地址"，hoist 到函数入口
2. 循环体头尾各插入一小段逻辑（选槽 / 计数），循环体核心 **load→vadd→store 一字不改**
3. 不拆循环、不加软件同步、不修改 pipe_barrier

**效果**：偶数迭代选 slot0，奇数迭代选 slot1。DMA_load(slot1) 和 DMA_store(slot0) 地址不同，硬件调度器自动并行发射，实现 DMA 与 VEC 的 overlap。

---

## 对应时间线

```
iter0: DMA_load(buf₀) ==barrier== VEC_vadd ==barrier== DMA_store(buf₀)
iter1:                               DMA_load(buf₁) ==barrier== VEC_vadd ==barrier== DMA_store(buf₁)
                                      ↑ buf₁ ≠ buf₀，地址无冲突
iter2:                                                                DMA_load(buf₀) ...
```

第 i 轮的 DMA_store 还在总线上时，第 i+1 轮的 DMA_load 已经可以往另一块 buffer 写了——WAR 假依赖消除，DMA 和 VEC 计算单元实现并行。
