# triton-ascend SeparateMemoryFromCompute 经典 Load-Compute 分离

来源：`third_party/ascend/unittest/.../SeparateMemoryFromCompute/basic_transform_test.mlir` — `@tc_b01_double_buffer_basic`

---

## 输入 IR（变换前）

单层 `scf.for`，`memref.copy`（DMA 搬运）→ `bufferization.to_tensor`（标记 `gm_load_bufferable`）→ `linalg.matmul`（计算）。

```mlir
func.func @tc_b01_double_buffer_basic(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32    = arith.constant 28    : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32   = arith.constant 128   : i32
  %c0_i32     = arith.constant 0     : i32
  %c128       = arith.constant 128   : index

  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
      iter_args(%acc = %arg1) -> tensor<128x128xf32> : i32 {

    // === ① DMA: 从 GM 搬运到 UB ===
    %iv_idx    = arith.index_cast %arg16 : i32 to index
    %row_off   = arith.muli %iv_idx, %c128 : index
    %q_src_rc  = memref.reinterpret_cast %arg0
                   to offset: [%row_off], sizes: [128, 128], strides: [128, 1]
                   : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %q_alloc   = memref.alloc () : memref<128x128xf16>
    memref.copy %q_src_rc, %q_alloc
      : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>

    // 标记：这个 load 要被 SeparateMemoryFromCompute 处理
    %q_tensor  = bufferization.to_tensor %q_alloc restrict writable
                   {gm_load_bufferable} : memref<128x128xf16>

    // === ② 计算: matmul ===
    %qk = linalg.matmul {input_precision = "ieee"}
            ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>)
           outs(%acc : tensor<128x128xf32>)
            -> tensor<128x128xf32>

    scf.yield %qk : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}
```

**关键特征**：
- `{gm_load_bufferable}` 属性是触发变换的标记（类比 BiShengIR 的 `hivm.multi_buffer`）
- 循环内 `memref.copy`（DMA load）→ `linalg.matmul`（compute），没有显式 store
- 这是 FlashAttention 风格的分块矩阵乘法，只做了 load 的 multibuffer

---

## 输出 IR（变换后）

Producer/Consumer 显式分离：循环体拆为两段，通过 flag 数组做软件握手。

```mlir
func.func @tc_b01_double_buffer_basic(
  %arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %arg1: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %c28_i32    = arith.constant 28    : i32
  %c65536_i32 = arith.constant 65536 : i32
  %c128_i32   = arith.constant 128   : i32
  %c0_i32     = arith.constant 0     : i32
  %c128       = arith.constant 128   : index
  %false      = arith.constant false
  %c0         = arith.constant 0     : index
  %c1         = arith.constant 1     : index
  %c2         = arith.constant 2     : index

  // ===================================================================
  // 新增 ①：循环前预分配 2 个 buffer slot
  // ===================================================================
  %Q_SLOT0 = memref.alloc() : memref<128x128xf16>
  %Q_SLOT1 = memref.alloc() : memref<128x128xf16>

  // Trip count = ceildiv(65536 - 0, 28)
  %trip = arith.ceildivui %c65536_i32, %c28_i32 : i32

  // ===================================================================
  // 循环签名扩展：原有 1 个 iter_args + 新增 4 个控制 iter_args
  //   flag0, flag1: 对应 slot0/slot1 是否"满"（可消费）
  //   prod: Producer 计数器（超前 consumer 多少个迭代）
  //   cons: Consumer 计数器（当前消费到第几个迭代）
  // ===================================================================
  %result = scf.for %arg16 = %c0_i32 to %c65536_i32 step %c28_i32
      iter_args(
        %acc   = %arg1,                          // 原有
        %flag0 = %false, %flag1 = %false,        // 新增: 2 个 flag
        %prod  = %c0,    %cons  = %c0            // 新增: prod/cons 计数器
      ) -> (tensor<128x128xf32>, i1, i1, index, index) : i32 {

    // ===============================================================
    // === Producer 段：提前搬运（超前 Consumer 1~2 个迭代）==========
    // ===============================================================

    // --- Slot 0 ---
    // 条件：flag0 == false（槽空）AND prod < trip_count（还有迭代要搬）
    %has_more_0 = arith.cmpi ult, %prod, %trip : index
    %can_fill_0 = arith.andi %flag0_inv, %has_more_0 : i1
    %new_flag0_0, %new_prod_0 = scf.if %can_fill_0 -> (i1, index) {
      // 计算超前 IV: lb + prod * step
      %prod_i32 = arith.index_cast %prod : index to i32
      %prod_virtual = arith.muli %prod_i32, %c28_i32 : i32
      %prod_iv = arith.addi %c0_i32, %prod_virtual : i32     // lb + prod*step
      %prod_iv_idx = arith.index_cast %prod_iv : i32 to index

      // 用超前 IV 计算地址
      %prefetch_off = arith.muli %prod_iv_idx, %c128 : index
      %q_src = memref.reinterpret_cast %arg0
                 to offset: [%prefetch_off], sizes: [128, 128], strides: [128, 1]
                 : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>

      // 提前 DMA 搬运到 slot0
      memref.copy %q_src, %Q_SLOT0
        : memref<128x128xf16, strided<[128, 1], offset: ?>>
            to memref<128x128xf16>

      %prod_inc = arith.addi %prod, %c1 : index
      scf.yield %true, %prod_inc : i1, index
    } else {
      scf.yield %flag0, %prod : i1, index
    }

    // --- Slot 1 ---（同理）
    %flag1_inv = arith.xori %flag1, %true : i1
    %has_more_1 = arith.cmpi ult, %new_prod_0, %trip : index
    %can_fill_1 = arith.andi %flag1_inv, %has_more_1 : i1
    %new_flag1_1, %new_prod_1 = scf.if %can_fill_1 -> (i1, index) {
      %prod_i32 = arith.index_cast %new_prod_0 : index to i32
      %prod_virtual = arith.muli %prod_i32, %c28_i32 : i32
      %prod_iv = arith.addi %c0_i32, %prod_virtual : i32
      %prod_iv_idx = arith.index_cast %prod_iv : i32 to index
      %prefetch_off = arith.muli %prod_iv_idx, %c128 : index
      %q_src = memref.reinterpret_cast %arg0
                 to offset: [%prefetch_off], sizes: [128, 128], strides: [128, 1]
                 : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      memref.copy %q_src, %Q_SLOT1
        : memref<128x128xf16, strided<[128, 1], offset: ?>>
            to memref<128x128xf16>
      %prod_inc = arith.addi %new_prod_0, %c1 : index
      scf.yield %true, %prod_inc : i1, index
    } else {
      scf.yield %flag1, %new_prod_0 : i1, index
    }

    // ===============================================================
    // === Consumer 段：消费 buffer slot，执行计算 ===================
    // ===============================================================

    // Round-robin 选槽: target = cons % 2
    %target = arith.remui %cons, %c2 : index
    %is_slot1 = arith.cmpi eq, %target, %c1 : index

    // 将选中的 slot 转为 tensor
    %q_tensor_0 = bufferization.to_tensor %Q_SLOT0
                    restrict writable : memref<128x128xf16>
    %q_tensor_1 = bufferization.to_tensor %Q_SLOT1
                    restrict writable : memref<128x128xf16>
    %q_tensor = arith.select %is_slot1, %q_tensor_1, %q_tensor_0
                  : tensor<128x128xf16>

    // 计算：matmul（原逻辑保留）
    %qk = linalg.matmul {input_precision = "ieee"}
            ins(%q_tensor, %q_tensor : tensor<128x128xf16>, tensor<128x128xf16>)
           outs(%acc : tensor<128x128xf32>)
            -> tensor<128x128xf32>

    // 释放 flag：消费完 slot 后标记为空
    %new_flag0_2 = arith.select %is_slot1, %new_flag0_0, %false : i1
    %new_flag1_2 = arith.select %is_slot1, %false, %new_flag1_1 : i1

    // Consumer 计数器 +1
    %cons_inc = arith.addi %cons, %c1 : index

    scf.yield %qk, %new_flag0_2, %new_flag1_2, %new_prod_1, %cons_inc
      : tensor<128x128xf32>, i1, i1, index, index
  }
  return %result : tensor<128x128xf32>
}
```

---

## 变换总结

```
输入                                        输出
───────────────────────────────────────    ───────────────────────────────────────
单段循环体                                 两段循环体: Producer + Consumer

memref.copy (循环内)                   →   memref.copy (Producer scf.if 内, 超前搬运)
bufferization.to_tensor {gm_load_...}  →   bufferization.to_tensor (Consumer 内, 选槽后)
linalg.matmul                           →   linalg.matmul (Consumer 内, 不变)

scf.for iter_args(%acc)                →   scf.for iter_args(%acc, flag0, flag1, prod, cons)
                                           +4 个控制 iter_args: 2 flags + prod + cons

无计数器                                  prod/cons 双计数器 (iter_args 内)
无同步机制                                flag 数组软件握手
```

**变换本质**：
1. 循环前预分配 2 个 buffer slot
2. 循环签名扩展 4 个 iter_args（2 flags + prod + cons）
3. 循环体拆为两段：
   - **Producer**：用 `prod` 计算超前 IV，提前 `memref.copy` 到空闲 slot，设 flag=true
   - **Consumer**：`cons % 2` 选槽 → to_tensor → matmul → 释放对应 flag
4. flag 握手保证：Producer 只在槽空时写入，Consumer 只在槽满时读取

---

## 时间线

```
iter0: [Producer: copy→SLOT0, flag0=true, prod=1]
       [Consumer: 无数据可消费，跳过]
iter1: [Producer: copy→SLOT1, flag1=true, prod=2]
       [Consumer: target=0, 选 SLOT0→tensor→matmul, flag0=false, cons=1]
        ↑ DMA(SLOT1) 和 MATMUL(SLOT0) 并行
iter2: [Producer: copy→SLOT0, flag0=true, prod=3]
       [Consumer: target=1, 选 SLOT1→tensor→matmul, flag1=false, cons=2]
        ↑ DMA(SLOT0) 和 MATMUL(SLOT1) 并行
...
```

---

## 与 BiShengIR EnableMultiBuffer 的核心差异

| | BiShengIR `multi_buffer_alloc_manual` | TA `tc_b01_double_buffer_basic` |
|---|---|---|
| **同步方式** | 无软件同步，硬件自动并行 | flag 数组 + prod/cons 软件握手 |
| **循环结构** | 单 `scf.for`，不拆体 | 拆为 Producer(if) + Consumer 两段 |
| **超前深度** | 1（相邻迭代 WAR 消除） | depth 个迭代（可配置，默认 2） |
| **计数器** | 函数级 `alloca<1xi64>` | 循环 `iter_args` 中的 prod/cons |
| **标志管理** | 不需要 flag | 每个 slot 一个 bool flag |
| **IV 投影** | 不修改地址计算 | Producer 用 `lb + prod*step` 计算超前地址 |
| **Store** | load + store 都做 | 仅 load（当前） |
| **适用循环** | scf.for + scf.while | scf.for（当前） |

**一句话**：BiShengIR 是"给两套房子，硬件自己调度"；triton-ascend 是"显式派搬家公司提前搬，主人后面算"。
