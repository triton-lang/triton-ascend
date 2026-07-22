# BiShengIR MultiBuffer vs triton-ascend SeparateMemoryFromCompute 方案对比

## 概述

两者都解决同一个问题：消除跨迭代的内存访问假依赖，让 DMA 和计算单元并行。但设计理念、IR 层次、变换方式截然不同。

---

## 1. 核心对比表

| 维度 | BiShengIR MultiBuffer | triton-ascend SeparateMemoryFromCompute |
|------|----------------------|----------------------------------------|
| **IR 层次** | HIVM（低层，接近硬件） | Triton IR（高层，tensor 级别） |
| **变换方式** | 不拆循环体，只改地址 | 拆循环体为 Producer + Consumer |
| **循环结构** | 保持单 `scf.for` 不变 | 拆成 Producer/Consumer 两段逻辑 |
| **超前深度** | 1 个迭代（只消除相邻 WAR） | depth 个迭代（可配置，默认 2） |
| **同步机制** | 无软件同步（硬件 DMA/VEC 自动并行） | flag 数组做 producer-consumer 握手 |
| **依赖分析** | 不需要（仅检查 alloc 是否在循环内） | 4 阶段 SSA 追溯依赖链 |
| **计数器** | 函数级 `memref.alloca<1xi64>` | 循环 `iter_args` 中的 prod/cons 计数器 |
| **支持的循环** | `scf.for` + `scf.while` | `scf.for`（当前） |
| **Store 处理** | load 和 store 的 buffer 都标记 | 当前只处理 load，store 不参与 overlap |
| **标记方式** | 自动扫描 + `annotation.mark` | `gm_load_bufferable` 属性标记 |
| **地址分配** | PlanMemory pass 独立分配不重叠物理地址 | 循环前手动 `memref.alloc` 每个 slot |
| **深度选择** | 固定 2（scope preload 为 4） | 动态分析 UB 容量决定（2 或 3） |
| **编译管线位置** | HIVM passes 中后期 | DynamicCVPipeline 第 8 步 |

---

## 2. 设计理念对比

### 2.1 BiShengIR：地址级解耦

```
理念：给两套房子轮流住，让上一轮的保洁和下一轮的搬家互不干扰

IR 变换：
  scf.for %iv = ... {              scf.for %iv = ... {
    %buf = pointer_cast(addr0,addr1)   %buf0 = pointer_cast(addr0)  // 拆到函数入口
    use(%buf)                            %buf1 = pointer_cast(addr1)
  }                                    %idx = counter % 2
                                        %buf = select(%idx, %buf1, %buf0)
                                        use(%buf)
                                        counter++
                                      }
```

**核心思想**：保持循环体结构完全不变，只在每轮迭代入口插入一个 `select` 选槽。DMA 和 VEC 的并行由硬件调度器自动完成——编译器不需要显式控制指令发射顺序。

### 2.2 triton-ascend：计算与内存显式分离

```
理念：专门派一个搬家公司提前搬，主人在后面慢慢收拾

IR 变换（简化）：
  scf.for %iv = ... {              scf.for ... iter_args(flags, prod, cons) {
    load → buf                        // === Producer（提前填充）===
    compute(buf)                       if flag0==false && prod<trip:
  }                                      copy → SLOT0; flag0=true; prod++
                                         if flag1==false && prod<trip:
                                           copy → SLOT1; flag1=true; prod++
                                       // === Consumer（落后消费）===
                                       target = cons % 2
                                       selected = select(target, SLOT0, SLOT1)
                                       compute(selected)
                                       if target==0: flag0=false;  // 释放
                                       if target==1: flag1=false;
                                       cons++
                                     }
```

**核心思想**：显式把循环体拆成 Producer（提前 DMA 搬数据）和 Consumer（落后执行计算），通过 flag 数组做软件握手。Producer 和 Consumer 使用不同的迭代计数器（prod/cons），实现任意深度的流水。

---

## 3. IR 层次差异

```
Triton 高层 IR（tensor 语义）
    │
    │  SeparateMemoryFromCompute（triton-ascend）
    │  在 tensor 级别操作：bufferization.to_tensor + linalg.matmul
    ▼
    │  （多层 lowering...）
    │
    ▼
HIVM 低层 IR（memref + 物理地址）
    │
    │  MarkMultiBuffer → EnableMultiBuffer → PlanMemory（BiShengIR）
    │  在 memref 级别操作：hivm.hir.load/store + pointer_cast
    ▼
硬件指令
```

**triton-ascend** 在较高的 Triton IR 层做变换，操作的是 `tensor`、`memref.copy`、`linalg.matmul`。优势是可以利用高层语义信息（如 tensor shape、compute 类型）做更智能的深度选择（2 buffer vs 3 buffer）。

**BiShengIR** 在较低的 HIVM 层做变换，操作的是 `memref`、`hivm.hir.load/store`、`pointer_cast`。优势是离硬件更近，可以直接控制物理地址分配，变换也更轻量。

---

## 4. 同步机制对比

### 4.1 BiShengIR：无软件同步

```
时间线（双缓冲）：
  iter0: DMA_load(buf₀) ──barrier── VEC_add ──barrier── DMA_store(buf₀)
  iter1:                               DMA_load(buf₁) ──barrier── VEC_add ──barrier── DMA_store(buf₁)
                                         ↑ buf₁ ≠ buf₀, 硬件自动并行
```

不需要软件 flag。DMA 引擎和 VEC 计算单元各自独立发射指令，只要地址不冲突（buf₀ ≠ buf₁），硬件调度器自动实现并行。编译器只负责分配不重叠的地址。

### 4.2 triton-ascend：Flag 握手

```
时间线（双缓冲，depth=2）：
  iter0: [Producer: copy→SLOT0, flag0=true, prod=1]
         [Consumer: 没有可消费数据，等待]
  iter1: [Producer: copy→SLOT1, flag1=true, prod=2]
         [Consumer: target=0, 选 SLOT0, compute, flag0=false, cons=1]
  iter2: [Producer: copy→SLOT0, flag0=true, prod=3]
         [Consumer: target=1, 选 SLOT1, compute, flag1=false, cons=2]
```

通过 `iter_args` 中的 flag 数组和 prod/cons 计数器显式控制流水节奏。Producer 只在 flag 为 false 时填充，Consumer 只在 flag 为 true 时消费。这种机制可以支持任意深度的流水（depth > 2）。

---

## 5. 超前深度对比

### BiShengIR：深度固定 = 1

```
只能重叠相邻迭代：
  DMA(iter i+1)  ||  VEC(iter i)
  
不能重叠：
  DMA(iter i+2)  ||  VEC(iter i)   ← 需要 3 buffer
```

### triton-ascend：深度可配置

```
depth=2（当前默认）：
  DMA(iter i+1)  ||  Compute(iter i)
  store 不参与 overlap

depth=3（如果 UB 容量允许）：
  DMA(iter i+2)  ||  Compute(iter i+1)  ||  Store(iter i)
  load、compute、store 三者可以同时进行
```

---

## 6. Store 处理差异

### BiShengIR

Load 的 dst buffer 和 Store 的 src buffer **都会被标记** multibuffer。也就是说 load 和 store 各自独立做双缓冲：

```mlir
scf.for %iv = ... {
  %buf = memref.alloca() : memref<8xf16, #hivm.address_space<ub>>
  // buf 被标记 multibuffer（因为它是 load 的 dst）
  hivm.hir.load ins(%gm) outs(%buf) ...
  
  %res = memref.alloca() : memref<8xf16, #hivm.address_space<ub>>
  // res 也被标记 multibuffer（因为它是 store 的 src）
  %r = compute(%buf) outs(%res)
  hivm.hir.store ins(%r) outs(%gm) ...
}
```

### triton-ascend

当前**只处理 load**，store 跟在 compute 后面串行执行：

```
load(N+1) to buf[1]  ||  compute(N) reads buf[0], store(N) writes buf[0] → GM
```

设计文档明确指出这是当前限制，未来可能扩展到 3 buffer 以覆盖 store 的 overlap。

---

## 7. 深度选择策略对比

### BiShengIR

- 默认 depth = 2（固定）
- Scope preload 场景 depth = 4
- 不做 UB 容量分析，不做自动降级
- 如果 UB 溢出，通过命令行 `--disable-multi-buffer-on-ub` 手动关闭

### triton-ascend

- 默认 depth = 2
- 通过 `BufferCountManager` 全局管理深度配置
- 分析 UB 容量决定是否可以升到 depth = 3
- 自动降级：3 buffer 溢出 → 降为 2 buffer → 仍溢出则跳过
- 考虑 tiling 和 buffer depth 的权衡

---

## 8. 适用场景

| 场景 | BiShengIR MultiBuffer | triton-ascend SeparateMemoryFromCompute |
|------|----------------------|----------------------------------------|
| Vector 核心逐元素计算 | ✅ 核心场景 | ✅ 支持 |
| Cube+Vector Mix 核心 | ✅ 通过 CVPipelining 集成 | ✅ 通过 DynamicCVPipeline |
| scf.while 循环 | ✅ 支持 | ❌ 当前不支持 |
| 深度流水（depth > 2） | ❌ 不支持 | ✅ 支持（UB 容量允许时） |
| 编译期未知大小的 buffer | ❌ 跳过 | ❌ 跳过 |
| Store overlap | ✅ load + store 都处理 | ❌ 仅 load |
| 大 tiling 场景 | ✅ 适用（1 层 overlap 足够） | ✅ 可自动降级 |

---

## 9. 总结

| | BiShengIR MultiBuffer | triton-ascend SeparateMemoryFromCompute |
|--|----------------------|----------------------------------------|
| **哲学** | 简单、局部、不拆体 | 显式分离、软件握手、深度可控 |
| **复杂度** | 低（4 个 pass，变换局部） | 高（依赖分析 + 循环拆分 + 状态机） |
| **泛化性** | for/while 统一 | 仅 for（当前） |
| **性能天花板** | 1 层 overlap | depth 层 overlap |
| **UB 压力** | 2× alloc | 2× 或 3× alloc（自动降级） |
| **可维护性** | 高（变换简单可预测） | 中（逻辑复杂，边界条件多） |
