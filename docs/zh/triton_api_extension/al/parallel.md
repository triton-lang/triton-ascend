# al.parallel

## 1. 函数概述

`parallel` 是一个专门用于多核心并行执行的迭代器，继承自 `range` 类，提供显式的多核心并行语义。

```python
al.parallel(arg1, arg2=None, step=None, num_stages=None,
            loop_unroll_factor=None, bind_sub_block: bool = False)
```

## 2. 规格

### 2.1 参数说明

| 参数                 | 类型               | 默认值  | 含义说明                              |
| -------------------- | ------------------ | :------ | ------------------------------------- |
| `arg1`               | `int` /`constexpr` | 必需    | 起始值（单参数时作为结束值，从0开始） |
| `arg2`               | `int`/`constexpr`  | -       | 结束值（不包含在范围内）              |
| `step`               | `int` /`constexpr` | `1`     | 每次迭代的步长增量                    |
| `num_stages`         | `int`              | -       | 流水线阶段数（同时执行的迭代数量）    |
| `loop_unroll_factor` | `int`              | -       | 循环展开因子（<2表示不展开）          |
| `bind_sub_block`     | `bool`             | `False` | 当前未生效（见 2.3）                  |

> **注意**：`parallel` 相比于 `range` 移除了以下参数：
>
> - `disallow_acc_multi_buffer`
> - `flatten`
> - `warp_specialize`
> - `disable_licm`

### 2.2 说明

`al.parallel` 对比 `tl.range`，在前端代码生成阶段唯一的区别是：会给对应的 `scf.for` 多打一个 `hivm.parallel_loop` unit attribute，其余（`num_stages`/`loop_unroll_factor` 等）走的是和 `range` 完全一样的处理逻辑。

`hivm.parallel_loop` 是打给 AscendNPU-IR 里同步分析相关 pass（`GraphSyncSolver`/`CrossCoreGSS` 这一族）的提示：**同一个循环、不同次迭代之间不存在需要跨迭代保序的数据依赖，可以跳过为此插入的同步指令**。

需要注意几点：

- 它不是分析手段，是分析的**跳过开关**。AscendNPU-IR 本身有一套基于读写集合（WAW/RAW/WAR）的自动依赖分析，不管标不标 `parallel_loop` 都会跑，很多真正无依赖的循环它自己就能判断出来。`hivm.parallel_loop` 的作用是在这套分析针对"跨迭代依赖"的检查**开始之前**就直接短路掉，强制得出"不需要同步"的结论。
- 它弥补的是自动分析故意保留的保守盲区：当不同迭代访问的是**同一块 buffer**（例如循环内复用的多缓冲 UB 临时 tile，或者写同一个 GM 输出指针）时，自动分析并不比较实际的偏移区间，只要命中同一块 buffer 就保守地判定为冲突、插入同步。`hivm.parallel_loop` 让用户用领域知识覆盖这个保守判断。
- 这是一个**用户向编译器做出的、未经验证的承诺**，不是编译器验证过的结论。如果循环实际存在跨迭代依赖（例如累加器模式，每次迭代读写同一个地址）却被标成 `parallel_loop`，编译器不会报错，会直接产出缺同步、结果错误的代码——正确性完全压在用户身上，语义上和 C 的 `restrict`、OpenMP 的 `#pragma omp parallel for` 是同一类契约。

### 2.3 特殊限制说明

`bind_sub_block` 目前是一个只存不用的死参数：Python 侧 `parallel.__init__` 里把它存成 `self.bind_sub_block`，但前端代码生成、以及下游 AscendNPU-IR 的所有 pass 都没有任何地方读取过这个字段，`bind_sub_block=True` 和 `bind_sub_block=False` 生成的 IR 完全一样。

### 2.4 dtype 支持

`al.parallel` 本身不携带 dtype 信息（循环边界只能是 int/constexpr），循环体内部数据的 dtype 支持与 `tl.range` 完全一致。

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| --- | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |
| Ascend 950 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |

除 `fp64` 编译报错外，其余类型（含 `uint16`/`uint32`/`uint64`）在 A2/A3 与 Ascend 950 上均可用，与仓库自带的 `docs/zh/python-api/_ascend_constraints.py`（额外标注 A2/A3 不支持 `uint16`/`uint32`/`uint64`）不一致，以此处为准。

### 2.5 与 `tl.range` 的实际差异示例

以下面第 3 节的 `parallel_kernel` 为例：`al.parallel(0, 2)` 循环里，每次迭代都会往同一块被标记为 `hivm.multi_buffer = 2`（2 个物理槽位做双缓冲）的 UB 临时 tile 里写入结果，再 store 到 GM。用

```bash
bishengir-compile kernel.mlir ... --mlir-print-ir-after=hivm-graph-sync-solver
```

拿到 `GraphSyncSolver` pass（当前负责插入 `hivm.hir.set_flag`/`wait_flag`/`pipe_barrier` 等同步指令的 pass）之后的 IR，分别对比 `tl.range(0, 2)` 和 `al.parallel(0, 2)` 两个版本：

**`tl.range` 版本**（循环体内多出一整套"多缓冲槽位复用保护"）：

```mlir
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
  %3 = hivm.hir.multi_buffer_counter -> i64            // 读运行时计数器，判断这次该用哪个物理缓冲槽
  %4 = arith.remui %3, %c2_i64 : i64
  %5 = arith.index_cast %4 : i64 to index
  %6 = arith.index_cast %5 : index to i1
  %7 = arith.select %6, %c0_i64_0, %c1_i64 : i64       // %7 = 本次迭代用的槽位编号（0 或 1）
  %8 = hivm.hir.pointer_cast(%c65536_i64, %c98304_i64) : memref<8192xf32, #hivm.address_space<ub>>
  annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<8192xf32, #hivm.address_space<ub>>
  %11 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%arg7]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %7]         // 等上一次用这个槽位的 store 做完，才能开始写
  func.call @parallel_kernel_outlined_vf_0(%11, %2, %8) {hivm.vector_function, no_inline}
      : (index, memref<16384xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_MTE3>]
  hivm.hir.store ins(%8 : memref<8192xf32, #hivm.address_space<ub>>)
                outs(%collapse_shape_2 : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %7]          // 通知"这个槽位这次的 store 完成了"
}
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
```

**`al.parallel` 版本**（跨迭代的缓冲复用保护整段消失，只保留同一次迭代内部 compute → store 的必需同步）：

```mlir
scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
  %3 = hivm.hir.pointer_cast(%c65536_i64, %c98304_i64) : memref<8192xf32, #hivm.address_space<ub>>
  annotation.mark %3 {hivm.multi_buffer = 2 : i32} : memref<8192xf32, #hivm.address_space<ub>>
  %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%arg7]
  func.call @parallel_kernel_outlined_vf_0(%6, %2, %3) {hivm.vector_function, no_inline}
      : (index, memref<16384xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%3 : memref<8192xf32, #hivm.address_space<ub>>)
                outs(%collapse_shape_1 : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
} {hivm.parallel_loop}
```

被砍掉的是：循环外两条预置 `set_flag`、循环内的 `multi_buffer_counter` 读取与槽位选择运算、`wait_flag[...,%槽位]`、`pipe_barrier[PIPE_MTE3]`、`set_flag[...,%槽位]`，以及循环外两条收尾 `wait_flag`——总计约 7 条同步指令 + 一整段槽位选择运算。同一迭代内部 compute → store 这条必需的同步（`set_flag[PIPE_V, PIPE_MTE3]`/`wait_flag[PIPE_V, PIPE_MTE3]`）两个版本都保留，没有被砍。

**这里被砍掉的同步在保护什么**：这块 UB 临时 tile 只开了 2 个物理槽位（`hivm.multi_buffer = 2`），而循环要跑 K 次；如果某次迭代要复用之前用过的槽位，必须等那次的 `store`（把数据搬去 GM）做完才能开始写入新数据，否则会在数据被搬走之前就被覆盖——这是一个典型的跨迭代 WAR 冒险。`tl.range` 版本里那套 `multi_buffer_counter` + 槽位编号 + 对应的 `wait_flag`/`set_flag` 就是在做这件事；`al.parallel` 直接假定不存在这种复用冲突，把这套保护全部去掉。

#### 什么情况下才会看到差异

只有当循环体内**跨迭代访问了同一块 buffer**（比如上面这个例子里被多缓冲复用的 UB tile，或者反复写同一个 GM 输出指针）时，`hivm.parallel_loop` 才会真的改变生成的 IR——因为只有这种场景才会触发自动依赖分析里"同一 buffer 即保守判定冲突"的分支，从而插入本可省略的同步指令，`parallel_loop` 才有东西可省。

如果循环体每次迭代访问的都是互不相干的独立 buffer（不同的 rootBuffer，彼此没有交集），自动依赖分析本来就能自己判断出没有依赖、不会插入任何多余同步——这种情况下不管标不标 `al.parallel`，`hivm-graph-sync-solver` 之后生成的 IR 除了那个 `hivm.parallel_loop` 属性本身之外，不会有其它区别。

本例满足"跨迭代复用同一 buffer"这个前提（多缓冲 UB tile 被 2 次迭代共用 2 个槽位），所以能观测到明显的同步指令差异，是一个有效、能体现 `al.parallel` 实际收益的例子。

## 3. 使用方法

```python
@triton.jit
def parallel_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # Load the full [M, N] block
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    block = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    SUB_M: tl.constexpr = M // 2
    for s in al.parallel(0, 2):
        sub = al.extract_slice(block, (s * SUB_M, 0), (SUB_M, N), (1, 1))
        sub = sub * 2.0
        offs_sub_m = s * SUB_M + tl.arange(0, SUB_M)
        out_ptrs = out_ptr + offs_sub_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, sub)
```
