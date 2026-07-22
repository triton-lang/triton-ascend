MultiBuffer 适用场景分析

1. 问题背景

Ascend NPU 上，单缓冲下的典型 Vector 循环：

    for i in range(N):
        buf = alloc_UB()
        load GM[i] -> buf      // DMA_load
        result = compute(buf)  // VEC 计算
        store result -> GM[i]  // DMA_store

瓶颈：第 i 轮 DMA_store(buf) 和第 i+1 轮 DMA_load(buf) 打到同一物理地址，形成 WAR（写后读）假依赖。虽然 DMA 引擎和 VEC 计算单元硬件独立，但地址冲突导致 DMA 和 VEC 无法并行。

方案：双缓冲 —— 分配 buf0 / buf1 两块不重叠物理地址，迭代轮流使用。IR 保持单 scf.for 不拆体，靠 round-robin 选槽 + 硬件调度器自动并行。

    双缓冲:
      iter0: DMA_load(buf0) --barrier-- VEC_vadd --barrier-- DMA_store(buf0)
      iter1:                                     DMA_load(buf1) --barrier-- VEC_vadd --barrier-- DMA_store(buf1)
                                                    buf1 != buf0, 无地址冲突, DMA 和 VEC 并行


2. 四阶段变换流程

阶段    Pass                      关键函数                              作用
Phase1  hivm-mark-multi-buffer    MarkMultiBuffer::matchAndRewrite      自动分析 buffer liveness，打 hivm.multi_buffer 标记
Phase2  中间 pass                  AllocToPointerCast 等                将 memref.alloca + annotation.mark 转为 hivm.hir.pointer_cast（携带多个物理地址）
Phase3  hivm-enable-multi-buffer  MultiBufferHelper::extMultiBuffer     拆分 pointer_cast + 插入 round-robin select 级联
Phase4  hivm-plan-memory          MemPlan::ExpandMultiBufferStorageEntry 为多缓冲 slot 分配不重叠物理地址

关键设计决策：

1. Alloca-based 迭代计数器：函数级 memref.alloca<1xi64> + 循环体内 load/store，统一支持 scf.for 和 scf.while，不修改循环签名
2. Loop ID 系统：每个 multibuffer 循环获得唯一 hivm.multi_buffer_loop_id 属性，计数器 alloca 带匹配的 hivm.multi_buffer_counter_for 属性，支持跨 pass 幂等发现
3. 幂等性：多次构造 MultiBufferLoopAdapter 不会重复创建计数器基础设施


3. 三种默认场景

3.1 Vector 搬运 —— Load/Store（GM <-> UB）

buffer 在 UB 空间，被 Load 或 Store 使用，对应端为 GM。是 multibuffer 的最典型场景：DMA 和 VEC 交错，WAR 假依赖消除后收益接近 2x。

条件：
- buffer 空间：UB（#hivm.address_space<ub>）
- buffer 大小：静态（memref.alloca() 无动态参数）
- 循环类型：scf.for / scf.while
- 使用者：hivm.hir.load 或 hivm.hir.store
- 方向：至少一端非 GM（GM->UB 标记 dst，UB->GM 标记 src）
- 核类型：AIV 或 MIX（Vector 组启用时）
- 命令行：--disable-multi-buffer-on-ub 不为 true
- 标记方式：hivm.multi_buffer = 2

IR 示例（mark-multi-buffer.mlir#L12-L37）：

    scf.for %i0 = %c0 to %c16 step %c4 {
        %tmp2 = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
        // 自动标记: annotation.mark %tmp2 {hivm.multi_buffer = 2 : i32}
        hivm.hir.load  ins(%in  : memref<8xf32, gm>) outs(%tmp2 : memref<8xf32, ub>)
        hivm.hir.store ins(%tmp4 : memref<8xf32, ub>) outs(%out : memref<8xf32, gm>)
    }


3.2 Cube 输入端 —— ND2NZ（GM -> L1 cbuf）

ND2NZ 从 GM 搬运数据到 L1 cbuf，matchAndRewrite 中 dst（L1）非 GM 触发 markBufferFunc 标记 L1 buffer。

条件：
- buffer 空间：L1（#hivm.address_space<cbuf>）
- buffer 大小：静态
- 循环类型：scf.for / scf.while
- 使用者：hivm.ND2NZ，dst 非 GM
- 核类型：MIX 且 allowCubeGroup = true
- 命令行：--disable-multi-buffer-on-l1 不为 true

- WAR 冲突：iter0 的 mmadL1 读 %cbuf vs iter1 的 nd2nz 写 %cbuf
- 双缓冲后：nd2nz（DMA）和 mmadL1（Cube）可并行
- 标记：hivm.multi_buffer = 2

Cube 输出端（Fixpipe，L0C -> UB/GM）默认不做 multibuffer。代码注册条件（MarkMultiBuffer.cpp#L301-L304）：

    if (limitAutoMultiBufferOfLocalBuffer != CUBE_NO_L0C && !disableMultiBufferOnL0C)
        patterns.insert<MarkMultiBuffer<hivm::FixpipeOp>>(...);  // 默认 CUBE_NO_L0C，不注册

Fixpipe 的 src 是 L0C 累加器，若进入 markBufferFunc 会尝试标记 L0C buffer，但 L0C 有跨迭代 RAW 真依赖不适合 multibuffer，所以默认被策略排除。对 UB 端的 Store（路径 A 中 fixpipe->store），Store 走 Vector 组自己的标记路径，不受此限制。


3.3 Workspace —— MIX 核 Fusion Junction 缓冲

Workspace 是 kernel 级预分配的 GM scratchpad buffer，通过 memref_ext.alloc_workspace 从函数参数切出。MIX 核的 cross-core fusion junction（CC/CV/VC/VV）中间结果 tensor.empty 无物理内存时，用 workspace 兜底。

独立于 3.1/3.2 的 MarkMultiBuffer，使用 MarkWorkspaceMultiBuffer pattern，追溯 AllocWorkspaceOp 而非 memref.alloca。

条件：
- buffer 来源：memref_ext.alloc_workspace 在循环内
- 函数类型：MIX 核
- 开关：limitAutoMultiBufferOnlyForLocalBuffer 为 false
- 写入操作：Store/Fixpipe 在循环内写入 workspace
- 标记方式：hivm.multi_buffer = workspaceMultiBufferNum（默认 2）

IR 示例（mark-multi-buffer.mlir#L97-L133）：

    scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 {
        %5 = memref_ext.alloc_workspace() from %arg1 : from memref<?xi8> to memref<16x16xf32>
        // 标记: annotation.mark %5 {hivm.multi_buffer = 2 : i32}
        hivm.hir.fixpipe ins(%4) outs(%6)
    }


4. 需注意的非默认场景

4.1 Scope Preload（仅在 CVPipeline Skew 模式下生效）

仅在开启 CVPipeline（--enablecv-pipeline-mode=Skew 或 --enable-preload=true）时生效。preload_num 由 CVPipelining 内部 createNewLoopsForPreloadWithScopes() 写入 scope 属性（--enable-preload=true 是强制 Skew 模式的向后兼容别名），MarkMultiBuffer 仅被动读取。

条件：
- scope：scope.scope 内部 buffer
- preload_num：> 0（由 CVPipelining 写入）
- core_type：VECTOR（非 CUBE）
- 使用者：buffer 被 V1（下游 scope）使用
- 地址空间：非 GM
- 标记：hivm.multi_buffer = 4 + hivm.preload_local_buffer = 1

IR 示例（mark-multi-buffer.mlir#L171-L198）：

    %40 = scope.scope : () -> memref<1x2048xf16, ub> {
        %39 = memref.alloc() : memref<1x2048xf16, ub>
        // 标记: annotation.mark %39 {hivm.multi_buffer = 4 : i32, hivm.preload_local_buffer = 1 : i32}
        hivm.hir.load ins(%arg0) outs(%39)
        scope.return %39
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}


4.2 嵌套循环 yield 传递 & scf.while 支持

通过 getParentLoop() 递归追踪 yielded value（EnableMultiBuffer.cpp#L54-L148），穿越嵌套 loop 和 scf.if，找到真正消费 buffer 的循环作为轮转锚点。scf.while 和 scf.for 同等支持（MarkMultiBuffer.cpp#L185-L192），依赖 alloca-based 迭代计数器统一处理。

这是跨场景共享机制，不是独立标记入口。见测试用例中的 test_for_yield、test_2for_yield、test_3for_yield（mark-multi-buffer.mlir#L40-L168）。


5. 不能做 MultiBuffer 的场景

5.1 L0C 空间 buffer（Cube 累加器）

#hivm.address_space<cc> 是 Cube 矩阵累加器，有独立流水线，不适合 multibuffer。
MarkMultiBuffer.cpp 中 markBufferFunc 会通过 tracebackMemRef 找到 allocOp，L0C 的 buffer 不会被 Load/Store 使用，且 Fixpipe 也有 CUBE_NO_L0C 控制。

5.2 动态大小 buffer

memref.alloca(%dyn) 编译期无法确定大小，无法静态分配多份 buffer slot。

    %tmp3 = memref.alloca(%d) : memref<?xf32, #hivm.address_space<ub>>
    // 不会被标记 -- 动态大小

5.3 不在循环内的 buffer

multibuffer 的核心是跨迭代轮转，buffer 必须在循环内。
代码位置：markBufferFunc 中检查 getParentLoop(allocOp->getResult(0))，如果返回 null 则跳过。

5.4 不支持的循环类型

只支持 scf.for 和 scf.while，不支持 scf.parallel、scf.forall 等。
代码位置（MarkMultiBuffer.cpp#L185-L192）：

    while (parentLoop) {
        if (!isa<scf::ForOp, scf::WhileOp>(parentLoop)) {
            return failure();
        }
        parentLoop = parentLoop->getParentOfType<LoopLikeOpInterface>();
    }

5.5 命令行禁用

通过命令行选项可按地址空间精细控制：

    --disable-multi-buffer-on-ub=true  -> UB buffer（Vector Load/Store）
    --disable-multi-buffer-on-l0c=true -> L0C Cube 累加器（Fixpipe）
    --disable-multi-buffer-on-l1=true  -> L1 cbuf（ND2NZ）

5.6 已标记过的 buffer

避免重复标记，检查 isMarked(allocOp)。

5.7 Buffer 两端都在 GM

multibuffer 针对片上 buffer（UB/L1），Load/Store 的 dst 和 src 必须有一端非 GM。
代码位置（MarkMultiBuffer.cpp#L213-L221）：

    if (getHIVMAddressSpace(src.getType()) != hivm::AddressSpace::GM) {
        return markBufferFunc(src);  // src 非 GM -> 标记 src
    }
    if (getHIVMAddressSpace(dst.getType()) != hivm::AddressSpace::GM) {
        return markBufferFunc(dst);  // dst 非 GM -> 标记 dst
    }
    return failure();  // src 和 dst 都是 GM -> 不标记

5.8 Pure tensor semantics 的 ops

markBufferFunc 要求 hasPureBufferSemantics()。memref 类型才有 memory_space 地址空间信息，tensor 类型没有。所以通用 MarkMultiBuffer 只在 bufferization 之后的 buffer 态生效。Workspace 的 MarkWorkspaceMultiBuffer 相反，要求 tensor 态。

5.9 Workspace 在非 MIX 核

MarkWorkspaceMultiBuffer 只在 isMixFuncCore 为 true 时注册 pattern。
代码位置（MarkMultiBuffer.cpp#L315-L318）：

    if (!limitAutoMultiBufferOnlyForLocalBuffer && isMixFuncCore)
        patterns.insert<MarkWorkspaceMultiBuffer<...>>(...);

5.10 Scope preload 的 Cube scope

MarkScopeMultiBuffer 过滤 tcore_type == CUBE 的 scope。

    if (!tCoreType || tCoreType.getTcoretype() == hivm::TCoreType::CUBE)
        return failure();

5.11 Pure Vector 核上 Cube 组被禁用

当 limitMixAutoMultiBufferBuffer == ONLY_VECTOR 时，Cube 组（ND2NZ、Fixpipe）的 pattern 不被注册。

5.12 Pure Cube 核上 Vector 组被禁用

当 limitMixAutoMultiBufferBuffer == ONLY_CUBE 时，Vector 组（Load、Store）的 pattern 不被注册。


6. 决策树总结

    buffer 是否在 scf.for/scf.while 循环内？
      +- 否 -> 不做
      +- 是 -> buffer 地址空间是什么？
          +- L0C (cc) -> 不做（Cube 累加器独立流水线）
          +- L1 (cbuf)
          |   +- 被 ND2NZ 使用 + dst 非 GM + 未禁用 L1 -> 做（multi_buffer = 2）
          |   +- 其他 -> 不做
          +- UB
          |   +- 被 Load/Store 使用 + 对应端是 GM + 未禁用 UB -> 做（multi_buffer = 2）
          |   +- 被 Fixpipe 使用 + 未禁用 L0C + CUBE_NO_L0C 未设 -> 做（multi_buffer = 2）
          |   +- 其他 -> 不做
          +- Workspace (MIX 核)
              +- 被 Store/Fixpipe 写入 + 循环内 + 未限制 local only -> 做
              +- 其他 -> 不做
    buffer 是否动态大小？
      +- 是 -> 不做
      +- 否 -> 继续
    buffer 是否已标记？
      +- 是 -> 不做
      +- 否 -> 继续
    Scope preload？
      +- scope 内 buffer + preload_num > 0 + VECTOR tcore + 下游 V1 使用 -> 做（multi_buffer = 4）
      +- 否 -> 不做 preload 标记


7. 关键代码位置

文件                   绝对路径                                                                      职责
设计文档               .../docs/MultiBuffer-CVPipeline-HIVMLevel.md                                 完整变换流程、动机
MarkMultiBuffer        .../lib/Dialect/HIVM/Transforms/MarkMultiBuffer.cpp                          Pass 1：自动分析标记
EnableMultiBuffer      .../lib/Dialect/HIVM/Transforms/EnableMultiBuffer.cpp                        Pass 3：round-robin 选槽
MultiBufferLoopAdapter .../lib/Dialect/HIVM/Utils/MultiBufferLoopAdapter.cpp                        统一迭代计数器
PlanMemory             .../lib/Dialect/HIVM/Transforms/PlanMemory.cpp                               Pass 4：地址分配
Pass Pipeline          .../lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp                             Pass 编排
