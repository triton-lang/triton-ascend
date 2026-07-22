TA 层 Multibuffer 设计方案（控制流后接入）

将 BiShengIR 的 round-robin 多缓冲逻辑上移到 TA 层，插在 DynamicCVPipeline 末尾（AddControlFlowCondition 之后、RemoveSsbufAttr 之前）。

1. 具体案例

1.1 算子层面

以 Vector 核心上的逐元素计算 kernel 为例（简化版 fused attention 中的残差+归一化）：

@triton.jit
def elemwise_kernel(in_ptr, out_ptr, N):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    for i in range(0, N // BLOCK):
        x = tl.load(in_ptr + offsets)      # DMA 写入临时 buffer
        x = x * 2.0 + 1.0                  # Vector 计算
        tl.store(out_ptr + offsets, x)      # DMA 读出临时 buffer
        offsets += BLOCK * grid_size

编译后，循环内的临时 buffer（x）被分配为 memref.alloc，每轮迭代复用同一块 UB 地址空间。load 写入它，store 从它读出——形成 WAR 假依赖：iter1 的 load 不能早于 iter0 的 store，因为两者打到同一物理地址。

1.2 ttadapter 层面

DynamicCVPipeline 运行到 AddControlFlowCondition 之后，IR 形态如下（已简化）：

func.func @kernel(%in: memref<128xf16, gm>, %out: memref<128xf16, gm>) {
  scf.for %iv = %c0 to %c128 step %c8 {ssbuffer.main_loop} {
    // 每个 block_id 的计算块被 scf.if 包裹（AddControlFlowCondition 产出）
    scf.if %cond3 {
      %buf = memref.alloc() : memref<8xf16, #hivm.address_space<ub>>
      hivm.copy ins(%in) outs(%buf) {ssbuffer.block_id = 3}
    }
    scf.if %cond5 {
      %res = hivm.copy ins(%buf_yielded) outs(%tmp) {ssbuffer.block_id = 5}
    }
    scf.if %cond7 {
      hivm.copy ins(%res_yielded) outs(%out) {ssbuffer.block_id = 7}
    }
  }
}

问题：%buf 在每轮迭代的 block_id=3 中重新 alloc，block_id=5 使用，block_id=7 store。iter1 的 load（block_id=3）需要等 iter0 的 store（block_id=7）完成——WAR 假依赖，DMA 引擎被迫串行。

期望效果：给 %buf 分配两块不重叠的物理地址（buf0、buf1），偶数迭代用 buf0，奇数迭代用 buf1。硬件 DMA 引擎检测到 iter1 的 load 目标地址（buf1）与 iter0 在途的 store 目标地址（buf0）不同 → 不阻塞 → 并行发射。

2. 需求说明

在 DynamicCVPipeline 的 AddControlFlowCondition 之后，对主循环（ssbuffer.main_loop 标记）内存在跨迭代 WAR 依赖的 memref.alloc buffer，自动做 round-robin 多缓冲变换：
1）识别：扫描主循环内的 memref.alloc，判断是否被 load 写入且被 store/compute 读出，存在跨迭代 WAR 依赖
2）拆分：为 buffer 创建 N 个独立的 slot（memref.alloc * N，hoist 到函数入口）
3）选槽：在循环体头部插入 counter % N -> arith.select 选择当前迭代使用的 buffer slot
4）计数：循环体尾部插入 counter += 1

关键约束：不修改循环体结构（不拆体、不引入 flag 握手），只替换 buffer 的引用为 round-robin 选槽结果。并行由硬件 DMA/VEC 独立队列自动实现。

3. 约束规格/预期收益

3.1 约束规格

buffer 类型：memref.alloc，在 scf.for / scf.while 的 body 内分配
跨迭代依赖：buffer 被 load（写）和 store/compute（读）使用，存在 WAR 假依赖
非 L0C 空间：Cube 累加器（L0C）有独立流水，不需要此优化
buffer 大小静态：编译期可确定，保证 slot 数可计算
slot 数量：默认 2（双缓冲），可通过属性配置
不影响已有控制流：scf.if 包裹的计算块条件不变
不影响 AllocMultiCache：跨 block 的 tensor 多缓冲和内层/外层 scope 多缓冲已在前面 pass 完成

3.2 预期收益

1）存在 WAR 依赖：
消除 DMA load 和上一轮 DMA store 之间的假阻塞，MTE2（load）、PIPE_V（compute）、MTE3（store）三条队列可以并行执行不同迭代的指令，性能相比单缓冲有正向收益（理论加速比接近 1.5x-2x，取决于 compute/load 时间比）

2）不存在 WAR 依赖（buffer 仅在单次迭代内使用，无跨迭代复用）：
MarkMultiBuffer 不会标记，EnableMultiBuffer 不会变换，不影响已有功能、性能

4. 方案设计

0、移除原有的 SeparateMemoryFromCompute

当前 DynamicCVPipeline 中 SeparateMemoryFromComputePass 已被移除（AddDynamicCVPipeline.cpp L95 直接跳到 AllocMultiCache）。本方案不再恢复该 pass，用 TA 层 round-robin 方案替代。

1、MarkMultiBuffer：分析并标记可多缓冲的 buffer

输入：AddControlFlowCondition 之后的 IR（主循环已标记 ssbuffer.main_loop，计算块已包裹 scf.if）

步骤：
1）找到带 ssbuffer.main_loop 属性的 scf.for / scf.while
2）遍历循环体内的所有 memref::AllocOp
3）对每个 alloc：
   - 追溯其 memref 的所有 user，判断是否存在 load（写）和 store/compute（读）操作
   - 排除 L0C 地址空间（AddressSpace::L0C）
   - 排除动态大小的 buffer
   - 排除已被标记的 buffer
   - 满足条件：添加 ssbuffer.multi_buffer = 2 : i32 属性

伪代码：

for (auto forOp : moduleOp.getOps<scf::ForOp>()) {
  if (!forOp->hasAttr("ssbuffer.main_loop")) continue;
  forOp.walk([&](memref::AllocOp allocOp) {
    if (allocOp->hasAttr("ssbuffer.multi_buffer")) return;
    if (isL0C(allocOp)) return;
    if (hasDynamicShape(allocOp)) return;
    if (hasWARPattern(allocOp, forOp)) {
      allocOp->setAttr("ssbuffer.multi_buffer",
                       IntegerAttr::get(i32Type, 2));
    }
  });
}

WAR 模式检测（hasWARPattern）：
- 存在 user 是 load 类操作（写 buffer）且在循环体内
- 存在 user 是 store/compute 类操作（读 buffer）且在循环体内
- load 和 store 在不同迭代间存在 WAR 依赖（相同 buffer，先 store 后 load）

2、EnableMultiBuffer：拆分 slot + 插入 round-robin 选槽

输入：MarkMultiBuffer 之后的 IR（带 ssbuffer.multi_buffer 属性的 memref.alloc）

步骤：

Step 1：Hoist buffer slot 分配到函数入口

// 在函数入口创建 N 个独立的 memref.alloc
for (int i = 0; i < numSlots; ++i) {
  auto slotAlloc = builder.create<memref::AllocOp>(
      loc, allocOp.getType());
  slotBuffers.push_back(slotAlloc);
}

Step 2：创建迭代计数器

// 函数入口
auto counterType = MemRefType::get({1}, i64Type);
auto counter = builder.create<memref::AllocaOp>(loc, counterType);
auto c0 = builder.create<arith::ConstantIntOp>(loc, 0, 64);
builder.create<memref::StoreOp>(loc, c0, counter, ValueRange{});

Step 3：循环体头部插入选槽级联

// 在循环体第一条指令前
auto iter = builder.create<memref::LoadOp>(loc, counter, ValueRange{c0});
auto idx = builder.create<arith::RemUIOp>(loc, iter, numSlotsConst);
// 选槽级联：默认选 slot0，逐级 arith.select
Value selected = slotBuffers[0];
for (int i = 1; i < numSlots; ++i) {
  auto ci = builder.create<arith::ConstantIntOp>(loc, i, 64);
  auto cond = builder.create<arith::CmpIOp>(loc, CmpIPredicate::eq, idx, ci);
  selected = builder.create<arith::SelectOp>(loc, cond, slotBuffers[i], selected);
}

Step 4：替换原 buffer 引用
将循环体内所有对原 memref.alloc 结果的引用替换为 arith.select 的结果。

Step 5：循环体尾部插入计数器递增

// 在循环体最后一条指令后
auto c1 = builder.create<arith::ConstantIntOp>(loc, 1, 64);
auto next = builder.create<arith::AddIOp>(loc, iter, c1);
builder.create<memref::StoreOp>(loc, next, counter, ValueRange{c0});

Step 6：删除原循环内的 memref.alloc

变换前后 IR 对比：

// ===== 变换前 =====
scf.for %iv = %c0 to %c128 step %c8 {ssbuffer.main_loop} {
  scf.if %cond3 {
    %buf = memref.alloc() {ssbuffer.multi_buffer = 2} : memref<8xf16, ub>
    hivm.copy ins(%in) outs(%buf) {ssbuffer.block_id = 3}
  }
  scf.if %cond5 {
    hivm.copy ins(%buf_yielded) outs(%tmp) {ssbuffer.block_id = 5}
  }
  scf.if %cond7 {
    hivm.copy ins(%res_yielded) outs(%out) {ssbuffer.block_id = 7}
  }
}

// ===== 变换后 =====
// 函数入口：独立 slot buffer
%buf0 = memref.alloc() : memref<8xf16, ub>
%buf1 = memref.alloc() : memref<8xf16, ub>

// 迭代计数器
%counter = memref.alloca() : memref<1xi64>
memref.store %c0, %counter[]

scf.for %iv = %c0 to %c128 step %c8 {ssbuffer.main_loop} {
  // Round-Robin 选槽
  %iter = memref.load %counter[]
  %idx = arith.remui %iter, %c2
  %cond = arith.cmpi eq, %idx, %c1
  %buf_selected = arith.select %cond, %buf1, %buf0

  scf.if %cond3 {
    hivm.copy ins(%in) outs(%buf_selected) {ssbuffer.block_id = 3}
  }
  scf.if %cond5 {
    hivm.copy ins(%buf_selected_yielded) outs(%tmp) {ssbuffer.block_id = 5}
  }
  scf.if %cond7 {
    hivm.copy ins(%res_yielded) outs(%out) {ssbuffer.block_id = 7}
  }

  // 计数器递增
  %next = arith.addi %iter, %c1
  memref.store %next, %counter[]
}

3、在 DynamicCVPipeline 中的插入位置

pm.addPass(createPreCheckAvailablePass());
pm.addPass(createStandardizeOpPass());
pm.addPass(createPlanComputeBlockPass());
pm.addPass(createComputeBlockOptPass());
pm.addPass(createSplitDataflowPass());
pm.addPass(createAnalyzeDataFlowPass());
pm.addPass(createAllocMultiCachePass());
pm.addPass(createAddControlFlowConditionPass());
pm.addPass(createMarkMultiBufferPass());      // 新增
pm.addPass(createEnableMultiBufferPass());     // 新增
pm.addPass(createRemoveSsbufAttrPass());

插入位置：AddControlFlowConditionPass 之后、RemoveSsbufAttrPass 之前。

为什么选这个位置：
- 所有计算块划分（PlanComputeBlock）已完成
- 控制流条件（AddControlFlowCondition）已添加，scf.if 结构稳定
- 跨核多缓冲（AllocMultiCache）已处理，不会冲突
- 在属性清理（RemoveSsbufAttr）之前，ssbuffer.* 属性仍可用于标记

4、嵌套循环 / yield 传递处理

buffer 可能通过 scf.yield 在嵌套循环间传递。处理方式：
1）从 buffer 的定义点出发，找到最近的 LoopLikeOpInterface 父节点
2）检查 buffer 是否在循环内被消费（load/store/compute）
3）如果 buffer 被 yield，追踪循环结果到更外层循环，继续查找消费点
4）对于通过 yield 传递的 buffer，在消费所在的循环做 slot 替换（替换 yield 值和循环结果）

5、PlanMemory 无需修改

TA 层的独立 memref.alloc（buf0、buf1）在 lowering 到 HIVM 后变为独立的 hivm.hir.pointer_cast。PlanMemory 为每个 pointer_cast 分配不重叠的物理地址——这正是多缓冲需要的。

TA 层         memref.alloc(buf0)       memref.alloc(buf1)
                ↓ (lowering)              ↓ (lowering)
HIVM 层       pointer_cast(addr0)       pointer_cast(addr1)
                ↓ (PlanMemory)            ↓ (PlanMemory)
硬件          0x000-0x00F               0x010-0x01F  (不重叠)

文件清单：
- lib/DynamicCVPipeline/MarkMultiBuffer.cpp：Phase 1，标记可多缓冲的 buffer
- lib/DynamicCVPipeline/EnableMultiBuffer.cpp：Phase 2，拆分 slot + round-robin 选槽
- include/DynamicCVPipeline/MarkMultiBuffer.h：MarkMultiBuffer 头文件
- include/DynamicCVPipeline/EnableMultiBuffer.h：EnableMultiBuffer 头文件
- include/DynamicCVPipeline/Passes.td：注册新 pass

与 BiShengIR 方案的差异：
- Buffer 创建：BiShengIR 用 pointer_cast(addr0, addr1) 打包多地址，TA 层用多个独立 memref.alloc()
- Slot 拆分：BiShengIR 需要 pointer_cast 拆为多个单地址 pointer_cast，TA 层天然独立无需拆分
- 选槽：都用 arith.select
- 计数器：都用 alloca-based 计数器
- PlanMemory：BiShengIR 需要 ExpandMultiBufferStorageEntry，TA 层无需修改，独立 alloc 自动独立地址
- 复杂度：TA 层省去 pointer_cast 打包/拆包步骤

总结：

本方案将 BiShengIR multibuffer 的 round-robin 选槽逻辑上移到 TA 层，插在 DynamicCVPipeline 末尾（控制流之后），改动最小：
- 新增 2 个 pass：MarkMultiBuffer + EnableMultiBuffer
- 新增 2 行注册：在 AddDynamicCVPipeline.cpp 中
- 不修改：PlanMemory、AllocMultiCache、AddControlFlowCondition 等现有 pass
- 不修改：CMakeLists.txt（新增文件加入即可）

核心优势：不拆循环体、不引入 flag 握手，选槽用 arith.select，地址分配交给 PlanMemory 自动处理。
