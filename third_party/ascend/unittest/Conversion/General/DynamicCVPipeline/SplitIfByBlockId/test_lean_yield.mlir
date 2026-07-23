// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ==========================================================================
  // 新设计验证：消除非必要的 yield 传递
  //
  // 核心原则：
  // 1. 中间 split-if 只 yield 自己产出的跨组值，不带原始 yield 类型
  // 2. 纯副作用组用 void if（零 result、无 else）
  // 3. 只有最后一个 split-if 携带原始 result 类型
  // 4. 非最后 group 产出的原始 yield 值，最后一个 if 跳跃引用（非逐级透传）
  // ==========================================================================

  // --------------------------------------------------------------------------
  // Case 1: then 侧三组，中间组纯副作用 → void if
  //
  // G0 (bid=10): 地址计算 → 产出 %offset (index)
  // G1 (bid=11): DMA 搬运 → 消费 %offset，纯副作用，无 SSA 产出
  // G2 (bid=12): 最终计算 → 产出 %result，即原始 yield 值
  //
  // 旧方案：3 个 if 全部带 (tensor<64xf32>, index) 两个 slot
  // 新方案：G0 只 yield index，G1 void if，G2 只 yield tensor
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_side_effect_void_if
  func.func @test_side_effect_void_if(%cond: i1, %src: memref<64xf32>, %dst: memref<64xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<64xf32>
    %fallback = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst : f32) outs(%t0 : tensor<64xf32>) -> tensor<64xf32>
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %c4 = arith.constant {ssbuffer.block_id = 14 : i32} 4 : index
    %a = arith.constant {ssbuffer.block_id = 14 : i32} 1.0 : f32
    %b = arith.constant {ssbuffer.block_id = 14 : i32} 2.0 : f32
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index

    // G0 (bid=10): pure side-effect group, void if (no result)
    // %offset is dead code so no cross-group values to yield
    // CHECK: scf.for
    // CHECK: scf.if %{{.*}} {
    // CHECK-NEXT: arith.addi {{.*}} {ssbuffer.block_id = 10 : i32}
    // CHECK: }

    // G1 (bid=11): void if, no result, no else
    // CHECK: scf.if {{.*}} {
    // CHECK-NOT: ->
    // CHECK-NEXT: memref.copy {{.*}} {ssbuffer.block_id = 11 : i32}
    // CHECK: }
    // G1 must NOT have an else block (Scene 2 with then-split)
    // CHECK-NOT: else

    // G2 (bid=12): last if, only original yield type
    // CHECK: [[G2:%.*]] = scf.if {{.*}} -> (tensor<64xf32>)
    // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 12 : i32}
    // CHECK: scf.yield
    // CHECK: else
    // CHECK: scf.yield {{.*}} : tensor<64xf32>

    scf.for %iv = %lb to %ub step %step {
      %result = scf.if %cond -> (tensor<64xf32>) {
        %offset = arith.addi %c0, %c4 {ssbuffer.block_id = 10 : i32} : index
        memref.copy %src, %dst {ssbuffer.block_id = 11 : i32} : memref<64xf32> to memref<64xf32>
        %val = arith.addf %a, %b {ssbuffer.block_id = 12 : i32} : f32
        %r = tensor.empty() {ssbuffer.block_id = 12 : i32} : tensor<64xf32>
        %result = linalg.fill {ssbuffer.block_id = 12 : i32} ins(%val : f32) outs(%r : tensor<64xf32>) -> tensor<64xf32>
        scf.yield %result : tensor<64xf32>
      } else {
        scf.yield %fallback : tensor<64xf32>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // --------------------------------------------------------------------------
  // Case 2: then 侧两组，G0 产出原始 yield slot 0 和跨组值，G1 产出 slot 1
  //
  // G0 (bid=20): 产出 %v1 (原始 slot 0)，同时被 G1 消费（跨组依赖）
  // G1 (bid=21): 消费 %v1，产出 %v2 (原始 slot 1)
  //
  // 旧方案：2 个 if 全部带 (tensor<16xf32>, tensor<16xf32>) 两个 slot
  // 新方案：G0 只 yield %v1，G1 带两个原始 slot，slot 0 跳跃引用 G0
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_nonlast_produces_original_yield
  func.func @test_nonlast_produces_original_yield(%cond: i1) {
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst : f32) outs(%t0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 14 : i32} 1.0 : f32
    %t1 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %f1 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst2 : f32) outs(%t1 : tensor<16xf32>) -> tensor<16xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index

    // G0 (bid=20): yields only %v1 (1 result), no augmented passthrough
    // CHECK: scf.for
    // CHECK: [[G0:%.*]] = scf.if {{.*}} -> (tensor<16xf32>)
    // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 20 : i32}
    // CHECK: scf.yield
    // CHECK: else
    // CHECK: scf.yield

    // G1 (bid=21): last if, yields both original slots
    // slot 0: jump reference from [[G0]] (in then block)
    // slot 1: %v2 produced by G1
    // else yields original else values (%f0 for both slots)
    // CHECK: [[G1:%.*]]:2 = scf.if {{.*}} -> (tensor<16xf32>, tensor<16xf32>)
    // CHECK-NEXT: arith.mulf [[G0]], {{.*}} {ssbuffer.block_id = 21 : i32}
    // CHECK: scf.yield [[G0]], {{.*}} : tensor<16xf32>, tensor<16xf32>
    // CHECK: else
    // CHECK: scf.yield {{%.*}}, {{%.*}} : tensor<16xf32>, tensor<16xf32>

    scf.for %iv = %lb to %ub step %step {
      %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
        %v1 = arith.addf %f1, %f0 {ssbuffer.block_id = 20 : i32} : tensor<16xf32>
        %v2 = arith.mulf %v1, %f1 {ssbuffer.block_id = 21 : i32} : tensor<16xf32>
        scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %f0, %f0 : tensor<16xf32>, tensor<16xf32>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // --------------------------------------------------------------------------
  // Case 3: V1/V2/V3 — V3 直接从 V1 取值，不经过中间 void if
  //
  // V1 (bid=30): 产出 %a (index) 和 %b (index) 两个跨组值
  // V2 (bid=31): 消费 %a，产出死代码 %unused（无下游 consumer）→ void if
  // V3 (bid=32): 消费 %b 直接从 V1，产出原始 yield 值
  //
  // 关键：V2 是 void if → 没有 result → 无法做 passthrough
  //       V3 引用 [[V1]]#1 证明值直接从 V1 跳到 V3，不经过 V2
  // --------------------------------------------------------------------------

  // V1 yields both %a and %b (2 results)
  // CHECK-LABEL: func.func @test_v1_to_v3_skip_void_v2
  // CHECK: scf.for
  // CHECK: [[V1:%.*]]:2 = scf.if {{.*}} -> (index, index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 30 : i32}
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 30 : i32}
  // CHECK: scf.yield

  // V2 is void if — no results, no else
  // CHECK: scf.if {{.*}} {
  // CHECK-NOT: ->
  // CHECK: memref.subview {{%.*}}[[[V1]]#0] {{.*}} {ssbuffer.block_id = 31 : i32}
  // CHECK: }
  // V2 must NOT have an else block
  // CHECK-NOT: else

  // V3 directly references [[V1]]#1 (NOT through V2)
  // CHECK: [[V3:%.*]] = scf.if {{.*}} -> (tensor<16xf32>)
  // CHECK: arith.index_cast [[V1]]#1 {ssbuffer.block_id = 32 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield

  func.func @test_v1_to_v3_skip_void_v2(%cond: i1, %src: memref<64xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %fallback = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst : f32) outs(%t0 : tensor<16xf32>) -> tensor<16xf32>
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %c4 = arith.constant {ssbuffer.block_id = 14 : i32} 4 : index
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %a = arith.constant {ssbuffer.block_id = 14 : i32} 1.0 : f32
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index

    scf.for %iv = %lb to %ub step %step {
      %result = scf.if %cond -> (tensor<16xf32>) {
        // V1 (bid=30): 产出 %a 和 %b 两个 index
        %off = arith.addi %c0, %c4 {ssbuffer.block_id = 30 : i32} : index
        %sz = arith.addi %c0, %c64 {ssbuffer.block_id = 30 : i32} : index
        // V2 (bid=31): 消费 %off，产出死代码 %unused
        %unused = memref.subview %src[%off] [4] [1] {ssbuffer.block_id = 31 : i32} : memref<64xf32> to memref<4xf32, strided<[1], offset: ?>>
        // V3 (bid=32): 消费 %sz 直接从 V1，产出 yield
        %idx = arith.index_cast %sz {ssbuffer.block_id = 32 : i32} : index to i32
        %val = arith.sitofp %idx {ssbuffer.block_id = 32 : i32} : i32 to f32
        %r = tensor.empty() {ssbuffer.block_id = 32 : i32} : tensor<16xf32>
        %result = linalg.fill {ssbuffer.block_id = 32 : i32} ins(%val : f32) outs(%r : tensor<16xf32>) -> tensor<16xf32>
        scf.yield %result : tensor<16xf32>
      } else {
        scf.yield %fallback : tensor<16xf32>
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }
}
