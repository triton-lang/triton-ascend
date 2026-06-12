// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Case B: 原始 if 有 2 个 yield slot, then 内 block_id={10, 11}
  // 跨组依赖: block 10 产出 %addf, block 11 消费 %addf (在 yield slot 0 中)
  // 期望: 拆分为 2 个链式 if, K=2 (无 augmentation)
  // ============================================================================

  // CHECK-LABEL: func.func @test_case_b_basic
  // 第一个 split if: 只含 block_id=10
  // CHECK: [[R0:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // 第二个 split if: 只含 block_id=11, 消费 [[R0]]#0
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.mulf [[R0]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R0]]#0,
  // CHECK: else
  // CHECK: scf.yield [[R0]]#0, [[R0]]#1
  // 原始 uses 替换为最后一个 if 的 result
  // CHECK: return [[R1]]#0, [[R1]]#1

  func.func @test_case_b_basic(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %3 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
      %v1 = arith.addf %3, %1 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
      %v2 = arith.mulf %v1, %3 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
      scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
    } else {
      scf.yield %1, %1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.block_id = 16 : i32}
    return %4#0, %4#1 : tensor<16xf32>, tensor<16xf32>
  }

  // ============================================================================
  // Case A: 原始 if 无 yield, then 内 block_id={5, 3}, 纯副作用
  // 跨组依赖: block 5 产出 maxsi 结果, block 3 消费 → 需要 1 个 augmented yield slot
  // ============================================================================

  // CHECK-LABEL: func.func @test_case_a_basic
  // 第一个 split if: 新增 yield slot 传递跨组值 (block_id=5)
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // 第二个 split if: 纯副作用, 消费 [[R0]] (block_id=3)
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK-NEXT: arith.index_cast [[R0]] {ssbuffer.block_id = 3 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 3 : i32}

  func.func @test_case_a_basic(%cond: i1) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f16
    %c64_i32 = arith.constant {ssbuffer.block_id = 13 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 13 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : index
    %alloc = memref.alloc() {ssbuffer.block_id = 13 : i32} : memref<64x64xf16>
    scf.if %cond {
      %v1 = arith.maxsi %c64_i32, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
      %v2 = arith.index_cast %v1 {ssbuffer.block_id = 3 : i32} : i32 to index
      %v3 = arith.muli %v2, %c64 {ssbuffer.block_id = 3 : i32} : index
      memref.dealloc %alloc {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
    }
    memref.dealloc %alloc {ssbuffer.block_id = 13 : i32} : memref<64x64xf16>
    return
  }

  // ============================================================================
  // Case B + augmentation: N=2 原始 yield slot, M=1 跨组值不在 yield 中
  // then 内 block_id={10, 11}, K=3 (N+M)
  // block 10 产出的 %cross 被 block 11 消费, 但不在原始 yield 中
  // ============================================================================

  // CHECK-LABEL: func.func @test_case_b_augment
  // 第一个 split if: K=3, block_id=10, augmented slot 2 放 %cross
  // CHECK: [[R0:%.*]]:3 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}
  // 第二个 split if: K=3, block_id=11, 消费 augmented slot [[R0]]#2
  // CHECK: [[R1:%.*]]:3 = scf.if
  // CHECK-NEXT: arith.addf [[R0]]#2, {{.*}} {ssbuffer.block_id = 11 : i32}
  // 原始 uses → 最后一个 if 的前 N=2 个 result
  // CHECK: return [[R1]]#0, [[R1]]#1

  func.func @test_case_b_augment(%cond: i1) -> (tensor<16xf32>, tensor<16x64xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%t0 : tensor<16xf32>) -> tensor<16xf32>
    %t1 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16x64xf32>
    %f1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%t1 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %cst1 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %t2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f2 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst1 : f32) outs(%t2 : tensor<16xf32>) -> tensor<16xf32>
    %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16x64xf32>) {
      %v1 = arith.addf %f2, %f0 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
      %cross = arith.mulf %f1, %f1 {ssbuffer.block_id = 10 : i32} : tensor<16x64xf32>
      %v2 = arith.addf %cross, %f1 {ssbuffer.block_id = 11 : i32} : tensor<16x64xf32>
      scf.yield %v1, %v2 : tensor<16xf32>, tensor<16x64xf32>
    } else {
      scf.yield %f0, %f1 : tensor<16xf32>, tensor<16x64xf32>
    } {ssbuffer.block_id = 16 : i32}
    return %4#0, %4#1 : tensor<16xf32>, tensor<16x64xf32>
  }

  // ============================================================================
  // 边界: 只有一个 block_id 的 if 不应该被拆分
  // ============================================================================

  // CHECK-LABEL: func.func @test_single_block_no_split
  // 只有一个 scf.if, 没有被拆分
  // CHECK: %{{.*}}:2 = scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK-NOT: scf.if

  func.func @test_single_block_no_split(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %3 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
      %v1 = arith.addf %3, %1 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
      %v2 = arith.mulf %v1, %3 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
      scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
    } else {
      scf.yield %1, %1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.block_id = 16 : i32}
    return %4#0, %4#1 : tensor<16xf32>, tensor<16xf32>
  }
}
