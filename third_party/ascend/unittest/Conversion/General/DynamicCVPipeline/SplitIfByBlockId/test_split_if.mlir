// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Case B: original if with 2 yield slots, then region has block_id={10, 11}
  // Cross-group dependency: block 10 produces %addf, block 11 consumes %addf (in yield slot 0)
  // Expected: split into 2 chained ifs, K=2 (no augmentation)
  // ============================================================================

  // CHECK-LABEL: func.func @test_case_b_basic
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: only block_id=10
  // CHECK: [[R0:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // second split if: only block_id=11, consumes [[R0]]#0
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.mulf [[R0]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R0]]#0,
  // CHECK: else
  // CHECK: scf.yield [[R0]]#0, [[R0]]#1
  // for yields the last split if's result:
  // CHECK: scf.yield [[R1]]#0, [[R1]]#1
  // original uses replaced with last if's result
  // CHECK: return

  func.func @test_case_b_basic(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %3 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %for:2 = scf.for %iv = %lb to %ub step %step iter_args(%a0 = %1, %a1 = %1) -> (tensor<16xf32>, tensor<16xf32>) {
      %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
        %v1 = arith.addf %3, %1 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
        %v2 = arith.mulf %v1, %3 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
        scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %1, %1 : tensor<16xf32>, tensor<16xf32>
      } {ssbuffer.block_id = 16 : i32}
      scf.yield %4#0, %4#1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.main_loop = 0 : i64}
    return %for#0, %for#1 : tensor<16xf32>, tensor<16xf32>
  }

  // ============================================================================
  // Case A: original if has no yield, then region has block_id={5, 3}, pure side effects
  // Cross-group dependency: block 5 produces maxsi result, block 3 consumes it → needs 1 augmented yield slot
  // ============================================================================

  // CHECK-LABEL: func.func @test_case_a_basic
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: augmented yield slot for cross-group value (block_id=5)
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // second split if: void, consumes [[R0]] (block_id=3)
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
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond {
        %v1 = arith.maxsi %c64_i32, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
        %v2 = arith.index_cast %v1 {ssbuffer.block_id = 3 : i32} : i32 to index
        %v3 = arith.muli %v2, %c64 {ssbuffer.block_id = 3 : i32} : index
        memref.dealloc %alloc {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
      }
    } {ssbuffer.main_loop = 0 : i64}
    memref.dealloc %alloc {ssbuffer.block_id = 13 : i32} : memref<64x64xf16>
    return
  }

  // ============================================================================
  // Case B + augmentation: N=2 original yield slots, M=1 cross-group value not in yield
  // then region has block_id={10, 11}, K=3 (N+M)
  // block 10 produces %cross consumed by block 11, but not in original yield
  // ============================================================================

  // CHECK-LABEL: func.func @test_case_b_augment
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: K=3, block_id=10, augmented slot 2 holds %cross
  // CHECK: [[R0:%.*]]:3 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}
  // second split if: K=3, block_id=11, consumes augmented slot [[R0]]#2
  // CHECK: [[R1:%.*]]:3 = scf.if
  // CHECK-NEXT: arith.addf [[R0]]#2, {{.*}} {ssbuffer.block_id = 11 : i32}
  // for yields first N=2 results of the last split if:
  // CHECK: scf.yield [[R1]]#0, [[R1]]#1
  // original uses → return:
  // CHECK: return

  func.func @test_case_b_augment(%cond: i1) -> (tensor<16xf32>, tensor<16x64xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%t0 : tensor<16xf32>) -> tensor<16xf32>
    %t1 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16x64xf32>
    %f1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%t1 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %cst1 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %t2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f2 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst1 : f32) outs(%t2 : tensor<16xf32>) -> tensor<16xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %for:2 = scf.for %iv = %lb to %ub step %step iter_args(%a0 = %f0, %a1 = %f1) -> (tensor<16xf32>, tensor<16x64xf32>) {
      %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16x64xf32>) {
        %v1 = arith.addf %f2, %f0 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
        %cross = arith.mulf %f1, %f1 {ssbuffer.block_id = 10 : i32} : tensor<16x64xf32>
        %v2 = arith.addf %cross, %f1 {ssbuffer.block_id = 11 : i32} : tensor<16x64xf32>
        scf.yield %v1, %v2 : tensor<16xf32>, tensor<16x64xf32>
      } else {
        scf.yield %f0, %f1 : tensor<16xf32>, tensor<16x64xf32>
      } {ssbuffer.block_id = 16 : i32}
      scf.yield %4#0, %4#1 : tensor<16xf32>, tensor<16x64xf32>
    } {ssbuffer.main_loop = 0 : i64}
    return %for#0, %for#1 : tensor<16xf32>, tensor<16x64xf32>
  }

  // ============================================================================
  // Edge case: if not in main_loop is skipped by the pass (isInsideMainLoop)
  // ============================================================================

  // CHECK-LABEL: func.func @test_not_in_main_loop_skip
  // only one scf.if, not split (skipped because not inside main_loop)
  // CHECK: %{{.*}}:2 = scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK-NOT: scf.if

  func.func @test_not_in_main_loop_skip(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
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

  // ============================================================================
  // Edge case: same block_id in main_loop → needsSplit() returns false, no split
  // ============================================================================

  // CHECK-LABEL: func.func @test_same_block_id_in_main_loop_no_split
  // inside main_loop, but all ops share block_id=10, only one if, not split
  // CHECK: scf.for
  // CHECK: %{{.*}}:2 = scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK-NOT: scf.if

  func.func @test_same_block_id_in_main_loop_no_split(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %3 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %for:2 = scf.for %iv = %lb to %ub step %step iter_args(%a0 = %1, %a1 = %1) -> (tensor<16xf32>, tensor<16xf32>) {
      %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
        %v1 = arith.addf %3, %1 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
        %v2 = arith.mulf %v1, %3 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
        scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %1, %1 : tensor<16xf32>, tensor<16xf32>
      } {ssbuffer.block_id = 16 : i32}
      scf.yield %4#0, %4#1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.main_loop = 0 : i64}
    return %for#0, %for#1 : tensor<16xf32>, tensor<16xf32>
  }
}
