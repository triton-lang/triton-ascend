// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ==========================================================================
  // Edge cases — scenarios where the pass should NOT split
  // ==========================================================================

  // --------------------------------------------------------------------------
  // If not inside a main_loop is skipped.
  // Pass only processes scf.if ops that are inside a loop tagged with
  // ssbuffer.main_loop attribute.
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_not_in_main_loop_skip
  // only one scf.if, not split (skipped because not inside main_loop)
  // CHECK: %{{.*}}:2 = scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 11 : i32}
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
      %v2 = arith.mulf %v1, %3 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
      scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
    } else {
      scf.yield %1, %1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.block_id = 16 : i32}
    return %4#0, %4#1 : tensor<16xf32>, tensor<16xf32>
  }

  // --------------------------------------------------------------------------
  // All ops share the same block_id inside main_loop → needsSplit()=false.
  // The pass only splits when there are ≥2 distinct block_id groups.
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_same_block_id_no_split
  // inside main_loop, but all ops share block_id=10, only one if, not split
  // CHECK: scf.for
  // CHECK: %{{.*}}:2 = scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK-NOT: scf.if

  func.func @test_same_block_id_no_split(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
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
