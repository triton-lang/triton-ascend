// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ==========================================================================
  // Scene 2 — else-side multi-group split, then side ≤1 group (or empty)
  //
  // Key difference from Scene 1: the condition must be negated, because the
  // ops originally in the else branch (condition=false) are placed in the
  // split if's then branch. An `arith.xori` negates the condition.
  // ==========================================================================

  // --------------------------------------------------------------------------
  // Case A: void if, then is empty, else region has block_id={5, 3}
  // Cross-group dep: block 5 produces maxsi, block 3 consumes it
  // Result: negate condition, split into 2 chained split-ifs
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_else_split_case_a
  // for inside main_loop:
  // CHECK: scf.for
  // negated condition
  // CHECK: arith.xori
  // first split if (block_id=5): augmented yield slot
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // second split if (block_id=3): void, consumes [[R0]]
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK-NEXT: arith.index_cast [[R0]] {ssbuffer.block_id = 3 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 3 : i32}

  func.func @test_else_split_case_a(%cond: i1) {
    %c64_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond {
      } else {
        %v1 = arith.maxsi %c64_i32, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
        %v2 = arith.index_cast %v1 {ssbuffer.block_id = 3 : i32} : i32 to index
        %v3 = arith.muli %v2, %c64 {ssbuffer.block_id = 3 : i32} : index
        memref.dealloc %alloc {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
      } {ssbuffer.block_id = 17 : i32}
    } {ssbuffer.main_loop = 0 : i64}
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }

  // --------------------------------------------------------------------------
  // Case B: yield if (N=2), then yields original seed values,
  // else region has block_id={10, 11} with cross-group dep via slot 0
  // Result: negate condition, split into 2 chained split-ifs
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_else_split_case_b
  // for inside main_loop:
  // CHECK: scf.for
  // negate condition
  // CHECK: arith.xori
  // first split if (block_id=10): yields only slot 0 (what it produces)
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (tensor<16xf32>)
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // second split if (block_id=11): last if carries original result types
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK: arith.mulf [[R0]], {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R0]],
  // CHECK: else
  // CHECK: scf.yield {{%.*}}, {{%.*}}
  // for yields the last split if's result:
  // CHECK: scf.yield [[R1]]#0, [[R1]]#1
  // CHECK: return

  func.func @test_else_split_case_b(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst : f32) outs(%t0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 14 : i32} 1.0 : f32
    %t1 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %f1 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst2 : f32) outs(%t1 : tensor<16xf32>) -> tensor<16xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %for:2 = scf.for %iv = %lb to %ub step %step iter_args(%a0 = %f0, %a1 = %f0) -> (tensor<16xf32>, tensor<16xf32>) {
      %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
        scf.yield %f0, %f0 : tensor<16xf32>, tensor<16xf32>
      } else {
        %v1 = arith.addf %f1, %f0 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
        %v2 = arith.mulf %v1, %f1 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
        scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
      } {ssbuffer.block_id = 17 : i32}
      scf.yield %4#0, %4#1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.main_loop = 0 : i64}
    return %for#0, %for#1 : tensor<16xf32>, tensor<16xf32>
  }
}
