// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ==========================================================================
  // Scene 1 — then-side multi-group split, else side ≤1 group (or empty)
  //
  // Case A: void if (no yield), cross-group SSA via augmented slots
  // Case B: yield if, original slots preserved + optional augmented slots
  //
  // These are the simplest split scenarios — only one side needs splitting.
  // ==========================================================================

  // --------------------------------------------------------------------------
  // Case B: N=2 original yield slots, then region has block_id={10, 11}
  // Cross-group dep: block 10 produces yield slot 0, block 11 consumes it
  // Result: 2 chained split-ifs, K=2 (no augmentation needed)
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_case_b_basic
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: block_id=10, yields only slot 0 (what it produces)
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (tensor<16xf32>)
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // second split if: block_id=11, last if carries all original result types
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK: arith.mulf [[R0]], {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R0]],
  // CHECK: else
  // CHECK: scf.yield {{%.*}}, {{%.*}}
  // for yields the last split if's result:
  // CHECK: scf.yield [[R1]]#0, [[R1]]#1
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

  // --------------------------------------------------------------------------
  // Case A: void if, then region has block_id={5, 3}, pure side effects
  // Cross-group dep: block 5 produces maxsi result, block 3 consumes it
  // Result: 2 chained split-ifs, 1 augmented yield slot for the cross-group value
  // --------------------------------------------------------------------------

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

  // --------------------------------------------------------------------------
  // Case B + augmentation: N=2 original yield slots,
  // M=1 cross-group value (%cross) consumed by block 11 but not in yield
  // Result: K=3 (N+M), augmented slot 2 carries %cross
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_case_b_augment
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: block_id=10, 2 outputs (1 augmented cross + original slot 0)
  // CHECK: [[R0:%.*]]:2 = scf.if
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield {{.*}}, {{.*}}
  // CHECK: else
  // CHECK: scf.yield
  // second split if: block_id=11, last if carries original result types
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK: arith.addf [[R0]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: else
  // CHECK: scf.yield {{%.*}}, {{%.*}}
  // for yields first N=2 results of the last split if:
  // CHECK: scf.yield [[R1]]#0, [[R1]]#1
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
}
