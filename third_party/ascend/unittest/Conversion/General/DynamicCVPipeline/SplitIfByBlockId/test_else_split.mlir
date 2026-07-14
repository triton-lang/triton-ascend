// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Else split (Case A): then is empty, else region has block_id={5, 3}
  // Expected: negate condition, remove original empty outer if, wrap else ops with negated condition and split
  // ============================================================================

  // CHECK-LABEL: func.func @test_else_split_basic
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
  // Note: original empty then if is gone, no nested scf.if wrapping at outer level

  func.func @test_else_split_basic(%cond: i1) {
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

  // ============================================================================
  // Else split (Case B): original if has yield (N=2), then yields original empty-then values
  // else region has block_id={10, 11}, cross-group dep: block 10 → block 11 (via slot 0)
  // ============================================================================

  // CHECK-LABEL: func.func @test_else_split_with_yield
  // for inside main_loop:
  // CHECK: scf.for
  // negate condition
  // CHECK: arith.xori
  // first split if (block_id=10): replaces original else→then
  // CHECK: [[R0:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // second split if (block_id=11): consumes [[R0]]#0
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.mulf [[R0]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R0]]#0,
  // for yields the last split if's result:
  // CHECK: scf.yield [[R1]]#0, [[R1]]#1
  // CHECK: return

  func.func @test_else_split_with_yield(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
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

  // ============================================================================
  // Nested else split: outer if else region has a nested if, which has different block_id ops in its else.
  // Iteration 1: inner if split into bid=5 and bid=3 split-ifs.
  // Iteration 2: outer if's else block now has multiple block_id nested ifs → outer also splits,
  //         outer result if wraps inner bid=5 split-if, outer void if wraps inner bid=3.
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_else_split
  // for inside main_loop:
  // CHECK: scf.for
  // outer negated condition
  // CHECK: arith.xori
  // outer result if: wraps block_id=5 split-if + inner condition + inner result
  // CHECK: [[R0:%.*]]:2 = scf.if
  // CHECK-SAME: -> (i1, i32)
  // inner negated condition
  // CHECK: [[INNER_COND:%.*]] = arith.xori
  // inner result if (block_id=5)
  // CHECK: [[INNER_R:%.*]] = scf.if [[INNER_COND]] -> (i32)
  // CHECK: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // outer then yield: inner condition + inner result
  // CHECK: scf.yield [[INNER_COND]], [[INNER_R]]
  // CHECK: else
  // CHECK: scf.yield
  // outer void if wrapping block_id=3 void if: consumes [[R0]]#0 (inner cond) + [[R0]]#1
  // CHECK: scf.if
  // CHECK-NOT: ->
  // inner void if (block_id=3)
  // CHECK: scf.if [[R0]]#0
  // CHECK: arith.index_cast [[R0]]#1 {ssbuffer.block_id = 3 : i32}

  func.func @test_nested_else_split(%outer: i1, %inner: i1) {
    %c64_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %outer {
      } else {
        scf.if %inner {
        } else {
          %v1 = arith.maxsi %c64_i32, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
          %v2 = arith.index_cast %v1 {ssbuffer.block_id = 3 : i32} : i32 to index
          %v3 = arith.muli %v2, %c64 {ssbuffer.block_id = 3 : i32} : index
          memref.dealloc %alloc {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
        } {ssbuffer.block_id = 16 : i32}
      } {ssbuffer.block_id = 17 : i32}
    } {ssbuffer.main_loop = 0 : i64}
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }
}
