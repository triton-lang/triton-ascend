// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ==========================================================================
  // Scene 4 — both sides have ≥2 groups, requiring two iterations
  //
  // Round 1: split side A, last split-if's else absorbs ALL of side B's ops
  // Round 2: after absorption, side B now has ≥2 groups in the else block
  //          → triggers a second split on side B (with negated condition)
  //
  // This is the most complex scenario.
  // ==========================================================================

  // --------------------------------------------------------------------------
  // Case A: then has block_id={5, 3}, else has block_id={8, 9}
  // Void if (no yield), pure side effects
  // Round 1: split then side, last if's else absorbs else ops
  // Round 2: absorbed block_id=8,9 triggers else-side split
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_scene4_case_a
  // for inside main_loop:
  // CHECK: scf.for
  // Round 1 first split if: block_id=5, result-bearing (cross-group dep augmented)
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // Round 2 — split else side (block_id=8, 9), negated condition:
  // CHECK: arith.constant true
  // CHECK: [[NEG:%.*]] = arith.xori
  // Round 2 first split if: block_id=8, result-bearing (negated condition)
  // CHECK: [[R2:%.*]] = scf.if [[NEG]]
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 8 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // Round 2 second split if: block_id=9 (void), else absorbs block_id=3
  // CHECK: scf.if [[NEG]]
  // CHECK-NOT: ->
  // then side = block_id=9:
  // CHECK-NEXT: arith.index_cast [[R2]] {ssbuffer.block_id = 9 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 9 : i32}
  // else side = absorbed original then block_id=3 ops:
  // CHECK: else
  // CHECK-NEXT: arith.index_cast [[R1]] {ssbuffer.block_id = 3 : i32}
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 3 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 3 : i32}

  func.func @test_scene4_case_a(%cond: i1) {
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
        // block_id=5: then group A
        %v1 = arith.maxsi %c64_i32, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
        // block_id=3: then group B (uses %v1 -> cross-group dep)
        %v2 = arith.index_cast %v1 {ssbuffer.block_id = 3 : i32} : i32 to index
        %v3 = arith.muli %v2, %c64 {ssbuffer.block_id = 3 : i32} : index
        memref.dealloc %alloc {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
      } else {
        // block_id=8: else group C
        %v4 = arith.maxsi %c0_i32, %c64_i32 {ssbuffer.block_id = 8 : i32} : i32
        // block_id=9: else group D
        %v5 = arith.index_cast %v4 {ssbuffer.block_id = 9 : i32} : i32 to index
        memref.dealloc %alloc {ssbuffer.block_id = 9 : i32} : memref<64x64xf16>
      } {ssbuffer.block_id = 12 : i32}
    } {ssbuffer.main_loop = 0 : i64}
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }

  // --------------------------------------------------------------------------
  // Case B: then has block_id={10, 11}, else has block_id={5, 3}
  // Yield if (N=2), both sides ≥2 groups
  // Round 1: split then side, last if's else absorbs else ops
  // Round 2: absorbed block_id=5,3 triggers else-side split
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_scene4_case_b
  // for inside main_loop:
  // CHECK: scf.for
  // Round 1 first split if: block_id=10 (then), yields only slot 0
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (tensor<16xf32>)
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // Round 2 — split else side (block_id=5, 3), negated condition:
  // CHECK: arith.constant true
  // CHECK: [[NEG:%.*]] = arith.xori
  // Round 2 first split if: block_id=5 (then, negated condition), yields slot 0
  // CHECK: [[R2:%.*]] = scf.if [[NEG]]
  // CHECK-SAME: -> (tensor<16xf32>)
  // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // Round 2 second split if: block_id=3 (then) + block_id=11 absorbed into else
  // CHECK: [[R3:%.*]]:2 = scf.if [[NEG]]
  // CHECK: arith.addf [[R2]], {{.*}} {ssbuffer.block_id = 3 : i32}
  // CHECK: scf.yield [[R2]],
  // CHECK: else
  // original then-side block_id=11 ops absorbed here as else:
  // CHECK: arith.mulf [[R1]], {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R1]],
  // for yields the last split if's result:
  // CHECK: scf.yield [[R3]]#0, [[R3]]#1
  // CHECK: return

  func.func @test_scene4_case_b(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %for:2 = scf.for %iv = %lb to %ub step %step iter_args(%a0 = %1, %a1 = %1) -> (tensor<16xf32>, tensor<16xf32>) {
      %res:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
        // block_id=10: then group A
        %a = arith.addf %1, %f0 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
        // block_id=11: then group B (uses %a -> cross-group dep)
        %b = arith.mulf %a, %f0 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
        scf.yield %a, %b : tensor<16xf32>, tensor<16xf32>
      } else {
        // block_id=5: else group C
        %c = arith.subf %f0, %f0 {ssbuffer.block_id = 5 : i32} : tensor<16xf32>
        // block_id=3: else group D (uses %c -> cross-group dep)
        %d = arith.addf %c, %1 {ssbuffer.block_id = 3 : i32} : tensor<16xf32>
        scf.yield %c, %d : tensor<16xf32>, tensor<16xf32>
      } {ssbuffer.block_id = 12 : i32}
      scf.yield %res#0, %res#1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.main_loop = 0 : i64}
    return %for#0, %for#1 : tensor<16xf32>, tensor<16xf32>
  }
}
