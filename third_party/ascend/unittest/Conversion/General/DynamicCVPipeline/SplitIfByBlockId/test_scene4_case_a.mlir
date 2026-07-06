// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Scene 4 Case A: then has block_id={5, 3}, else has block_id={8, 9}
  // Both sides have >=2 groups, void if (no yield), pure side effects
  // Two iterations:
  //   Round 1: split then side, last if's else absorbs all else-side ops
  //   Round 2: after absorption, else has 2 groups, triggers else-side split
  // ============================================================================

  // CHECK-LABEL: func.func @test_scene4_case_a

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

  // Expected structure:
  //
  // Round 1 — split then side (block_id=5, 3):
  // for inside main_loop:
  // CHECK: scf.for
  //
  // Placeholder seed for round 1:
  // CHECK: arith.constant {ssbuffer.block_id = -1 : i32}
  //
  // Round 1 first split if: block_id=5, result-bearing (cross-group dep augmented)
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // Round 2 — split else side (block_id=8, 9), negated condition:
  //
  // CHECK: arith.constant true
  // CHECK: [[NEG:%.*]] = arith.xori
  //
  // Placeholder seed for round 2:
  // CHECK: arith.constant {ssbuffer.block_id = -1 : i32}
  //
  // Round 2 first split if: block_id=8, result-bearing (negated condition)
  // CHECK: [[R2:%.*]] = scf.if [[NEG]]
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 8 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
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
}
