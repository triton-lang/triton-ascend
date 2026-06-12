// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // 场景4 Case A: then 有 block_id={5, 3}, else 有 block_id={8, 9}
  // 两边都有 >=2 组, void if (无 yield), 纯副作用
  // 两轮迭代:
  //   Round 1: 拆分 then 侧, 最后 if 的 else 吸收 else 侧所有 ops
  //   Round 2: 吸收后 else 内有 2 组, 触发 else 侧拆分
  // ============================================================================

  // CHECK-LABEL: func.func @test_scene4_case_a

  func.func @test_scene4_case_a(%cond: i1) {
    %c64_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
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
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }

  // 期望结构:
  //
  // Round 1 — 拆分 then 侧 (block_id=5, 3):
  //
  // Placeholder seed for round 1:
  // CHECK: arith.constant {ssbuffer.block_id = -1 : i32}
  //
  // Round 1 第一个 split if: block_id=5, result-bearing (跨组依赖 augmented)
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // Round 2 — 拆分 else 侧 (block_id=8, 9), 条件取反:
  //
  // CHECK: arith.constant true
  // CHECK: [[NEG:%.*]] = arith.xori
  //
  // Placeholder seed for round 2:
  // CHECK: arith.constant {ssbuffer.block_id = -1 : i32}
  //
  // Round 2 第一个 split if: block_id=8, result-bearing (取反条件)
  // CHECK: [[R2:%.*]] = scf.if [[NEG]]
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 8 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // Round 2 第二个 split if: block_id=9 (void), else 吸收 block_id=3
  // CHECK: scf.if [[NEG]]
  // CHECK-NOT: ->
  // then 侧 = block_id=9:
  // CHECK-NEXT: arith.index_cast [[R2]] {ssbuffer.block_id = 9 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 9 : i32}
  // else 侧 = 吸收原始 then block_id=3 ops:
  // CHECK: else
  // CHECK-NEXT: arith.index_cast [[R1]] {ssbuffer.block_id = 3 : i32}
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 3 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 3 : i32}
}
