// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // 场景3 Case A: then 有 block_id={5, 3}, else 有 block_id={8} actual ops
  // 原始 if 无 yield (void if), 纯副作用
  // 跨组依赖: block 5 产出 maxsi 结果, block 3 消费 → augmented yield slot
  // then 拆分为 2 个 split if, 最后 if 的 else 吸收 block_id=8 的 ops
  // ============================================================================

  // CHECK-LABEL: func.func @test_scene3_case_a

  func.func @test_scene3_case_a(%cond: i1) {
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
      // block_id=8: else group C (single group)
      %v4 = arith.maxsi %c0_i32, %c64_i32 {ssbuffer.block_id = 8 : i32} : i32
      memref.dealloc %alloc {ssbuffer.block_id = 8 : i32} : memref<64x64xf16>
    } {ssbuffer.block_id = 12 : i32}
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }

  // 期望结构:
  //
  // Round 1 — 拆分 then 侧 (block_id=5, 3), else 侧 (block_id=8) 吸收:
  //
  // Placeholder seed:
  // CHECK: arith.constant {ssbuffer.block_id = -1 : i32}
  //
  // 第一个 split if: block_id=5, result-bearing (跨组依赖 augmented slot)
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // 第二个 split if: block_id=3 (void if), else 吸收 block_id=8
  // CHECK: scf.if
  // CHECK-NOT: ->
  // then 侧 = block_id=3 ops:
  // CHECK-NEXT: arith.index_cast [[R1]] {ssbuffer.block_id = 3 : i32}
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 3 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 3 : i32}
  // else 侧 = 吸收 block_id=8 ops:
  // CHECK: else
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 8 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 8 : i32}
}
