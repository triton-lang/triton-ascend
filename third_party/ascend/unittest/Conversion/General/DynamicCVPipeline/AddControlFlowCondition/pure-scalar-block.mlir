// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// A block that has neither a cross-core nor an intra-core interaction is a
// pure scalar computation: it does not exchange any data with the other
// blocks. It is not part of the flag chain of the loop: it neither takes the
// flag of the loop as a condition nor updates it. The chain passes through it,
// the block that comes after it still reads the flag of the last block that
// took part in the chain.
//   block 1 produces attribute group 1 (three producers)
//   block 2 consumes group 1 and produces group 0
//   block 3 is a pure scalar computation, it has no dependency at all
//   block 4 consumes group 0

// CHECK-LABEL: func.func @pure_scalar_block
// CHECK: scf.while
// CHECK: ^bb0
// The flag of the loop starts with a constant zero.
// CHECK: %[[FLAG:.*]] = arith.constant 0 : i32
// The first block of the chain reads the flag the loop starts with: its check
// is always true and is left out.
// CHECK: %[[GUARD1:.*]] = arith.andi %{{.*}}, %{{.*}}
// CHECK: %[[IF1:[0-9]+]]:2 = scf.if %[[GUARD1]]
// The flag of the loop is computed with a select after the block.
// CHECK: ssbuffer.if = 1 : i32}
// CHECK: %[[ONE1:.*]] = arith.constant 1 : i32
// CHECK: %[[FLAG1:.*]] = arith.select %[[GUARD1]], %[[ONE1]], %[[FLAG]] : i32
// The second block of the chain checks the flag of the first one.
// CHECK: %[[FLAGCMP2:.*]] = arith.cmpi eq, %[[FLAG1]], %{{.*}} : i32
// CHECK: %[[GUARD2:.*]] = arith.andi %{{.*}}, %[[FLAGCMP2]]
// The flag of the second block is cumulative: it takes the flag of the first
// block as its false value.
// CHECK: ssbuffer.if = 2 : i32}
// CHECK: %[[ONE2:.*]] = arith.constant 1 : i32
// CHECK: %[[FLAG2:.*]] = arith.select %[[GUARD2]], %[[ONE2]], %[[FLAG1]] : i32
// The third block is a pure scalar computation: it does not check the flag of
// the loop and it does not update it, the chain passes through it.
// CHECK: %[[GUARD3:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i32
// CHECK: %[[IF3:[0-9]+]] = scf.if %[[GUARD3]] -> (i32) {
// CHECK: hivm.hir.copy {{.*}} {ssbuffer.block_id = 3 : i32}
// CHECK-NOT: arith.select
// The fourth block checks the flag of the second one: the pure scalar block
// did not break the chain.
// CHECK: %[[FLAGCMP4:.*]] = arith.cmpi eq, %[[FLAG2]], %{{.*}} : i32
// CHECK: %[[GUARD4:.*]] = arith.andi %{{.*}}, %[[FLAGCMP4]]
// CHECK: %[[IF4:[0-9]+]]:2 = scf.if %[[GUARD4]]

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pure_scalar_block(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32

    scope.scope : () -> () {
      %alloc0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem0 = memref.memory_space_cast %alloc0 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem1 = memref.memory_space_cast %alloc1 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>

      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        // Producers of attribute group 1. Its consumer is in the next block.
        %t1_p0 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1_p0 : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]}
        %t1_p1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1_p1 : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]}
        %t1_p2 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1_p2 : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]}

        // First intra consumer (attribute group 1 -> DEP0). Also produces group 0.
        %t2_in = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t2_in : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]}
        %t2_out = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t2_out : tensor<128xf32>) outs(%mem1 : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]}

        // Pure scalar computation: it has no dependency at all.
        %t3 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t3 : tensor<128xf32>) outs(%mem1 : memref<128xf32>) {ssbuffer.block_id = 3 : i32}

        // Second intra consumer (attribute group 0 -> DEP1).
        %t4_in = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t4_in : tensor<128xf32>) outs(%mem1 : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        %tp = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tp : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [20 : i32, 1 : i32]}
        %tc = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tc : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [20 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
