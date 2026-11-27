// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// VECTOR and CUBE each have one main scf.for and one intraDeps pair. The two
// pairs use different intra groups: one group lists every producer, and
// calculateFactor looks those producers up inside the current for. Block 5's
// mulf is inside one splitted_if.
// The normal scheme does not wrap the ssbuffer.if: the outer scf.if of the
// block carries the block condition and the ssbuffer.if marker, and the
// splitted_if stays inside it as the inner data dependent scf.if. The counter
// of the block is incremented in the then branch of the outer if, outside the
// inner splitted_if, so it advances on every round of the loop even when the
// data dependent condition of the block is false. The counter is the
// induction variable of the scf.while, so a counter that only advances under
// the data dependent condition would stop the loop.
// ProcessArgs appends one shared iter_arg, then UpdateLoopOps appends block
// counters. The first counter is the third iter_arg, initialized to the for
// lower bound.
// UpdateLoopIterTimes rewrites the main loop into an scf.while before wrapping:
// the induction variable becomes the first carried value, so the counters move
// to the fourth and the fifth one. The while keeps the upper bound of the for
// loop and its condition is the OR of the counters of the ssbuffer.if blocks
// compared against that upper bound. SplittedIfWrapSsbufIf skips this while,
// so the payload predicate stays inside ssbuffer.if 5. The block counter only
// advances when the ready/credit guard runs; its else preserves the counter.

// CHECK-LABEL: func.func @wrap_for_splitted_if
// CHECK: scf.while (%[[IV:[^ ]+]] = %[[LB:[^ ,]+]], %{{[^ ,]+}} = %{{[^ ,]+}}, %{{[^ ,]+}} = %{{[^ ,]+}}, %[[CTR1:[^ ,]+]] = %[[LB]], %[[CTR2:[^ ,]+]] = %[[LB]], %{{[^ ,]+}} = %{{[^ ,]+}}, %{{[^ ,]+}} = %{{[^ ,]+}}) : (i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32)
// CHECK: %[[CMP1:[^ ]+]] = arith.cmpi slt, %[[CTR1]], %[[UB:[^ ,]+]] : i32
// CHECK: %[[CMP2:[^ ]+]] = arith.cmpi slt, %[[CTR2]], %[[UB]] : i32
// CHECK: arith.ori %[[CMP1]], %[[CMP2]] : i1
// CHECK: scf.condition
// CHECK: arith.addi %[[IV]], %[[STEP:[^ ]+]] : i32
// CHECK: %[[BLOCK5:[^ :]+]]:3 = scf.if %{{[^ ]+}} -> (i32, i32, i32) {
// CHECK: %[[SPLIT:[^ ]+]] = arith.cmpi sgt, %[[CTR1]], {{.*}}ssbuffer.block_id = 5 : i32
// CHECK: scf.if %[[SPLIT]]
// CHECK: arith.mulf {{.*}} : tensor<4xf32>
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: arith.addi %[[CTR1]], %[[STEP]]
// CHECK: } {{.*}}ssbuffer.if = 5 : i32}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @wrap_for_splitted_if() attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %one = arith.constant dense<1.000000e+00> : tensor<4xf32>

    scope.scope : () -> () {
      // The pipeline of the kernel: the vector core writes a tile into a cross
      // core buffer that the cube core reads, the cube core writes a tile of
      // its own that the vector core reads back. The three transfers make the
      // cross core dependency of the module a VECTOR -> CUBE -> VECTOR -> CUBE
      // chain, the shape the normal preload scheme is enabled on. The scheme
      // only counts the cross core blocks that carry a condition, so the
      // blocks of the chain are the blocks of the main loop. The transfers
      // themselves sit outside the loop: they give the module the pipeline
      // shape without changing the conditions of the loop the test checks.
      %cbuf_t2 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_5 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_t3 = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_t4 = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_7 = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %tile_v2 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_v2 : tensor<8x4x16x16xf16>) outs(%cbuf_t2 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_v2 : tensor<8x4x16x16xf16>) outs(%dup_tcb_5 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 1 : i32]}
      %tile_v3 = bufferization.to_tensor %cbuf_t3 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %tile_v4 = tensor.empty() {ssbuffer.block_id = 7 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_v4 : tensor<8x4x16x16xf16>) outs(%cbuf_t4 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_v4 : tensor<8x4x16x16xf16>) outs(%dup_tcb_7 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 1 : i32]}
      %0 = scf.for %arg0 = %c0_i32 to %c16_i32 step %c1_i32
          iter_args(%arg1 = %c0_i32) -> (i32) : i32 {
        %split = arith.cmpi sgt, %arg0, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
        scf.if %split {
          %t = arith.mulf %zero, %one : tensor<4xf32>
        } {ssbuffer.block_id = 5 : i32, ssbuffer.splitted_if = 0 : i32}
        %prod = arith.addi %arg1, %c0_i32 {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]} : i32
        %cons = arith.addi %arg1, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : i32
        scf.yield %cons : i32
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %cbuf_c2 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_c3 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_6 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_c4 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %tile_c2 = bufferization.to_tensor %cbuf_c2 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %tile_c3 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_c3 : tensor<8x4x16x16xf16>) outs(%cbuf_c3 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_c3 : tensor<8x4x16x16xf16>) outs(%dup_tcb_6 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 1 : i32]}
      %tile_c4 = bufferization.to_tensor %cbuf_c4 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %0 = scf.for %arg0 = %c0_i32 to %c16_i32 step %c1_i32
          iter_args(%arg1 = %c0_i32) -> (i32) : i32 {
        %prod = arith.addi %arg1, %c0_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]} : i32
        %cons = arith.addi %arg1, %c1_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]} : i32
        scf.yield %cons : i32
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
