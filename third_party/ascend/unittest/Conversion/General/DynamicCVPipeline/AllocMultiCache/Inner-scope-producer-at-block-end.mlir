// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// Producer transfer is inserted at the end of the producer block (just
// before the terminator), not immediately after the dep def op.
//
// Setup inside main_loop body:
//   block 5 (producer):
//     %dep = arith.addf %arg, %arg {block_id = 5}        <- dep def op
//     %intra_use = arith.mulf %dep, %dep {block_id = 5}  <- intra-block use
//   block 6 (consumer):
//     %consumed = arith.addf %dep, %dep {block_id = 6}   <- cross-block read
//
// Expected: producer copy chain for block 5 is positioned just before
// scf.yield, AFTER the intra-block use and the cross-block consumer.

// CHECK-LABEL: func.func @test_producer_at_block_end
// CHECK: scf.for
// Intra-block use (%intra_use) must follow the dep def directly,
// with NO producer copy chain in between.
// CHECK: arith.addf %{{.*}}, %{{.*}} {ssbuffer.block_id = 5
// CHECK-NEXT: arith.mulf %{{.*}}, %{{.*}} {ssbuffer.block_id = 5
// Producer copy for block 5: closing brace with intra_buffer attribute
// must precede scf.yield (the producer transfer is at end of block).
// CHECK: } {ssbuffer.block_id = 5 : i32, ssbuffer.intra_buffer}
// CHECK: scf.yield

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_producer_at_block_end() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %dep = arith.addf %arg, %arg {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %intra_use = arith.mulf %dep, %dep {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %consumed = arith.addf %dep, %dep {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        scf.yield %intra_use : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}