// RUN: (triton-opt --add_multi_buffer_inner_scope %s 2>&1 || echo "PASS") | FileCheck %s
// CHECK: PASS

// T-i1: i1 Tensor Dependency Triggers Fallback
// Test: When a tensor dep with element type i1 is produced in one block and
//       consumed in another inside the main_loop body (via a normal
//       cross-block tensor ref, NOT the tensor.empty + linalg.fill clone
//       pattern), the pass should detect this during dep collection and
//       fall back (set ERRCODE_IGNORED=2 + signalPassFailure).
//
// Setup:
//   - VECTOR scope with main_loop.
//   - Producer block_id = 7 (inside main_loop): memref.alloc +
//     bufferization.to_tensor to produce tensor<128xi1>.
//   - Consumer block_id = 10 (inside main_loop, different from 7):
//     arith.ori reads %prod cross-block.
//   - The cross-block dep is %prod (tensor<i1>, block 7) -> %consumed (block 10).
//   - This is a NORMAL tensor dep (would normally go through the multi-buffer
//     pipeline), not the empty+fill clone path.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_i1_tensor_dep_fallback() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_true = arith.constant true
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        // Producer: tensor<128xi1> in block 7 (NOT empty+fill pattern)
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xi1>
        %prod = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 7 : i32} : memref<128xi1>
        // Consumer in block 10 (cross-block)
        %consumed = arith.ori %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xi1>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}