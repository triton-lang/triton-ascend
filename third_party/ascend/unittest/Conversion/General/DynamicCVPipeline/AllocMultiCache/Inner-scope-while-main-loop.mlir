// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// WhileOp as a main_loop: verifies that scf.while carrying both
// ssbuffer.main_loop and ssbuffer.block_id reuses its first do-region
// block arg as the iteration counter (the user's existing init value,
// typically 0, already increments each iter via scf.yield). The pass
// therefore does NOT inject a fresh arith.constant 0 / arith.addi
// pair — it just reads the block arg directly. Cross-block tensor
// deps inside the do region get the standard multi-buffer pattern
// (arith.remsi/arith.cmpi/scf.if with intra_buffer / intraDeps)
// anchored to that block arg.
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_while_double_buffer
  // Multi-buffer: 2 allocs + memspacecasts inserted before the whileOp
  // CHECK-DAG: memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps = [0 : i32, 1 : i32]}
  // CHECK-DAG: memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps = [0 : i32, 1 : i32]}
  // The first do-region block arg IS the iter counter — no fresh counter
  // insertion. arith.remsi/arith.cmpi ping-pong gates use it directly.
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // Producer scf.if + intra_buffer mark — present somewhere in do region
  // CHECK-DAG: } {ssbuffer.block_id = 6 : i32, ssbuffer.intra_buffer}
  // Consumer scf.if + intra_buffer / intraDeps mark
  // CHECK-DAG: } {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32], ssbuffer.intra_buffer}
  // User's existing arith.addi inside do region remains untouched
  // CHECK: arith.addi
  // CHECK: scf.yield
  // CHECK: } attributes {ssbuffer.block_id = 5 : i32, ssbuffer.main_loop
  func.func @test_while_double_buffer() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %0:2 = scf.while (%i = %c0_i32, %arg_t = %prod) : (i32, tensor<128xf32>) -> (i32, tensor<128xf32>) {
        %cond = arith.cmpi slt, %i, %c100_i32 : i32
        scf.condition(%cond) %i, %arg_t : i32, tensor<128xf32>
      } do {
      ^bb0(%i: i32, %arg_t: tensor<128xf32>):
        %consumed = arith.addf %arg_t, %arg_t {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %next = arith.addi %i, %c1_i32 : i32
        scf.yield %next, %new_prod : i32, tensor<128xf32>
      } attributes {ssbuffer.block_id = 5 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
