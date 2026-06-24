// RUN: triton-opt --pre-check-blacklist --debug %s 2>&1 | FileCheck %s

// Test Case: Module with scf.while operation (blacklist op)
// Should be detected by PreCheckBlacklistPass

//CHECK: SSBUFFER will be skipped because scf.while operation was found
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_while_blacklist() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32

    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>

    %result:2 = scf.while (%arg0 = %init, %arg1 = %c0_i32) : (tensor<32x32xf32>, i32) -> (tensor<32x32xf32>, i32) {
      %cond = arith.cmpi slt, %arg1, %c32_i32 : i32
      scf.condition(%cond) %arg0, %arg1 : tensor<32x32xf32>, i32
    } do {
    ^bb0(%arg0: tensor<32x32xf32>, %arg1: i32):
      %next_idx = arith.addi %arg1, %c1_i32 : i32
      scf.yield %arg0, %next_idx : tensor<32x32xf32>, i32
    }

    return
  }
}