// RUN: triton-opt %s --split-input-file --ub-usage-opt --verify-each | FileCheck %s --enable-var-scope

// Yield forwarding must not act as an artificial sink. Cyclic iter_arg
// dependencies must terminate chain discovery and preserve unprofitable cuts.

// CHECK-LABEL: func.func @self_cycle(
func.func @self_cycle(%n: index, %input: tensor<64x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64xf32> {
    %big = math.exp %input {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 10 : i32
    %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [0] {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %first = arith.addf %small, %acc {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    scf.yield %first : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @two_node_cycle(
func.func @two_node_cycle(%n: index, %input: tensor<64x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64xf32> {
    %big = math.exp %input {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 110 : i32
    %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [0] {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %first = arith.addf %small, %acc {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    // CHECK: math.exp {{.*}} {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %second = math.exp %first {ssbuffer.block_id = 111 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    scf.yield %second : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @three_node_cycle(
func.func @three_node_cycle(%n: index, %input: tensor<64x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64xf32> {
    %big = math.exp %input {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 210 : i32
    %small = linalg.reduce ins(%big : tensor<64x64xf32>) outs(%init : tensor<64xf32>) dimensions = [0] {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %first = arith.addf %small, %acc {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    // CHECK: math.exp {{.*}} {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %second = math.exp %first {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %third = arith.mulf %second, %init {ssbuffer.block_id = 211 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    scf.yield %third : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @self_cycle_no_gain(
func.func @self_cycle_no_gain(%n: index, %input: tensor<1x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64xf32> {
    %big = math.exp %input {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x64xf32>
    // CHECK: linalg.reduce {{.*}}ssbuffer.block_id = 311 : i32
    %small = linalg.reduce ins(%big : tensor<1x64xf32>) outs(%init : tensor<64xf32>) dimensions = [0] {ssbuffer.block_id = 311 : i32, ssbuffer.core_type = "VECTOR"}
      (%x: f32, %y: f32) {
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 311 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %first = arith.addf %small, %acc {ssbuffer.block_id = 311 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    scf.yield %first : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
