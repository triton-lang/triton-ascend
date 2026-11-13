// RUN: triton-opt %s --split-input-file --plan-vector-block --verify-each | FileCheck %s --enable-var-scope

// Tensor uses inside a void region op must wait for the backedge producer.
// CHECK-LABEL: func.func @void_if_then(
func.func @void_if_then(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>, %dst: memref<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    // CHECK: scf.if
    // CHECK: bufferization.materialize_in_destination {{.*}} {ssbuffer.block_id = [[DEFERRED:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    scf.if %cond {
      bufferization.materialize_in_destination %acc in writable %dst {ssbuffer.core_type = "VECTOR"} : (tensor<8x8xf32>, memref<8x8xf32>) -> ()
    }
    // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[INDEPENDENT:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    %cube = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[DEFERRED]] : i32, ssbuffer.core_type = "VECTOR"}
    %next = arith.addf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    scf.yield %next : tensor<8x8xf32>
  }
  return %r : tensor<8x8xf32>
}

// -----

// Tensor uses inside a void region op must wait for the backedge producer.
// CHECK-LABEL: func.func @void_if_else(
func.func @void_if_else(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>, %dst: memref<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    // CHECK: scf.if
    // CHECK: bufferization.materialize_in_destination {{.*}} {ssbuffer.block_id = [[DEFERRED:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    scf.if %cond {
    } else {
      bufferization.materialize_in_destination %acc in writable %dst {ssbuffer.core_type = "VECTOR"} : (tensor<8x8xf32>, memref<8x8xf32>) -> ()
    }
    // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[INDEPENDENT:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    %cube = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[DEFERRED]] : i32, ssbuffer.core_type = "VECTOR"}
    %next = arith.addf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    scf.yield %next : tensor<8x8xf32>
  }
  return %r : tensor<8x8xf32>
}

// -----

// Tensor uses inside a void region op must wait for the backedge producer.
// CHECK-LABEL: func.func @void_if_nested(
func.func @void_if_nested(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>, %dst: memref<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    // CHECK: scf.if
    // CHECK: bufferization.materialize_in_destination {{.*}} {ssbuffer.block_id = [[DEFERRED:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    scf.if %cond {
    } else {
      scf.if %cond {
        bufferization.materialize_in_destination %acc in writable %dst {ssbuffer.core_type = "VECTOR"} : (tensor<8x8xf32>, memref<8x8xf32>) -> ()
      }
    }
    // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = [[INDEPENDENT:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    %cube = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = [[DEFERRED]] : i32, ssbuffer.core_type = "VECTOR"}
    %next = arith.addf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    scf.yield %next : tensor<8x8xf32>
  }
  return %r : tensor<8x8xf32>
}
