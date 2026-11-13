// RUN: triton-opt %s --split-input-file --merge-small-block --verify-each | FileCheck %s --enable-var-scope

// Infer region core types without requiring an attribute on the scf.if.

// CHECK-LABEL: func.func @scalar_to_vector_if(
func.func @scalar_to_vector_if(%i: i32, %a: tensor<8xf32>, %dst: memref<8xf32>) {
  %c0 = arith.constant 0 : i32
  // CHECK: arith.cmpi {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  %cond = arith.cmpi eq, %i, %c0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : i32
  scf.if %cond {
    bufferization.materialize_in_destination %a in writable %dst {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<8xf32>, memref<8xf32>) -> ()
  } {ssbuffer.block_id = 2 : i32}
  return
}

// -----

// CHECK-LABEL: func.func @scalar_to_cube_if(
func.func @scalar_to_cube_if(%i: i32, %a: tensor<8xf32>, %dst: memref<8xf32>) {
  %c0 = arith.constant 0 : i32
  // CHECK: arith.cmpi {{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
  %cond = arith.cmpi eq, %i, %c0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : i32
  scf.if %cond {
    bufferization.materialize_in_destination %a in writable %dst {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : (tensor<8xf32>, memref<8xf32>) -> ()
  } {ssbuffer.block_id = 2 : i32}
  return
}

// -----

// CHECK-LABEL: func.func @scalar_to_mixed_if(
func.func @scalar_to_mixed_if(%i: i32, %a: tensor<8xf32>, %dst: memref<8xf32>) {
  %c0 = arith.constant 0 : i32
  // CHECK: arith.cmpi {{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
  %cond = arith.cmpi eq, %i, %c0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : i32
  scf.if %cond {
    bufferization.materialize_in_destination %a in writable %dst {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<8xf32>, memref<8xf32>) -> ()
  } else {
    bufferization.materialize_in_destination %a in writable %dst {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : (tensor<8xf32>, memref<8xf32>) -> ()
  } {ssbuffer.block_id = 2 : i32}
  return
}
