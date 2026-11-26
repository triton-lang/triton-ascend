// RUN: triton-opt --exp-load-pattern %s | FileCheck %s

// An `exp(load) -> broadcast -> mul` chain must end up in the block of the
// multiplication: the load unit, the exp and the broadcast are moved together.
// After PackMemref the load unit is one atomic `ssbuffer.load` scope, before it
// (or for a unit PackMemref left unpacked) it is the raw alloc/copy/to_tensor
// chain.

module {
  // The load unit is packed: the scope travels as one op and its body keeps no
  // block id.
  // CHECK-LABEL: func.func @packed_load_scope(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<64xf32> {
  // CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<64xf32>
  // CHECK-NEXT: memref.copy {{.*}}, %[[ALLOC]] : memref<64xf32> to memref<64xf32>
  // CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<64xf32>
  // CHECK-NEXT: scope.return %[[T]] : tensor<64xf32>
  // CHECK-NEXT: } {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: math.exp %[[SCOPE]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: linalg.broadcast {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  func.func @packed_load_scope(%gm: memref<64xf32>,
                               %arg: tensor<8x64xf32>) -> tensor<8x64xf32> {
    %0 = scope.scope : () -> tensor<64xf32> {
      %alloc = memref.alloc() : memref<64xf32>
      memref.copy %gm, %alloc : memref<64xf32> to memref<64xf32>
      %t = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
      scope.return %t : tensor<64xf32>
    } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
    %e = math.exp %0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<8x64xf32>
    %b = linalg.broadcast ins(%e : tensor<64xf32>) outs(%empty : tensor<8x64xf32>) dimensions = [0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
    %m = arith.mulf %b, %arg {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x64xf32>
    return %m : tensor<8x64xf32>
  }

  // Fallback: a unit PackMemref left unpacked keeps the raw chain, which is
  // matched and moved the same way.
  // CHECK-LABEL: func.func @raw_load_unit(
  // CHECK: %[[ALLOC2:.*]] = memref.alloc() {ssbuffer.block_id = 22 : i32} : memref<64xf32>
  // CHECK-NEXT: memref.copy {{.*}}, %[[ALLOC2]] {ssbuffer.block_id = 22 : i32} : memref<64xf32> to memref<64xf32>
  // CHECK-NEXT: %[[T2:.*]] = bufferization.to_tensor %[[ALLOC2]] restrict writable {ssbuffer.block_id = 22 : i32}
  // CHECK-NEXT: math.exp %[[T2]] {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: linalg.broadcast {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"}
  func.func @raw_load_unit(%gm: memref<64xf32>,
                           %arg: tensor<8x64xf32>) -> tensor<8x64xf32> {
    %alloc = memref.alloc() {ssbuffer.block_id = 21 : i32} : memref<64xf32>
    memref.copy %gm, %alloc {ssbuffer.block_id = 21 : i32} : memref<64xf32> to memref<64xf32>
    %t = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 21 : i32} : memref<64xf32> to tensor<64xf32>
    %e = math.exp %t {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %empty = tensor.empty() {ssbuffer.block_id = 21 : i32} : tensor<8x64xf32>
    %b = linalg.broadcast ins(%e : tensor<64xf32>) outs(%empty : tensor<8x64xf32>) dimensions = [0] {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"}
    %m = arith.mulf %b, %arg {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x64xf32>
    return %m : tensor<8x64xf32>
  }
}
