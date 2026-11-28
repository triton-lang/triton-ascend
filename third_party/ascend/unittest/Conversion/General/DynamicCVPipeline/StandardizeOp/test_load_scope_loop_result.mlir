// RUN: triton-opt --ssbuf-standardize-op-pack-memref %s | FileCheck %s

// A load may use a scalar result of a multi-result loop. Capture the result
// without moving or cloning the loop and its tensor computation into the load.
// CHECK-LABEL: func.func @load_after_multi_result_loop(
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK: %[[LOAD:.*]] = scope.scope
// CHECK-NOT: scf.for
// CHECK: memref.reinterpret_cast {{.*}} to offset: [%[[LOOP]]#0]
// CHECK-NOT: scf.for
// CHECK: scope.return
// CHECK-NEXT: } {ssbuffer.load}
// CHECK-NOT: scf.for
// CHECK: return %[[LOAD]], %[[LOOP]]#1
func.func @load_after_multi_result_loop(%gm: memref<?xf32>, %n: index, %init: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %loop:2 = scf.for %i = %c0 to %n step %c1 iter_args(%offset = %c0, %acc = %init) -> (index, tensor<4xf32>) {
    %next = arith.addi %offset, %c4 : index
    %sum = arith.addf %acc, %init : tensor<4xf32>
    scf.yield %next, %sum : index, tensor<4xf32>
  }
  %view = memref.reinterpret_cast %gm to offset: [%loop#0], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<4xf32>
  memref.copy %view, %alloc : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
  %t = bufferization.to_tensor %alloc restrict writable : memref<4xf32> to tensor<4xf32>
  return %t, %loop#1 : tensor<4xf32>, tensor<4xf32>
}

// Preserve external users of a shared source view while packing an independent
// view chain and its scalar address calculation into the load scope.
// CHECK-LABEL: func.func @shared_source_view(
// CHECK: %[[OUTSIDE:.*]] = memref.reinterpret_cast
// CHECK: %[[LOAD:.*]] = scope.scope
// CHECK: arith.addi
// CHECK: %[[SOURCE:.*]] = memref.reinterpret_cast
// CHECK: %[[SUB:.*]] = memref.subview %[[SOURCE]]
// CHECK: memref.copy %[[SUB]],
// CHECK: scope.return
// CHECK-NEXT: } {ssbuffer.load}
// CHECK: return %[[LOAD]], %[[OUTSIDE]]
func.func @shared_source_view(%gm: memref<?xf32>, %offset: index) -> (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) {
  %c1 = arith.constant 1 : index
  %next = arith.addi %offset, %c1 : index
  %view = memref.reinterpret_cast %gm to offset: [%next], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  %sub = memref.subview %view[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<1xf32>
  memref.copy %sub, %alloc : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
  %t = bufferization.to_tensor %alloc restrict writable : memref<1xf32> to tensor<1xf32>
  return %t, %view : tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>
}
