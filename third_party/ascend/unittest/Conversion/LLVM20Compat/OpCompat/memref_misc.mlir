// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @memref_misc(
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[LD0:.*]] = memref.load %arg0[%[[C0]]] : memref<4xf32>
// CHECK-NEXT: memref.store %[[C1]], %arg0[%[[C0]]] : memref<4xf32>
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT: memref.store %[[C1]], %[[ALLOC]][%[[C0]]] : memref<4xf32>
// CHECK-NEXT: %[[LD1:.*]] = memref.load %[[ALLOC]][%[[C0]]] : memref<4xf32>
// CHECK-NEXT: memref.dealloc %[[ALLOC]] : memref<4xf32>
// CHECK-NEXT: return %[[LD0]] : f32
func.func @memref_misc(%arg0: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1.000000e+00 : f32
  %ld0 = memref.load %arg0[%c0] : memref<4xf32>
  memref.store %c1, %arg0[%c0] : memref<4xf32>

  %alloc = memref.alloc() : memref<4xf32>
  memref.store %c1, %alloc[%c0] : memref<4xf32>
  %ld1 = memref.load %alloc[%c0] : memref<4xf32>
  memref.dealloc %alloc : memref<4xf32>

  return %ld0 : f32
}

// -----

// CHECK-LABEL: func.func @memref_extract_aligned_pointer_as_index(
// CHECK-NEXT: %[[P:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?xi32> -> index
// CHECK-NEXT: return %[[P]] : index
func.func @memref_extract_aligned_pointer_as_index(%arg0: memref<?xi32>) -> index {
  %p = memref.extract_aligned_pointer_as_index %arg0 : memref<?xi32> -> index
  return %p : index
}
