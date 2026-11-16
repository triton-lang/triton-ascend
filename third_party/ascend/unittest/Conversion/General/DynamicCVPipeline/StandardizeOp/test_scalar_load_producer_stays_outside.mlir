// RUN: triton-opt --ssbuf-standardize-op-pack-memref %s | FileCheck %s

module {
  // ============================================================================
  // scalar_load_producer_stays_outside
  // A producer that reads memory has a scalar result but is still not part of
  // the packed unit: moving or cloning it would repeat the read at a point
  // where the buffer may already hold another value. The original therefore
  // stays outside the scope and the packed body reads it across the boundary.
  // ============================================================================
  // CHECK-LABEL: func.func @scalar_load_producer_stays_outside(
  // CHECK: %[[IDX:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<4xi32>
  // CHECK: %[[OFF:.*]] = arith.index_cast %[[IDX]] : i32 to index
  // CHECK: %[[SCOPE:.*]] = scope.scope
  // CHECK-NOT: memref.load
  // CHECK: } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
  func.func @scalar_load_producer_stays_outside(
      %arg0: memref<?xf16>, %idx: memref<4xi32>) -> tensor<32x64xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %raw = memref.load %idx[%c0] : memref<4xi32>
    %off = arith.index_cast %raw : i32 to index
    %alloc = memref.alloc() : memref<32x64xf16>
    scf.for %i = %c0 to %c32 step %c1 {
      %i2 = arith.addi %i, %off : index
      %subview = memref.subview %alloc[%i2, 0] [1, 64] [1, 1] : memref<32x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
      %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
      memref.copy %src, %subview : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    } {ExtractedLoadOrStore}
    %lhs = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16> to tensor<32x64xf16>
    return %lhs : tensor<32x64xf16>
  }
}
