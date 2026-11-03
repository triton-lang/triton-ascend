// RUN: triton-opt --ssbuf-unpack-scopeop %s | FileCheck %s

// clang-format off: MLIR, and the C++ formatter mangles its signatures.

// A packed unit's tags describe all of its work, not only the ops that sit
// directly in the scope body: the work inside a loop keeps its own region, and
// the passes that read `block_id` / `core_type` look at an op's own attributes
// only. Unpacking therefore has to stamp every nested op as well.
//
// `ssbuffer.load` marks the unit itself, so it must not survive the unpack.
module {
  // CHECK-LABEL: func.func @unpack_stamps_nested_ops(
  // CHECK-NOT: ssbuffer.load
  // CHECK-NOT: scope.scope
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: scf.for
  // CHECK: %[[ROW:.*]] = memref.subview %[[ALLOC]]{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: memref.copy {{.*}}, %[[ROW]] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
  func.func @unpack_stamps_nested_ops(%gm: memref<4x16xf16>) {
    %0 = scope.scope : () -> tensor<4x16xf16> {
      %alloc = memref.alloc() : memref<4x16xf16>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %row = memref.subview %alloc[%i, 0] [1, 16] [1, 1] : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
        %src = memref.subview %gm[%i, 0] [1, 16] [1, 1] : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
        memref.copy %src, %row : memref<1x16xf16, strided<[16, 1], offset: ?>> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      }
      %t = bufferization.to_tensor %alloc restrict writable : memref<4x16xf16> to tensor<4x16xf16>
      scope.return %t : tensor<4x16xf16>
    } {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
    return
  }
}
// clang-format on
