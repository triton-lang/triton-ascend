// RUN: triton-opt --op-classifier %s | FileCheck %s

// Regression test for `OpClassifierPass` (GEN-1571):
//   A boundary-masked load lowers to
//     %alloc = memref.alloc()
//     scf.if %rare { linalg.fill ... outs(%alloc) }   // zero-pad guard
//     memref.copy %src, %subview(%alloc)
//     %t = bufferization.to_tensor %alloc
//   When %t feeds both a CUBE consumer (linalg.matmul) and a VECTOR consumer
//   (arith.addf), the buffer chain is CUBE_AND_VECTOR and handleCubeAndVector
//   splits it so each core computes into its own buffer copy. The result-less
//   guard (and its fill) used to stay CUBE_ONLY: the guard never appears on a
//   value def chain and the fill is not the buffer's last writer (the copy
//   is), so VECTOR propagation never reached it. The VECTOR-side buffer copy
//   was therefore never zero-filled and its out-of-bounds region read
//   uninitialized memory (GEN-1571: 2D float32 unaligned shape (31, 63),
//   mismatch=2/31).
//
//   The fix upgrades such guards to CUBE_AND_VECTOR before the split, so the
//   guard (with its fill) is cloned to the VECTOR side as well and each
//   side's copy of the buffer is guarded.

module {
  // CHECK-LABEL: func.func @fill_guard_duplicated_for_vector_side
  func.func @fill_guard_duplicated_for_vector_side(
      %src: memref<16x16xf32>,
      %rhs: tensor<16x16xf32>,
      %rare: i1) -> (tensor<16x16xf32>, tensor<16x16xf32>) {

    // The buffer chain is split into a CUBE copy and a VECTOR copy.
    // CHECK: [[VALLOC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<16x16xf32>
    // CHECK: [[CALLOC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<16x16xf32>
    // The guard is duplicated: the VECTOR-side guard zero-fills the
    // VECTOR-side buffer (before the fix this guard stayed CUBE-only and
    // the VECTOR-side copy was never zero-filled)...
    // CHECK: scf.if
    // CHECK: linalg.fill {ssbuffer.core_type = "VECTOR"} ins({{.*}} : f32) outs([[VALLOC]] : memref<16x16xf32>)
    // ...and the CUBE-side guard keeps filling the CUBE-side buffer.
    // CHECK: scf.if
    // CHECK: linalg.fill {ssbuffer.core_type = "CUBE"} ins({{.*}} : f32) outs([[CALLOC]] : memref<16x16xf32>)
    // Each side copies GM into its own guarded buffer and exposes it as a
    // tensor.
    // CHECK: memref.copy {{.*}} {ssbuffer.core_type = "VECTOR"}
    // CHECK: memref.copy {{.*}} {ssbuffer.core_type = "CUBE"}
    // CHECK: [[VT:%[A-Za-z0-9_]+]] = bufferization.to_tensor [[VALLOC]] restrict writable {ssbuffer.core_type = "VECTOR"}
    // CHECK: [[CT:%[A-Za-z0-9_]+]] = bufferization.to_tensor [[CALLOC]] restrict writable {ssbuffer.core_type = "CUBE"}
    // The CUBE matmul reads the CUBE copy, the VECTOR addf reads the guarded
    // VECTOR copy.
    // CHECK: linalg.matmul {{.*}}{ssbuffer.core_type = "CUBE"} ins([[CT]], {{.*}}
    // CHECK: arith.addf [[VT]], [[VT]] {ssbuffer.core_type = "VECTOR"}
    // Negative guard: no CUBE_AND_VECTOR residue is left behind.
    // CHECK-NOT: ssbuffer.core_type = "CUBE_AND_VECTOR"

    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    scf.if %rare {
      linalg.fill ins(%zero : f32) outs(%alloc : memref<16x16xf32>)
    }
    %src_view = memref.subview %src[%c0, %c0] [%c16, %c16] [1, 1]
      : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
    %dst_view = memref.subview %alloc[%c0, %c0] [%c16, %c16] [1, 1]
      : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
    memref.copy %src_view, %dst_view
      : memref<?x?xf32, strided<[16, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
    %t = bufferization.to_tensor %alloc restrict writable
      : memref<16x16xf32> to tensor<16x16xf32>
    %out = tensor.empty() : tensor<16x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %mm = linalg.matmul ins(%t, %rhs : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %acc = arith.addf %t, %t : tensor<16x16xf32>
    return %mm, %acc : tensor<16x16xf32>, tensor<16x16xf32>
  }

  // Negative shape: a guard holding TWO fills is not the single-fill bug
  // pattern and must NOT be upgraded. It keeps its CUBE_ONLY mark, no
  // VECTOR-side guard is cloned, and the VECTOR-side buffer stays unguarded
  // (conservative by design: only the exact single-fill shape is upgraded).
  // CHECK-LABEL: func.func @multi_fill_guard_not_upgraded
  func.func @multi_fill_guard_not_upgraded(
      %src: memref<16x16xf32>,
      %rhs: tensor<16x16xf32>,
      %rare: i1) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
    // The buffer chain is still split into a CUBE copy and a VECTOR copy...
    // CHECK: [[MVALLOC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<16x16xf32>
    // CHECK: [[MCALLOC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<16x16xf32>
    // ...but only ONE guard remains: it keeps the CUBE mark and both of its
    // fills still write the CUBE-side buffer.
    // CHECK: scf.if
    // CHECK: linalg.fill {ssbuffer.core_type = "CUBE"} ins({{.*}} : f32) outs([[MCALLOC]] : memref<16x16xf32>)
    // CHECK: linalg.fill {ssbuffer.core_type = "CUBE"} ins({{.*}} : f32) outs([[MCALLOC]] : memref<16x16xf32>)
    // No VECTOR-side guard is cloned for this richer shape.
    // CHECK-NOT: linalg.fill {ssbuffer.core_type = "VECTOR"}

    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    scf.if %rare {
      linalg.fill ins(%zero : f32) outs(%alloc : memref<16x16xf32>)
      linalg.fill ins(%zero : f32) outs(%alloc : memref<16x16xf32>)
    }
    %src_view = memref.subview %src[%c0, %c0] [%c16, %c16] [1, 1]
      : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
    %dst_view = memref.subview %alloc[%c0, %c0] [%c16, %c16] [1, 1]
      : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1], offset: ?>>
    memref.copy %src_view, %dst_view
      : memref<?x?xf32, strided<[16, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1], offset: ?>>
    %t = bufferization.to_tensor %alloc restrict writable
      : memref<16x16xf32> to tensor<16x16xf32>
    %out = tensor.empty() : tensor<16x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %mm = linalg.matmul ins(%t, %rhs : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %acc = arith.addf %t, %t : tensor<16x16xf32>
    return %mm, %acc : tensor<16x16xf32>, tensor<16x16xf32>
  }
}
