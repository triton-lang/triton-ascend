// RUN: triton-opt --op-classifier %s | FileCheck %s

// A matmul slice used by VECTOR compute must not seed CUBE colouring or
// duplicate the shared loop indices. Check raw classifier output without DCE.
module {
  // CHECK-LABEL: func.func @slice_for_vector_compute
  func.func @slice_for_vector_compute(%a: tensor<64x64xf16>,
      %b: tensor<64x256xf16>, %init: tensor<64x256xf32>)
      -> tensor<64x256xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
    %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x256xf16>)
        outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
    // CHECK: scf.for
    // CHECK-NEXT: %[[OFFSET:.*]] = arith.muli {{.*}} {ssbuffer.core_type = "VECTOR"}
    // CHECK-NEXT: %[[IDX:.*]] = arith.index_cast %[[OFFSET]] {ssbuffer.core_type = "VECTOR"}
    // CHECK-NEXT: %[[ACC:.*]] = tensor.extract_slice {{.*}}[%[[IDX]], 0] {{.*}} {ssbuffer.core_type = "VECTOR"}
    // CHECK-NEXT: %[[PV:.*]] = tensor.extract_slice %[[MM]][%[[IDX]], 0] {{.*}} {ssbuffer.core_type = "VECTOR"}
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[ACC]], %[[PV]] {ssbuffer.core_type = "VECTOR"}
    // CHECK-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[ADD]] {{.*}} {ssbuffer.core_type = "VECTOR"}
    // CHECK-NEXT: scf.yield {ssbuffer.core_type = "VECTOR"} %[[INSERT]]
    // CHECK-NEXT: } {ssbuffer.core_type = "VECTOR"}
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init)
        -> (tensor<64x256xf32>) : i32 {
      %offset = arith.muli %i, %c16 : i32
      %idx = arith.index_cast %offset : i32 to index
      %acc_i = tensor.extract_slice %acc[%idx, 0] [16, 256] [1, 1]
          : tensor<64x256xf32> to tensor<16x256xf32>
      %pv_i = tensor.extract_slice %mm[%idx, 0] [16, 256] [1, 1]
          : tensor<64x256xf32> to tensor<16x256xf32>
      %add = arith.addf %acc_i, %pv_i : tensor<16x256xf32>
      %insert = tensor.insert_slice %add into %acc[%idx, 0] [16, 256] [1, 1]
          : tensor<16x256xf32> into tensor<64x256xf32>
      scf.yield %insert : tensor<64x256xf32>
    }
    return %result : tensor<64x256xf32>
  }

  // Preserve the intended matmul -> slice -> materialize CUBE path.
  // CHECK-LABEL: func.func @slice_for_cube_store
  func.func @slice_for_cube_store(%a: tensor<64x64xf16>,
      %b: tensor<64x256xf16>, %init: tensor<64x256xf32>,
      %dest: memref<16x256xf32>, %offset: index) {
    // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
    %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x256xf16>)
        outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
    // CHECK-NEXT: %[[SLICE:.*]] = tensor.extract_slice %[[MM]]{{.*}} {ssbuffer.core_type = "CUBE"}
    %slice = tensor.extract_slice %mm[%offset, 0] [16, 256] [1, 1]
        : tensor<64x256xf32> to tensor<16x256xf32>
    // CHECK-NEXT: bufferization.materialize_in_destination %[[SLICE]] {{.*}} {ssbuffer.core_type = "CUBE"}
    bufferization.materialize_in_destination %slice in writable %dest
        : (tensor<16x256xf32>, memref<16x256xf32>) -> ()
    return
  }

  // Follow consecutive slices across blocks to a materialize.
  // CHECK-LABEL: func.func @two_slices_for_cube_store
  func.func @two_slices_for_cube_store(%a: tensor<64x64xf16>, %b: tensor<64x256xf16>, %init: tensor<64x256xf32>, %dest: memref<16x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
    %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x256xf16>)
        outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
    // CHECK-NEXT: scf.for
    scf.for %i = %c0 to %c1 step %c1 {
      // CHECK-NEXT: %[[S0:.*]] = tensor.extract_slice %[[MM]]{{.*}} {ssbuffer.core_type = "CUBE"}
      %s0 = tensor.extract_slice %mm[0, 0] [32, 256] [1, 1]
          : tensor<64x256xf32> to tensor<32x256xf32>
      // CHECK-NEXT: %[[S1:.*]] = tensor.extract_slice %[[S0]]{{.*}} {ssbuffer.core_type = "CUBE"}
      %s1 = tensor.extract_slice %s0[0, 0] [16, 256] [1, 1]
          : tensor<32x256xf32> to tensor<16x256xf32>
      // CHECK-NEXT: bufferization.materialize_in_destination %[[S1]] {{.*}} {ssbuffer.core_type = "CUBE"}
      bufferization.materialize_in_destination %s1 in writable %dest
          : (tensor<16x256xf32>, memref<16x256xf32>) -> ()
    }
    return
  }

  // Follow consecutive slices across blocks to a hivm store.
  // CHECK-LABEL: func.func @three_slices_for_cube_store
  func.func @three_slices_for_cube_store(%a: tensor<64x64xf16>, %b: tensor<64x256xf16>, %init: tensor<64x256xf32>, %dest: memref<8x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
    %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x256xf16>)
        outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
    // CHECK-NEXT: scf.for
    scf.for %i = %c0 to %c1 step %c1 {
      // CHECK-NEXT: %[[S0:.*]] = tensor.extract_slice %[[MM]]{{.*}} {ssbuffer.core_type = "CUBE"}
      %s0 = tensor.extract_slice %mm[0, 0] [32, 256] [1, 1]
          : tensor<64x256xf32> to tensor<32x256xf32>
      // CHECK-NEXT: %[[S1:.*]] = tensor.extract_slice %[[S0]]{{.*}} {ssbuffer.core_type = "CUBE"}
      %s1 = tensor.extract_slice %s0[0, 0] [16, 256] [1, 1]
          : tensor<32x256xf32> to tensor<16x256xf32>
      // CHECK-NEXT: %[[S2:.*]] = tensor.extract_slice %[[S1]]{{.*}} {ssbuffer.core_type = "CUBE"}
      %s2 = tensor.extract_slice %s1[0, 0] [8, 256] [1, 1]
          : tensor<16x256xf32> to tensor<8x256xf32>
      // CHECK-NEXT: hivm.hir.store ins(%[[S2]] {{.*}} {ssbuffer.core_type = "CUBE"}
      hivm.hir.store ins(%s2 : tensor<8x256xf32>) outs(%dest : memref<8x256xf32>)
    }
    return
  }

  // Preserve the original slice seeding in the matmul's own block, including
  // the CUBE copy when the slice is also required by VECTOR compute.
  // CHECK-LABEL: func.func @two_slices_for_vector_compute
  func.func @two_slices_for_vector_compute(%a: tensor<64x64xf16>, %b: tensor<64x256xf16>, %init: tensor<64x256xf32>, %other: tensor<16x256xf32>) -> tensor<16x256xf32> {
    // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
    %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x256xf16>)
        outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
    // CHECK-NEXT: %[[S0:.*]] = tensor.extract_slice %[[MM]]{{.*}} {ssbuffer.core_type = "VECTOR"}
    %s0 = tensor.extract_slice %mm[0, 0] [32, 256] [1, 1]
        : tensor<64x256xf32> to tensor<32x256xf32>
    // CHECK-NEXT: %{{.*}} = tensor.extract_slice %[[MM]]{{.*}} {ssbuffer.core_type = "CUBE"}
    // CHECK-NEXT: %[[S1:.*]] = tensor.extract_slice %[[S0]]{{.*}} {ssbuffer.core_type = "VECTOR"}
    %s1 = tensor.extract_slice %s0[0, 0] [16, 256] [1, 1]
        : tensor<32x256xf32> to tensor<16x256xf32>
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[S1]], %{{.*}} {ssbuffer.core_type = "VECTOR"}
    %add = arith.addf %s1, %other : tensor<16x256xf32>
    // CHECK-NEXT: return %[[ADD]]
    return %add : tensor<16x256xf32>
  }

  // Follow consecutive slices, stopping at the VECTOR compute boundary.
  // CHECK-LABEL: func.func @two_slices_for_vector_compute_across_blocks
  func.func @two_slices_for_vector_compute_across_blocks(%a: tensor<64x64xf16>, %b: tensor<64x256xf16>, %init: tensor<64x256xf32>, %other: tensor<16x256xf32>) -> tensor<16x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
    %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x256xf16>)
        outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
    // CHECK-NEXT: %[[RESULT:.*]] = scf.for
    %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %other)
        -> (tensor<16x256xf32>) {
      // CHECK-NEXT: %[[S0:.*]] = tensor.extract_slice %[[MM]]{{.*}} {ssbuffer.core_type = "VECTOR"}
      %s0 = tensor.extract_slice %mm[0, 0] [32, 256] [1, 1]
          : tensor<64x256xf32> to tensor<32x256xf32>
      // CHECK-NEXT: %[[S1:.*]] = tensor.extract_slice %[[S0]]{{.*}} {ssbuffer.core_type = "VECTOR"}
      %s1 = tensor.extract_slice %s0[0, 0] [16, 256] [1, 1]
          : tensor<32x256xf32> to tensor<16x256xf32>
      // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[S1]], %{{.*}} {ssbuffer.core_type = "VECTOR"}
      %add = arith.addf %s1, %acc : tensor<16x256xf32>
      // CHECK-NEXT: scf.yield {ssbuffer.core_type = "VECTOR"} %[[ADD]]
      scf.yield %add : tensor<16x256xf32>
    }
    // CHECK: return %[[RESULT]]
    return %result : tensor<16x256xf32>
  }

  // Follow scf.yield to the real user, then compare its block with matmul's.
  // CHECK-LABEL: func.func @slice_after_matmul_loop
  func.func @slice_after_matmul_loop(%a: tensor<64x64xf16>,
      %b: tensor<64x256xf16>, %init: tensor<64x256xf32>,
      %other: tensor<16x256xf32>) -> tensor<16x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[RESULT:.*]] = scf.for
    %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %init)
        -> (tensor<64x256xf32>) {
      // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
      %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x256xf16>)
          outs(%acc : tensor<64x256xf32>) -> tensor<64x256xf32>
      // CHECK-NEXT: scf.yield {ssbuffer.core_type = "CUBE"} %[[MM]]
      scf.yield %mm : tensor<64x256xf32>
    }
    // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[RESULT]]{{.*}} {ssbuffer.core_type = "VECTOR"}
    %slice = tensor.extract_slice %result[0, 0] [16, 256] [1, 1]
        : tensor<64x256xf32> to tensor<16x256xf32>
    // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[SLICE]], %{{.*}} {ssbuffer.core_type = "VECTOR"}
    %add = arith.addf %slice, %other : tensor<16x256xf32>
    // CHECK-NEXT: return %[[ADD]]
    return %add : tensor<16x256xf32>
  }

}
