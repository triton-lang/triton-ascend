// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// DIAG-LABEL: [cv-split] Function: cube_and_vector
// DIAG: [cv-split] Classification: 4C 4V
// DIAG-NOT: Loop must contain both CUBE and VECTOR ops, skip

// IR-NOT: ssbuffer.core_type
// IR-LABEL: func.func @cube_and_vector
// IR: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @cube_and_vector(%lhs_src: memref<32x16xf16>,
                             %rhs: tensor<16x16xf16>,
                             %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %lhs_buffer = memref.alloc() : memref<32x16xf16>
    memref.copy %lhs_src, %lhs_buffer : memref<32x16xf16> to memref<32x16xf16>
    %lhs = bufferization.to_tensor %lhs_buffer restrict writable :
        memref<32x16xf16> to tensor<32x16xf16>
    scf.for %iv = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %result = math.exp %matmul : tensor<32x16xf32>
    }
    return
  }
}
