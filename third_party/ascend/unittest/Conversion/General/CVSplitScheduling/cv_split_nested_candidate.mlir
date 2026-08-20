// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s

// A function may contain enclosing control loops as long as it has exactly one
// innermost candidate loop. Verify that the inner mixed CUBE/VECTOR loop is
// transformed while its enclosing loop remains intact.

// CHECK-LABEL: func.func @nested_mixed_candidate
// CHECK: scf.for
// CHECK: scope.scope
// CHECK: scf.for
// CHECK: scope.return
// CHECK: scope.scope
// CHECK: scf.for
// CHECK: scope.return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @nested_mixed_candidate(
      %lhs_src: memref<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %lhs_buffer = memref.alloc() : memref<32x16xf16>
    memref.copy %lhs_src, %lhs_buffer : memref<32x16xf16> to memref<32x16xf16>
    %lhs = bufferization.to_tensor %lhs_buffer restrict writable :
        memref<32x16xf16> to tensor<32x16xf16>
    scf.for %outer = %c0 to %c2 step %c1 {
      scf.for %inner = %c0 to %c16 step %c1 {
        %matmul = linalg.matmul
            ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
            outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
        %vector = math.exp %matmul : tensor<32x16xf32>
      }
    }
    return
  }
}
