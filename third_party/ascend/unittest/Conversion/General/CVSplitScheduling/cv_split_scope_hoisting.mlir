// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s

// Verify that the CUBE scope hoists an iteration-invariant L1 layout view and
// its memory-space cast before the cloned loop.

// CHECK-LABEL: func.func @hoist_cube_layout_view
// CHECK: scope.scope : () -> () {
// CHECK: %[[VIEW:.*]] = hivm.hir.convert_layout
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[VIEW]]
// CHECK: scf.for
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: scope.return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @hoist_cube_layout_view(
      %lhs_src: memref<32x16xf16>, %init: tensor<32x16xf32>) {
    %lhs_buffer = memref.alloc() : memref<32x16xf16>
    memref.copy %lhs_src, %lhs_buffer : memref<32x16xf16> to memref<32x16xf16>
    %lhs = bufferization.to_tensor %lhs_buffer restrict writable :
        memref<32x16xf16> to tensor<32x16xf16>
    %l1 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %view = hivm.hir.convert_layout %l1 output_shape [16, 16]
          {dstLayout = #hivm.data_layout<ND>,
           srcLayout = #hivm.data_layout<ND>} :
          (memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) ->
          memref<16x16xf16, #hivm.address_space<cbuf>>
      %plain = memref.memory_space_cast %view :
          memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      %rhs = bufferization.to_tensor %plain restrict writable :
          memref<16x16xf16> to tensor<16x16xf16>
      %matmul = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %matmul : tensor<32x16xf32>
    }
    return
  }
}
