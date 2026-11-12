// RUN: triton-opt --ssbuf-standardize-op-pack-memref --plan-compute-block %s | FileCheck %s

// C is only needed by the post-matmul add. Its annotation must not introduce
// a memory dependency into the unrelated bias load and retain C in the first
// vector block. The C load must be scheduled after the matmul with its extf.
module {
  // CHECK-LABEL: func.func @refine_annotated_load(
  // CHECK: %[[BIAS:.*]] = scope.scope : () -> tensor<4xf16> {
  // CHECK: linalg.matmul
  // CHECK: %[[C:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK: annotation.mark {{.*}} {MayImplicitTransposeWithLastAxis}
  // CHECK: scope.return
  // CHECK-NEXT: } {ssbuffer.block_id = [[C_BLOCK:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: arith.extf %[[C]] {ssbuffer.block_id = [[C_BLOCK]] : i32, ssbuffer.core_type = "VECTOR"}
  func.func @refine_annotated_load(%c: memref<4x4xf16>, %bias: memref<4xf16>, %a: tensor<4x4xf16>, %b: tensor<4x4xf16>, %count: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %count step %c1 {
      %src = memref.subview %c[0, 0] [4, 4] [1, 1] : memref<4x4xf16> to memref<4x4xf16, strided<[4, 1]>>
      %cbuf = memref.alloc() : memref<4x4xf16>
      memref.copy %src, %cbuf : memref<4x4xf16, strided<[4, 1]>> to memref<4x4xf16>
      %old = bufferization.to_tensor %cbuf restrict writable : memref<4x4xf16> to tensor<4x4xf16>
      annotation.mark %old {MayImplicitTransposeWithLastAxis} : tensor<4x4xf16>
      %biasbuf = memref.alloc() : memref<4xf16>
      memref.copy %bias, %biasbuf : memref<4xf16> to memref<4xf16>
      %bias_t = bufferization.to_tensor %biasbuf restrict writable : memref<4xf16> to tensor<4xf16>
      %bias_f32 = arith.extf %bias_t : tensor<4xf16> to tensor<4xf32>
      %empty = tensor.empty() : tensor<4x4xf32>
      %init = linalg.broadcast ins(%bias_f32 : tensor<4xf32>) outs(%empty : tensor<4x4xf32>) dimensions = [0]
      %mm = linalg.matmul ins(%a, %b : tensor<4x4xf16>, tensor<4x4xf16>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
      %out = arith.truncf %mm : tensor<4x4xf32> to tensor<4x4xf16>
      bufferization.materialize_in_destination %out in writable %src : (tensor<4x4xf16>, memref<4x4xf16, strided<[4, 1]>>) -> ()
      %old_f32 = arith.extf %old : tensor<4x4xf16> to tensor<4x4xf32>
      %sum = arith.addf %mm, %old_f32 : tensor<4x4xf32>
      %sum_f16 = arith.truncf %sum : tensor<4x4xf32> to tensor<4x4xf16>
      bufferization.materialize_in_destination %sum_f16 in writable %src : (tensor<4x4xf16>, memref<4x4xf16, strided<[4, 1]>>) -> ()
    }
    return
  }
}
