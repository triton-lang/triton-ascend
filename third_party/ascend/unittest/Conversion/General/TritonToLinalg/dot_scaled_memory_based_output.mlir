// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s --check-prefix=A3 --implicit-check-not=tensor.insert_slice
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s --check-prefix=A5 --implicit-check-not=bufferization.materialize_in_destination

// C is an arbitrary input snapshot. A fresh output buffer is fully written by
// disjoint 16x32 tiles; K still advances by 32 in the original order. In
// particular, the second N tile must read the original C, not unwritten output.
// A3-LABEL: func.func @two_n_tiles
// A3-SAME: %[[A:.*]]: tensor<16x64xf16>, %[[B:.*]]: tensor<64x64xbf16>, %[[C:.*]]: tensor<16x64xf32>
// A3-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// A3-DAG: %[[STEP:.*]] = arith.constant 32 : index
// A3-DAG: %[[END:.*]] = arith.constant 64 : index
// A3: %[[OUT:.*]] = memref.alloc() : memref<16x64xf32>
// A3: scf.for %[[N:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[END]] step %[[STEP]] {
// A3: tensor.extract_slice %[[C]][0, %[[N]]] [16, 32] [1, 1]
// A3: scf.for {{.*}} iter_args
// A3: linalg.matmul {{.*}}input_precision = "ieee"
// A3-SAME: outs({{.*}} : tensor<16x32xf32>)
// A3: %[[DST:.*]] = memref.subview %[[OUT]][0, %[[N]]] [16, 32] [1, 1]
// A3: bufferization.materialize_in_destination {{.*}} in writable %[[DST]]
// A3: bufferization.to_tensor %[[OUT]] restrict writable
// A3-NOT: tensor.insert_slice
// A5-LABEL: func.func @two_n_tiles
// A5-NOT: memref.alloc
// A5: scf.for {{.*}} iter_args
// A5: scf.for {{.*}} iter_args
// A5: linalg.matmul {{.*}}input_precision = "ieee"
// A5: tensor.insert_slice
// A5-NOT: bufferization.materialize_in_destination
func.func @two_n_tiles(%a: tensor<16x64xf16>, %b: tensor<64x64xbf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = bf16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// Both M and N have multiple tiles. No full-output tensor is carried through
// these loops on A3, while the K reduction still carries its 32x32 accumulator.
// A3-LABEL: func.func @both_axes
// A3-SAME: %[[A:.*]]: tensor<64x64xf16>, %[[B:.*]]: tensor<64x64xbf16>, %[[C:.*]]: tensor<64x64xf32>
// A3-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// A3-DAG: %[[STEP:.*]] = arith.constant 32 : index
// A3-DAG: %[[END:.*]] = arith.constant 64 : index
// A3: %[[OUT:.*]] = memref.alloc() : memref<64x64xf32>
// A3: scf.for %[[M:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[END]] step %[[STEP]] {
// A3: scf.for %[[N:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[END]] step %[[STEP]] {
// A3: tensor.extract_slice %[[C]][%[[M]], %[[N]]] [32, 32] [1, 1]
// A3: scf.for {{.*}} iter_args
// A3: linalg.matmul {{.*}}input_precision = "ieee"
// A3: %[[DST:.*]] = memref.subview %[[OUT]][%[[M]], %[[N]]] [32, 32] [1, 1]
// A3: bufferization.materialize_in_destination {{.*}} in writable %[[DST]]
// A3: bufferization.to_tensor %[[OUT]] restrict writable
// A3-NOT: tensor.insert_slice
// A5-LABEL: func.func @both_axes
// A5-NOT: memref.alloc
// A5: scf.for {{.*}} iter_args
// A5: scf.for {{.*}} iter_args
// A5: scf.for {{.*}} iter_args
// A5: linalg.matmul {{.*}}input_precision = "ieee"
// A5: tensor.insert_slice
func.func @both_axes(%a: tensor<64x64xf16>, %b: tensor<64x64xbf16>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = bf16 {fastMath = false} : tensor<64x64xf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// The M-only case also covers a dimension smaller than the tile bound.
// A3-LABEL: func.func @narrow_n
// A3-SAME: %[[A:.*]]: tensor<64x32xf16>, %[[B:.*]]: tensor<32x16xbf16>, %[[C:.*]]: tensor<64x16xf32>
// A3-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// A3-DAG: %[[STEP:.*]] = arith.constant 32 : index
// A3-DAG: %[[END:.*]] = arith.constant 64 : index
// A3: %[[OUT:.*]] = memref.alloc() : memref<64x16xf32>
// A3: scf.for %[[M:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[END]] step %[[STEP]] {
// A3: tensor.extract_slice %[[C]][%[[M]], 0] [32, 16] [1, 1]
// A3: linalg.matmul {{.*}}input_precision = "ieee"
// A3: %[[DST:.*]] = memref.subview %[[OUT]][%[[M]], 0] [32, 16] [1, 1]
// A3: bufferization.materialize_in_destination {{.*}} in writable %[[DST]]
// A3: bufferization.to_tensor %[[OUT]] restrict writable
// A5-LABEL: func.func @narrow_n
// A5: scf.for {{.*}} iter_args
// A5: linalg.matmul {{.*}}input_precision = "ieee"
// A5: tensor.insert_slice
func.func @narrow_n(%a: tensor<64x32xf16>, %b: tensor<32x16xbf16>, %c: tensor<64x16xf32>) -> tensor<64x16xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = bf16 {fastMath = false} : tensor<64x32xf16> * tensor<32x16xbf16> -> tensor<64x16xf32>
  return %d : tensor<64x16xf32>
}
