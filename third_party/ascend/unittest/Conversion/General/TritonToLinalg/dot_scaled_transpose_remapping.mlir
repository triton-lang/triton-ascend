// RUN: triton-opt %s --split-input-file --triton-to-linalg="compile-on-910-95=true" --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @already_lowered_transpose
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>) outs({{.*}} : tensor<128x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return

func.func @already_lowered_transpose(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @converted_transpose_single_read
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>) outs({{.*}} : tensor<128x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return

func.func @converted_transpose_single_read(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// The original tt.trans has two real dot users; it is not a pending
// single-use conversion result even if the new transpose temporarily has none.
// CHECK-LABEL: func.func @converted_transpose_two_dots
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>) outs({{.*}} : tensor<128x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return

func.func @converted_transpose_two_dots(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  %d2 = tt.dot_scaled %a, %b, %d lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d, %d2 : tensor<16x64xf32>, tensor<16x64xf32>
}

// -----

// The input also feeds a separate already-lowered transpose. Ignore only
// the exact old/new replacement pair, not this additional logical read.
// CHECK-LABEL: func.func @converted_transpose_shared_input
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>) outs({{.*}} : tensor<128x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return

func.func @converted_transpose_shared_input(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %b2 = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  %d2 = tt.dot_scaled %a, %b2, %d lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d, %d2 : tensor<16x64xf32>, tensor<16x64xf32>
}
