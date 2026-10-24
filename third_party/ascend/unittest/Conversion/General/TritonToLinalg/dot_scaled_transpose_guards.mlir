// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// Plain snapshot paths remain whole. The indirect-gate test separately exercises
// each transpose guard while an eligible indirect operand triggers native tiling.

// -----
// CHECK-LABEL: func.func @bulk_transpose_volatile
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = true}
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>) outs({{.*}} : tensor<128x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return
func.func @bulk_transpose_volatile(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = true} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----
// CHECK-LABEL: func.func @bulk_transpose_unmarked
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x128xf8E4M3FN>) outs({{.*}} : tensor<128x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<16x4xi8>, tensor<64x4xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return
func.func @bulk_transpose_unmarked(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----
// CHECK-LABEL: func.func @bulk_transpose_escape
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
func.func private @escape(memref<64x128xf8E4M3FN>)
func.func @bulk_transpose_escape(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  func.call @escape(%buf) : (memref<64x128xf8E4M3FN>) -> ()
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----
// CHECK-LABEL: func.func @bulk_transpose_external_write
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
func.func private @external_write(memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>)
func.func @bulk_transpose_external_write(%base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  func.call @external_write(%src) : (memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>) -> ()
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----
// CHECK-LABEL: func.func @bulk_transpose_nested_write
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
func.func @bulk_transpose_nested_write(%condition: i1, %base: memref<?xf8E4M3FN>, %row_stride: index, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %base_offset = arith.constant 5 : index
  %src = memref.reinterpret_cast %base to offset: [%base_offset], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>> to memref<64x128xf8E4M3FN>
  scf.if %condition {
    memref.store %v, %src[%zero, %zero] : memref<64x128xf8E4M3FN, strided<[?, 2], offset: ?>>
  }
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
