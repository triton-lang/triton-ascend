// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// Keep unknown i64 indirect offsets and their complete original snapshots.
// Native MX and its accumulator remain whole, including shared/volatile reads.

// -----

// CHECK-LABEL: func.func @native_indirect_lhs
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_lhs(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_volatile
// CHECK: call @triton_indirect_load{{.*}} -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = true}
func.func @native_indirect_volatile(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = true} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_shared
// CHECK: call @triton_indirect_load{{.*}} -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_shared(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x128xf8E4M3FN>) {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d, %a : tensor<64x64xf32>, tensor<64x128xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @native_indirect_alias_write
// CHECK: call @triton_indirect_load{{.*}} -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_alias_write(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  memref.store %v, %base[%zero] : memref<?xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_unknown_write
// CHECK: call @triton_indirect_load{{.*}} -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func private @write(memref<?xf8E4M3FN>)
func.func @native_indirect_unknown_write(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  func.call @write(%base) : (memref<?xf8E4M3FN>) -> ()
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_opaque_slice
// CHECK: call @triton_indirect_load{{.*}} -> tensor<128x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<128x128xi64>) -> tensor<128x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_opaque_slice(%base: memref<?xf8E4M3FN>, %offsets: tensor<128x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %full = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<128x128xi64>) -> tensor<128x128xf8E4M3FN>
  %a = tensor.extract_slice %full[0, 0] [64, 128] [1, 1] : tensor<128x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_scale_only_indirect
// CHECK: call @triton_indirect_load{{.*}} -> tensor<64x4xi8>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xi8>, tensor<64x4xi64>) -> tensor<64x4xi8> attributes {isVolatile = false}
func.func @native_scale_only_indirect(%base: memref<?xi8>, %offsets: tensor<64x4xi64>, %a: tensor<64x128xf8E4M3FN>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %sa = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xi8>, tensor<64x4xi64>) -> tensor<64x4xi8>
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN>, tensor<64x4xi8> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_broadcast_does_not_shrink
// CHECK: call @triton_indirect_load{{.*}} -> tensor<1x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<1x128xi64>) -> tensor<1x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_broadcast_does_not_shrink(%base: memref<?xf8E4M3FN>, %offsets: tensor<1x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %small = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<1x128xi64>) -> tensor<1x128xf8E4M3FN>
  %a = tt.broadcast %small : tensor<1x128xf8E4M3FN> -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// tt.broadcast lowers before the dot to collapse_shape/linalg.broadcast.
// That opaque snapshot path is not an existing slice() optimization.
// CHECK-LABEL: func.func @native_lowered_broadcast_snapshot
// CHECK: call @triton_indirect_load{{.*}} -> tensor<64x1xf8E4M3FN>
// CHECK: linalg.broadcast
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-NOT: scf.for
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x1xi64>) -> tensor<64x1xf8E4M3FN> attributes {isVolatile = false}
func.func @native_lowered_broadcast_snapshot(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x1xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %small = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x1xi64>) -> tensor<64x1xf8E4M3FN>
  %a = tt.broadcast %small : tensor<64x1xf8E4M3FN> -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_rhs
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<128x64xi64>) -> tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<128x64xi64>) -> tensor<128x64xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_rhs(%base: memref<?xf8E4M3FN>, %offsets: tensor<128x64xi64>, %a: tensor<16x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %b = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<128x64xi64>) -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_triggered_bulk_plain
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_plain(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>, %row_stride: index) -> tensor<64x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_triggered_bulk_volatile
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_volatile(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [256, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = true} : memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_triggered_bulk_unmarked
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_unmarked(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [256, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf : memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_triggered_bulk_shared
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_shared(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x128xf8E4M3FN>) {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [256, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d, %loaded : tensor<64x64xf32>, tensor<64x128xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @native_triggered_bulk_alias_write
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_alias_write(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [256, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  memref.store %v, %bbase[%zero] : memref<?xf8E4M3FN>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_triggered_bulk_unknown_write
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func private @write(memref<?xf8E4M3FN>)
func.func @native_triggered_bulk_unknown_write(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [256, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  func.call @write(%bbase) : (memref<?xf8E4M3FN>) -> ()
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_transpose
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_transpose(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %a: tensor<16x128xf8E4M3FN>, %extra: tensor<64x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %loaded = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_indirect_transpose_mixed_snapshot
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_transpose_mixed_snapshot(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %a: tensor<16x128xf8E4M3FN>, %extra: tensor<64x128xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %loaded = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %mix = arith.addf %loaded, %extra : tensor<64x128xf8E4M3FN>
  %b = tt.trans %mix {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @native_triggered_bulk_nested_write
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_nested_write(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>, %condition: i1) -> tensor<64x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [256, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  scf.if %condition {
    memref.store %v, %bbase[%zero] : memref<?xf8E4M3FN>
  }
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}


// -----

// CHECK-LABEL: func.func @native_triggered_bulk_escaping_buffer
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_escaping_buffer(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>) -> (tensor<64x64xf32>, memref<64x128xf8E4M3FN>) {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [256, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[256, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d, %buf : tensor<64x64xf32>, memref<64x128xf8E4M3FN>
}


// -----
// Reducing N does not reduce this already-small A address tensor.

// CHECK-LABEL: func.func @native_indirect_does_not_shrink
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<16x128xi64>) -> tensor<16x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x128xi64>) -> tensor<16x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_indirect_does_not_shrink(%base: memref<?xf8E4M3FN>, %offsets: tensor<16x128xi64>, %b: tensor<128x64xf8E4M3FN>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x128xi64>) -> tensor<16x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}


// -----

// CHECK-LABEL: func.func @native_triggered_bulk_lowered_transpose
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_lowered_transpose(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>, %row_stride: index) -> tensor<64x64xf32> {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d : tensor<64x64xf32>
}


// -----

// CHECK-LABEL: func.func @native_triggered_bulk_two_dot_users
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_two_dot_users(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>, %row_stride: index) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  %a2 = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d2 = tt.dot_scaled %a2, %b, %d lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d, %d2 : tensor<64x64xf32>, tensor<64x64xf32>
}


// -----

// CHECK-LABEL: func.func @native_triggered_bulk_extra_transpose_user
// CHECK: memref.copy {{.*}} to memref<64x128xf8E4M3FN>
// CHECK: linalg.transpose {{.*}} tensor<64x128xf8E4M3FN>{{.*}}tensor<128x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.for
// CHECK: call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN> attributes {isVolatile = false}
func.func @native_triggered_bulk_extra_transpose_user(%base: memref<?xf8E4M3FN>, %offsets: tensor<64x128xi64>, %bbase: memref<?xf8E4M3FN>, %c: tensor<64x64xf32>, %row_stride: index) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %zero = arith.constant 0 : index
  %v = arith.constant 0.0 : f8E4M3FN
  %src = memref.reinterpret_cast %bbase to offset: [5], sizes: [64, 128], strides: [%row_stride, 2] : memref<?xf8E4M3FN> to memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>>
  %buf = memref.alloc() : memref<64x128xf8E4M3FN>
  memref.copy %src, %buf {isVolatile = false} : memref<64x128xf8E4M3FN, strided<[?, 2], offset: 5>> to memref<64x128xf8E4M3FN>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<64x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
  %init = tensor.empty() : tensor<128x64xf8E4M3FN>
  %b = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
  %b2 = linalg.transpose ins(%loaded : tensor<64x128xf8E4M3FN>) outs(%init : tensor<128x64xf8E4M3FN>) permutation = [1, 0]
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  %a2 = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<64x128xi64>) -> tensor<64x128xf8E4M3FN>
  %d2 = tt.dot_scaled %a2, %b2, %d lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN> * tensor<128x64xf8E4M3FN> -> tensor<64x64xf32>
  return %d, %d2 : tensor<64x64xf32>, tensor<64x64xf32>
}
