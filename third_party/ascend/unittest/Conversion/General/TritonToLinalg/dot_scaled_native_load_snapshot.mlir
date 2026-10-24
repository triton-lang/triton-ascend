// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// Preserve full-shape signed i32 address arithmetic, mask, and other.
// Removing only its redundant sign extension retains the whole native product.
// CHECK-LABEL: func.func @native_masked_bounded_snapshot
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloc
// CHECK: %[[PRODUCT:[^ ]+]] = arith.muli {{.*}} : tensor<16x512xi32>
// CHECK-NOT: arith.extsi
// CHECK: call @triton_indirect_load{{.*}}(%{{[^,]+}}, %[[PRODUCT]], {{.*}}) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x512xi32>, tensor<16x512xi1>, tensor<16x512xf8E4M3FN>) -> tensor<16x512xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK-NOT: tensor.insert_slice
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x512xf8E4M3FN>, tensor<512x32xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK: return
func.func private @triton_indirect_load_masked(memref<?xf8E4M3FN>, tensor<16x512xi64>, tensor<16x512xi1>, tensor<16x512xf8E4M3FN>) -> tensor<16x512xf8E4M3FN> attributes {isVolatile = false}
func.func @native_masked_bounded_snapshot(%base: memref<?xf8E4M3FN>, %indices: tensor<16x512xi32>, %stride: tensor<16x512xi32>, %mask: tensor<16x512xi1>, %other: tensor<16x512xf8E4M3FN>, %b: tensor<512x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %product = arith.muli %indices, %stride : tensor<16x512xi32>
  %offsets = arith.extsi %product : tensor<16x512xi32> to tensor<16x512xi64>
  %a = func.call @triton_indirect_load_masked(%base, %offsets, %mask, %other) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x512xi64>, tensor<16x512xi1>, tensor<16x512xf8E4M3FN>) -> tensor<16x512xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x512xf8E4M3FN> * tensor<512x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Shared address arithmetic remains one computation used by the whole read
// and its other consumer.
// CHECK-LABEL: func.func @native_shared_address
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloc
// CHECK: %[[ORIGINAL:[^ ]+]] = arith.muli {{.*}} : tensor<16x512xi32>
// CHECK-NOT: arith.extsi
// CHECK: call @triton_indirect_load{{.*}}({{.*}}, %[[ORIGINAL]]) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x512xi32>) -> tensor<16x512xf8E4M3FN>
// CHECK-NOT: arith.muli
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x512xf8E4M3FN>, tensor<512x32xf8E4M3FN>
// CHECK-SAME: outs({{.*}} : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK: return {{.*}}, %[[ORIGINAL]] : tensor<16x32xf32>, tensor<16x512xi32>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x512xi64>) -> tensor<16x512xf8E4M3FN> attributes {isVolatile = false}
func.func @native_shared_address(%base: memref<?xf8E4M3FN>, %indices: tensor<16x512xi32>, %stride: tensor<16x512xi32>, %b: tensor<512x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x512xi32>) {
  %product = arith.muli %indices, %stride : tensor<16x512xi32>
  %offsets = arith.extsi %product : tensor<16x512xi32> to tensor<16x512xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x512xi64>) -> tensor<16x512xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x512xf8E4M3FN> * tensor<512x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d, %product : tensor<16x32xf32>, tensor<16x512xi32>
}
