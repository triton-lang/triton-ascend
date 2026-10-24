// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @signed_i32_arithmetic
// CHECK-NOT: scf.for
// CHECK: %[[MUL:[^ ]+]] = arith.muli {{.*}} : tensor<16x64xi32>
// CHECK-NOT: arith.extsi
// CHECK: %[[A:[^ ]+]] = call @[[CALLEE:[^(]+]]({{.*}}, %[[MUL]]) {isVolatile = false, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi32>) -> tensor<16x64xf8E4M3FN>
// CHECK-NOT: call @triton_indirect_load
// CHECK: hfusion.matmul_mx
// CHECK-SAME: ins(%[[A]],
// CHECK: return
// CHECK: func.func private @[[CALLEE]](memref<?xf8E4M3FN>, tensor<16x64xi32>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false, test.helper = "preserved"}
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false, test.helper = "preserved"}
func.func @signed_i32_arithmetic(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %product = arith.muli %indices, %stride : tensor<16x64xi32>
  %offsets = arith.extsi %product : tensor<16x64xi32> to tensor<16x64xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @unsigned_offsets
// CHECK: arith.extui
// CHECK: call @triton_indirect_load({{.*}}) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false, test.helper = "preserved"}
func.func @unsigned_offsets(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %offsets = arith.extui %indices : tensor<16x64xi32> to tensor<16x64xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @wide_sum
// CHECK: arith.extsi
// CHECK: call @triton_indirect_load({{.*}}) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false, test.helper = "preserved"}
func.func @wide_sum(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %wide = arith.extsi %indices : tensor<16x64xi32> to tensor<16x64xi64>
  %one = arith.constant dense<1> : tensor<16x64xi64>
  %offsets = arith.addi %wide, %one : tensor<16x64xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @volatile_snapshot_before_write
// CHECK: %[[A:[^ ]+]] = call @[[CALLEE:[^(]+]]({{.*}}) {isVolatile = true, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi32>) -> tensor<16x64xf8E4M3FN>
// CHECK: memref.store
// CHECK-NOT: call @triton_indirect_load
// CHECK: hfusion.matmul_mx
// CHECK-SAME: ins(%[[A]],
// CHECK: return
// CHECK: func.func private @[[CALLEE]](memref<?xf8E4M3FN>, tensor<16x64xi32>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = true, test.helper = "preserved"}
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = true, test.helper = "preserved"}
func.func @volatile_snapshot_before_write(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>, %zero: f8E4M3FN) -> tensor<16x32xf32> {
  %offsets = arith.extsi %indices : tensor<16x64xi32> to tensor<16x64xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = true, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %index = arith.constant 0 : index
  memref.store %zero, %base[%index] : memref<?xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @shared_read_once
// CHECK: %[[A:[^ ]+]] = call @triton_indirect_load{{.*}} : (memref<?xf8E4M3FN>, tensor<16x64xi32>) -> tensor<16x64xf8E4M3FN>
// CHECK-NOT: call @triton_indirect_load
// CHECK: hfusion.matmul_mx
// CHECK-SAME: ins(%[[A]],
// CHECK-NOT: call @triton_indirect_load
// CHECK: return {{.*}}, %[[A]] : tensor<16x32xf32>, tensor<16x64xf8E4M3FN>
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false, test.helper = "preserved"}
func.func @shared_read_once(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x64xf8E4M3FN>) {
  %offsets = arith.extsi %indices : tensor<16x64xi32> to tensor<16x64xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d, %a : tensor<16x32xf32>, tensor<16x64xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @shared_callee_unchanged
// CHECK: call @triton_indirect_load{{[_0-9]+}}({{.*}}) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x64xi32>) -> tensor<16x64xf8E4M3FN>
// CHECK: call @triton_indirect_load({{.*}}) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
// CHECK: hfusion.matmul_mx
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false, test.helper = "preserved"}
func.func @shared_callee_unchanged(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x64xf8E4M3FN>) {
  %offsets = arith.extsi %indices : tensor<16x64xi32> to tensor<16x64xi64>
  %a = func.call @triton_indirect_load(%base, %offsets) {isVolatile = false, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %unsigned = arith.extui %stride : tensor<16x64xi32> to tensor<16x64xi64>
  %other = func.call @triton_indirect_load(%base, %unsigned) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d, %other : tensor<16x32xf32>, tensor<16x64xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @mask_and_other
// CHECK-SAME: %[[MASK:[^ :]+]]: tensor<16x64xi1>, %[[OTHER:[^ :]+]]: tensor<16x64xf8E4M3FN>
// CHECK: call @triton_indirect_load{{.*}}({{.*}}, %[[MASK]], %[[OTHER]]) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x64xi32>, tensor<16x64xi1>, tensor<16x64xf8E4M3FN>) -> tensor<16x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK: return
func.func private @triton_indirect_load(memref<?xf8E4M3FN>, tensor<16x64xi64>, tensor<16x64xi1>, tensor<16x64xf8E4M3FN>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false}
func.func @mask_and_other(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>, %mask: tensor<16x64xi1>, %other: tensor<16x64xf8E4M3FN>) -> tensor<16x32xf32> {
  %offsets = arith.extsi %indices : tensor<16x64xi32> to tensor<16x64xi64>
  %a = func.call @triton_indirect_load(%base, %offsets, %mask, %other) {isVolatile = false} : (memref<?xf8E4M3FN>, tensor<16x64xi64>, tensor<16x64xi1>, tensor<16x64xf8E4M3FN>) -> tensor<16x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @other_external_name
// CHECK: arith.extsi
// CHECK: call @unrelated_read({{.*}}) {{.*}} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
// CHECK: hfusion.matmul_mx
// CHECK: return
func.func private @unrelated_read(memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN> attributes {isVolatile = false, test.helper = "preserved"}
func.func @other_external_name(%base: memref<?xf8E4M3FN>, %indices: tensor<16x64xi32>, %stride: tensor<16x64xi32>, %b: tensor<64x32xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %offsets = arith.extsi %indices : tensor<16x64xi32> to tensor<16x64xi64>
  %a = func.call @unrelated_read(%base, %offsets) {isVolatile = false, test.call = "preserved"} : (memref<?xf8E4M3FN>, tensor<16x64xi64>) -> tensor<16x64xf8E4M3FN>
  %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
