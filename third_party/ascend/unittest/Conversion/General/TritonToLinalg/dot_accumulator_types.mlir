// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// FP64 accumulation must retain its precision without FP32 casts.
// CHECK-LABEL: func.func @dot_f64
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: linalg.matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<16x16xf64>, tensor<16x16xf64>)
// CHECK-SAME: outs(%{{.*}} : tensor<16x16xf64>) -> tensor<16x16xf64>
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: return %{{.*}} : tensor<16x16xf64>
func.func @dot_f64(%a: tensor<16x16xf64>, %b: tensor<16x16xf64>, %c: tensor<16x16xf64>) -> tensor<16x16xf64> {
  %d = tt.dot %a, %b, %c : tensor<16x16xf64> * tensor<16x16xf64> -> tensor<16x16xf64>
  return %d : tensor<16x16xf64>
}

// -----

// CHECK-LABEL: func.func @batch_dot_f64
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<2x16x16xf64>, tensor<2x16x16xf64>)
// CHECK-SAME: outs(%{{.*}} : tensor<2x16x16xf64>) -> tensor<2x16x16xf64>
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: return %{{.*}} : tensor<2x16x16xf64>
func.func @batch_dot_f64(%a: tensor<2x16x16xf64>, %b: tensor<2x16x16xf64>, %c: tensor<2x16x16xf64>) -> tensor<2x16x16xf64> {
  %d = tt.dot %a, %b, %c : tensor<2x16x16xf64> * tensor<2x16x16xf64> -> tensor<2x16x16xf64>
  return %d : tensor<2x16x16xf64>
}

// -----

// Narrow floating-point accumulators still use FP32 internally.
// CHECK-LABEL: func.func @dot_f16
// CHECK: %[[WIDE:.*]] = arith.extf %{{.*}} : tensor<16x16xf16> to tensor<16x16xf32>
// CHECK: %[[DOT:.*]] = linalg.matmul
// CHECK-SAME: outs(%[[WIDE]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %[[RESULT:.*]] = arith.truncf %[[DOT]] {{.*}} : tensor<16x16xf32> to tensor<16x16xf16>
// CHECK: return %[[RESULT]] : tensor<16x16xf16>
func.func @dot_f16(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %d = tt.dot %a, %b, %c : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf16>
  return %d : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @dot_bf16
// CHECK: %[[WIDE:.*]] = arith.extf %{{.*}} : tensor<16x16xbf16> to tensor<16x16xf32>
// CHECK: %[[DOT:.*]] = linalg.matmul
// CHECK-SAME: outs(%[[WIDE]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %[[RESULT:.*]] = arith.truncf %[[DOT]] {{.*}} : tensor<16x16xf32> to tensor<16x16xbf16>
// CHECK: return %[[RESULT]] : tensor<16x16xbf16>
func.func @dot_bf16(%a: tensor<16x16xbf16>, %b: tensor<16x16xbf16>, %c: tensor<16x16xbf16>) -> tensor<16x16xbf16> {
  %d = tt.dot %a, %b, %c : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xbf16>
  return %d : tensor<16x16xbf16>
}

// -----

// CHECK-LABEL: func.func @dot_f32
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: linalg.matmul
// CHECK-SAME: outs(%{{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: return %{{.*}} : tensor<16x16xf32>
func.func @dot_f32(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %d = tt.dot %a, %b, %c : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @dot_i32
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: linalg.matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<16x16xi8>, tensor<16x16xi8>)
// CHECK-SAME: outs(%{{.*}} : tensor<16x16xi32>) -> tensor<16x16xi32>
// CHECK: return %{{.*}} : tensor<16x16xi32>
func.func @dot_i32(%a: tensor<16x16xi8>, %b: tensor<16x16xi8>, %c: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %d = tt.dot %a, %b, %c : tensor<16x16xi8> * tensor<16x16xi8> -> tensor<16x16xi32>
  return %d : tensor<16x16xi32>
}
