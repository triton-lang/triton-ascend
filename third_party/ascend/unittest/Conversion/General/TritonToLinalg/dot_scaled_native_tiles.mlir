// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// Plain native inputs retain one full MX, including all 256 logical K elements.
// CHECK-LABEL: func.func @large_native_fp4
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<128x128xi8>, tensor<128x128xi8>, tensor<128x8xi8>, tensor<128x8xi8>) outs({{.*}} : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NOT: scf.for
// CHECK: return
func.func @large_native_fp4(%a: tensor<128x128xi8>, %b: tensor<128x128xi8>, %sa: tensor<128x8xi8>, %sb: tensor<128x8xi8>, %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8>, tensor<128x8xi8> * tensor<128x128xi8>, tensor<128x8xi8> -> tensor<128x128xf32>
  return %d : tensor<128x128xf32>
}

// -----

// Optional scales cover the full native FP8 operands without extra M/N loops.
// CHECK-LABEL: func.func @large_native_fp8
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<128x256xf8E5M2>, tensor<256x32xf8E4M3FN>, tensor<128x8xi8>, tensor<32x8xi8>) outs({{.*}} : tensor<128x32xf32>) -> tensor<128x32xf32>
// CHECK-NOT: scf.for
// CHECK: return
func.func @large_native_fp8(%a: tensor<128x256xf8E5M2>, %b: tensor<256x32xf8E4M3FN>, %c: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = e5m2 rhs = e4m3 {fastMath = false} : tensor<128x256xf8E5M2> * tensor<256x32xf8E4M3FN> -> tensor<128x32xf32>
  return %d : tensor<128x32xf32>
}
