// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s

// Without an FP16 input, decode FP8 to BF16 before expanding its E8M0 scale.
// CHECK-LABEL: func.func @fp8_bf16
// CHECK: %[[A:.*]] = arith.extf %arg0 : tensor<16x32xf8E4M3FN> to tensor<16x32xbf16>
// CHECK: arith.mulf %[[A]], {{.*}} : tensor<16x32xbf16>
// CHECK: linalg.matmul {{.*}}ins({{.*}} : tensor<16x32xbf16>, tensor<32x32xbf16>) outs({{.*}} : tensor<16x32xf32>)
func.func @fp8_bf16(%a: tensor<16x32xf8E4M3FN>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = e4m3 rhs = bf16 {fastMath = false} : tensor<16x32xf8E4M3FN>, tensor<16x1xi8> * tensor<32x32xbf16> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Convert an unscaled RHS too; both dot operands must use the compute type.
// CHECK-LABEL: func.func @bf16_fp8
// CHECK: arith.extf %arg1 : tensor<32x32xf8E5M2> to tensor<32x32xbf16>
// CHECK: arith.mulf {{.*}} : tensor<16x32xbf16>
// CHECK: linalg.matmul {{.*}}ins({{.*}} : tensor<16x32xbf16>, tensor<32x32xbf16>) outs({{.*}} : tensor<16x32xf32>)
func.func @bf16_fp8(%a: tensor<16x32xbf16>, %b: tensor<32x32xf8E5M2>, %sa: tensor<16x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = bf16 rhs = e5m2 {fastMath = false} : tensor<16x32xbf16>, tensor<16x1xi8> * tensor<32x32xf8E5M2> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// An FP16 LHS selects FP16 for both operands and their scaling operations.
// CHECK-LABEL: func.func @fp16_bf16
// CHECK: %[[WIDE:.*]] = arith.extf %arg1 : tensor<32x32xbf16> to tensor<32x32xf32>
// CHECK: %[[B:.*]] = arith.truncf %[[WIDE]] {round_mode = #hfusion.round_mode<rint>} : tensor<32x32xf32> to tensor<32x32xf16>
// CHECK: arith.mulf %[[B]], {{.*}} : tensor<32x32xf16>
// CHECK: arith.mulf %arg0, {{.*}} : tensor<16x32xf16>
// CHECK: linalg.matmul {{.*}}ins({{.*}} : tensor<16x32xf16>, tensor<32x32xf16>) outs({{.*}} : tensor<16x32xf32>)
func.func @fp16_bf16(%a: tensor<16x32xf16>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = bf16 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xbf16>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// The RHS format must also participate in compute-type selection.
// CHECK-LABEL: func.func @bf16_fp16
// CHECK: %[[WIDE:.*]] = arith.extf %arg0 : tensor<16x32xbf16> to tensor<16x32xf32>
// CHECK: %[[A:.*]] = arith.truncf %[[WIDE]] {round_mode = #hfusion.round_mode<rint>} : tensor<16x32xf32> to tensor<16x32xf16>
// CHECK: arith.mulf %[[A]], {{.*}} : tensor<16x32xf16>
// CHECK: linalg.matmul {{.*}}ins({{.*}} : tensor<16x32xf16>, tensor<32x32xf16>) outs({{.*}} : tensor<16x32xf32>)
func.func @bf16_fp16(%a: tensor<16x32xbf16>, %b: tensor<32x32xf16>, %sa: tensor<16x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = bf16 rhs = fp16 {fastMath = false} : tensor<16x32xbf16>, tensor<16x1xi8> * tensor<32x32xf16> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// FP8 is decoded to FP16 when paired with an FP16 operand.
// CHECK-LABEL: func.func @fp16_fp8
// CHECK: %[[B:.*]] = arith.extf %arg1 : tensor<32x32xf8E5M2> to tensor<32x32xf16>
// CHECK: arith.mulf %[[B]], {{.*}} : tensor<32x32xf16>
// CHECK: linalg.matmul {{.*}}ins({{.*}} : tensor<16x32xf16>, tensor<32x32xf16>) outs({{.*}} : tensor<16x32xf32>)
func.func @fp16_fp8(%a: tensor<16x32xf16>, %b: tensor<32x32xf8E5M2>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = e5m2 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xf8E5M2>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
