// RUN: not triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file --mlir-disable-threading %s 2>&1 | FileCheck %s

// Reject invalid scale shapes before the lowering can consume them.
// CHECK: scales K dimension must match the operand K divided by the scale factor
func.func @bad_lhs_groups(%a: tensor<16x32xf16>, %b: tensor<32x32xf16>, %sa: tensor<16x3xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x32xf16>, tensor<16x3xi8> * tensor<32x32xf16>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK: scales K dimension must match the operand K divided by the scale factor
func.func @bad_rhs_groups(%a: tensor<16x32xf16>, %b: tensor<32x32xf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x3xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xf16>, tensor<32x3xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK: scales N dimension must match the operand N dimension
func.func @bad_rhs_rank(%a: tensor<16x32xf16>, %b: tensor<32x32xf16>, %sa: tensor<16x1xi8>, %sb: tensor<1x32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xf16>, tensor<1x32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
