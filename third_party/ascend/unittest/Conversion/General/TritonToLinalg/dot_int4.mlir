// RUN: triton-opt --triton-to-linalg="named-ops=True" --verify-each --split-input-file %s | FileCheck %s --implicit-check-not=tt.dot --implicit-check-not=enable_i4

// CHECK-LABEL: func.func @dot_s4_acc
// CHECK-SAME: %[[A:[^ ,)]+]]: tensor<16x32xi8>, %[[B:[^ ,)]+]]: tensor<64x16xi8>, %[[ACC:[^ ,)]+]]: tensor<16x32xi32>
// CHECK: %[[A4:.*]] = builtin.unrealized_conversion_cast %[[A]] : tensor<16x32xi8> to tensor<16x64xi4>
// CHECK: %[[B4:.*]] = builtin.unrealized_conversion_cast %[[B]] : tensor<64x16xi8> to tensor<64x32xi4>
// CHECK: %[[DOT:.*]] = linalg.matmul
// CHECK-SAME: input_precision = "ieee"
// CHECK-SAME: ins(%[[A4]], %[[B4]] : tensor<16x64xi4>, tensor<64x32xi4>)
// CHECK-SAME: outs(%[[ACC]] : tensor<16x32xi32>) -> tensor<16x32xi32>
// CHECK: return %[[DOT]] : tensor<16x32xi32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @dot_s4_acc(%a: tensor<16x32xi8>, %b: tensor<64x16xi8>, %acc: tensor<16x32xi32>) -> tensor<16x32xi32> {
    %a4 = builtin.unrealized_conversion_cast %a : tensor<16x32xi8> to tensor<16x64xi4>
    %b4 = builtin.unrealized_conversion_cast %b : tensor<64x16xi8> to tensor<64x32xi4>
    %result = tt.dot %a4, %b4, %acc : tensor<16x64xi4> * tensor<64x32xi4> -> tensor<16x32xi32>
    return %result : tensor<16x32xi32>
  }
}
