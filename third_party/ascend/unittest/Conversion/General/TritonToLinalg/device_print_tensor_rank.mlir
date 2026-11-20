// RUN: triton-opt --triton-to-linalg="named-ops=true" --split-input-file %s | FileCheck %s

// Only the batch dimensions are collapsed, independently for each argument.
// Scalar, vector, matrix, and rank-three arguments keep their types.
// CHECK: func.func private @triton_print(i32, tensor<8xi32>, tensor<16x32xf16>, tensor<2x16x32xf16>, tensor<6x16x32xbf16>, tensor<24x16x32xi8>, tensor<120x16x32xf32>) attributes {hex = true, prefix = "mixed"}
// CHECK-LABEL: func.func @mixed_ranks
// CHECK: %[[R4:.*]] = tensor.reshape %arg4({{.*}}) : (tensor<2x3x16x32xbf16>, tensor<3xi64>) -> tensor<6x16x32xbf16>
// CHECK: %[[R5:.*]] = tensor.reshape %arg5({{.*}}) : (tensor<2x3x4x16x32xi8>, tensor<3xi64>) -> tensor<24x16x32xi8>
// CHECK: %[[R6:.*]] = tensor.reshape %arg6({{.*}}) : (tensor<2x3x4x5x16x32xf32>, tensor<3xi64>) -> tensor<120x16x32xf32>
// CHECK: call @triton_print(%arg0, %arg1, %arg2, %arg3, %[[R4]], %[[R5]], %[[R6]]) : (i32, tensor<8xi32>, tensor<16x32xf16>, tensor<2x16x32xf16>, tensor<6x16x32xbf16>, tensor<24x16x32xi8>, tensor<120x16x32xf32>) -> ()
module {
  func.func @mixed_ranks(%scalar: i32, %vector: tensor<8xi32>, %matrix: tensor<16x32xf16>, %batch: tensor<2x16x32xf16>, %r4: tensor<2x3x16x32xbf16>, %r5: tensor<2x3x4x16x32xi8>, %r6: tensor<2x3x4x5x16x32xf32>) {
    tt.print "mixed" {hex = true, isSigned = array<i32: 1, 1, 1, 1, 1, 1, 1>} : %scalar, %vector, %matrix, %batch, %r4, %r5, %r6 : i32, tensor<8xi32>, tensor<16x32xf16>, tensor<2x16x32xf16>, tensor<2x3x16x32xbf16>, tensor<2x3x4x16x32xi8>, tensor<2x3x4x5x16x32xf32>
    return
  }
}

// -----

// Reproduce tl.dot's rank-four reshape/transpose chain and print all three
// values after the dot. The function signatures and call operands must agree.
// CHECK: func.func private @triton_print(tensor<4x32x32xbf16>) attributes {hex = false, prefix = "x"}
// CHECK: func.func private @triton_print_0(tensor<4x32x32xbf16>) attributes {hex = false, prefix = "y"}
// CHECK: func.func private @triton_print_1(tensor<4x32x32xf32>) attributes {hex = false, prefix = "z"}
// CHECK-LABEL: func.func @dot_rank4
// CHECK: linalg.transpose
// CHECK: linalg.batch_matmul {{.*}} ins(%[[A:.*]], %[[B:.*]] : tensor<4x32x32xbf16>, tensor<4x32x32xbf16>)
// CHECK: call @triton_print(%[[A]]) : (tensor<4x32x32xbf16>) -> ()
// CHECK: call @triton_print_0(%[[B]]) : (tensor<4x32x32xbf16>) -> ()
// CHECK: call @triton_print_1({{.*}}) : (tensor<4x32x32xf32>) -> ()
module {
  func.func @dot_rank4(%x: tensor<2x2x32x32xbf16>, %y: tensor<2x2x32x32xbf16>) {
    %a = tt.trans %x {order = array<i32: 0, 1, 3, 2>} : tensor<2x2x32x32xbf16> -> tensor<2x2x32x32xbf16>
    %a3 = tt.reshape %a : tensor<2x2x32x32xbf16> -> tensor<4x32x32xbf16>
    %b3 = tt.reshape %y : tensor<2x2x32x32xbf16> -> tensor<4x32x32xbf16>
    %zero = arith.constant dense<0.0> : tensor<4x32x32xf32>
    %dot = tt.dot %a3, %b3, %zero, inputPrecision = ieee : tensor<4x32x32xbf16> * tensor<4x32x32xbf16> -> tensor<4x32x32xf32>
    %z = tt.reshape %dot : tensor<4x32x32xf32> -> tensor<2x2x32x32xf32>
    tt.print "x" {hex = false, isSigned = array<i32: 1>} : %a : tensor<2x2x32x32xbf16>
    tt.print "y" {hex = false, isSigned = array<i32: 1>} : %y : tensor<2x2x32x32xbf16>
    tt.print "z" {hex = false, isSigned = array<i32: 1>} : %z : tensor<2x2x32x32xf32>
    return
  }
}

// -----

// Printing before the dot must use the same batch view as the dot as well.
// CHECK: func.func private @triton_print(tensor<4x32x32xbf16>) attributes {hex = false, prefix = "before"}
// CHECK-LABEL: func.func @print_before_dot
// CHECK: call @triton_print(%[[B:.*]]) : (tensor<4x32x32xbf16>) -> ()
// CHECK: linalg.batch_matmul {{.*}} ins({{.*}}, %[[B]] : tensor<4x32x32xbf16>, tensor<4x32x32xbf16>)
module {
  func.func @print_before_dot(%x: tensor<2x2x32x32xbf16>, %y: tensor<2x2x32x32xbf16>) -> tensor<4x32x32xf32> {
    tt.print "before" {hex = false, isSigned = array<i32: 1>} : %y : tensor<2x2x32x32xbf16>
    %a3 = tt.reshape %x : tensor<2x2x32x32xbf16> -> tensor<4x32x32xbf16>
    %b3 = tt.reshape %y : tensor<2x2x32x32xbf16> -> tensor<4x32x32xbf16>
    %zero = arith.constant dense<0.0> : tensor<4x32x32xf32>
    %dot = tt.dot %a3, %b3, %zero, inputPrecision = ieee : tensor<4x32x32xbf16> * tensor<4x32x32xbf16> -> tensor<4x32x32xf32>
    return %dot : tensor<4x32x32xf32>
  }
}

// -----

// A user function with the same name prefix is not a generated print stub.
// CHECK-LABEL: func.func private @triton_print_user(tensor<2x3x16x32xf32>)
// CHECK-LABEL: func.func @user_call
// CHECK-NOT: tensor.reshape
// CHECK: call @triton_print_user(%arg0) : (tensor<2x3x16x32xf32>) -> ()
module {
  func.func private @triton_print_user(tensor<2x3x16x32xf32>)
  func.func @user_call(%x: tensor<2x3x16x32xf32>) {
    call @triton_print_user(%x) : (tensor<2x3x16x32xf32>) -> ()
    return
  }
}
