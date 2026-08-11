// RUN: not triton-opt --triton-control-flow-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @if_tensor_ptr_different_base(%base0: !tt.ptr<f32>, %base1: !tt.ptr<f32>, %cond: i1) -> tensor<4x!tt.ptr<f32>> {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %off1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %off2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %splat0 = tt.splat %base0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %splat1 = tt.splat %base1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %selected = scf.if %cond -> (tensor<4x!tt.ptr<f32>>) {
      %then_ptr = tt.addptr %splat0, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %then_ptr : tensor<4x!tt.ptr<f32>>
    } else {
      %else_ptr = tt.addptr %splat1, %off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %else_ptr : tensor<4x!tt.ptr<f32>>
    }
    tt.return %selected : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK: error: failed to analyze pointer components across control flow
