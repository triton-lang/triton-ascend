// RUN: not triton-opt --triton-control-flow-opt %s 2>&1 | FileCheck %s

module {
  tt.func public @if_block_ptr_different_base(%base0: !tt.ptr<f32>, %base1: !tt.ptr<f32>, %cond: i1) -> !tt.ptr<tensor<16xf32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %ptr0 = tt.make_tensor_ptr %base0, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %ptr1 = tt.make_tensor_ptr %base1, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %selected = scf.if %cond -> (!tt.ptr<tensor<16xf32>>) {
      %then_ptr = tt.advance %ptr0, [%c1_i32] : !tt.ptr<tensor<16xf32>>
      scf.yield %then_ptr : !tt.ptr<tensor<16xf32>>
    } else {
      %else_ptr = tt.advance %ptr1, [%c2_i32] : !tt.ptr<tensor<16xf32>>
      scf.yield %else_ptr : !tt.ptr<tensor<16xf32>>
    }
    tt.return %selected : !tt.ptr<tensor<16xf32>>
  }
}

// CHECK: error: failed to analyze pointer components across control flow
