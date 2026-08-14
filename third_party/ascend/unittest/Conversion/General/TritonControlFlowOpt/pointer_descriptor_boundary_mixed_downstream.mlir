// RUN: triton-opt --triton-control-flow-opt %s | FileCheck %s --check-prefix=CFO
// RUN: triton-opt --triton-control-flow-opt --triton-to-linalg %s -verify-each | FileCheck %s --check-prefix=LINALG --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast --implicit-check-not=PointerDescriptorBoundary

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @expanded_block_ptr_with_invariant_tensor_ptr(
      %block_base: !tt.ptr<f32>, %tensor_base: !tt.ptr<f32>,
      %output: !tt.ptr<f32>, %upper: index) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %block = tt.make_tensor_ptr %block_base, [%c4_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<4xf32>>
    %tensor_base_splat = tt.splat %tensor_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %tensor = tt.addptr %tensor_base_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %results:2 = scf.for %iv = %c0 to %upper step %c1 iter_args(%block_arg = %block, %tensor_arg = %tensor) -> (!tt.ptr<tensor<4xf32>>, tensor<4x!tt.ptr<f32>>) {
      %next_block = tt.advance %block_arg, [%c1_i32] : !tt.ptr<tensor<4xf32>>
      scf.yield %next_block, %tensor_arg : !tt.ptr<tensor<4xf32>>, tensor<4x!tt.ptr<f32>>
    }
    %value = tt.load %results#1 : tensor<4x!tt.ptr<f32>>
    %output_splat = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %value : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CFO-LABEL: tt.func public @expanded_block_ptr_with_invariant_tensor_ptr
// CFO:       %[[LOOP:.*]]:5 = scf.for
// CFO-SAME:  -> (i64, i64, i64, i32, tensor<4x!tt.ptr<f32>>) {
// CFO:       } {PointerDescriptorBoundary = array<i32: 0, 1, 2, 3>}
// CFO:       tt.load %[[LOOP]]#4 : tensor<4x!tt.ptr<f32>>

// LINALG-LABEL: func.func @expanded_block_ptr_with_invariant_tensor_ptr
// LINALG:       scf.for
// LINALG:       return
