// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure --triton-to-linalg %s -verify-each | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_pointer_descriptor_loop(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<1> : tensor<4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %initial = tt.addptr %base_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %final = scf.for %iv = %c0 to %upper step %c1 iter_args(%ptr = %initial) -> (tensor<4x!tt.ptr<f32>>) {
      %next = tt.addptr %ptr, %delta : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next : tensor<4x!tt.ptr<f32>>
    }
    %value = tt.load %final : tensor<4x!tt.ptr<f32>>
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptr, %value : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @tensor_pointer_descriptor_loop
// CHECK:       scf.for
// CHECK-SAME:  tensor<4xi32>
// CHECK-NOT:   tensor<4x!tt.ptr
// CHECK-NOT:   unrealized_conversion_cast
