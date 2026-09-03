// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s --check-prefix=CFO
// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure --triton-to-linalg %s -verify-each | FileCheck %s --check-prefix=LINALG

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

  // A CFO descriptor loop may also carry an integer address tensor that is
  // not one of the descriptor slots. Preserve the affine row/column layout
  // through the legacy BlockData rewrite instead of lowering the load as an
  // indirect gather.
  tt.func public @marked_affine_index_carrier(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %row_stride: i32,
      %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %rows = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %stride = tt.splat %row_stride : i32 -> tensor<4x1xi32>
    %scaled_rows = arith.muli %rows, %stride : tensor<4x1xi32>
    %row_offsets = tt.broadcast %scaled_rows : tensor<4x1xi32> -> tensor<4x4xi32>
    %columns = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %column_offsets = tt.broadcast %columns : tensor<1x4xi32> -> tensor<4x4xi32>
    %initial_offsets = arith.addi %row_offsets, %column_offsets : tensor<4x4xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %initial_output = tt.addptr %output_base, %initial_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %advance = arith.constant dense<16> : tensor<4x4xi32>
    %result:2 = scf.for %iv = %c0 to %upper step %c1
        iter_args(%offsets = %initial_offsets, %output_ptrs = %initial_output)
        -> (tensor<4x4xi32>, tensor<4x4x!tt.ptr<f32>>) {
      %input_ptrs = tt.addptr %input_base, %offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %value = tt.load %input_ptrs : tensor<4x4x!tt.ptr<f32>>
      tt.store %output_ptrs, %value : tensor<4x4x!tt.ptr<f32>>
      %next_offsets = arith.addi %offsets, %advance : tensor<4x4xi32>
      %next_output = tt.addptr %output_ptrs, %advance : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      scf.yield %next_offsets, %next_output : tensor<4x4xi32>, tensor<4x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CFO-LABEL: tt.func public @tensor_pointer_descriptor_loop
// CFO:       scf.for
// CFO-SAME:  i32
// CFO:       tt.addptr {{.*}}PointerDescriptorOffsetForm = "strided_1d"
// CFO-SAME:  PointerDescriptorStructuredAxes = array<i32: 1>

// LINALG-LABEL: func.func @tensor_pointer_descriptor_loop
// LINALG:       scf.for
// LINALG-SAME:  i32
// LINALG-NOT:   tensor<4x!tt.ptr
// LINALG-NOT:   unrealized_conversion_cast

// CFO-LABEL: tt.func public @marked_affine_index_carrier
// CFO:       scf.for
// CFO:       PointerDescriptorBoundary

// LINALG-LABEL: func.func @marked_affine_index_carrier
// LINALG:       scf.for
// LINALG-NOT:   triton_indirect_load
// LINALG:       return
