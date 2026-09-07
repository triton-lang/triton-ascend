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

  // This is the one tiled rank-2 carrier optimized by the narrow T2U path.
  // The ordinary offset slot is initialized from complementary broadcasted
  // axes and advances by 64 alongside two CFO-owned pointer descriptors.
  tt.func public @marked_rank_two_tiled_offset_carrier(
      %input: !tt.ptr<f32>, %dense: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %row_stride: i32) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %row_range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %rows = tt.expand_dims %row_range {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %stride = tt.splat %row_stride : i32 -> tensor<32x1xi32>
    %scaled_rows = arith.muli %rows, %stride : tensor<32x1xi32>
    %row_offsets = tt.broadcast %scaled_rows : tensor<32x1xi32> -> tensor<32x64xi32>
    %column_range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %columns = tt.expand_dims %column_range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %column_offsets = tt.broadcast %columns : tensor<1x64xi32> -> tensor<32x64xi32>
    %initial_offsets = arith.addi %row_offsets, %column_offsets : tensor<32x64xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %dense_base = tt.splat %dense : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %initial_dense = tt.addptr %dense_base, %initial_offsets : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %initial_output = tt.addptr %output_base, %initial_offsets : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %column_advance = arith.constant dense<64> : tensor<64xi32>
    %offset_advance = arith.constant dense<64> : tensor<32x64xi32>
    %result:4 = scf.for %iv = %c0 to %c256 step %c64
        iter_args(%loop_columns = %column_range, %offsets = %initial_offsets,
                  %dense_ptrs = %initial_dense, %output_ptrs = %initial_output)
        -> (tensor<64xi32>, tensor<32x64xi32>, tensor<32x64x!tt.ptr<f32>>, tensor<32x64x!tt.ptr<f32>>) {
      %input_ptrs = tt.addptr %input_base, %offsets : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
      %input_value = tt.load %input_ptrs : tensor<32x64x!tt.ptr<f32>>
      %dense_value = tt.load %dense_ptrs : tensor<32x64x!tt.ptr<f32>>
      %value = arith.addf %input_value, %dense_value : tensor<32x64xf32>
      tt.store %output_ptrs, %value : tensor<32x64x!tt.ptr<f32>>
      %next_columns = arith.addi %loop_columns, %column_advance : tensor<64xi32>
      %next_offsets = arith.addi %offsets, %offset_advance : tensor<32x64xi32>
      %next_dense = tt.addptr %dense_ptrs, %offset_advance : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
      %next_output = tt.addptr %output_ptrs, %offset_advance : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
      scf.yield %next_columns, %next_offsets, %next_dense, %next_output : tensor<64xi32>, tensor<32x64xi32>, tensor<32x64x!tt.ptr<f32>>, tensor<32x64x!tt.ptr<f32>>
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

// CFO-LABEL: tt.func public @marked_rank_two_tiled_offset_carrier
// CFO:       scf.for
// CFO:       PointerDescriptorBoundary

// LINALG-LABEL: func.func @marked_rank_two_tiled_offset_carrier
// LINALG:       scf.for
// LINALG-NOT:   triton_indirect_load
// LINALG:       return
