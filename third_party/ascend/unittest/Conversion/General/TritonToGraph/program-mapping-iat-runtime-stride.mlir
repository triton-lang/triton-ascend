// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1048576 device-core-count=1 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
//
// The same dynamic head-stride form used by the safety no-op test becomes an
// IAT candidate only when the exact value has been carried through the
// versioned, cache-keyed scalar-specialization contract.  The graph pass must
// consume that internal attr before output; a generic unknown stride remains
// covered by program-mapping-iat-unknown-stride.mlir.

// CHECK: hacc.independent_axis_tensorize
// CHECK: hacc.program_grid_transforms = {{.*}}axis = 1 : i32{{.*}}persistent_coverage = false
// CHECK-NOT: hacc.program_mapping_scalar_specialization
// CHECK-LABEL: tt.func @_merge_split_states_kernel
// CHECK: arith.constant dense<4>
// CHECK-NOT: %arg2
// CHECK: tt.store
module attributes {
  hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 65 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64},
  hacc.program_mapping_scalar_specialization = {arguments = [{index = 99 : i64, name = "stride", value = 4 : i64}], version = 2 : i64}
} {
  tt.func @_merge_split_states_kernel(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %stride: i32 loc("stride")) attributes {
    hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 65 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64},
    hacc.program_mapping_scalar_specialization = {arguments = [{index = 99 : i64, name = "stride", value = 4 : i64}], version = 2 : i64}
  } {
    %head = tt.get_program_id y : i32
    %splits = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %head_offset = arith.muli %head, %stride : i32
    %input_base = tt.addptr %input, %head_offset : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %input_broadcast = tt.broadcast %input_splat : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %dims_expand = tt.expand_dims %dims {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %dims_broadcast = tt.broadcast %dims_expand : tensor<1x4xi32> -> tensor<2x4xi32>
    %input_ptrs = tt.addptr %input_broadcast, %dims_broadcast : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %states = tt.load %input_ptrs : tensor<2x4x!tt.ptr<f32>>
    %sum = "tt.reduce"(%states) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<2x4xf32>) -> tensor<4xf32>
    %output_base = tt.addptr %output, %head_offset : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
