// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1048576 device-core-count=1 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s

// CHECK: hacc.independent_axis_tensorize
// CHECK-NOT: hacc.program_mapping_scalar_specialization
// CHECK: hacc.program_grid_transforms
// CHECK: factor = 8 : i64
// CHECK: logical_extent = 65 : i64
// CHECK-LABEL: tt.func @structural_merge_split_entry
// CHECK: arith.constant 512 : i32
// CHECK: arith.cmpi slt
// CHECK: tt.load {{.*}} : tensor<2x8x4x!tt.ptr<f32>>
// CHECK: tt.reduce
// CHECK-SAME: axis = 0 : i32
// CHECK: tensor<8x4xf32>
//
// The source ordinal intentionally does not name a retained TTIR argument:
// v2 must find the dynamic output-token stride by NameLoc, replace it with
// the cache-keyed value, then consume the bridge attribute before IAT runs.
module attributes {
  hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 65 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64},
  hacc.program_mapping_scalar_specialization = {arguments = [{index = 99 : i64, name = "stride_out_token", value = 512 : i64}], version = 2 : i64}
} {
  tt.func @structural_merge_split_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %stride_out_token: i32 loc("stride_out_token")) attributes {
    hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 65 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64},
    hacc.program_mapping_scalar_specialization = {arguments = [{index = 99 : i64, name = "stride_out_token", value = 512 : i64}], version = 2 : i64}
  } {
    %c4 = arith.constant 4 : i32
    %token = tt.get_program_id x : i32
    %head = tt.get_program_id y : i32
    %splits = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %token_offset = arith.muli %token, %stride_out_token : i32
    %head_offset = arith.muli %head, %c4 : i32
    %base = arith.addi %token_offset, %head_offset : i32
    %input_base = tt.addptr %input, %base : !tt.ptr<f32>, i32
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
    %output_base = tt.addptr %output, %base : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
