// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1048576 device-core-count=1 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
//
// A runtime head stride has no statically proved lower bound relative to the
// four-lane store interval below.  IAT must therefore retain the original
// program mapping instead of assuming that distinct program ids imply
// non-overlapping addresses.

// CHECK-NOT: hacc.independent_axis_tensorize
// CHECK-NOT: hacc.program_grid_transforms
// CHECK-LABEL: tt.func @_merge_split_states_kernel
// CHECK: arith.muli {{.*}} : i32
// CHECK: tt.store
module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 65 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @_merge_split_states_kernel(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %stride: i32) {
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
