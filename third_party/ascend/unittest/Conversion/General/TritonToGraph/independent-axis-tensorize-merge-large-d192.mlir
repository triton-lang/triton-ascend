// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=131072 mapping-ub-capacity-bytes=262144 device-core-count=56 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
//
// Phase B is only legal for the structural D256 padding form: an exact
// ``range(0, 256) < 192`` mask feeding the same MergeSplit state load/store
// cone selected by large-factor IAT.  The rule must retain F16/grid ownership
// while removing the 64 masked-off D lanes.

// CHECK: hacc.independent_axis_tensorize
// CHECK: hacc.program_grid_transforms = {{.*}}axis = 1 : i32{{.*}}factor = 16 : i64{{.*}}persistent_coverage = false
// CHECK-NOT: hacc.coalesce_
// CHECK-LABEL: tt.func @merge_split_large_d192
// CHECK: tt.get_program_id x
// CHECK: tt.get_program_id y
// CHECK: tt.make_range {end = 16 : i32, start = 0 : i32}
// CHECK-NOT: tt.make_range {end = 256 : i32, start = 0 : i32}
// CHECK: tt.make_range {end = 192 : i32, start = 0 : i32}
// CHECK: tt.load {{.*}} : tensor<2x16x192x!tt.ptr<f32>>
// CHECK: tt.reduce
// CHECK-SAME: axis = 0 : i32
// CHECK: tensor<16x192xf32>
// CHECK: tt.store {{.*}} : tensor<16x192x!tt.ptr<f32>>

module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 64 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @merge_split_large_d192(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c192 = arith.constant 192 : i32
    %c12288 = arith.constant 12288 : i32
    %token = tt.get_program_id x : i32
    %head = tt.get_program_id y : i32
    %splits = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %dims = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %bound = arith.constant dense<192> : tensor<256xi32>
    %dim_ok = arith.cmpi slt, %dims, %bound : tensor<256xi32>
    %token_offset = arith.muli %token, %c12288 : i32
    %head_offset = arith.muli %head, %c192 : i32
    %base = arith.addi %token_offset, %head_offset : i32
    %input_base = tt.addptr %input, %base : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %input_broadcast = tt.broadcast %input_splat : tensor<2x1x!tt.ptr<f32>> -> tensor<2x256x!tt.ptr<f32>>
    %dims_expand = tt.expand_dims %dims {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %dims_broadcast = tt.broadcast %dims_expand : tensor<1x256xi32> -> tensor<2x256xi32>
    %input_ptrs = tt.addptr %input_broadcast, %dims_broadcast : tensor<2x256x!tt.ptr<f32>>, tensor<2x256xi32>
    %mask_expand = tt.expand_dims %dim_ok {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
    %mask = tt.broadcast %mask_expand : tensor<1x256xi1> -> tensor<2x256xi1>
    %zero = arith.constant dense<0.000000e+00> : tensor<2x256xf32>
    %states = tt.load %input_ptrs, %mask, %zero : tensor<2x256x!tt.ptr<f32>>
    %sum = "tt.reduce"(%states) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<2x256xf32>) -> tensor<256xf32>
    %output_base = tt.addptr %output, %base : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %output_ptrs, %sum, %dim_ok : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
