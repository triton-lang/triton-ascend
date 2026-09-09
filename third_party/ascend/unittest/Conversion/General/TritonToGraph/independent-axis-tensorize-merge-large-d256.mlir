// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=131072 mapping-ub-capacity-bytes=262144 device-core-count=56 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
//
// This uses the primary MergeSplit D256 lane shape and the runtime mapping UB
// budget.  The final F16 candidate is legal but its generic live-byte score is
// negative, so it guards that the dedicated small-grid policy does not fall
// back to F8 merely because the generic score is below zero.

// CHECK: hacc.independent_axis_tensorize
// CHECK: hacc.program_grid_transforms = {{.*}}axis = 1 : i32{{.*}}factor = 16 : i64{{.*}}persistent_coverage = false
// CHECK-NOT: hacc.coalesce_
// CHECK-LABEL: tt.func @merge_split_large_d256
// CHECK: tt.get_program_id x
// CHECK: tt.get_program_id y
// CHECK: arith.muli
// CHECK: tt.make_range {end = 16 : i32, start = 0 : i32}
// CHECK: arith.cmpi slt
// CHECK: tt.make_range {end = 256 : i32, start = 0 : i32}
// CHECK: tt.load {{.*}} : tensor<2x16x256x!tt.ptr<f32>>
// CHECK: tt.reduce
// CHECK-SAME: axis = 0 : i32
// CHECK: tensor<16x256xf32>
module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 64 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @merge_split_large_d256(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c256 = arith.constant 256 : i32
    %c16384 = arith.constant 16384 : i32
    %token = tt.get_program_id x : i32
    %head = tt.get_program_id y : i32
    %splits = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %dims = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %token_offset = arith.muli %token, %c16384 : i32
    %head_offset = arith.muli %head, %c256 : i32
    %base = arith.addi %token_offset, %head_offset : i32
    %input_base = tt.addptr %input, %base : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %input_broadcast = tt.broadcast %input_splat : tensor<2x1x!tt.ptr<f32>> -> tensor<2x256x!tt.ptr<f32>>
    %dims_expand = tt.expand_dims %dims {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %dims_broadcast = tt.broadcast %dims_expand : tensor<1x256xi32> -> tensor<2x256xi32>
    %input_ptrs = tt.addptr %input_broadcast, %dims_broadcast : tensor<2x256x!tt.ptr<f32>>, tensor<2x256xi32>
    %states = tt.load %input_ptrs : tensor<2x256x!tt.ptr<f32>>
    %sum = "tt.reduce"(%states) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<2x256xf32>) -> tensor<256xf32>
    %output_base = tt.addptr %output, %base : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %output_ptrs, %sum : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
