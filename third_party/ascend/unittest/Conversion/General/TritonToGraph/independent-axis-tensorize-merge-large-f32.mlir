// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1048576 device-core-count=16 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
//
// With a 16-vector-core target, F16 would leave 32 programs and is outside the
// one-wave small-grid policy.  F32 leaves exactly 16 nonpersistent programs,
// so this fixture exercises the F32 materializer and its one IAT contract.

// CHECK: hacc.independent_axis_tensorize
// CHECK: hacc.program_grid_transforms = {{.*}}axis = 1 : i32{{.*}}factor = 32 : i64{{.*}}persistent_coverage = false
// CHECK-NOT: hacc.coalesce_
// CHECK-LABEL: tt.func @merge_split_large_f32
// CHECK: tt.get_program_id x
// CHECK: tt.get_program_id y
// CHECK: arith.muli
// CHECK: tt.make_range {end = 32 : i32, start = 0 : i32}
// CHECK: arith.cmpi slt
// CHECK: tt.load {{.*}} : tensor<2x32x32x!tt.ptr<f32>>
// CHECK: tt.reduce
// CHECK-SAME: axis = 0 : i32
// CHECK: tensor<32x32xf32>
module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 64 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @merge_split_large_f32(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c32 = arith.constant 32 : i32
    %c2048 = arith.constant 2048 : i32
    %token = tt.get_program_id x : i32
    %head = tt.get_program_id y : i32
    %splits = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %dims = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %token_offset = arith.muli %token, %c2048 : i32
    %head_offset = arith.muli %head, %c32 : i32
    %base = arith.addi %token_offset, %head_offset : i32
    %input_base = tt.addptr %input, %base : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %input_broadcast = tt.broadcast %input_splat : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
    %dims_expand = tt.expand_dims %dims {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %dims_broadcast = tt.broadcast %dims_expand : tensor<1x32xi32> -> tensor<2x32xi32>
    %input_ptrs = tt.addptr %input_broadcast, %dims_broadcast : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
    %states = tt.load %input_ptrs : tensor<2x32x!tt.ptr<f32>>
    %sum = "tt.reduce"(%states) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<2x32xf32>) -> tensor<32xf32>
    %output_base = tt.addptr %output, %base : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %output_ptrs, %sum : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
