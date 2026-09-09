// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=2048 ub-capacity-bytes=1048576 device-core-count=1 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
//
// PTSM needs the same exact-address proof as IAT.  A runtime token stride is
// specialized only for this cache-keyed compilation; without the scalar
// contract it remains unknown and PTSM must stay a no-op.

// CHECK: hacc.persistent_task_strip_mining
// CHECK: hacc.program_grid_transforms = {{.*}}axis = 0 : i32{{.*}}persistent_coverage = true
// CHECK-NOT: hacc.program_mapping_scalar_specialization
// CHECK-LABEL: tt.func @structural_persistent_stride_entry
// CHECK: arith.constant dense<4>
// CHECK-NOT: %arg2
// CHECK: tt.get_num_programs x
// CHECK: scf.for
// CHECK: tt.store
module attributes {
  hacc.grid_specialization = {grid_0 = 19 : i64, grid_1 = 1 : i64, grid_2 = 1 : i64, rule_mask = 2048 : i64, version = 1 : i64},
  hacc.program_mapping_scalar_specialization = {arguments = [{index = 2 : i64, value = 4 : i64}], version = 1 : i64}
} {
  tt.func @structural_persistent_stride_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %stride: i32) attributes {
    hacc.grid_specialization = {grid_0 = 19 : i64, grid_1 = 1 : i64, grid_2 = 1 : i64, rule_mask = 2048 : i64, version = 1 : i64},
    hacc.program_mapping_scalar_specialization = {arguments = [{index = 2 : i64, value = 4 : i64}], version = 1 : i64}
  } {
    %token = tt.get_program_id x : i32
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = arith.muli %token, %stride : i32
    %input_base = tt.addptr %input, %base : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %input_ptrs = tt.addptr %input_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %values = tt.load %input_ptrs : tensor<4x!tt.ptr<f32>>
    %squares = arith.mulf %values, %values : tensor<4xf32>
    %sum = "tt.reduce"(%squares) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<4xf32>) -> f32
    %scale = tt.splat %sum : f32 -> tensor<4xf32>
    %result = arith.mulf %values, %scale : tensor<4xf32>
    %output_base = tt.addptr %output, %base : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %result : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
