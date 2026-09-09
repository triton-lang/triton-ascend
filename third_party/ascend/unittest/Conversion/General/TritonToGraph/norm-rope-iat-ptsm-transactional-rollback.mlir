// RUN: split-file %s %t
// RUN: triton-opt %t/iat-success.mlir --verify-each -graph-optimize='rule-mask=512 mapping-ub-capacity-bytes=80 device-core-count=56 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s --check-prefix=IAT
// RUN: triton-opt %t/rollback.mlir --verify-each -graph-optimize='rule-mask=2560 mapping-ub-capacity-bytes=80 device-core-count=56 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s --check-prefix=ROLLBACK
//
// The first invocation proves the valid Norm+RoPE IAT materializer succeeds
// under the 64-byte effective mapping budget. The second gives the same TTIR
// both bits: IAT is materialized in a detached sandbox, but every PTSM block
// candidate exceeds the final budget. The transaction must leave no IAT body,
// PTSM body, grid transform, or mapping marker on the real module.

// IAT: hacc.independent_axis_tensorize
// IAT-NOT: hacc.persistent_task_strip_mining
// IAT: hacc.program_grid_transforms = {{.*}}axis = 1 : i32{{.*}}factor = 2 : i64{{.*}}persistent_coverage = false
// IAT: tt.get_program_id x
// IAT: tt.get_program_id y

// ROLLBACK-NOT: hacc.independent_axis_tensorize
// ROLLBACK-NOT: hacc.persistent_task_strip_mining
// ROLLBACK-NOT: hacc.program_grid_transforms
// ROLLBACK: tt.func @structural_rollback_joint_entry
// ROLLBACK: tt.get_program_id x
// ROLLBACK: tt.get_program_id y
// ROLLBACK-NOT: tt.get_num_programs
// ROLLBACK-NOT: scf.for
// ROLLBACK: tt.store

//--- iat-success.mlir
module attributes {
  hacc.grid_specialization = {grid_0 = 4096 : i64, grid_1 = 16 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}
} {
  tt.func @structural_rollback_iat_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) attributes {
    hacc.grid_specialization = {grid_0 = 4096 : i64, grid_1 = 16 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}
  } {
    %token = tt.get_program_id x : i32
    %head = tt.get_program_id y : i32
    %token_stride = arith.constant 64 : i32
    %head_stride = arith.constant 4 : i32
    %token_offset = arith.muli %token, %token_stride : i32
    %head_offset = arith.muli %head, %head_stride : i32
    %base = arith.addi %token_offset, %head_offset : i32
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %input_base = tt.addptr %input, %base : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %input_ptrs = tt.addptr %input_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %values = tt.load %input_ptrs : tensor<4x!tt.ptr<f32>>
    %sum = "tt.reduce"(%values) <{axis = 0 : i32}> ({
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

//--- rollback.mlir
module attributes {
  hacc.grid_specialization = {grid_0 = 4096 : i64, grid_1 = 16 : i64, grid_2 = 1 : i64, rule_mask = 2560 : i64, version = 1 : i64}
} {
  tt.func @structural_rollback_joint_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) attributes {
    hacc.grid_specialization = {grid_0 = 4096 : i64, grid_1 = 16 : i64, grid_2 = 1 : i64, rule_mask = 2560 : i64, version = 1 : i64}
  } {
    %token = tt.get_program_id x : i32
    %head = tt.get_program_id y : i32
    %token_stride = arith.constant 64 : i32
    %head_stride = arith.constant 4 : i32
    %token_offset = arith.muli %token, %token_stride : i32
    %head_offset = arith.muli %head, %head_stride : i32
    %base = arith.addi %token_offset, %head_offset : i32
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %input_base = tt.addptr %input, %base : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %input_ptrs = tt.addptr %input_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %values = tt.load %input_ptrs : tensor<4x!tt.ptr<f32>>
    %sum = "tt.reduce"(%values) <{axis = 0 : i32}> ({
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
