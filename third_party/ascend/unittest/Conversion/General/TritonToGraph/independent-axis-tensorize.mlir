// RUN: triton-opt --split-input-file %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1048576 device-core-count=1 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
//
// This fixture uses the same direct two-PID / affine-pointer form as the
// unchanged-DSL kernels.  It intentionally keeps the tensors small so the
// rule's shape and launch-contract checks are independent of a device UB
// budget.

// CHECK: hacc.independent_axis_tensorize
// CHECK: hacc.program_grid_transforms
// CHECK: factor = 8 : i64
// CHECK: logical_extent = 65 : i64
// CHECK: persistent_coverage = false
// CHECK-LABEL: tt.func @structural_merge_split_entry
// CHECK: arith.cmpi slt
// CHECK: tt.load {{.*}} : tensor<2x8x4x!tt.ptr<f32>>
// CHECK: tt.reduce
// CHECK-SAME: axis = 0 : i32
// CHECK: tensor<8x4xf32>
module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 65 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @structural_merge_split_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c4 = arith.constant 4 : i32
    %head = tt.get_program_id y : i32
    %splits = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %head_offset = arith.muli %head, %c4 : i32
    // MergeSplit also has scalar peak/total reductions. They must not make
    // the later [split, dim] reduction look like the Norm structural form.
    %aux = arith.constant dense<1.000000e+00> : tensor<2xf32>
    %aux_total = "tt.reduce"(%aux) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<2xf32>) -> f32
    %aux_total_splat = tt.splat %aux_total : f32 -> tensor<4xf32>
    %input_base = tt.addptr %input, %head_offset : !tt.ptr<f32>, i32
    %input_splat = tt.splat %input_base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %input_broadcast = tt.broadcast %input_splat : tensor<2x1x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %dims_expand = tt.expand_dims %dims {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %dims_broadcast = tt.broadcast %dims_expand : tensor<1x4xi32> -> tensor<2x4xi32>
    %input_ptrs = tt.addptr %input_broadcast, %dims_broadcast : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %states = tt.load %input_ptrs : tensor<2x4x!tt.ptr<f32>>
    %main_sum = "tt.reduce"(%states) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<2x4xf32>) -> tensor<4xf32>
    %sum = arith.addf %main_sum, %aux_total_splat : tensor<4xf32>
    %output_base = tt.addptr %output, %head_offset : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK: hacc.independent_axis_tensorize
// CHECK: logical_extent = 16 : i64
// CHECK-LABEL: tt.func @structural_norm_iat_entry
// CHECK: arith.cmpi slt
// CHECK: tensor<{{(2|4|8)}}x4xf32>
// CHECK: tt.reduce
// CHECK-SAME: axis = 1 : i32
// CHECK: tensor<{{(2|4|8)}}xf32>
module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 16 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @structural_norm_iat_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c4 = arith.constant 4 : i32
    %head = tt.get_program_id y : i32
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %head_offset = arith.muli %head, %c4 : i32
    %input_base = tt.addptr %input, %head_offset : !tt.ptr<f32>, i32
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
    %output_base = tt.addptr %output, %head_offset : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %result : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A K-shaped head extent of one is a deliberate no-op even when all other
// conditions match the Q form.
// CHECK-NOT: hacc.independent_axis_tensorize
// CHECK-LABEL: module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 1 : i64
// CHECK-LABEL: tt.func @structural_norm_head_one_entry
// CHECK: tt.get_program_id y
// CHECK: tt.reduce
// CHECK-SAME: axis = 0 : i32
module attributes {hacc.grid_specialization = {grid_0 = 8 : i64, grid_1 = 1 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @structural_norm_head_one_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c4 = arith.constant 4 : i32
    %head = tt.get_program_id y : i32
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %head_offset = arith.muli %head, %c4 : i32
    %input_base = tt.addptr %input, %head_offset : !tt.ptr<f32>, i32
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
    %output_base = tt.addptr %output, %head_offset : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %result : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Norm+RoPE IAT is conservative above its validated transformed launch range.
// The factor-eight candidate would make this 32768 x 16 input become 65536
// logical programs, so it must remain unmodified instead of reaching the NPU
// vector-core runtime with an unsupported launch range.
// CHECK-NOT: hacc.independent_axis_tensorize
// CHECK-NOT: hacc.program_grid_transforms
// CHECK-LABEL: module attributes {hacc.grid_specialization = {grid_0 = 32768 : i64, grid_1 = 16 : i64
// CHECK-LABEL: tt.func @structural_norm_launch_limit_entry
// CHECK: tt.get_program_id y
// CHECK: tt.store
module attributes {hacc.grid_specialization = {grid_0 = 32768 : i64, grid_1 = 16 : i64, grid_2 = 1 : i64, rule_mask = 512 : i64, version = 1 : i64}} {
  tt.func @structural_norm_launch_limit_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c4 = arith.constant 4 : i32
    %head = tt.get_program_id y : i32
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %head_offset = arith.muli %head, %c4 : i32
    %input_base = tt.addptr %input, %head_offset : !tt.ptr<f32>, i32
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
    %output_base = tt.addptr %output, %head_offset : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %result : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
