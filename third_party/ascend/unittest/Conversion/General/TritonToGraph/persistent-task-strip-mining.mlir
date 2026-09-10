// RUN: triton-opt --split-input-file %s --verify-each -graph-optimize='rule-mask=2048 ub-capacity-bytes=1048576 device-core-count=1 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simd_simt_template' -o - | FileCheck %s
// RUN: triton-opt --split-input-file %s --verify-each -graph-optimize='rule-mask=2048 ub-capacity-bytes=1048576 device-core-count=1 min-programs-per-core=1 ub-safety-percent=80 compile-mode=simt_only' -o - | FileCheck %s --check-prefix=SIMT
//
// PTSM owns the token (x) axis.  The input/output pointers are token-derived,
// while the weight load is invariant and must stay outside the strip-mined
// loop.  The small synthetic shape makes the tail and grid-stride contract
// observable without relying on an NPU runtime.

// CHECK: hacc.persistent_task_strip_mining
// CHECK: hacc.program_grid_transforms
// CHECK: grid_stride_abi_verified = true
// CHECK: logical_extent = 19 : i64
// CHECK: persistent_coverage = true
// CHECK-LABEL: tt.func @structural_persistent_entry
// CHECK: tt.get_program_id x
// CHECK: tt.get_num_programs x
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: tt.store
// SIMT-NOT: hacc.persistent_task_strip_mining
// SIMT-NOT: hacc.program_grid_transforms
// SIMT-LABEL: tt.func @structural_persistent_entry
// SIMT: tt.get_program_id x
module attributes {hacc.grid_specialization = {grid_0 = 19 : i64, grid_1 = 1 : i64, grid_2 = 1 : i64, rule_mask = 2048 : i64, version = 1 : i64}} {
  tt.func @structural_persistent_entry(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %weight: !tt.ptr<f32>) attributes {hacc.grid_specialization = {grid_0 = 19 : i64, grid_1 = 1 : i64, grid_2 = 1 : i64, rule_mask = 2048 : i64, version = 1 : i64}} {
    %c4 = arith.constant 4 : i32
    %token = tt.get_program_id x : i32
    %dims = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = arith.muli %token, %c4 : i32
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
    %weight_splat = tt.splat %weight : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %weight_ptrs = tt.addptr %weight_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %weights = tt.load %weight_ptrs : tensor<4x!tt.ptr<f32>>
    %scale = tt.splat %sum : f32 -> tensor<4xf32>
    %result = arith.mulf %values, %scale : tensor<4xf32>
    %weighted = arith.mulf %result, %weights : tensor<4xf32>
    %output_base = tt.addptr %output, %base : !tt.ptr<f32>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_splat, %dims : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptrs, %weighted : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
