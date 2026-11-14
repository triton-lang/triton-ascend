// RUN: %PYTHON %S/Inputs/rewrite.py %s --factor 3 --axis -1 --layout semantic_only | FileCheck %s

// CHECK: compile_status: REWRITTEN
// CHECK: selected_axis: 2
// CHECK: hacc.block_mapping_owner = "block_batching"
// CHECK: tt.func public @arbitrary_name
// CHECK-SAME: %__hacc_original_grid_x: i32
// CHECK-SAME: %__hacc_original_grid_y: i32
// CHECK-SAME: %__hacc_original_grid_z: i32
// CHECK: tensor<3x4xf32>
// CHECK-NOT: tt.get_num_programs
// CHECK: tt.return

module {
  tt.func public @arbitrary_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id z : i32
    %g = tt.get_num_programs z : i32
    %gf = arith.sitofp %g : i32 to f32
    %gv = tt.splat %gf : f32 -> tensor<4xf32>
    %b = arith.constant 4 : i32
    %base = arith.muli %p, %b : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %basev = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %basev, %range : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %idx, %nv : tensor<4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ya = tt.addptr %yp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    %v = tt.load %xa, %mask, %zero : tensor<4x!tt.ptr<f32>>
    %result = arith.addf %v, %gv : tensor<4xf32>
    tt.store %ya, %result, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
