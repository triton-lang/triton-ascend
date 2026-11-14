// RUN: %PYTHON %S/Inputs/rewrite.py %s --cv | FileCheck %s

// CHECK: compile_status: REWRITTEN
// CHECK: cv_lowering: spatial_m_v1
// CHECK: tt.func public @not_a_kernel_name
// CHECK: tt.dot {{.*}} : tensor<8x4xf32> * tensor<4x4xf32> -> tensor<8x4xf32>
// CHECK-NOT: tt.dot
// CHECK: tt.store

module {
  tt.func public @not_a_kernel_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %z: !tt.ptr<f32>, %limit: i32) {
    %p = tt.get_program_id x : i32
    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %basev = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %basev, %range : tensor<4xi32>
    %rcol = tt.expand_dims %idx {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %rrow = tt.expand_dims %idx {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %nrow = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %krow = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %kcol = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %stride = arith.constant dense<64> : tensor<4x1xi32>
    %aoff0 = arith.muli %rcol, %stride : tensor<4x1xi32>
    %aoff1 = tt.broadcast %aoff0 : tensor<4x1xi32> -> tensor<4x4xi32>
    %aoff2 = tt.broadcast %krow : tensor<1x4xi32> -> tensor<4x4xi32>
    %aoff = arith.addi %aoff1, %aoff2 : tensor<4x4xi32>
    %boff0 = arith.muli %kcol, %stride : tensor<4x1xi32>
    %boff1 = tt.broadcast %boff0 : tensor<4x1xi32> -> tensor<4x4xi32>
    %boff2 = tt.broadcast %nrow : tensor<1x4xi32> -> tensor<4x4xi32>
    %boff = arith.addi %boff1, %boff2 : tensor<4x4xi32>
    %coff0 = arith.muli %rcol, %stride : tensor<4x1xi32>
    %coff1 = tt.broadcast %coff0 : tensor<4x1xi32> -> tensor<4x4xi32>
    %coff = arith.addi %coff1, %boff2 : tensor<4x4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %zp = tt.splat %z : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %ap = tt.addptr %xp, %aoff : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %bp = tt.addptr %yp, %boff : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %cp = tt.addptr %zp, %coff : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %lim = tt.splat %limit : i32 -> tensor<4xi32>
    %mv = arith.cmpi slt, %idx, %lim : tensor<4xi32>
    %me = tt.expand_dims %mv {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
    %mask = tt.broadcast %me : tensor<4x1xi1> -> tensor<4x4xi1>
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %a = tt.load %ap : tensor<4x4x!tt.ptr<f32>>
    %b = tt.load %bp : tensor<4x4x!tt.ptr<f32>>

    %dot = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
    tt.store %cp, %dot, %mask : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}
