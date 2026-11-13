// RUN: %PYTHON %S/Inputs/rewrite.py %s --cv --batch-matmul | FileCheck %s

// CHECK: compile_status: REWRITTEN
// CHECK: cv_lowering: rank3_uniform_control_batch_v1
// CHECK: scf.for
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} :
// CHECK: tt.dot {{.*}}tensor<2x4x4xf32>
// CHECK-NOT: tt.dot
// CHECK: tt.return

module {
  tt.func public @arbitrary_transposed_dot(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, %trips: i32, %limit: i32) {
    %p = tt.get_program_id x : i32
    %c32 = arith.constant 32 : i32
    %base = arith.muli %p, %c32 : i32
    %r4 = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r8 = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %m = tt.expand_dims %r4 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %k = tt.expand_dims %r8 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %mm = tt.broadcast %m : tensor<4x1xi32> -> tensor<4x8xi32>
    %kk = tt.broadcast %k : tensor<1x8xi32> -> tensor<4x8xi32>
    %s4 = arith.constant dense<4> : tensor<4x8xi32>
    %ks = arith.muli %kk, %s4 : tensor<4x8xi32>
    %offset = arith.addi %mm, %ks : tensor<4x8xi32>
    %bs = tt.splat %base : i32 -> tensor<4x8xi32>
    %at = arith.addi %bs, %offset : tensor<4x8xi32>
    %ap = tt.splat %a : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %ptr = tt.addptr %ap, %at : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>

    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %z = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %dot = scf.for %iv = %z to %trips step %one iter_args(%carry = %zero) -> (tensor<4x4xf32>) : i32 {

      %bound = tt.splat %limit : i32 -> tensor<4x8xi32>
      %mask = arith.cmpi slt, %kk, %bound : tensor<4x8xi32>
      %pad = arith.constant dense<0.0> : tensor<4x8xf32>
      %av = tt.load %ptr, %mask, %pad : tensor<4x8x!tt.ptr<f32>>
    %bv = tt.trans %av {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
    %next = tt.dot %av, %bv, %carry : tensor<4x8xf32> * tensor<8x4xf32> -> tensor<4x4xf32>
    scf.yield %next : tensor<4x4xf32>
    }
    %row = tt.broadcast %m : tensor<4x1xi32> -> tensor<4x4xi32>
    %n = tt.expand_dims %r4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %col = tt.broadcast %n : tensor<1x4xi32> -> tensor<4x4xi32>
    %cs4 = arith.constant dense<4> : tensor<4x4xi32>
    %rs = arith.muli %row, %cs4 : tensor<4x4xi32>
    %co = arith.addi %rs, %col : tensor<4x4xi32>
    %cbase = tt.splat %base : i32 -> tensor<4x4xi32>
    %ct = arith.addi %cbase, %co : tensor<4x4xi32>
    %cp = tt.splat %c : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %cptr = tt.addptr %cp, %ct : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    tt.store %cptr, %dot : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}
