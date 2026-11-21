// RUN: split-file %s %t
// RUN: %PYTHON %S/Inputs/rewrite.py %t/cv-spatial-m.mlir --cv | FileCheck %s --check-prefix=M
// RUN: %PYTHON %S/Inputs/rewrite.py %t/cv-spatial-n.mlir --cv | FileCheck %s --check-prefix=N
// RUN: %PYTHON %S/Inputs/rewrite.py %t/cv-spatial-loop.mlir --cv --batch-matmul | FileCheck %s --check-prefix=SPATIAL-LOOP
// RUN: %PYTHON %S/Inputs/rewrite.py %t/cv-batch-loop.mlir --cv --batch-matmul | FileCheck %s --check-prefix=BATCH-LOOP

//--- cv-spatial-m.mlir
// M: compile_status: REWRITTEN
// M: cv_lowering: spatial_m_v1
// M: tt.func public @not_a_kernel_name
// M: tt.dot {{.*}} : tensor<8x4xf32> * tensor<4x4xf32> -> tensor<8x4xf32>
// M-NOT: tt.dot
// M: tt.store

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

//--- cv-spatial-n.mlir
// N: compile_status: REWRITTEN
// N: cv_lowering: spatial_n_v1
// N: tt.func public @not_a_kernel_name
// N: tt.dot {{.*}} : tensor<4x4xf32> * tensor<4x8xf32> -> tensor<4x8xf32>
// N-NOT: tt.dot
// N: tt.store

module {
  tt.func public @not_a_kernel_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %z: !tt.ptr<f32>, %limit: i32) {
    %p = tt.get_program_id x : i32
    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %basev = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %basev, %range : tensor<4xi32>
    %rcol = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %rrow = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %nrow = tt.expand_dims %idx {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
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
    %me = tt.expand_dims %mv {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
    %mask = tt.broadcast %me : tensor<1x4xi1> -> tensor<4x4xi1>
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %a = tt.load %ap : tensor<4x4x!tt.ptr<f32>>
    %b = tt.load %bp : tensor<4x4x!tt.ptr<f32>>

    %dot = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
    tt.store %cp, %dot, %mask : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

//--- cv-spatial-loop.mlir
// SPATIAL-LOOP: compile_status: REWRITTEN
// SPATIAL-LOOP: cv_lowering: spatial_m_uniform_control_v1
// SPATIAL-LOOP: scf.for
// SPATIAL-LOOP-SAME: tensor<8x4xf32>
// SPATIAL-LOOP: scf.if
// SPATIAL-LOOP: tt.dot {{.*}} : tensor<8x4xf32> * tensor<4x4xf32> -> tensor<8x4xf32>
// SPATIAL-LOOP-NOT: tt.dot
// SPATIAL-LOOP: tt.return

module {
  tt.func public @not_a_kernel_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %z: !tt.ptr<f32>, %limit: i32, %trips: i32, %flag: i32) {
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

    %loop_zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %dot = scf.for %iv = %loop_zero to %trips step %one iter_args(%carry = %zero) -> (tensor<4x4xf32>) : i32 {

      %choose = arith.cmpi sgt, %flag, %loop_zero : i32
      %picked = scf.if %choose -> (tensor<4x4xf32>) {
            %a = tt.load %ap : tensor<4x4x!tt.ptr<f32>>
    %b = tt.load %bp : tensor<4x4x!tt.ptr<f32>>

    %next = tt.dot %a, %b, %carry, inputPrecision = ieee : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>

        scf.yield %next : tensor<4x4xf32>
      } else {
        scf.yield %carry : tensor<4x4xf32>
      }
    scf.yield %picked : tensor<4x4xf32>
    }
    tt.store %cp, %dot, %mask : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

//--- cv-batch-loop.mlir
// BATCH-LOOP: compile_status: REWRITTEN
// BATCH-LOOP: cv_lowering: static_uniform_control_lanes_v1
// BATCH-LOOP-NOT: tensor<2x4x4xf32>
// BATCH-LOOP: scf.for
// BATCH-LOOP: tt.load {{.*}}, {{.*}}, {{.*}} :
// BATCH-LOOP: tt.dot {{.*}}tensor<4x4xf32>
// BATCH-LOOP: scf.for
// BATCH-LOOP: tt.load {{.*}}, {{.*}}, {{.*}} :
// BATCH-LOOP: tt.dot {{.*}}tensor<4x4xf32>
// BATCH-LOOP-NOT: tt.dot
// BATCH-LOOP: tt.return

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
