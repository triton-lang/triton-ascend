// RUN: split-file %s %t
// RUN: %PYTHON %S/Inputs/rewrite.py %t/loop-overflow.mlir  | FileCheck %s --check-prefix=OVERFLOW
// RUN: %PYTHON %S/Inputs/rewrite.py %t/unsupported-attribute.mlir  | FileCheck %s --check-prefix=ATTRIBUTE
// RUN: %PYTHON %S/Inputs/rewrite.py %t/varying-control.mlir --cv | FileCheck %s --check-prefix=CONTROL

//--- loop-overflow.mlir
// The grouped induction step would overflow i32. It must not select
// sequential iteration grouping even though the original loop is valid.
// OVERFLOW-NOT: vv_lowering: sequential_iterations_v1
// OVERFLOW: module
// OVERFLOW: tt.return

module {
  tt.func public @static_recurrence(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id x : i32
    %xp = tt.addptr %x, %p : !tt.ptr<f32>, i32
    %op = tt.addptr %o, %p : !tt.ptr<f32>, i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 2147483647 : i32
    %step = arith.constant 1073741824 : i32
    %zero = arith.constant 0.0 : f32
    %sum = scf.for %iv = %lo to %hi step %step
        iter_args(%acc = %zero) -> (f32) : i32 {
      %v = tt.load %xp : !tt.ptr<f32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    }
    tt.store %op, %sum : !tt.ptr<f32>
    tt.return
  }
}

//--- unsupported-attribute.mlir
// ATTRIBUTE: compile_status: REJECTED
// ATTRIBUTE: reason: UNSUPPORTED_OPERATION_ATTRIBUTE
// ATTRIBUTE: unchanged: true

module {
  tt.func public @arbitrary_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id x : i32
    %g = tt.get_num_programs x : i32
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
    %result = "arith.addf"(%v, %gv) {unknown_batch_semantics = true} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    tt.store %ya, %result, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

//--- varying-control.mlir
// CONTROL: compile_status: REJECTED
// CONTROL: reason: VARYING_CONTROL_CONDITION
// CONTROL: unchanged: true

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

      %choose = arith.cmpi sgt, %p, %loop_zero : i32
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
