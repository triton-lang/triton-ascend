// RUN: %PYTHON %S/Inputs/rewrite.py %s  | FileCheck %s

// The grouped induction step would overflow i32. It must not select
// sequential iteration grouping even though the original loop is valid.
// CHECK-NOT: vv_lowering: sequential_iterations_v1
// CHECK: module
// CHECK: tt.return

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
