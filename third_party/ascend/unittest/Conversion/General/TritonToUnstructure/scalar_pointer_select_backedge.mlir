// RUN: triton-opt %s --triton-to-unstructure -verify-each | FileCheck %s

// Different pointer bases selected inside a loop must travel together as a
// complete address, including the initial value and the backedge.
tt.func public @for_select_backedge(%initial: !tt.ptr<i32>,
    %alternate: !tt.ptr<i32>, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %count step %c1
      iter_args(%ptr = %initial) -> (!tt.ptr<i32>) {
    %condition = arith.cmpi eq, %i, %c0 : index
    %next = arith.select %condition, %alternate, %ptr : !tt.ptr<i32>
    scf.yield %next : !tt.ptr<i32>
  }
  %loaded = tt.load %result : !tt.ptr<i32>
  tt.return %loaded : i32
}

// CHECK-LABEL: tt.func public @for_select_backedge
// CHECK:       tt.ptr_to_int
// CHECK:       scf.for
// CHECK-SAME:  -> (i64)
// CHECK:       tt.int_to_ptr
// CHECK:       arith.select {{.*}} : !tt.ptr<i32>
// CHECK:       tt.ptr_to_int
// CHECK:       scf.yield {{.*}} : i64

tt.func public @while_select_backedge(%initial: !tt.ptr<i32>,
    %alternate: !tt.ptr<i32>, %count: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %result:2 = scf.while (%i = %c0, %ptr = %initial)
      : (i32, !tt.ptr<i32>) -> (i32, !tt.ptr<i32>) {
    %continue = arith.cmpi slt, %i, %count : i32
    scf.condition(%continue) %i, %ptr : i32, !tt.ptr<i32>
  } do {
  ^bb0(%i: i32, %ptr: !tt.ptr<i32>):
    %next_i = arith.addi %i, %c1 : i32
    %condition = arith.cmpi eq, %i, %c0 : i32
    %next = arith.select %condition, %alternate, %ptr : !tt.ptr<i32>
    scf.yield %next_i, %next : i32, !tt.ptr<i32>
  }
  %loaded = tt.load %result#1 : !tt.ptr<i32>
  tt.return %loaded : i32
}

// CHECK-LABEL: tt.func public @while_select_backedge
// CHECK:       scf.while
// CHECK-SAME:  (i32, i64) -> (i32, i64)
// CHECK:       scf.condition
// CHECK-SAME:  i32, i64
// CHECK:       arith.select {{.*}} : !tt.ptr<i32>
// CHECK:       tt.ptr_to_int
// CHECK:       scf.yield {{.*}} : i32, i64
