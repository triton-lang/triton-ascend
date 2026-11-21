// RUN: triton-opt %s --triton-to-unstructure -verify-each | FileCheck %s

// Include the initial displacement on zero trips and accumulate both signed
// element increments on every subsequent trip. The sum does not fit in i8.
// CHECK-LABEL: tt.func public @chained_cursor(
// CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<i32>
// CHECK-DAG: %[[START:.*]] = arith.constant 7 : i64
// CHECK-DAG: %[[STEP:.*]] = arith.constant -129 : i64
// CHECK: %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[OFFSET:.*]] = %[[START]]) -> (i64)
// CHECK: %[[PTR:.*]] = tt.addptr %[[BASE]], %[[OFFSET]] : !tt.ptr<i32>, i64
// CHECK: tt.load %[[PTR]]
// CHECK: %[[NEXT:.*]] = arith.addi %[[OFFSET]], %[[STEP]] : i64
// CHECK: scf.yield %[[NEXT]] : i64
// CHECK: %[[FINAL:.*]] = tt.addptr %[[BASE]], %[[RESULT]] : !tt.ptr<i32>, i64
// CHECK: tt.load %[[FINAL]]
tt.func public @chained_cursor(%base: !tt.ptr<i32>, %out: !tt.ptr<i32>,
    %count: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %min = arith.constant -128 : i8
  %minus_one = arith.constant -1 : i8
  %c7 = arith.constant 7 : i32
  %start = tt.addptr %base, %c7 : !tt.ptr<i32>, i32
  %result = scf.for %i = %c0 to %count step %c1
      iter_args(%p = %start) -> (!tt.ptr<i32>) : i32 {
    %v = tt.load %p : !tt.ptr<i32>
    tt.store %out, %v : !tt.ptr<i32>
    %q = tt.addptr %p, %min : !tt.ptr<i32>, i8
    %next = tt.addptr %q, %minus_one : !tt.ptr<i32>, i8
    scf.yield %next : !tt.ptr<i32>
  }
  %last = tt.load %result : !tt.ptr<i32>
  tt.return %last : i32
}

// Both possible pointers have the same root, but the numeric analysis treats
// the selected SSA as a fresh base. Its zero offset cannot describe this edge.
// CHECK-LABEL: tt.func public @opaque_offset(
// CHECK: tt.ptr_to_int
// CHECK: scf.for
// CHECK: %[[SELECT:.*]] = arith.select {{.*}} : !tt.ptr<i32>
// CHECK: %[[ADDRESS:.*]] = tt.ptr_to_int %[[SELECT]]
// CHECK: scf.yield %[[ADDRESS]] : i64
tt.func public @opaque_offset(%base: !tt.ptr<i32>, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %step = arith.constant 1 : i32
  %result = scf.for %i = %c0 to %count step %c1
      iter_args(%p = %base) -> (!tt.ptr<i32>) {
    %advanced = tt.addptr %p, %step : !tt.ptr<i32>, i32
    %first = arith.cmpi eq, %i, %c0 : index
    %next = arith.select %first, %advanced, %p : !tt.ptr<i32>
    scf.yield %next : !tt.ptr<i32>
  }
  %last = tt.load %result : !tt.ptr<i32>
  tt.return %last : i32
}

// The condition forwards an offset to both the after region and loop results.
// The backedge resets the cursor relative to the invariant base.
// CHECK-LABEL: tt.func public @while_cursor(
// CHECK-SAME: %[[WBASE:[^:]+]]: !tt.ptr<i32>
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[RESET:.*]] = arith.constant 7 : i64
// CHECK: %[[R:.*]]:2 = scf.while ({{.*}}, %[[CUR:.*]] = {{.*}}) : (i32, i64) -> (i32, i64)
// CHECK: %[[EXIT:.*]] = arith.addi %[[CUR]], %[[TWO]] : i64
// CHECK: scf.condition({{.*}}) {{.*}}, %[[EXIT]] : i32, i64
// CHECK: scf.yield {{.*}}, %[[RESET]] : i32, i64
// CHECK: %[[ENDPTR:.*]] = tt.addptr %[[WBASE]], %[[R]]#1 : !tt.ptr<i32>, i64
// CHECK: tt.load %[[ENDPTR]]
tt.func public @while_cursor(%base: !tt.ptr<i32>, %count: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c7 = arith.constant 7 : i32
  %start = tt.addptr %base, %c1 : !tt.ptr<i32>, i32
  %result:2 = scf.while (%i = %c0, %p = %start)
      : (i32, !tt.ptr<i32>) -> (i32, !tt.ptr<i32>) {
    %more = arith.cmpi slt, %i, %count : i32
    %forward = tt.addptr %p, %c2 : !tt.ptr<i32>, i32
    scf.condition(%more) %i, %forward : i32, !tt.ptr<i32>
  } do {
  ^bb0(%i: i32, %q: !tt.ptr<i32>):
    %next = arith.addi %i, %c1 : i32
    %reset = tt.addptr %base, %c7 : !tt.ptr<i32>, i32
    scf.yield %next, %reset : i32, !tt.ptr<i32>
  }
  %last = tt.load %result#1 : !tt.ptr<i32>
  tt.return %last : i32
}
