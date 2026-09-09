// RUN: triton-opt --split-input-file %s --verify-each -graph-optimize='rule-mask=0' -o - | FileCheck %s
//
// Negative and transactional templates stay valid TTIR. With all new rules
// disabled they must preserve their baseline mapping; future rules add their
// own positive checks rather than silently changing this control.

// CHECK-LABEL: tt.func @task0005_negative_group_one(
// CHECK: tt.get_program_id z
// CHECK: tt.return
tt.func @task0005_negative_group_one() {
  %group = tt.get_program_id z : i32
  tt.return
}

// -----

// CHECK-LABEL: tt.func @task0005_transactional_store_control(
// CHECK-NOT: hacc.coalesce
// CHECK: tt.return
tt.func @task0005_transactional_store_control() {
  tt.return
}
