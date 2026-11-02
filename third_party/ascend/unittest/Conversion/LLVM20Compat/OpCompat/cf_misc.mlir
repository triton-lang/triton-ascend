// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @cf_branch_misc(%arg0: i32) -> i32 {
// CHECK-NEXT: cf.br ^bb1(%arg0 : i32)
// CHECK-NEXT: ^bb1(%[[V:.*]]: i32):
// CHECK-NEXT: return %[[V]] : i32
func.func @cf_branch_misc(%arg0: i32) -> i32 {
  cf.br ^bb1(%arg0 : i32)
^bb1(%v: i32):
  return %v : i32
}
