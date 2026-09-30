// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func private @callee(i32) -> i32
func.func private @callee(%arg0: i32) -> i32

// CHECK-LABEL: func.func @func_return_misc(%arg0: i32) -> i32 {
// CHECK-NEXT: %[[R:.*]] = call @callee(%arg0) : (i32) -> i32
// CHECK-NEXT: return %[[R]] : i32
func.func @func_return_misc(%arg0: i32) -> i32 {
  %0 = call @callee(%arg0) : (i32) -> i32
  return %0 : i32
}
