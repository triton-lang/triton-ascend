// RUN: triton-opt -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @ub_misc() -> i32 {
// CHECK-NEXT: %[[P:.*]] = "ub.poison"() : () -> i32
// CHECK-NEXT: return %[[P]] : i32
func.func @ub_misc() -> i32 {
  %0 = "ub.poison"() : () -> i32
  return %0 : i32
}
