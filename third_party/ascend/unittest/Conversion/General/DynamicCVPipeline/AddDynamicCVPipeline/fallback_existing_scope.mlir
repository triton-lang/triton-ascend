// RUN: triton-opt '--add_dynamic_cv_pipeline=compile-on-910-95=True' %s | FileCheck %s

// The return code, plus the seed decision the pass publishes on every exit --
// here "nothing was chosen, and nothing was asked for", which is what a module
// the pipeline declined has to report rather than leaving the fields absent.
// Matched by name instead of by the whole dictionary: the pass has every right
// to publish more, and pinning the braces makes an addition look like a
// regression in the fallback this test is actually about.
// CHECK: module attributes {{.*}}dynamic_cv_pipeline.chosen_seed = -1 : i64
// CHECK-SAME: dynamic_cv_pipeline.rc = 2 : i32
// CHECK-SAME: dynamic_cv_pipeline.search_status = 0 : i64
module {
  // CHECK-LABEL: func.func @fallback_existing_scope
  // CHECK: arith.addi
  // CHECK: scope.scope
  // CHECK: scope.return
  // CHECK: return
  func.func @fallback_existing_scope(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    scope.scope : () -> () {
      scope.return
    }
    return %0 : i32
  }
}
