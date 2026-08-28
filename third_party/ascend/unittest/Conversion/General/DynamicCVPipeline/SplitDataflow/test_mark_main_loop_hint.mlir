// RUN: triton-opt --mark-main-loop %s | FileCheck %s --implicit-check-not="tt.compile_hint"

// An explicit frontend hint must select the outer loop even though the
// Fixpipe/Copy heuristic would otherwise select the inner loop.
// CHECK-LABEL: func.func @explicit_outer_main_loop
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:   } {test.loop = "candidate"}
// CHECK: } {ssbuffer.main_loop = 0 : i32, test.loop = "selected"}
// CHECK-NOT: ssbuffer.main_loop
func.func @explicit_outer_main_loop(%src: tensor<4xf32>, %dst: memref<4xf32>,
                                    %lb: index, %ub: index, %step: index) {
  scf.for %outer = %lb to %ub step %step {
    scf.for %inner = %lb to %ub step %step {
      hivm.hir.copy ins(%src : tensor<4xf32>) outs(%dst : memref<4xf32>)
    } {test.loop = "candidate"}
  } {test.loop = "selected", tt.compile_hint = "main_loop"}
  return
}

