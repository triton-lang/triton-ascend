// RUN: triton-opt --mark-main-loop %s | FileCheck %s

// Unhinted kernels retain the existing innermost selection policy.
// CHECK-LABEL: func.func @automatic_inner
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.copy
// CHECK-NEXT: } {ssbuffer.main_loop = {{[0-9]+}} : i32}
// CHECK-NEXT: }
module attributes {ssbuffer.outer_main_loop} {
func.func @automatic_inner(%src: tensor<16x16xf16>, %dst: memref<16x16xf16, #hivm.address_space<cbuf>>, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      hivm.hir.copy ins(%src : tensor<16x16xf16>) outs(%dst : memref<16x16xf16, #hivm.address_space<cbuf>>)
    }
  }
  return
}

// An explicit scheduling domain takes priority over inner automatic candidates.
// CHECK-LABEL: func.func @explicit_outer
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.copy
// CHECK-NEXT: }
// CHECK-NEXT: } {ssbuffer.explicit_main_loop, ssbuffer.main_loop = {{[0-9]+}} : i32}
func.func @explicit_outer(%src: tensor<16x16xf16>, %dst: memref<16x16xf16, #hivm.address_space<cbuf>>, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      hivm.hir.copy ins(%src : tensor<16x16xf16>) outs(%dst : memref<16x16xf16, #hivm.address_space<cbuf>>)
    } {ssbuffer.main_loop = 7 : i32}
  } {ssbuffer.explicit_main_loop}
  return
}
}
