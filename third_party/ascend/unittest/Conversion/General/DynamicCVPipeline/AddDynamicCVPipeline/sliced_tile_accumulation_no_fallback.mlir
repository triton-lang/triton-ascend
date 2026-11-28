// Negative control for sliced_tile_accumulation_fallback.mlir: a scf.for
// that carries tensor iter_args but never slices the carried value keeps
// the dynamic CV pipeline available (no fallback attribute).

// RUN: triton-opt %s --pre-check-blacklist | FileCheck %s --check-prefix=PASS

// PASS-LABEL: func.func @carried_not_sliced
// PASS: scf.for
// PASS: arith.addf
// PASS-NOT: triton_ascend.dynamic_cv_pipeline.rc
module {
  func.func @carried_not_sliced(%arg0: tensor<64x256xf32>,
                                %arg1: tensor<64x256xf32>)
      -> tensor<64x256xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %acc = scf.for %i = %c0 to %c4 step %c1
        iter_args(%tile = %arg0) -> (tensor<64x256xf32>) : i32 {
      %next = arith.addf %tile, %arg1 : tensor<64x256xf32>
      scf.yield %next : tensor<64x256xf32>
    }
    return %acc : tensor<64x256xf32>
  }
}
