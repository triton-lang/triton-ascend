// Negative controls for sliced_tile_accumulation_fallback.mlir: kernels that
// slice carried tensors in shapes OTHER than the paired same-window
// read-modify-write must keep the dynamic CV pipeline (no fallback).
//
//   1. @carried_not_sliced: tensor iter_args never sliced at all.
//   2. @fill_by_rows: insert-only moving window (gather / discretized-load
//      shape, e.g. index_select filling buf[scatterRow, :] row by row);
//      there is no matching extract of the carried tensor, so this is not
//      manual UB-driven sub-tiling.
//   3. @copy_through: an extract/insert pair at the same window whose
//      inserted value is the extracted value itself - a pure copy, not an
//      accumulation.

// RUN: triton-opt %s --pre-check-blacklist | FileCheck %s --check-prefix=PASS

// PASS-LABEL: func.func @carried_not_sliced
// PASS: scf.for
// PASS: arith.addf
// PASS-LABEL: func.func @fill_by_rows
// PASS: scf.for
// PASS: tensor.insert_slice
// PASS-LABEL: func.func @copy_through
// PASS: tensor.extract_slice
// PASS: tensor.insert_slice
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

  func.func @fill_by_rows(%arg0: tensor<16x64xf32>, %arg1: tensor<1x64xf32>)
      -> tensor<16x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %buf = scf.for %i = %c0 to %c16 step %c1
        iter_args(%tile = %arg0) -> (tensor<16x64xf32>) : index {
      %next = tensor.insert_slice %arg1 into %tile[%i, 0] [1, 64] [1, 1]
          : tensor<1x64xf32> into tensor<16x64xf32>
      scf.yield %next : tensor<16x64xf32>
    }
    return %buf : tensor<16x64xf32>
  }

  func.func @copy_through(%arg0: tensor<64x256xf32>, %arg1: tensor<64x256xf32>)
      -> tensor<64x256xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_idx = arith.constant 0 : index
    %acc = scf.for %i = %c0 to %c4 step %c1
        iter_args(%tile = %arg0) -> (tensor<64x256xf32>) : i32 {
      %off = arith.muli %i, %c16_i32 : i32
      %off_i = arith.index_cast %off : i32 to index
      %sub = tensor.extract_slice %tile[%off_i, %c0_idx] [16, 256] [1, 1]
          : tensor<64x256xf32> to tensor<16x256xf32>
      %next = tensor.insert_slice %sub into %tile[%off_i, %c0_idx] [16, 256] [1, 1]
          : tensor<16x256xf32> into tensor<64x256xf32>
      scf.yield %next : tensor<64x256xf32>
    }
    return %acc : tensor<64x256xf32>
  }
}
