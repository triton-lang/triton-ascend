// A scf.for carrying tensor iter_args whose body slices the carried tensor
// itself (tensor.extract_slice / tensor.insert_slice on the loop-carried
// value) is the sliced-tile accumulation idiom: the kernel sub-tiles its
// accumulator manually to fit the legacy SSBUFFER UB budget (e.g. the
// HEAD_DIM >= 256 fused-attention update). The dynamic CV pipeline would
// plan whole-tile cross-core UB buffers and multi-buffer them, overflowing
// UB at the BiShengIR regbase stage, so the kernel must fall back.

// RUN: triton-opt %s --pre-check-blacklist | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %s --pre-check-dynamic-cv-pipeline-available | FileCheck %s --check-prefix=DECLINE

// DECLINE: module attributes
// DECLINE-SAME: triton_ascend.dynamic_cv_pipeline.rc = 2 : i32
// DECLINE-LABEL: func.func @sliced_accumulation
// DECLINE: scf.for
// DECLINE: tensor.extract_slice
// DECLINE: tensor.insert_slice
module {
  func.func @matmul_anchor(%arg0: memref<16x16xf16>,
                           %arg1: memref<16x16xf16>,
                           %arg2: memref<16x16xf32>) {
    linalg.matmul
      ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x16xf16>)
      outs(%arg2 : memref<16x16xf32>)
    return
  }

  func.func @sliced_accumulation(%arg0: tensor<64x256xf32>)
      -> tensor<64x256xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %acc = scf.for %i = %c0 to %c4 step %c1
        iter_args(%tile = %arg0) -> (tensor<64x256xf32>) : i32 {
      %off = arith.muli %i, %c16_i32 : i32
      %off_i = arith.index_cast %off : i32 to index
      %row = tensor.extract_slice %tile[%off_i, 0] [16, 256] [1, 1]
          : tensor<64x256xf32> to tensor<16x256xf32>
      %upd = arith.addf %row, %row : tensor<16x256xf32>
      %next = tensor.insert_slice %upd into %tile[%off_i, 0] [16, 256] [1, 1]
          : tensor<16x256xf32> into tensor<64x256xf32>
      scf.yield %next : tensor<64x256xf32>
    }
    return %acc : tensor<64x256xf32>
  }
}
