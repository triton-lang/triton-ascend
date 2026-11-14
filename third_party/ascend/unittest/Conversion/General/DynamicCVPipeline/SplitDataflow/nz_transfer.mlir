// RUN: triton-opt %s --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --canonicalize | FileCheck %s
// Transfer FP16 NZ scores and reuse the NZ dot operand across core boundaries.
// CHECK-LABEL: func.func @nz_transfer
// CHECK: hivm.hir.fixpipe
// CHECK-SAME: memref<16x4x16x16xf16, #hivm.address_space<ub>>
// CHECK-NOT: linalg.transpose
// CHECK: hivm.hir.copy
// CHECK-SAME: tensor<16x4x16x16xf16>
// CHECK-NOT: linalg.transpose
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nz_transfer(%q: tensor<64x64xf16>, %k: tensor<64x256xf16>, %v: tensor<256x64xf16>) {
    %empty = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x256xf32>
    %s = linalg.matmul {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<64x64xf16>, tensor<64x256xf16>) outs(%empty : tensor<64x256xf32>) -> tensor<64x256xf32>
    %nz = hivm.hir.convert_layout %s output_shape [16, 4, 16, 16] {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE", dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<64x256xf32>) -> tensor<16x4x16x16xf32>
    %p = arith.truncf %nz {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x4x16x16xf32> to tensor<16x4x16x16xf16>
    %nd = hivm.hir.convert_layout %p output_shape [64, 256] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, dstLayout = #hivm.data_layout<ND>} : (tensor<16x4x16x16xf16>) -> tensor<64x256xf16>
    %out = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
    %pv = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%nd, %v : tensor<64x256xf16>, tensor<256x64xf16>) outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
    return
  }
}
