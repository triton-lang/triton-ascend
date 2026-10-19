// RUN: triton-opt %s --split-input-file --data-dependency-analysis --inter-core-transfer-and-sync | FileCheck %s --implicit-check-not=arith.mulf --implicit-check-not=arith.truncf --implicit-check-not=hivm.hir.convert_layout
// RUN: triton-opt %s --split-input-file --data-dependency-analysis --inter-core-transfer-and-sync --canonicalize --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// RUN: sed '/^    %nz = hivm.hir.convert_layout %value/i\    %observer = arith.negf %value {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x256xf32>' %s > %t.shared
// RUN: triton-opt %t.shared --split-input-file --data-dependency-analysis --inter-core-transfer-and-sync | FileCheck %s --check-prefix=SHARED

// Fold a CUBE-resident matmul -> ND2NZ -> scalar scale -> f32-to-f16 cast
// into QF322F16_PRE. The scalar remains a real FIXPIPE operand; the temporary
// scale tensor and elementwise multiply disappear.

// CHECK-LABEL: func.func @fixpipe_scaled_prequant_c2v
// CHECK-SAME: %[[SCALE:[a-zA-Z0-9_]+]]: f32
// CHECK: memref.alloc() {{.*}} : memref<16x8x16x16xf16, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>
// CHECK-SAME: ins({{%.*}} : tensor<128x256xf32>)
// CHECK-SAME: outs({{%.*}} : memref<16x8x16x16xf16, #hivm.address_space<ub>>)
// CHECK-SAME: quant_scale = %[[SCALE]] : f32
// CHECK: arith.negf {{%.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x8x16x16xf16>
// GENERIC: "hivm.hir.fixpipe"({{%.*}}, {{%.*}}, %{{.*}}) <{dma_mode = #hivm.dma_mode<normal>, {{.*}}pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @fixpipe_scaled_prequant_c2v(%q: tensor<128x128xf8E4M3FN>, %k: tensor<128x256xf8E4M3FN>, %scale: f32) -> tensor<16x8x16x16xf16> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x256xf32>
    %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
    %mm = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%init : tensor<128x256xf32>) -> tensor<128x256xf32>
    %nz = hivm.hir.convert_layout %mm output_shape [16, 8, 16, 16] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
    %scale_empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x8x16x16xf32>
    %scale_splat = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%scale : f32) outs(%scale_empty : tensor<16x8x16x16xf32>) -> tensor<16x8x16x16xf32>
    %value = arith.mulf %nz, %scale_splat {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x8x16x16xf32>
    %half = arith.truncf %value {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x8x16x16xf32> to tensor<16x8x16x16xf16>
    %consumer = arith.negf %half {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x8x16x16xf16>
    return %consumer : tensor<16x8x16x16xf16>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @fixpipe_scaled_before_layout
  // CHECK-SAME: %[[SCALE_BEFORE:[a-zA-Z0-9_]+]]: f32
  // CHECK: hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>
  // CHECK-SAME: quant_scale = %[[SCALE_BEFORE]] : f32
  // CHECK: arith.negf %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
  // SHARED-LABEL: func.func @fixpipe_scaled_before_layout
  // SHARED: %[[SCALED:.*]] = arith.mulf
  // SHARED: arith.negf %[[SCALED]] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
  // SHARED: hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>
  func.func @fixpipe_scaled_before_layout(%q: tensor<128x128xf8E4M3FN>, %k: tensor<128x256xf8E4M3FN>, %scale: f32) -> tensor<16x8x16x16xf16> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x256xf32>
    %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
    %mm = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%init : tensor<128x256xf32>) -> tensor<128x256xf32>
    %scale_empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x256xf32>
    %scale_splat = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%scale : f32) outs(%scale_empty : tensor<128x256xf32>) -> tensor<128x256xf32>
    %value = arith.mulf %mm, %scale_splat {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x256xf32>
    %nz = hivm.hir.convert_layout %value output_shape [16, 8, 16, 16] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
    %half = arith.truncf %nz {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x8x16x16xf32> to tensor<16x8x16x16xf16>
    %consumer = arith.negf %half {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x8x16x16xf16>
    return %consumer : tensor<16x8x16x16xf16>
  }
}
