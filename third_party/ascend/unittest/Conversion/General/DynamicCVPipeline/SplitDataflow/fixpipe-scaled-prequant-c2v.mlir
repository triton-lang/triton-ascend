// RUN: split-file %s %t
// RUN: triton-opt %t/double.mlir --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop --separate-cv-scope --add_multi_buffer_outer_scope --clone-ops --verify-each > %t/double.out
// RUN: FileCheck %s --check-prefix=DOUBLE --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc < %t/double.out
// RUN: FileCheck %s --check-prefix=RECEIVER < %t/double.out
// RUN: triton-opt %s --split-input-file --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --verify-each | FileCheck %s --implicit-check-not=tensor.insert_slice --implicit-check-not=arith.mulf --implicit-check-not=arith.truncf --implicit-check-not=hivm.hir.convert_layout
// RUN: triton-opt %s --split-input-file --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --canonicalize --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// RUN: sed '/^    %nz = hivm.hir.convert_layout %value/i\    %observer = arith.negf %value {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x256xf32>' %s > %t.shared
// RUN: triton-opt %t.shared --split-input-file --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync | FileCheck %s --check-prefix=SHARED

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
//--- direct.mlir
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

// -----

// Three unequal ND pieces share one allocation pair, including double buffering.
// CHECK-LABEL: func.func @concat_three_transfers
// CHECK-COUNT-2: memref.alloc() {{.*}} : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK: memref.subview {{.*}}[0, 0] [32, 32] [1, 1]
// CHECK: hivm.hir.fixpipe
// CHECK: memref.subview {{.*}}[0, 32] [32, 64] [1, 1]
// CHECK: hivm.hir.fixpipe
// CHECK: memref.subview {{.*}}[0, 96] [32, 32] [1, 1]
// CHECK: hivm.hir.fixpipe
// CHECK: memref.memory_space_cast
// CHECK-SAME: ssbuffer.crossCoreDeps = [{{.*}}, 0 : i32, {{.*}}, 0 : i32, {{.*}}, 0 : i32]
// CHECK: bufferization.to_tensor {{.*}} : memref<32x128xf32>
// DOUBLE-LABEL: func.func @concat_three_transfers
// DOUBLE-COUNT-4: memref.alloc() {{.*}} : memref<32x128xf32, #hivm.address_space<ub>>
// DOUBLE: arith.select {{.*}} : memref<
// DOUBLE-COUNT-3: hivm.hir.fixpipe
// DOUBLE-NOT: hivm.hir.fixpipe
// RECEIVER-LABEL: func.func @concat_three_transfers
// RECEIVER: scf.if {{.*}} -> (tensor<32x128xf32>)
// RECEIVER: ssbuffer.crossCoreDeps = [{{.*}}, 0 : i32, {{.*}}, 0 : i32, {{.*}}, 0 : i32]
//--- double.mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
  func.func @concat_three_transfers(%a: tensor<32x16xf32>, %b: tensor<16x32xf32>, %c: tensor<16x64xf32>, %out: memref<32x128xf32>, %n: i32) {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %empty0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
      %init0 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f32) outs(%empty0 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %empty1 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x64xf32>
      %init1 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f32) outs(%empty1 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %x = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<32x16xf32>, tensor<16x32xf32>) outs(%init0 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %y = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %c : tensor<32x16xf32>, tensor<16x64xf32>) outs(%init1 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %z = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<32x16xf32>, tensor<16x32xf32>) outs(%init0 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
      %r0 = tensor.insert_slice %x into %empty[0, 0] [32, 32] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> into tensor<32x128xf32>
      %r1 = tensor.insert_slice %y into %r0[0, 32] [32, 64] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x64xf32> into tensor<32x128xf32>
      %r2 = tensor.insert_slice %z into %r1[0, 96] [32, 32] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> into tensor<32x128xf32>
      %value = arith.negf %r2 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
      hivm.hir.copy ins(%value : tensor<32x128xf32>) outs(%out : memref<32x128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
    }
    return
  }
}

// -----

//--- nz.mlir
// PRE quantization and NZ geometry remain attached to each original transfer.
// CHECK-LABEL: func.func @concat_prequant_nz
// CHECK-COUNT-2: memref.alloc() {{.*}} : memref<4x2x16x16xf16, #hivm.address_space<ub>>
// CHECK: memref.subview {{.*}}[0, 0, 0, 0] [2, 2, 16, 16]
// CHECK: hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>
// CHECK-SAME: quant_scale = %{{.*}} : f32
// CHECK: memref.subview {{.*}}[2, 0, 0, 0] [2, 2, 16, 16]
// CHECK: hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>
// CHECK-SAME: quant_scale = %{{.*}} : f32
// CHECK: bufferization.to_tensor {{.*}} : memref<4x2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @concat_prequant_nz(%a: tensor<32x16xf8E4M3FN>, %b: tensor<16x32xf8E4M3FN>, %c: tensor<16x32xf8E4M3FN>, %scale: f32) -> tensor<4x2x16x16xf16> {
    %zero = arith.constant 0.0 : f32
    %empty0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f32) outs(%empty0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %scale_empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2x16x16xf32>
    %scale_tensor = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%scale : f32) outs(%scale_empty : tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32>
    %x = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<32x16xf8E4M3FN>, tensor<16x32xf8E4M3FN>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %nx = hivm.hir.convert_layout %x output_shape [2, 2, 16, 16] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<32x32xf32>) -> tensor<2x2x16x16xf32>
    %sx = arith.mulf %nx, %scale_tensor {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2x16x16xf32>
    %hx = arith.truncf %sx {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2x16x16xf32> to tensor<2x2x16x16xf16>
    %y = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %c : tensor<32x16xf8E4M3FN>, tensor<16x32xf8E4M3FN>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %ny = hivm.hir.convert_layout %y output_shape [2, 2, 16, 16] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<32x32xf32>) -> tensor<2x2x16x16xf32>
    %sy = arith.mulf %ny, %scale_tensor {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2x16x16xf32>
    %hy = arith.truncf %sy {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2x16x16xf32> to tensor<2x2x16x16xf16>
    %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x2x16x16xf16>
    %r0 = tensor.insert_slice %hx into %empty[0, 0, 0, 0] [2, 2, 16, 16] [1, 1, 1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2x16x16xf16> into tensor<4x2x16x16xf16>
    %r1 = tensor.insert_slice %hy into %r0[2, 0, 0, 0] [2, 2, 16, 16] [1, 1, 1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2x16x16xf16> into tensor<4x2x16x16xf16>
    %value = arith.negf %r1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x2x16x16xf16>
    return %value : tensor<4x2x16x16xf16>
  }
}
