// RUN: triton-opt --op-classifier %s | FileCheck %s

// A FIXPIPE-compatible cast after the score layout conversion must stay on
// CUBE while its arbitrary consumer stays on VECTOR. The producer is
// intentionally multi-use: foldability is a property of this result edge, not
// of an entire single-use graph.

// CHECK-LABEL: func.func @fixpipe_output_cast
// CHECK: [[MM:%.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"}
// CHECK: arith.negf [[MM]] {ssbuffer.core_type = "VECTOR"}
// CHECK: [[NZ:%.*]] = hivm.hir.convert_layout [[MM]] {{.*}}ssbuffer.core_type = "CUBE"
// CHECK: [[HALF:%.*]] = arith.truncf [[NZ]] {ssbuffer.core_type = "CUBE"} : tensor<16x8x16x16xf32> to tensor<16x8x16x16xf16>
// CHECK-NOT: arith.truncf
// CHECK: arith.negf [[HALF]] {ssbuffer.core_type = "VECTOR"}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @fixpipe_output_cast(%q: tensor<128x128xf8E4M3FN>, %k: tensor<128x256xf8E4M3FN>) -> tensor<16x8x16x16xf16> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x256xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
    %mm = linalg.matmul ins(%q, %k : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%init : tensor<128x256xf32>) -> tensor<128x256xf32>
    %other_user = arith.negf %mm : tensor<128x256xf32>
    %nz = hivm.hir.convert_layout %mm output_shape [16, 8, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
    %half = arith.truncf %nz : tensor<16x8x16x16xf32> to tensor<16x8x16x16xf16>
    %consumer = arith.negf %half : tensor<16x8x16x16xf16>
    return %consumer : tensor<16x8x16x16xf16>
  }

  // CHECK-LABEL: func.func @scaled_after_layout
  // CHECK-SAME: %[[SCALE:[a-zA-Z0-9_]+]]: f32
  // CHECK: [[NZ:%.*]] = hivm.hir.convert_layout {{.*}}ssbuffer.core_type = "CUBE"
  // CHECK: [[SPLAT:%.*]] = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%[[SCALE]] : f32)
  // CHECK: [[SCALED:%.*]] = arith.mulf [[NZ]], [[SPLAT]] {ssbuffer.core_type = "CUBE"}
  // CHECK: [[HALF:%.*]] = arith.truncf [[SCALED]] {ssbuffer.core_type = "CUBE"
  // CHECK: arith.negf [[HALF]] {ssbuffer.core_type = "VECTOR"}
  func.func @scaled_after_layout(%q: tensor<128x128xf8E4M3FN>, %k: tensor<128x256xf8E4M3FN>, %scale: f32) -> tensor<16x8x16x16xf16> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x256xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
    %mm = linalg.matmul ins(%q, %k : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%init : tensor<128x256xf32>) -> tensor<128x256xf32>
    %nz = hivm.hir.convert_layout %mm output_shape [16, 8, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
    %scale_empty = tensor.empty() : tensor<16x8x16x16xf32>
    %scale_splat = linalg.fill ins(%scale : f32) outs(%scale_empty : tensor<16x8x16x16xf32>) -> tensor<16x8x16x16xf32>
    %scaled = arith.mulf %nz, %scale_splat : tensor<16x8x16x16xf32>
    %half = arith.truncf %scaled : tensor<16x8x16x16xf32> to tensor<16x8x16x16xf16>
    %consumer = arith.negf %half : tensor<16x8x16x16xf16>
    return %consumer : tensor<16x8x16x16xf16>
  }

  // CHECK-LABEL: func.func @scaled_before_layout
  // CHECK-SAME: %[[SCALE:[a-zA-Z0-9_]+]]: f32
  // CHECK: [[SPLAT:%.*]] = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%[[SCALE]] : f32)
  // CHECK: [[SCALED:%.*]] = arith.mulf {{%.*}}, [[SPLAT]] {ssbuffer.core_type = "CUBE"}
  // CHECK: [[NZ:%.*]] = hivm.hir.convert_layout [[SCALED]] {{.*}}ssbuffer.core_type = "CUBE"
  // CHECK: [[HALF:%.*]] = arith.truncf [[NZ]] {ssbuffer.core_type = "CUBE"
  // CHECK: arith.negf [[HALF]] {ssbuffer.core_type = "VECTOR"}
  func.func @scaled_before_layout(%q: tensor<128x128xf8E4M3FN>, %k: tensor<128x256xf8E4M3FN>, %scale: f32) -> tensor<16x8x16x16xf16> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x256xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
    %mm = linalg.matmul ins(%q, %k : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%init : tensor<128x256xf32>) -> tensor<128x256xf32>
    %scale_empty = tensor.empty() : tensor<128x256xf32>
    %scale_splat = linalg.fill ins(%scale : f32) outs(%scale_empty : tensor<128x256xf32>) -> tensor<128x256xf32>
    %scaled = arith.mulf %mm, %scale_splat : tensor<128x256xf32>
    %nz = hivm.hir.convert_layout %scaled output_shape [16, 8, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
    %half = arith.truncf %nz : tensor<16x8x16x16xf32> to tensor<16x8x16x16xf16>
    %consumer = arith.negf %half : tensor<16x8x16x16xf16>
    return %consumer : tensor<16x8x16x16xf16>
  }
}
