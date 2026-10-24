// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,A5
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,A3

// Mixed FP4/FP8 must decode the packed bytes before scale/NaN processing.
// CHECK-LABEL: func.func @mixed_fp4_fp8
// CHECK-NOT: hfusion.matmul_mx
// CHECK: arith.bitcast {{.*}} : tensor<16x16xi32> to tensor<16x16xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<16x16x2xf32> into tensor<16x32xf32>
// CHECK: arith.extf {{.*}} : tensor<32x32xf8E5M2> to tensor<32x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-SAME: tensor<16x32xf32>, tensor<32x32xf32>
// CHECK-NOT: tt.dot_scaled
func.func @mixed_fp4_fp8(%a: tensor<16x16xi8>, %b: tensor<32x32xf8E5M2>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e5m2 {fastMath = false} : tensor<16x16xi8>, tensor<16x1xi8> * tensor<32x32xf8E5M2>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// M/N packing doubles the non-K axes, not K. No implicit E8M0 scale is needed.
// CHECK-LABEL: func.func @fp4_non_k_packed
// CHECK-NOT: hfusion.matmul_mx
// CHECK: tensor.collapse_shape {{.*}} : tensor<8x2x32xf32> into tensor<16x32xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<32x16x2xf32> into tensor<32x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-SAME: tensor<16x32xf32>, tensor<32x32xf32>
func.func @fp4_non_k_packed(%a: tensor<8x32xi8>, %b: tensor<32x16xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = false, rhs_k_pack = false} : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// NVFP4 uses numerical E4M3 scales repeated in groups of 16, not E8M0 bits.
// CHECK-LABEL: func.func @nvfp4
// CHECK-NOT: hfusion.matmul_mx
// CHECK: arith.extf {{.*}} : tensor<16x2xf8E4M3FN> to tensor<16x2xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<16x2x16xf32> into tensor<16x32xf32>
// CHECK: arith.extf {{.*}} : tensor<32x2xf8E4M3FN> to tensor<32x2xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<32x2x16xf32> into tensor<32x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
func.func @nvfp4(%a: tensor<16x16xi8>, %b: tensor<16x32xi8>, %sa: tensor<16x2xf8E4M3FN>, %sb: tensor<32x2xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x16xi8>, tensor<16x2xf8E4M3FN> * tensor<16x32xi8>, tensor<32x2xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Half operands scale in f32, preserving their products and FP32 accumulators.
// E8M0 encodings 0 and 255 map to 2^-127 and NaN respectively.
// CHECK-LABEL: func.func @mixed_half_scales
// CHECK-DAG: arith.constant 4194304 : i32
// CHECK-DAG: arith.constant 2143289344 : i32
// CHECK: arith.extf {{.*}} : tensor<16x32xf16> to tensor<16x32xf32>
// CHECK: arith.mulf {{.*}} : tensor<16x32xf32>
// CHECK: arith.extf {{.*}} : tensor<32x32xbf16> to tensor<32x32xf32>
// CHECK: arith.mulf {{.*}} : tensor<32x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
func.func @mixed_half_scales(%a: tensor<16x32xf16>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = bf16 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xbf16>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Batched packed operands preserve the batch axis and each batch accumulator.
// CHECK-LABEL: func.func @batched_fp4
// CHECK: scf.for {{.*}} : i32
// CHECK: arith.index_cast {{.*}} : i32 to index
// CHECK: tensor.extract_slice {{.*}} : tensor<2x16x32xf32> to tensor<16x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: tensor.insert_slice {{.*}} : tensor<16x32xf32> into tensor<2x16x32xf32>
// CHECK-NOT: linalg.batch_matmul
func.func @batched_fp4(%a: tensor<2x16x16xi8>, %b: tensor<2x16x32xi8>, %sa: tensor<2x16x1xi8>, %sb: tensor<2x32x1xi8>, %c: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<2x16x16xi8>, tensor<2x16x1xi8> * tensor<2x16x32xi8>, tensor<2x32x1xi8> -> tensor<2x16x32xf32>
  return %d : tensor<2x16x32xf32>
}

// -----

// Large M/N-packed inputs are sliced before decoding; no full f32 input tile.
// CHECK-LABEL: func.func @large_non_k_packed
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<128x256xf32> to tensor<32x32xf32>
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x256xi8> to tensor<16x32xi8>
// CHECK: tensor.extract_slice {{.*}} : tensor<256x128xi8> to tensor<32x16xi8>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-SAME: tensor<32x32xf32>, tensor<32x32xf32>
// A5: tensor.insert_slice
// A3: memref.subview
// A3: bufferization.materialize_in_destination
// A3: bufferization.to_tensor
// CHECK-NOT: tt.dot_scaled
func.func @large_non_k_packed(%a: tensor<64x256xi8>, %b: tensor<256x128xi8>, %sa: tensor<128x8xi8>, %sb: tensor<256x8xi8>, %c: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = false, rhs_k_pack = false} : tensor<64x256xi8>, tensor<128x8xi8> * tensor<256x128xi8>, tensor<256x8xi8> -> tensor<128x256xf32>
  return %d : tensor<128x256xf32>
}
