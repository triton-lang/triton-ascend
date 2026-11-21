// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s --implicit-check-not="arith.cmpi uge" --implicit-check-not="arith.cmpi ugt"

// -----

// CHECK-LABEL: func.func @tf32x3_matrix
// CHECK-NOT: tt.dot
// CHECK: arith.cmpi sge
// CHECK-COUNT-3: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: return
func.func @tf32x3_matrix(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = tf32x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @tf32x3_batch
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK-NOT: tt.dot
// CHECK: arith.cmpi sge
// CHECK-COUNT-3: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: return
func.func @tf32x3_batch(%a: tensor<2x16x16xf32>, %b: tensor<2x16x16xf32>, %c: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = tf32x3 : tensor<2x16x16xf32> * tensor<2x16x16xf32> -> tensor<2x16x16xf32>
  return %d : tensor<2x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @bf16x3_matrix
// CHECK-NOT: tt.dot
// CHECK: arith.cmpi sgt
// CHECK-COUNT-3: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: return
func.func @bf16x3_matrix(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @bf16x3_batch
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK-NOT: tt.dot
// CHECK: arith.cmpi sgt
// CHECK-COUNT-3: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: return
func.func @bf16x3_batch(%a: tensor<2x16x16xf32>, %b: tensor<2x16x16xf32>, %c: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x3 : tensor<2x16x16xf32> * tensor<2x16x16xf32> -> tensor<2x16x16xf32>
  return %d : tensor<2x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @bf16x6_matrix
// CHECK-NOT: tt.dot
// CHECK: arith.cmpi sgt
// CHECK-COUNT-6: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: return
func.func @bf16x6_matrix(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x6 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @bf16x6_batch
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK-NOT: tt.dot
// CHECK: arith.cmpi sgt
// CHECK-COUNT-6: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: return
func.func @bf16x6_batch(%a: tensor<2x16x16xf32>, %b: tensor<2x16x16xf32>, %c: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x6 : tensor<2x16x16xf32> * tensor<2x16x16xf32> -> tensor<2x16x16xf32>
  return %d : tensor<2x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @f16_tf32x3
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @f16_tf32x3(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = tf32x3 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @f16_bf16x3
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @f16_bf16x3(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x3 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @f16_bf16x6
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @f16_bf16x6(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x6 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
  return %d : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @f64_tf32x3
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @f64_tf32x3(%a: tensor<16x16xf64>, %b: tensor<16x16xf64>, %c: tensor<16x16xf64>) -> tensor<16x16xf64> {
  %d = tt.dot %a, %b, %c, inputPrecision = tf32x3 : tensor<16x16xf64> * tensor<16x16xf64> -> tensor<16x16xf64>
  return %d : tensor<16x16xf64>
}

// -----

// CHECK-LABEL: func.func @f64_bf16x3
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @f64_bf16x3(%a: tensor<16x16xf64>, %b: tensor<16x16xf64>, %c: tensor<16x16xf64>) -> tensor<16x16xf64> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x3 : tensor<16x16xf64> * tensor<16x16xf64> -> tensor<16x16xf64>
  return %d : tensor<16x16xf64>
}

// -----

// CHECK-LABEL: func.func @f64_bf16x6
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @f64_bf16x6(%a: tensor<16x16xf64>, %b: tensor<16x16xf64>, %c: tensor<16x16xf64>) -> tensor<16x16xf64> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x6 : tensor<16x16xf64> * tensor<16x16xf64> -> tensor<16x16xf64>
  return %d : tensor<16x16xf64>
}

// -----

// CHECK-LABEL: func.func @narrow_acc_tf32x3
// CHECK: arith.extf
// CHECK-COUNT-3: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: arith.truncf
// CHECK: return
func.func @narrow_acc_tf32x3(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %d = tt.dot %a, %b, %c, inputPrecision = tf32x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf16>
  return %d : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @narrow_acc_bf16x3
// CHECK: arith.extf
// CHECK-COUNT-3: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: arith.truncf
// CHECK: return
func.func @narrow_acc_bf16x3(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf16>
  return %d : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @narrow_acc_bf16x6
// CHECK: arith.extf
// CHECK-COUNT-6: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.addf
// CHECK: arith.truncf
// CHECK: return
func.func @narrow_acc_bf16x6(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %d = tt.dot %a, %b, %c, inputPrecision = bf16x6 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf16>
  return %d : tensor<16x16xf16>
}
