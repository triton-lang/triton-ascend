// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @plain
// CHECK-NOT: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: scf.for
// CHECK: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<32x32xf16>
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = false}
func.func @plain(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = false} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
    %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @alias_store
// CHECK: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: {{(func\.)?}}call @triton_indirect_load
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = false}
func.func @alias_store(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = false} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
    memref.store %one, %src[%zero] : memref<?xf16>
    %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @shared
// CHECK: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: {{(func\.)?}}call @triton_indirect_load
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = false}
func.func @shared(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, f16) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = false} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
    %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    %snapshot = tensor.extract %b[%zero, %zero] : tensor<64x64xf16>
    return %d, %snapshot : tensor<16x64xf32>, f16
}

// -----

// CHECK-LABEL: func.func @volatile
// CHECK: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: {{(func\.)?}}call @triton_indirect_load
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = true}
func.func @volatile(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = true} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
    %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @unknown
// CHECK: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: {{(func\.)?}}call @triton_indirect_load
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = false}
func.func private @external_write(memref<?xf16>)
func.func @unknown(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = false} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
    func.call @external_write(%src) : (memref<?xf16>) -> ()
    %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @atomic
// CHECK: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: {{(func\.)?}}call @triton_indirect_load
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = false}
func.func @atomic(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = false} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
    %old = memref.atomic_rmw addf %one, %src[%zero] : (f16, memref<?xf16>) -> f16
    %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    return %d : tensor<16x64xf32>
}

// -----


// CHECK-LABEL: func.func @shared_cast
// CHECK: {{(func\.)?}}call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: {{(func\.)?}}call @triton_indirect_load
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = false}
func.func @shared_cast(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, f16) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = false} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
    %neg = arith.negf %b : tensor<64x64xf16>
    %d = tt.dot_scaled %a, %neg, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    %snapshot = tensor.extract %neg[%zero, %zero] : tensor<64x64xf16>
    return %d, %snapshot : tensor<16x64xf32>, f16
}

// -----

// CHECK-LABEL: func.func @recursive_alias_store
// CHECK: call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// CHECK: linalg.generic
// CHECK: scf.for
// CHECK-NOT: call @triton_indirect_load
func.func private @triton_indirect_load(memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16> attributes {isVolatile = false}
func.func @recursive_alias_store(%src: memref<?xf16>, %offsets: tensor<64x64xi64>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %b = func.call @triton_indirect_load(%src, %offsets) {isVolatile = false} : (memref<?xf16>, tensor<64x64xi64>) -> tensor<64x64xf16>
  %fresh = memref.alloc() : memref<1xf16>
  linalg.generic {indexing_maps = [affine_map<(i) -> (i)>], iterator_types = ["parallel"]} outs(%fresh : memref<1xf16>) {
  ^bb0(%v: f16):
    memref.store %one, %src[%zero] : memref<?xf16>
    linalg.yield %one : f16
  }
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
