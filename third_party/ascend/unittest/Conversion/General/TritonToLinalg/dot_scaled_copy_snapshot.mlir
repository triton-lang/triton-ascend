// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @plain_copy
// CHECK-NOT: memref.alloc() : memref<64x64xf16>
// CHECK: scf.for
// CHECK: memref.alloc() : memref<32x32xf16>
// CHECK: memref.copy
// CHECK: linalg.matmul
func.func @plain_copy(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output {isVolatile = false} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @alias_copy
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
func.func @alias_copy(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output {isVolatile = false} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  memref.store %v, %src[%zero] : memref<?xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @shared_copy
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
func.func @shared_copy(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> (tensor<16x64xf32>, f16) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output {isVolatile = false} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  %snapshot = tensor.extract %b[%zero, %zero] : tensor<64x64xf16>
  return %d, %snapshot : tensor<16x64xf32>, f16
}

// -----

// CHECK-LABEL: func.func @escape_copy
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
func.func private @escape(memref<64x64xf16>)
func.func @escape_copy(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output {isVolatile = false} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  func.call @escape(%buf) : (memref<64x64xf16>) -> ()
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @extra_write_copy
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
func.func @extra_write_copy(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output {isVolatile = false} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.store %v, %buf[%r, %zero] : memref<64x64xf16>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @row_volatile
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
func.func @row_volatile(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output {isVolatile = true} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @row_unmarked
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
func.func @row_unmarked(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @row_external_write
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
func.func private @external_write(memref<?xf16>)
func.func @row_external_write(%src: memref<?xf16>, %a: tensor<16x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 64 : index
  %v = arith.constant 1.0 : f16
  %buf = memref.alloc() : memref<64x64xf16>
  scf.for %r = %zero to %end step %one {
    %offset = arith.muli %r, %end : index
    %input = memref.reinterpret_cast %src to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    %output = memref.subview %buf[%r, 0] [1, 64] [1, 1] : memref<64x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    memref.copy %input, %output {isVolatile = false} : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  } {ExtractedLoadOrStore, hivm.parallel_loop}
  func.call @external_write(%src) : (memref<?xf16>) -> ()
  %b = bufferization.to_tensor %buf restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
