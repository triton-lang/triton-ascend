// RUN: triton-opt %s -triton-to-linalg="compile-on-910-95=true" -verify-each -split-input-file | FileCheck %s

// Packed allocations need 32B alignment, including complete rows.
// CHECK-LABEL: func.func @packed_64
// CHECK: memref.alloc() {alignment = 32 : i64} : memref<128xi8>
// CHECK: hfusion.packed_mask_view
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  tt.func public @packed_64(%src: !tt.ptr<i8>) -> tensor<4x64xi1> {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %rows = arith.constant 4 : i64
    %cols = arith.constant 8 : i64
    %ptr = tt.make_tensor_ptr %src, [%rows, %cols], [%cols, %one], [%zero, %zero] {order = array<i32: 1, 0>} : <tensor<4x8xi8>>
    %bytes = tt.load %ptr : !tt.ptr<tensor<4x8xi8>>
    %bits = hfusion.packed_mask_view %bytes : tensor<4x8xi8> -> tensor<4x64xi1>
    tt.return %bits : tensor<4x64xi1>
  }
}

// -----

// Packed allocations need 32B alignment, including complete rows.
// CHECK-LABEL: func.func @packed_128
// CHECK: memref.alloc() {alignment = 32 : i64} : memref<128xi8>
// CHECK: hfusion.packed_mask_view
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  tt.func public @packed_128(%src: !tt.ptr<i8>) -> tensor<4x128xi1> {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %rows = arith.constant 4 : i64
    %cols = arith.constant 16 : i64
    %ptr = tt.make_tensor_ptr %src, [%rows, %cols], [%cols, %one], [%zero, %zero] {order = array<i32: 1, 0>} : <tensor<4x16xi8>>
    %bytes = tt.load %ptr : !tt.ptr<tensor<4x16xi8>>
    %bits = hfusion.packed_mask_view %bytes : tensor<4x16xi8> -> tensor<4x128xi1>
    tt.return %bits : tensor<4x128xi1>
  }
}

// -----

// Packed allocations need 32B alignment, including complete rows.
// CHECK-LABEL: func.func @packed_256
// CHECK: memref.alloc() {alignment = 32 : i64} : memref<4x32xi8>
// CHECK: hfusion.packed_mask_view
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  tt.func public @packed_256(%src: !tt.ptr<i8>) -> tensor<4x256xi1> {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %rows = arith.constant 4 : i64
    %cols = arith.constant 32 : i64
    %ptr = tt.make_tensor_ptr %src, [%rows, %cols], [%cols, %one], [%zero, %zero] {order = array<i32: 1, 0>} : <tensor<4x32xi8>>
    %bytes = tt.load %ptr : !tt.ptr<tensor<4x32xi8>>
    %bits = hfusion.packed_mask_view %bytes : tensor<4x32xi8> -> tensor<4x256xi1>
    tt.return %bits : tensor<4x256xi1>
  }
}

// -----

// Packed allocations need 32B alignment, including complete rows.
// CHECK-LABEL: func.func @packed_512
// CHECK: memref.alloc() {alignment = 32 : i64} : memref<4x64xi8>
// CHECK: hfusion.packed_mask_view
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  tt.func public @packed_512(%src: !tt.ptr<i8>) -> tensor<4x512xi1> {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %rows = arith.constant 4 : i64
    %cols = arith.constant 64 : i64
    %ptr = tt.make_tensor_ptr %src, [%rows, %cols], [%cols, %one], [%zero, %zero] {order = array<i32: 1, 0>} : <tensor<4x64xi8>>
    %bytes = tt.load %ptr : !tt.ptr<tensor<4x64xi8>>
    %bits = hfusion.packed_mask_view %bytes : tensor<4x64xi8> -> tensor<4x512xi1>
    tt.return %bits : tensor<4x512xi1>
  }
}

// -----

// Ordinary byte loads retain their existing allocation policy.
// CHECK-LABEL: func.func @ordinary_bytes
// CHECK: memref.alloc() : memref<4x32xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  tt.func public @ordinary_bytes(%src: !tt.ptr<i8>) -> tensor<4x32xi8> {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %rows = arith.constant 4 : i64
    %cols = arith.constant 32 : i64
    %ptr = tt.make_tensor_ptr %src, [%rows, %cols], [%cols, %one], [%zero, %zero] {order = array<i32: 1, 0>} : <tensor<4x32xi8>>
    %bytes = tt.load %ptr : !tt.ptr<tensor<4x32xi8>>
    tt.return %bytes : tensor<4x32xi8>
  }
}
