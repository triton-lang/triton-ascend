// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=false' --verify-each | FileCheck %s --check-prefix=OTHER
//
// Reduced from one byte-selection round of _radix_topk_threshold, BLOCK_R=8192.
// Preserve its u32 key extraction, zero parking and bin-zero correction.
// All four radix rounds use this structure. This is not an end-to-end kernel.
//
// CHECK-LABEL: tt.func @radix_topk_threshold_round(
// CHECK: arith.constant dense<-1> : tensor<8192xi32>
// CHECK: arith.shrui {{.*}} : tensor<8192xi32>
// CHECK-NOT: tt.reduce
// CHECK: tt.histogram
// CHECK-NOT: tt.reduce
// CHECK-NOT: arith.subi
// CHECK: tt.return
// OTHER-LABEL: tt.func @radix_topk_threshold_round(
// OTHER: arith.shrui {{.*}} : tensor<8192xi32>
// OTHER: tt.histogram
// OTHER: tt.reduce
// OTHER: arith.subi
tt.func @radix_topk_threshold_round(%keys: tensor<8192xi64>, %live: tensor<8192xi1>) -> tensor<256xi32> {
  %u32mask = arith.constant dense<4294967295> : tensor<8192xi64>
  %shift = arith.constant dense<24> : tensor<8192xi64>
  %byte_mask = arith.constant dense<255> : tensor<8192xi64>
  %zero = arith.constant dense<0> : tensor<8192xi32>
  %true = arith.constant dense<true> : tensor<8192xi1>
  %zero_bins = arith.constant dense<0> : tensor<256xi32>
  %bounded = arith.andi %keys, %u32mask : tensor<8192xi64>
  %shifted = arith.shrui %bounded, %shift : tensor<8192xi64>
  %byte = arith.andi %shifted, %byte_mask : tensor<8192xi64>
  %indices = arith.trunci %byte : tensor<8192xi64> to tensor<8192xi32>
  %parked = arith.select %live, %indices, %zero : tensor<8192xi1>, tensor<8192xi32>
  %hist = tt.histogram %parked : tensor<8192xi32> -> tensor<256xi32>
  %invalid = arith.xori %live, %true : tensor<8192xi1>
  %counts = arith.extui %invalid : tensor<8192xi1> to tensor<8192xi32>
  %count = "tt.reduce"(%counts) ({
  ^bb0(%a: i32, %b: i32):
    %sum = arith.addi %a, %b : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<8192xi32>) -> i32
  %splat = tt.splat %count : i32 -> tensor<256xi32>
  %bins = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %iszero = arith.cmpi eq, %bins, %zero_bins : tensor<256xi32>
  %correction = arith.select %iszero, %splat, %zero_bins : tensor<256xi1>, tensor<256xi32>
  %result = arith.subi %hist, %correction : tensor<256xi32>
  tt.return %result : tensor<256xi32>
}

// CHECK-LABEL: tt.func @keep_signed_shift(
// CHECK: arith.shrsi {{.*}} : tensor<8192xi64>
// OTHER-LABEL: tt.func @keep_signed_shift(
// OTHER: arith.shrsi {{.*}} : tensor<8192xi64>
tt.func @keep_signed_shift(%keys: tensor<8192xi64>) -> tensor<8192xi64> {
  %shift = arith.constant dense<24> : tensor<8192xi64>
  %result = arith.shrsi %keys, %shift : tensor<8192xi64>
  tt.return %result : tensor<8192xi64>
}
