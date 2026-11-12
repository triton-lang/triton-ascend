// RUN: triton-opt %s -triton-combine | FileCheck %s


// CHECK-LABEL: @test_combine_broadcast_mul_reduce
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
// CHECK: %[[RES:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[CST]] : tensor<32x16xf32> * tensor<16x32xf32> -> tensor<32x32xf32>
// CHECK-NOT: inputPrecision = tf32
// CHECK: tt.return %[[RES]] : tensor<32x32xf32>
tt.func @test_combine_broadcast_mul_reduce(%arg0: tensor<32x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<32x32xf32> {
    %0 = tt.expand_dims %arg0 {axis = 2 : i32} : tensor<32x16xf32> -> tensor<32x16x1xf32>
    %1 = tt.broadcast %0 : tensor<32x16x1xf32> -> tensor<32x16x32xf32>
    %2 = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<16x32xf32> -> tensor<1x16x32xf32>
    %3 = tt.broadcast %2 : tensor<1x16x32xf32> -> tensor<32x16x32xf32>
    %4 = arith.mulf %1, %3 : tensor<32x16x32xf32>
    %5 = "tt.reduce"(%4) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %6 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %6 : f32
    }) : (tensor<32x16x32xf32>) -> tensor<32x32xf32>
    tt.return %5 : tensor<32x32xf32>
}
