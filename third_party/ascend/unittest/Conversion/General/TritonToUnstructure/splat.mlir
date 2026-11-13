// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s

// CHECK-LABEL: tt.func @test_unstructure_splatandloadscenario
// CHECK: %[[TEN:.*]] = arith.constant 10 : i64
// CHECK-NOT: scf.for
// CHECK: %[[PTR:.*]] = tt.addptr %{{.*}}, %[[TEN]] : !tt.ptr<f32>, i64
// CHECK: %[[VAL1:.*]] = tt.load %[[PTR]] : !tt.ptr<f32>
// CHECK: %[[VAL2:.*]] = tt.splat %[[VAL1]] : f32 -> tensor<128xf32>
tt.func @test_unstructure_splatandloadscenario(%base: !tt.ptr<f32>) -> tensor<128xf32> {
    %offset = arith.constant 10 : i64
    %offset_tensor = tt.splat %offset : i64 -> tensor<128xi64>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %base_tensor, %offset_tensor : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
    %val = tt.load %ptr : tensor<128x!tt.ptr<f32>>
    tt.return %val : tensor<128xf32>
}
