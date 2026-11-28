// RUN: triton-opt %s -triton-expand-packed-mask-load -split-input-file | FileCheck %s

// CHECK-LABEL: @block_pointer
// CHECK: %[[BYTES:.*]] = tt.load %arg0 {{.*}} : !tt.ptr<tensor<32x16xi8>>
// CHECK: %[[BITS:.*]] = hfusion.packed_mask_view %[[BYTES]] : tensor<32x16xi8> -> tensor<32x128xi1>
// CHECK: tt.return %[[BITS]] : tensor<32x128xi1>
tt.func @block_pointer(%arg0: !tt.ptr<tensor<32x16xi8>>) -> tensor<32x128xi1> {
  %bits = "ascend.load_packed_mask"(%arg0) <{boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = true, operandSegmentSizes = array<i32: 1, 0, 0>, padding = 1 : i32}> : (!tt.ptr<tensor<32x16xi8>>) -> tensor<32x128xi1>
  tt.return %bits : tensor<32x128xi1>
}

// -----

// CHECK-LABEL: @masked_pointer
// CHECK: %[[BYTES:.*]] = tt.load %arg0, %arg1, %{{.*}} : tensor<32x32x!tt.ptr<i8>>
// CHECK: hfusion.packed_mask_view %[[BYTES]] : tensor<32x32xi8> -> tensor<32x256xi1>
tt.func @masked_pointer(%arg0: tensor<32x32x!tt.ptr<i8>>, %arg1: tensor<32x32xi1>) -> tensor<32x256xi1> {
  %zero = arith.constant dense<0> : tensor<32x32xi8>
  %bits = "ascend.load_packed_mask"(%arg0, %arg1, %zero) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<32x32x!tt.ptr<i8>>, tensor<32x32xi1>, tensor<32x32xi8>) -> tensor<32x256xi1>
  tt.return %bits : tensor<32x256xi1>
}
