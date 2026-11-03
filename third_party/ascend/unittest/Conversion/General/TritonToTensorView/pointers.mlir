// RUN: triton-opt %s -triton-to-tensor-view -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @pointer_addresses
// CHECK: %[[BASE:.*]] = tt.ptr_to_int %{{.*}} : !tt.ptr<f32> -> i64
// CHECK: %[[SPLAT:.*]] = tt.splat %[[BASE]] : i64 -> tensor<2x3xi64>
// CHECK: %[[WIDE:.*]] = arith.extsi %{{.*}} : tensor<2x3xi32> to tensor<2x3xi64>
// CHECK: %[[SCALED:.*]] = arith.muli %[[WIDE]], %{{.*}} : tensor<2x3xi64>
// CHECK: %[[ADDRESSES:.*]] = arith.addi %[[SPLAT]], %[[SCALED]] : tensor<2x3xi64>
// CHECK: tv.ptr_store %[[ADDRESSES]],
func.func @pointer_addresses(%base: !tt.ptr<f32>, %offsets: tensor<2x3xi32>, %data: tensor<4xf32>) {
  %rows = arith.constant dense<[0, 0, 1, 1]> : tensor<4xindex>
  %cols = arith.constant dense<[0, 2, 1, 2]> : tensor<4xindex>
  %splat = tt.splat %base : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %pointers = tt.addptr %splat, %offsets : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %addresses = tt.ptr_to_int %pointers : tensor<2x3x!tt.ptr<f32>> -> tensor<2x3xi64>
  tv.ptr_store %addresses, %data, indices = [%rows, %cols] : tensor<2x3xi64>, tensor<4xf32>, tensor<4xindex>, tensor<4xindex>
  return
}

// CHECK-LABEL: func.func @selected_addresses
// CHECK: %[[SELECTED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi64>
// CHECK: tv.ptr_store %[[SELECTED]],
func.func @selected_addresses(%a: tensor<4xi64>, %b: tensor<4xi64>, %condition: tensor<4xi1>, %data: tensor<4xf32>) {
  %indices = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
  %pa = tt.int_to_ptr %a : tensor<4xi64> -> tensor<4x!tt.ptr<f32>>
  %pb = tt.int_to_ptr %b : tensor<4xi64> -> tensor<4x!tt.ptr<f32>>
  %pointers = arith.select %condition, %pa, %pb : tensor<4xi1>, tensor<4x!tt.ptr<f32>>
  %addresses = tt.ptr_to_int %pointers : tensor<4x!tt.ptr<f32>> -> tensor<4xi64>
  tv.ptr_store %addresses, %data, indices = [%indices] : tensor<4xi64>, tensor<4xf32>, tensor<4xindex>
  return
}
