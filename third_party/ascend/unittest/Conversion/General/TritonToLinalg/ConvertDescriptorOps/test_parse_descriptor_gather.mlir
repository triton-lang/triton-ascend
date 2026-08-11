// RUN: triton-opt --convert-descriptor-ops --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_descriptor_gather_rows
// CHECK-NOT: tt.descriptor_gather
// CHECK: %[[DIM:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<32xi32>
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[DIM]] step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<32x32xf32>)
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<32xi32>
// CHECK: %[[PTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}], [%[[EXTRACTED]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<1x32xf32>>
// CHECK: %[[ROW:.*]] = tt.load %[[PTR]] {DiscreteMemAccess, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<1x32xf32>>
// CHECK: tensor.insert_slice %[[ROW]] into %{{.*}}[%{{.*}}, 0] [1, 32] [1, 1] {DiscreteMemAccess} : tensor<1x32xf32> into tensor<32x32xf32>
// CHECK: scf.yield

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_descriptor_gather_rows(%in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %y: i32) attributes {noinline = false} {
    %desc = arith.constant 1 : i64
    %desc_0 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %desc_4 = tt.make_tensor_descriptor %in_ptr, [%c128_i32, %c128_i32], [%desc_0, %desc] : <f32>, <tensor<1x32xf32>>
    %out = tt.descriptor_gather %desc_4[%idx, %y] : (!tt.tensordesc<tensor<1x32xf32>>, tensor<32xi32>, i32) -> tensor<32x32xf32>
    tt.return
  }
}
