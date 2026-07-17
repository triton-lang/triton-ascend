// RUN: triton-opt --convert-descriptor-ops --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_descriptor_scatter_rows_kernel
// CHECK-NOT: tt.descriptor_scatter
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<32xi32>
// CHECK: %[[PTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}], [%[[EXTRACTED]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<1x32xf32>>
// CHECK: %[[ROW:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
// CHECK: tt.store %[[PTR]], %[[ROW]] {DiscreteMemAccess, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x32xf32>>

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_descriptor_scatter_rows_kernel(%out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %data: tensor<32x32xf32>, %y: i32) attributes {noinline = false} {
    %desc = arith.constant 1 : i64
    %desc_0 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %desc_13 = tt.make_tensor_descriptor %out_ptr, [%c128_i32, %c128_i32], [%desc_0, %desc] : <f32>, <tensor<1x32xf32>>
    tt.descriptor_scatter %desc_13[%idx, %y], %data : !tt.tensordesc<tensor<1x32xf32>>, tensor<32xi32>, i32, tensor<32x32xf32>
    tt.return
  }
}
