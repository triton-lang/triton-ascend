// RUN: triton-opt --convert-descriptor-ops --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_descriptor_load
// CHECK-NOT: tt.descriptor_load
// CHECK: %[[PTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
// CHECK: tt.load %[[PTR]] {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xf32>>

module {
  tt.func public @tensor_descriptor_load(%in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %row_idx: i32, %col_idx: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %in_ptr, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <tensor<128x128xf32>>
    %out = tt.descriptor_load %desc[%row_idx, %col_idx] : !tt.tensordesc<tensor<128x128xf32>> -> tensor<128x128xf32>
    tt.return
  }
}
