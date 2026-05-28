// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @scalar_ceil_meta_use
// CHECK-NOT: tensor<1xf32>
// CHECK: math.ceil %{{.*}} : f32
// CHECK-NOT: tensor<1xf32>
// CHECK: arith.fptosi %{{.*}} : f32 to i32
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tensor<1xf32>

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @scalar_ceil_meta_use(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %T: i32) attributes {noinline = false} {
    %cst = arith.constant 3.200000e+01 : f32
    %zero = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %T_i64 = arith.extsi %T : i32 to i64
    %T_f32 = arith.sitofp %T : i32 to f32
    %div = arith.divf %T_f32, %cst : f32
    %ceil = math.ceil %div : f32
    %ceil_i32 = arith.fptosi %ceil : f32 to i32
    %next = arith.muli %ceil_i32, %c32_i32 : i32
    %last = arith.subi %next, %c32_i32 : i32
    %ptr = tt.make_tensor_ptr %x, [%T_i64], [%c1_i64], [%last] {order = array<i32: 0>} : <tensor<32xf32>>
    %loaded = tt.load %ptr {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
    scf.for %iv = %c0_i32 to %last step %c32_i32 : i32 {
      %out_ptr = tt.addptr %out, %iv : !tt.ptr<f32>, i32
      tt.store %out_ptr, %zero : !tt.ptr<f32>
    }
    %mask = arith.cmpi slt, %last, %T : i32
    %out_ptr = tt.addptr %out, %last : !tt.ptr<f32>, i32
    %val = tensor.extract %loaded[%c0] : tensor<32xf32>
    tt.store %out_ptr, %val, %mask : !tt.ptr<f32>
    tt.return
  }
}
