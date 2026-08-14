// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-control-flow-opt --triton-to-linalg --split-input-file %s -verify-each | FileCheck %s --check-prefix=LINALG

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @local_addptr_base(%base: !tt.ptr<f32>) -> tensor<16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %shifted = tt.addptr %base, %c3_i32 : !tt.ptr<f32>, i32
    %ptr = tt.make_tensor_ptr %shifted, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %value = tt.load %ptr : !tt.ptr<tensor<16xf32>>
    tt.return %value : tensor<16xf32>
  }
}

// CHECK-LABEL: tt.func public @local_addptr_base
// CHECK:       %[[SHIFTED:.*]] = tt.addptr
// CHECK:       tt.make_tensor_ptr %[[SHIFTED]],
// CHECK-NOT:   tt.ptr_to_int

// LINALG-LABEL: func.func @local_addptr_base
// LINALG:       return

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @region_local_addptr_base(%base: !tt.ptr<f32>, %cond: i1) -> tensor<16xf32> {
    %result = scf.if %cond -> (tensor<16xf32>) {
      %c0_i32 = arith.constant 0 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1_i64 = arith.constant 1 : i64
      %c16_i64 = arith.constant 16 : i64
      %shifted = tt.addptr %base, %c3_i32 : !tt.ptr<f32>, i32
      %ptr = tt.make_tensor_ptr %shifted, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
      %loaded = tt.load %ptr : !tt.ptr<tensor<16xf32>>
      scf.yield %loaded : tensor<16xf32>
    } else {
      %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
      scf.yield %zero : tensor<16xf32>
    }
    tt.return %result : tensor<16xf32>
  }
}

// CHECK-LABEL: tt.func public @region_local_addptr_base
// CHECK:       scf.if
// CHECK:         %[[SHIFTED:.*]] = tt.addptr
// CHECK:         tt.make_tensor_ptr %[[SHIFTED]],
// CHECK-NOT:   tt.ptr_to_int

// LINALG-LABEL: func.func @region_local_addptr_base
// LINALG:       return

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @loop_carried_addptr_base(%base: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %shifted = tt.addptr %base, %c3_i32 : !tt.ptr<f32>, i32
    %initial = tt.make_tensor_ptr %shifted, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %final = scf.for %iv = %c0 to %c2 step %c1 iter_args(%ptr = %initial) -> (!tt.ptr<tensor<16xf32>>) {
      %next = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<16xf32>>
      scf.yield %next : !tt.ptr<tensor<16xf32>>
    }
    %value = tt.load %final : !tt.ptr<tensor<16xf32>>
    %output_ptr = tt.make_tensor_ptr %output, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    tt.store %output_ptr, %value : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @loop_carried_addptr_base
// CHECK:       %[[SHIFTED:.*]] = tt.addptr
// CHECK:       %[[ADDRESS:.*]] = tt.ptr_to_int %[[SHIFTED]]
// CHECK:       %[[FOR:.*]]:4 = scf.for
// CHECK-SAME:      iter_args(%{{.*}} = %[[ADDRESS]], %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
// CHECK-SAME:      -> (i64, i64, i64, i32)
// CHECK:       %[[BASE:.*]] = tt.int_to_ptr %[[FOR]]#0
// CHECK:       tt.make_tensor_ptr %[[BASE]], [%[[FOR]]#1], [%[[FOR]]#2], [%[[FOR]]#3]

// LINALG-LABEL: func.func @loop_carried_addptr_base
// LINALG:       scf.for
// LINALG-NOT:   tt.addptr
// LINALG-NOT:   tt.make_tensor_ptr
// LINALG-NOT:   tt.ptr_to_int
// LINALG-NOT:   tt.int_to_ptr
// LINALG:       return
