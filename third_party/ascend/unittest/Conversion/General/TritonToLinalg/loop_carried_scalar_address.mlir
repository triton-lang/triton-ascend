// RUN: triton-opt --triton-to-linalg --split-input-file %s -verify-each | FileCheck %s \
// RUN:   --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast

// The tensor update is not a BlockData expression. Reducing it to an integer
// scalar makes the address offset opaque to BlockDataParser, so the tensor
// must remain an ordinary loop-carried value.
// CHECK-LABEL: func.func @scalar_address_from_carried_tensor
// CHECK: scf.for
// CHECK-SAME: -> (tensor<4xi32>)
// CHECK: linalg.reduce
// CHECK: scf.yield
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @scalar_address_from_carried_tensor(%output: !tt.ptr<i32>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0> : tensor<4xi32>
    %lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %result = scf.for %iv = %c0 to %upper step %c1 iter_args(%counts = %zero) -> (tensor<4xi32>) {
      %active = arith.cmpi sle, %counts, %lanes : tensor<4xi32>
      %increment = arith.extui %active : tensor<4xi1> to tensor<4xi32>
      %next = arith.addi %counts, %increment : tensor<4xi32>
      %sum = "tt.reduce"(%next) <{axis = 0 : i32}> ({
      ^bb0(%lhs: i32, %rhs: i32):
        %add = arith.addi %lhs, %rhs : i32
        tt.reduce.return %add : i32
      }) : (tensor<4xi32>) -> i32
      %ptr = tt.addptr %output, %sum : !tt.ptr<i32>, i32
      tt.store %ptr, %sum : !tt.ptr<i32>
      scf.yield %next : tensor<4xi32>
    }
    tt.return
  }
}

// -----

// The affine offset still needs legacy decomposition. The numerical carrier
// in the same loop must not be selected, including when its scalar reduction
// is splatted back into a structured tensor address.
// CHECK-LABEL: func.func @mixed_affine_and_numerical_carriers
// CHECK: scf.for
// CHECK-SAME: tensor<4xi32>
// CHECK-SAME: index
// CHECK: linalg.reduce
// CHECK: scf.yield
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @mixed_affine_and_numerical_carriers(%output: !tt.ptr<i32>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0> : tensor<4xi32>
    %step = arith.constant dense<4> : tensor<4xi32>
    %lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %output : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %result:2 = scf.for %iv = %c0 to %upper step %c1
        iter_args(%offsets = %lanes, %counts = %zero) -> (tensor<4xi32>, tensor<4xi32>) {
      %active = arith.cmpi sle, %counts, %lanes : tensor<4xi32>
      %increment = arith.extui %active : tensor<4xi1> to tensor<4xi32>
      %next_counts = arith.addi %counts, %increment : tensor<4xi32>
      %sum = "tt.reduce"(%next_counts) <{axis = 0 : i32}> ({
      ^bb0(%lhs: i32, %rhs: i32):
        %add = arith.addi %lhs, %rhs : i32
        tt.reduce.return %add : i32
      }) : (tensor<4xi32>) -> i32
      %uniform = tt.splat %sum : i32 -> tensor<4xi32>
      %address = arith.addi %offsets, %uniform : tensor<4xi32>
      %ptr = tt.addptr %base, %address : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      tt.store %ptr, %next_counts : tensor<4x!tt.ptr<i32>>
      %next_offsets = arith.addi %offsets, %step : tensor<4xi32>
      scf.yield %next_offsets, %next_counts : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}
