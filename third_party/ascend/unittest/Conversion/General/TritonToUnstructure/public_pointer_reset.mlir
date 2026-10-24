// RUN: triton-opt %s --triton-to-unstructure -verify-each | FileCheck %s

// Preparse reaches a local add whose RHS resets the public LHS cache.
module {
  tt.func @reset_during_local_integer_parse(%base: !tt.ptr<i32>, %offsets: tensor<2x4xi64>, %lb: index, %ub: index, %step: index) -> tensor<8xi32> {
    %zero = arith.constant dense<0> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %p = tt.splat %base : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    %ptrs = tt.addptr %p, %offsets : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi64>
    %flat = tt.reshape %ptrs allow_reorder : tensor<2x4x!tt.ptr<i32>> -> tensor<8x!tt.ptr<i32>>
    %loaded = tt.load %flat : tensor<8x!tt.ptr<i32>>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%state = %zero) -> (tensor<8xi32>) {
      %sum = arith.addi %range, %loaded : tensor<8xi32>
      %next = arith.addi %sum, %state : tensor<8xi32>
      scf.yield %next : tensor<8xi32>
    }
    tt.return %result : tensor<8xi32>
  }
}
// CHECK-LABEL: @reset_during_local_integer_parse
// CHECK-NOT: allow_reorder
// CHECK: tt.reshape {{.*}} : tensor<2x4xi64> -> tensor<8xi64>
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.addi {{.*}} : tensor<8xi32>
// CHECK: scf.yield {{.*}} : tensor<8xi32>
// CHECK: tt.return
