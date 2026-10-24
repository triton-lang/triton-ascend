// RUN: triton-opt %s --triton-to-unstructure -verify-each | FileCheck %s --check-prefix=LOOP
// RUN: triton-opt %s --triton-to-unstructure=compile-on-910-95=true -verify-each | FileCheck %s --check-prefix=TEMPLATE
// RUN: triton-opt %s --triton-control-flow-opt --triton-to-structured --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg -verify-each | FileCheck %s --check-prefix=LOWERED --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast
// RUN: sed 's/Ascend910B2/Ascend950PR_9579/g' %s | triton-opt --triton-control-flow-opt --triton-to-structured --discrete-mask-access-conversion=compile-on-910-95=true --triton-to-annotation --triton-to-unstructure=compile-on-910-95=true --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg=compile-on-910-95=true -verify-each | FileCheck %s --check-prefix=LOWERED --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast

// Use kernel-style output stores: standalone tensor-return conversion has an
// existing tensor/memref ABI mismatch unrelated to pointer handoff.
// The public affine proof is valid, but these original tensor producers are
// outside BlockDataParser support. Unstructure must consume their exact O.
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @affine_integer_select(%base: !tt.ptr<i32>, %out: !tt.ptr<i32>, %cond: i1, %start: i64) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %r64 = arith.extsi %range : tensor<8xi32> to tensor<8xi64>
    %s = tt.splat %start : i64 -> tensor<8xi64>
    %shifted = arith.addi %r64, %s : tensor<8xi64>
    %offset = arith.select %cond, %r64, %shifted : tensor<8xi64>
    %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %p, %offset : tensor<8x!tt.ptr<i32>>, tensor<8xi64>
    %v = tt.load %ptr : tensor<8x!tt.ptr<i32>>
    %op = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ops = tt.addptr %op, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ops, %v : tensor<8x!tt.ptr<i32>>
    tt.return
  }
  tt.func public @unsigned_extend(%base: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %offset = arith.extui %range : tensor<8xi32> to tensor<8xi64>
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %ptr = tt.addptr %p, %offset : tensor<8x!tt.ptr<i32>>, tensor<8xi64>
  %v = tt.load %ptr : tensor<8x!tt.ptr<i32>>
  %op = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ops = tt.addptr %op, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ops, %v : tensor<8x!tt.ptr<i32>>
    tt.return
  }
  tt.func public @truncate(%base: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %narrow = arith.trunci %range : tensor<8xi32> to tensor<8xi8>
  %offset = arith.extsi %narrow : tensor<8xi8> to tensor<8xi32>
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %ptr = tt.addptr %p, %offset : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %v = tt.load %ptr : tensor<8x!tt.ptr<i32>>
  %op = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ops = tt.addptr %op, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ops, %v : tensor<8x!tt.ptr<i32>>
    tt.return
  }
  tt.func public @index_cast(%base: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %idx = arith.index_cast %range : tensor<8xi32> to tensor<8xindex>
  %two = arith.constant dense<2> : tensor<8xindex>
  %scaled = arith.muli %idx, %two : tensor<8xindex>
  %offset = arith.index_cast %scaled : tensor<8xindex> to tensor<8xi64>
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %ptr = tt.addptr %p, %offset : tensor<8x!tt.ptr<i32>>, tensor<8xi64>
  %v = tt.load %ptr : tensor<8x!tt.ptr<i32>>
  %op = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ops = tt.addptr %op, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ops, %v : tensor<8x!tt.ptr<i32>>
    tt.return
  }
  tt.func public @unsigned_index_cast(%base: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %idx = arith.index_castui %range : tensor<8xi32> to tensor<8xindex>
  %two = arith.constant dense<2> : tensor<8xindex>
  %scaled = arith.muli %idx, %two : tensor<8xindex>
  %offset = arith.index_cast %scaled : tensor<8xindex> to tensor<8xi64>
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %ptr = tt.addptr %p, %offset : tensor<8x!tt.ptr<i32>>, tensor<8xi64>
  %v = tt.load %ptr : tensor<8x!tt.ptr<i32>>
  %op = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ops = tt.addptr %op, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ops, %v : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
// LOOP-LABEL: @affine_integer_select
// LOOP: scf.for
// LOOP: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @affine_integer_select
// TEMPLATE: ascend.indirect_load
// LOWERED-LABEL: func.func @affine_integer_select
// LOOP-LABEL: @unsigned_extend
// LOOP: scf.for
// LOOP: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @unsigned_extend
// TEMPLATE: ascend.indirect_load
// LOWERED-LABEL: func.func @unsigned_extend
// LOOP-LABEL: @truncate
// LOOP: scf.for
// LOOP: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @truncate
// TEMPLATE: ascend.indirect_load
// LOWERED-LABEL: func.func @truncate
// LOOP-LABEL: @index_cast
// LOOP: scf.for
// LOOP: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @index_cast
// TEMPLATE: ascend.indirect_load
// LOWERED-LABEL: func.func @index_cast
// LOOP-LABEL: @unsigned_index_cast
// LOOP: scf.for
// LOOP: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @unsigned_index_cast
// TEMPLATE: ascend.indirect_load
// LOWERED-LABEL: func.func @unsigned_index_cast
