// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// VECTOR and CUBE each have one main scf.for and one intraDeps pair, so the
// pipeline reaches SplittedIfWrapSsbufIf. The two pairs use different intra
// groups: one group lists every producer, and calculateFactor looks those
// producers up inside the current for. Block 5's mulf is inside one
// splitted_if. The new if's else adds the block counter and the for step.
// ProcessArgs appends one shared iter_arg, then UpdateLoopOps appends block
// counters. The first counter is the third iter_arg, initialized to the for
// lower bound.
// UpdateLoopIterTimes runs last and rewrites the main loop into an scf.while:
// the induction variable becomes the first carried value, so the counters move
// to the fourth and the fifth one. The while keeps the upper bound of the for
// loop and its condition is the OR of the counters of the ssbuffer.if blocks
// compared against that upper bound.

// CHECK-LABEL: func.func @wrap_for_splitted_if
// CHECK: scf.while (%[[IV:[^ ]+]] = %[[LB:[^ ,]+]], %{{[^ ,]+}} = %{{[^ ,]+}}, %{{[^ ,]+}} = %{{[^ ,]+}}, %[[CTR1:[^ ,]+]] = %[[LB]], %[[CTR2:[^ ,]+]] = %[[LB]], %{{[^ ,]+}} = %{{[^ ,]+}}, %{{[^ ,]+}} = %{{[^ ,]+}}) : (i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32)
// CHECK: %[[CMP1:[^ ]+]] = arith.cmpi slt, %[[CTR1]], %[[UB:[^ ,]+]] : i32
// CHECK: %[[CMP2:[^ ]+]] = arith.cmpi slt, %[[CTR2]], %[[UB]] : i32
// CHECK: arith.ori %[[CMP1]], %[[CMP2]] : i1
// CHECK: scf.condition
// CHECK: arith.addi %[[IV]], %[[STEP:[^ ]+]] : i32
// CHECK: %[[SPLIT:[^ ]+]] = arith.cmpi sgt, {{.*}}ssbuffer.block_id = 5 : i32
// CHECK: scf.if %[[SPLIT]]
// CHECK: arith.mulf {{.*}} : tensor<4xf32>
// CHECK: arith.addi %[[CTR1]], %[[STEP]]
// CHECK: } {{.*}}ssbuffer.if = 5 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @wrap_for_splitted_if() attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %one = arith.constant dense<1.000000e+00> : tensor<4xf32>

    scope.scope : () -> () {
      %0 = scf.for %arg0 = %c0_i32 to %c16_i32 step %c1_i32
          iter_args(%arg1 = %c0_i32) -> (i32) : i32 {
        %split = arith.cmpi sgt, %arg0, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
        scf.if %split {
          %t = arith.mulf %zero, %one : tensor<4xf32>
        } {ssbuffer.block_id = 5 : i32, ssbuffer.splitted_if = 0 : i32}
        %prod = arith.addi %arg1, %c0_i32 {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]} : i32
        %cons = arith.addi %arg1, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : i32
        scf.yield %cons : i32
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %0 = scf.for %arg0 = %c0_i32 to %c16_i32 step %c1_i32
          iter_args(%arg1 = %c0_i32) -> (i32) : i32 {
        %prod = arith.addi %arg1, %c0_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]} : i32
        %cons = arith.addi %arg1, %c1_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]} : i32
        scf.yield %cons : i32
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
