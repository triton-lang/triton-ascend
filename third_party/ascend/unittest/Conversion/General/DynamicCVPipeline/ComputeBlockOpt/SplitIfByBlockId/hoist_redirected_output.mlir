// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s
//
// Tests for hoistAndUnifyRedirectedOutput():
//   - @hoist_caseb_live_body_bails   : the safety gate refuses a destructive
//                                      Case-B hoist (leaves the IR untouched).
//   - @hoist_casec_redirects_result  : the positive path really redirects a
//                                      pass-through result to the producing
//                                      split-if and preserves the rebuilt
//                                      if-op's attributes.

// Case B (every result is a pass-through) where the last split-if still holds
// real work: the group-96 `arith.mulf` lives in the then body but is not part of
// any yield. Hoisting would erase the whole if-op and with it the mulf, so the
// gate must leave the pass-through untouched: all three split-ifs survive.
// CHECK-LABEL: func.func @hoist_caseb_live_body_bails
// CHECK: ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.splitted_if = 0 : i32
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
// CHECK: ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "CUBE", ssbuffer.splitted_if = 0 : i32
func.func @hoist_caseb_live_body_bails(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) -> tensor<2x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = tensor.empty() : tensor<2x2xf32>
  %res = scf.for %iv = %c0 to %c1 step %c1 iter_args(%arg = %init) -> (tensor<2x2xf32>) {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    %r = scf.if %cond -> (tensor<2x2xf32>) {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %m, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %m : tensor<2x2xf32>
    } else {
      scf.yield %arg : tensor<2x2xf32>
    }
    scf.yield %r : tensor<2x2xf32>
  }
  return %res : tensor<2x2xf32>
}

// -----

// Case C (partial hoist): slot 0 is produced by the group-95 CUBE split-if and
// forwarded by the last split-if, slot 1 is produced by the last group itself.
// After hoisting:
//   * the loop yields the group-95 if result (%M) directly instead of the
//     last if's pass-through slot 0;
//   * the rebuilt group-96 if has a single result whose body consumes %M.
// (Attribute inheritance onto the rebuilt if-op is deliberately NOT done yet,
// so no block_id/splitted_if check here.)
// CHECK-LABEL: func.func @hoist_casec_redirects_result
// CHECK: ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.splitted_if = 0 : i32
// CHECK-NEXT: %[[M:.*]] = scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: %[[LAST:.*]] = scf.if
// CHECK-NEXT: %{{.*}} = arith.mulf %[[M]]
// CHECK: scf.yield %[[M]], %[[LAST]]
func.func @hoist_casec_redirects_result(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %cond: i1) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.for %iv = %c0 to %c1 step %c1 iter_args(%a0 = %arg0, %a1 = %arg1) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    %res:2 = scf.if %cond -> (tensor<2x2xf32>, tensor<2x2xf32>) {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %m, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %m, %v2 : tensor<2x2xf32>, tensor<2x2xf32>
    } else {
      scf.yield %a0, %a1 : tensor<2x2xf32>, tensor<2x2xf32>
    }
    scf.yield %res#0, %res#1 : tensor<2x2xf32>, tensor<2x2xf32>
  }
  return %r#0, %r#1 : tensor<2x2xf32>, tensor<2x2xf32>
}
