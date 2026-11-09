// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s

// Case B, slot produced by an earlier group: its else operand is the original
// else value (an external loop argument), so the slot must NOT be relayed
// through the last split-if. slot[0] is produced by the last group and stays
// there; slot[1] is produced by the CUBE group and is read directly.
// CHECK-LABEL: func.func @caseb_relay_redirect
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: %[[CUBE:[0-9]+]] = scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: scf.yield {{.*}} : tensor<2x2xf32>
// The last split-if keeps only the slot its own group produces.
// CHECK: %[[LAST:[0-9]+]] = scf.if
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
// CHECK: scf.yield {{.*}} : tensor<2x2xf32>
// The loop reads the CUBE result directly instead of forwarding it.
// CHECK: scf.yield %[[LAST]], %[[CUBE]] : tensor<2x2xf32>, tensor<2x2xf32>
func.func @caseb_relay_redirect(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %init: tensor<2x2xf32>, %cond: i1) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res:2 = scf.for %iv = %c0 to %c1 step %c1 iter_args(%arg0 = %init, %arg1 = %init) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    %r:2 = scf.if %cond -> (tensor<2x2xf32>, tensor<2x2xf32>) {
      %v2 = arith.addf %v1, %c {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m2 = linalg.matmul ins(%v2, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%m : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v3 = arith.mulf %m2, %d {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %v3, %m2 : tensor<2x2xf32>, tensor<2x2xf32>
    } else {
      scf.yield %arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>
    }
    scf.yield %r#0, %r#1 : tensor<2x2xf32>, tensor<2x2xf32>
  }
  return %res#0, %res#1 : tensor<2x2xf32>, tensor<2x2xf32>
}

// -----

// Every original slot is produced by an earlier group: the last group carries
// no result at all and must degrade to a void if instead of relaying, while
// keeping the work of its own group.
// CHECK-LABEL: func.func @caseb_all_redirected
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: %[[CUBE:[0-9]+]]:2 = scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<2x2xf32>, tensor<2x2xf32>
// CHECK: scf.if %{{.*}} {
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
// CHECK: } {ssbuffer.block_id = 96
// The loop yield follows the slot, not the result position: slot[0] is the
// second CUBE result, slot[1] the first one.
// CHECK: scf.yield %[[CUBE]]#1, %[[CUBE]]#0 : tensor<2x2xf32>, tensor<2x2xf32>
func.func @caseb_all_redirected(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %init: tensor<2x2xf32>, %cond: i1) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res:2 = scf.for %iv = %c0 to %c1 step %c1 iter_args(%arg0 = %init, %arg1 = %init) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    %r:2 = scf.if %cond -> (tensor<2x2xf32>, tensor<2x2xf32>) {
      %v2 = arith.addf %v1, %c {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m2 = linalg.matmul ins(%v2, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%m : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %m3 = linalg.matmul ins(%v2, %c : tensor<2x2xf32>, tensor<2x2xf32>) outs(%m2 : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v3 = arith.mulf %m3, %d {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %m2, %m3 : tensor<2x2xf32>, tensor<2x2xf32>
    } else {
      scf.yield %arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>
    }
    scf.yield %r#0, %r#1 : tensor<2x2xf32>, tensor<2x2xf32>
  }
  return %res#0, %res#1 : tensor<2x2xf32>, tensor<2x2xf32>
}
