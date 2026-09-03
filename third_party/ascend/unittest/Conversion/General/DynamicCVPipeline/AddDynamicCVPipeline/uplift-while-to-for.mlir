// RUN: triton-opt %s --uplift-while-to-for -split-input-file | FileCheck %s

// DynamicCVPipeline thin wrapper around
// scf::populateUpliftWhileToForPatterns (same as HFusion
// hfusion-uplift-while-to-for).

// -----
// CHECK-LABEL: func.func @uplift_for_shaped_while
//  CHECK-SAME: (%[[BEGIN:.*]]: index, %[[END:.*]]: index, %[[STEP:.*]]: index) -> index
//   CHECK-NOT: scf.while
//       CHECK: scf.for %[[I:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]]
func.func @uplift_for_shaped_while(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    %added = arith.addi %arg3, %arg2 : index
    scf.yield %added : index
  }
  return %0 : index
}

// -----
// CHECK-LABEL: func.func @keep_data_driven_while
//       CHECK: scf.while
//   CHECK-NOT: scf.for
func.func @keep_data_driven_while(%arg0: i1, %arg1: index) -> index {
  %0 = scf.while (%arg2 = %arg1) : (index) -> (index) {
    scf.condition(%arg0) %arg2 : index
  } do {
  ^bb0(%arg2: index):
    %c1 = arith.constant 1 : index
    %added = arith.addi %arg2, %c1 : index
    scf.yield %added : index
  }
  return %0 : index
}
