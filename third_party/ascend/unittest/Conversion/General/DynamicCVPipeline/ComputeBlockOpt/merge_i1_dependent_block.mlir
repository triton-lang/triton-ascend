// RUN: triton-opt --split-input-file --merge-i1-dependent-block %s | FileCheck %s
// RUN: triton-opt --split-input-file --merge-i1-dependent-block --reorder-ops-by-block-id --verify-each %s | FileCheck %s

// Merge whole blocks, including members with no i1 operands, all consumers,
// and transitive i1 dependencies. Leave the unrelated block alone.
// CHECK-LABEL: func.func @chain_and_multiple_users
func.func @chain_and_multiple_users(%a: tensor<4xf32>, %b: tensor<4xf32>) {
  // CHECK-DAG: arith.addf {{.*}}ssbuffer.block_id = 10 : i32
  %sum = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK-DAG: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.cmpf ogt, %sum, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK-DAG: arith.mulf {{.*}}ssbuffer.block_id = 20 : i32
  %unrelated = arith.mulf %a, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK-DAG: arith.select {{.*}}ssbuffer.block_id = 10 : i32
  %selected = arith.select %mask, %a, %b {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>, tensor<4xf32>
  // CHECK-DAG: arith.addf {{.*}}ssbuffer.block_id = 10 : i32
  %extra = arith.addf %selected, %a {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK-DAG: arith.xori {{.*}}ssbuffer.block_id = 10 : i32
  %next = arith.xori %mask, %mask {ssbuffer.block_id = 40 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>
  // CHECK-DAG: arith.select {{.*}}ssbuffer.block_id = 10 : i32
  %last = arith.select %next, %a, %b {ssbuffer.block_id = 50 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>, tensor<4xf32>
  // CHECK: return
  return
}

// -----

// A scalar i1, a non-i1 tensor and a vector of i1 must not trigger fusion.
// CHECK-LABEL: func.func @other_types
func.func @other_types(%a: f32, %b: f32, %ta: tensor<4xf32>, %tb: tensor<4xf32>, %va: vector<4xf32>, %vb: vector<4xf32>) {
  // CHECK-DAG: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %cond = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : f32
  // CHECK-DAG: arith.select {{.*}}ssbuffer.block_id = 20 : i32
  %selected = arith.select %cond, %a, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : f32
  // CHECK-DAG: arith.addf {{.*}}ssbuffer.block_id = 30 : i32
  %tensor = arith.addf %ta, %tb {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK-DAG: arith.mulf {{.*}}ssbuffer.block_id = 40 : i32
  %result = arith.mulf %tensor, %tb {ssbuffer.block_id = 40 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK-DAG: arith.cmpf {{.*}}ssbuffer.block_id = 50 : i32
  %mask = arith.cmpf ogt, %va, %vb {ssbuffer.block_id = 50 : i32, ssbuffer.core_type = "VECTOR"} : vector<4xf32>
  // CHECK-DAG: arith.select {{.*}}ssbuffer.block_id = 60 : i32
  %vector = arith.select %mask, %va, %vb {ssbuffer.block_id = 60 : i32, ssbuffer.core_type = "VECTOR"} : vector<4xi1>, vector<4xf32>
  // CHECK: return
  return
}

// -----

// Do not mix CUBE and VECTOR compute blocks.
// CHECK-LABEL: func.func @different_cores
func.func @different_cores(%a: tensor<4xf32>, %b: tensor<4xf32>) {
  // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 20 : i32
  %selected = arith.select %mask, %a, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : tensor<4xi1>, tensor<4xf32>
  return
}

// -----

// A -> B -> C plus an i1 edge A -> C: merging A and C creates a cycle.
// CHECK-LABEL: func.func @ssa_cycle
func.func @ssa_cycle(%a: tensor<4xf32>, %b: tensor<4xf32>) {
  // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 10 : i32
  %sum = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20 : i32
  %middle = arith.mulf %sum, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 30 : i32
  %selected = arith.select %mask, %middle, %b {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>, tensor<4xf32>
  return
}

// -----

// The A -> C merge is initially blocked by A -> B -> C. Merging B and C
// through their i1 edge makes A -> C safe on the next iteration.
// CHECK-LABEL: func.func @retry_after_merge
func.func @retry_after_merge(%a: tensor<4xf32>, %b: tensor<4xf32>) {
  // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 10 : i32
  %sum = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %middle = arith.cmpf olt, %sum, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: arith.andi {{.*}}ssbuffer.block_id = 10 : i32
  %both = arith.andi %mask, %middle {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>
  return
}

// -----

// The dependency cycle also needs to account for memory effects: A stores,
// B loads and C consumes B's load as well as A's i1 tensor.
// CHECK-LABEL: func.func @memory_cycle
func.func @memory_cycle(%a: tensor<4xf32>, %b: tensor<4xf32>, %mem: memref<f32>, %value: f32) {
  // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK: memref.store {{.*}}ssbuffer.block_id = 10 : i32
  memref.store %value, %mem[] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<f32>
  // CHECK: memref.load {{.*}}ssbuffer.block_id = 20 : i32
  %loaded = memref.load %mem[] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<f32>
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 30 : i32
  %sum = arith.addf %loaded, %value {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : f32
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 30 : i32
  %selected = arith.select %mask, %a, %b {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>, tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @sync_boundary
func.func @sync_boundary(%a: tensor<4xf32>, %b: tensor<4xf32>) {
  // CHECK-DAG: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  // CHECK-DAG: gpu.barrier {ssbuffer.block_id = 20 : i32
  gpu.barrier {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.external_sync = 1 : i32}
  // CHECK-DAG: arith.select {{.*}}ssbuffer.block_id = 30 : i32
  %selected = arith.select %mask, %a, %b {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>, tensor<4xf32>
  // CHECK: return
  return
}

// -----

// Region-bearing and multi-result producers must also be handled. This case
// uses rank-zero tensors and cannot be handled by sink-i1-producers-into-users.
// CHECK-LABEL: func.func @multi_result_if
func.func @multi_result_if(%cond: i1, %a: tensor<f32>, %b: tensor<f32>) {
  // CHECK: scf.if
  %mask, %value = scf.if %cond -> (tensor<i1>, tensor<f32>) {
    // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
    %cmp = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<f32>
    scf.yield %cmp, %a : tensor<i1>, tensor<f32>
  } else {
    // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
    %cmp = arith.cmpf olt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<f32>
    scf.yield %cmp, %b : tensor<i1>, tensor<f32>
  // CHECK: } {ssbuffer.block_id = 10 : i32}
  } {ssbuffer.block_id = 10 : i32}
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 10 : i32
  %selected = arith.select %mask, %value, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<i1>, tensor<f32>
  return
}

// -----

// Resolve a nested consumer to its tagged enclosing block, updating all
// members of that block, including those in the nested region.
// CHECK-LABEL: func.func @nested_consumer
func.func @nested_consumer(%cond: i1, %a: tensor<2x4xf32>, %b: tensor<2x4xf32>) {
  // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x4xf32>
  // CHECK: scf.if
  scf.if %cond {
    // CHECK: arith.select {{.*}}ssbuffer.block_id = 10 : i32
    %selected = arith.select %mask, %a, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x4xi1>, tensor<2x4xf32>
  // CHECK: } {ssbuffer.block_id = 10 : i32}
  } {ssbuffer.block_id = 20 : i32}
  return
}

// -----

// Untagged producers/users and block arguments do not identify compute blocks.
// CHECK-LABEL: func.func @unassigned_ops
func.func @unassigned_ops(%arg_mask: tensor<4xi1>, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK: arith.cmpf ogt, {{.*}} : tensor<4xf32>
  // CHECK-NOT: ssbuffer.block_id
  %unassigned = arith.cmpf ogt, %a, %b : tensor<4xf32>
  // CHECK: arith.andi {{.*}}ssbuffer.block_id = 10 : i32
  %mask = arith.andi %unassigned, %arg_mask {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>
  // CHECK: arith.select {{.*}} : tensor<4xi1>, tensor<4xf32>
  // CHECK-NOT: ssbuffer.block_id
  %selected = arith.select %mask, %a, %b : tensor<4xi1>, tensor<4xf32>
  // CHECK: return
  return %mask : tensor<4xi1>
}

// -----

// CHECK: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 1 : i32}
module attributes {triton_ascend.dynamic_cv_pipeline.rc = 1 : i32} {
  // CHECK-LABEL: func.func @fallback
  func.func @fallback(%a: tensor<4xf32>, %b: tensor<4xf32>) {
    // CHECK: arith.cmpf {{.*}}ssbuffer.block_id = 10 : i32
    %mask = arith.cmpf ogt, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    // CHECK: arith.select {{.*}}ssbuffer.block_id = 20 : i32
    %selected = arith.select %mask, %a, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xi1>, tensor<4xf32>
    return
  }
}
