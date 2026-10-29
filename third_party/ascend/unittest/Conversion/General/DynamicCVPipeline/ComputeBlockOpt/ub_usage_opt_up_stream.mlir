// RUN: triton-opt --split-input-file --ub-usage-opt-up-stream %s | FileCheck %s
// RUN: triton-opt --split-input-file --ub-usage-opt-up-stream --ub-usage-opt-up-stream %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @dynamic_slices(
func.func @dynamic_slices(%input: tensor<2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %flag = arith.cmpi ult, %iv, %n {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    %size = arith.index_castui %flag {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : i1 to index
    // CHECK: arith.cmpi {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    // CHECK: arith.index_castui {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    %slice35 = tensor.extract_slice %input[0] [%size] [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    %slice36 = tensor.extract_slice %input[0] [%size] [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    %result1 = tensor.insert_slice %slice35 into %input[0] [%size] [1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    %result2 = tensor.insert_slice %slice36 into %result1[0] [%size] [1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    // A shared scalar consumer prevents moving %size with either slice.
    %shared = arith.addi %size, %iv {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @static_transfer_reduction(
func.func @static_transfer_reduction(%input: tensor<4xf16>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    %wide = arith.extf %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    %result = arith.addf %wide, %wide {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @shared_producer(
func.func @shared_producer(%input: tensor<2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %slice = tensor.extract_slice %input[0] [%n] [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    %result1 = tensor.insert_slice %slice into %input[0] [%n] [1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    %result2 = tensor.insert_slice %slice into %input[0] [%n] [1] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @cycle_rejected(
func.func @cycle_rejected(%input: tensor<2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %slice = tensor.extract_slice %input[0] [%n] [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    %middle = arith.addf %slice, %slice {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "CUBE"} : tensor<?xf32>
    %result1 = tensor.insert_slice %slice into %input[0] [%n] [1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    %result2 = tensor.insert_slice %middle into %result1[0] [%n] [1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @sync_rejected(
func.func @sync_rejected(%input: tensor<2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %slice = tensor.extract_slice %input[0] [%n] [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    gpu.barrier {ssbuffer.external_sync = 1 : i32}
    %result = tensor.insert_slice %slice into %input[0] [%n] [1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @cross_core_rejected(
func.func @cross_core_rejected(%input: tensor<2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %slice = tensor.extract_slice %input[0] [%n] [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    %result = tensor.insert_slice %slice into %input[0] [%n] [1] {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "CUBE"} : tensor<?xf32> into tensor<2xf32>
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @equal_cost_unchanged(
func.func @equal_cost_unchanged(%input: tensor<2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: math.exp {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %keep_input = arith.addf %input, %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32>
    %exp = math.exp %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32>
    %result = arith.addf %exp, %exp {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32>
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @memory_effect_rejected(
func.func @memory_effect_rejected(%input: memref<2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: memref.load {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %scalar = memref.load %input[%c0] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : memref<2xf32>
    %value = arith.addf %scalar, %scalar {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : f32
    scf.yield
  }
  return
}

// -----
// CHECK-LABEL: func.func @cycle_with_private_producer(
func.func @cycle_with_private_producer(%input: tensor<2xf32>, %n: index, %indices: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %offset = arith.addi %iv, %c0 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : index
    %loaded = memref.load %indices[%offset] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : memref<2xindex>
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %slice = tensor.extract_slice %input[%loaded] [%n] [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2xf32> to tensor<?xf32>
    %result = tensor.insert_slice %slice into %input[0] [%n] [1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32> into tensor<2xf32>
    scf.yield
  }
  return
}

// -----
// The first reverse cut increases bytes. Moving both operations improves it.
// CHECK-LABEL: func.func @reverse_chain(
func.func @reverse_chain(%input: tensor<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    // CHECK: arith.truncf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %small = arith.truncf %wide {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32> to tensor<4xf16>
    %result = arith.addf %small, %small {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16>
    scf.yield
  }
  return
}

// -----
// Structured tensor operations may carry scalar regions; these are not control flow.
// CHECK-LABEL: func.func @tensor_region(
func.func @tensor_region(%value: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: tensor.empty() {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %empty = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    // CHECK: linalg.fill {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %filled = linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} ins(%value : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %result = arith.addf %filled, %filled {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    scf.yield
  }
  return
}

// -----
// A yielded producer is also a loop-carried source, not a private producer.
// CHECK-LABEL: func.func @loop_carried_rejected(
func.func @loop_carried_rejected(%input: tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %iv = %c0 to %c1 step %c1 iter_args(%arg = %input) -> tensor<?xf32> {
    // CHECK: math.exp {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %exp = math.exp %arg {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
    %use = arith.addf %exp, %exp {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
    scf.yield %exp : tensor<?xf32>
  }
  return
}
