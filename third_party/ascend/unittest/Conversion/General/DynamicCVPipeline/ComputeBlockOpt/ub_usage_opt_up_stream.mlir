// RUN: triton-opt --split-input-file --ub-usage-opt-up-stream %s > %t.once
// RUN: FileCheck %s < %t.once
// RUN: triton-opt --split-input-file --ub-usage-opt-up-stream --ub-usage-opt-up-stream %s > %t.twice
// RUN: FileCheck %s < %t.twice
// RUN: diff %t.once %t.twice

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
    // Keep the source group while its tensor chain moves.
    // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
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
    // Keep the source group while its tensor chain moves.
    // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
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
// Scalar-only initialization roots are auxiliary, not tensor data chains.
// CHECK-LABEL: func.func @tensor_region(
func.func @tensor_region(%value: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %empty = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    // CHECK: linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
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

// -----
// Equal minimum costs keep the first cut closest to the consumer. Moving exp
// as well would transfer the same f16 input, which the source still needs.
// CHECK-LABEL: func.func @first_equal_cut_wins(
func.func @first_equal_cut_wins(%input: tensor<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %retained = arith.addf %input, %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16>
    // CHECK: math.exp {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %exp = math.exp %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16>
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %exp {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    %use = arith.addf %wide, %wide {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  }
  return
}

// -----
// A write-only empty destination is auxiliary. Neither it nor the scalar-only
// fill is moved; the multi-input consumer is allowed to own the tensor chain.
// CHECK-LABEL: func.func @broadcast_keeps_auxiliary_producers(
func.func @broadcast_keeps_auxiliary_producers(%input: tensor<4xf32>, %value: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
  %empty = tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x4xf32>
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %scalar = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %filled = linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} ins(%value : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: linalg.broadcast {{.*}}ssbuffer.block_id = 14 : i32
    %broadcast = linalg.broadcast ins(%input : tensor<4xf32>) outs(%empty : tensor<4x4xf32>) dimensions = [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %use = arith.addf %broadcast, %filled {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x4xf32>
  }
  return
}

// -----
// Two different tensor inputs are a real branch, even when both producers are
// individually pure and could lower the boundary cost if moved together.
// CHECK-LABEL: func.func @genuine_branch_is_not_a_chain(
func.func @genuine_branch_is_not_a_chain(%left: tensor<4xf16>, %right: tensor<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %a = arith.extf %left {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %b = arith.extf %right {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %sum = arith.addf %a, %b {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    %use = arith.addf %sum, %sum {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  }
  return
}

// -----
// A profitable singleton move would empty the source group, so it is rejected.
// CHECK-LABEL: func.func @source_group_is_preserved(
func.func @source_group_is_preserved(%input: tensor<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    %use = arith.addf %wide, %wide {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  }
  return
}

// -----
// A zero dimension wins over overflow in an earlier dimension. Both boundary
// values are empty and changing precision must not invent a saving.
// CHECK-LABEL: func.func @zero_dimension_before_size_estimation(
func.func @zero_dimension_before_size_estimation(%input: tensor<2305843009213693952x0xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2305843009213693952x0xf16> to tensor<2305843009213693952x0xf32>
    %use = arith.addf %wide, %wide {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2305843009213693952x0xf32>
  }
  return
}

// -----
// An explicitly grouped yield is still a loop-state boundary, never a normal
// target consumer. The retained scalar makes this independent of group size.
// CHECK-LABEL: func.func @annotated_loop_state_is_pinned(
func.func @annotated_loop_state_is_pinned(%input: tensor<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %iv = %c0 to %c1 step %c1 iter_args(%state = %input) -> tensor<4xf16> {
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: math.exp {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %exp = math.exp %state {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16>
    %wide = arith.extf %exp {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    "scf.yield"(%exp) {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<4xf16>) -> ()
  }
  return
}

// -----
// A nested user's own block id cannot be replaced by its scf.if ancestor's id.
// CHECK-LABEL: func.func @nested_user_is_not_a_private_consumer(
func.func @nested_user_is_not_a_private_consumer(%input: tensor<4xf16>, %condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    %direct = arith.addf %wide, %wide {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    scf.if %condition {
      %nested = arith.addf %wide, %wide {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    } {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
  }
  return
}

// -----
// A destination read by the payload is a real second data input, even if its
// value comes from tensor.empty. Only write-only destinations are auxiliary.
// CHECK-LABEL: func.func @read_destination_is_data(
func.func @read_destination_is_data(%input: tensor<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  scf.for %iv = %c0 to %c1 step %c1 {
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: linalg.generic {{.*}}ssbuffer.block_id = 12 : i32
    %result = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} ins(%input : tensor<4xf16>) outs(%empty : tensor<4xf32>) {
    ^bb0(%element: f16, %initial: f32):
      %wide = arith.extf %element : f16 to f32
      %sum = arith.addf %wide, %initial : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    %use = arith.addf %result, %result {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  }
  return
}

// -----
// Each reverse chain stays in its source group. Group 11 still needs its
// original input, so moving its narrowing operation is not profitable.
// CHECK-LABEL: func.func @source_group_boundary_stops_the_chain(
func.func @source_group_boundary_stops_the_chain(%input: tensor<4xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %keep11 = arith.addf %input, %input {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf64>
    // CHECK: arith.truncf {{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
    %base = arith.truncf %input {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf64> to tensor<4xf16>
    %keep12 = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %base {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    // CHECK: math.exp {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %exp = math.exp %wide {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    %use = arith.addf %exp, %exp {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  }
  return
}

// -----
// Before and after regions may have different arities. Forwarded/yielded
// producers stay pinned; a non-state chain using an after argument may move.
// CHECK-LABEL: func.func @while_arguments_are_boundary_inputs(
func.func @while_arguments_are_boundary_inputs(%input: tensor<4xf16>, %condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:3 = scf.while (%first = %input, %second = %input) : (tensor<4xf16>, tensor<4xf16>) -> (tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) {
    %keep_before = arith.addi %c0, %c1 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"}
    %forwarded = arith.addf %second, %second {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16>
    %before_use = arith.extf %forwarded {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    "scf.condition"(%condition, %forwarded, %first, %first) {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : (i1, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) -> ()
  } do {
  ^bb0(%left: tensor<4xf16>, %right: tensor<4xf16>, %extra: tensor<4xf16>):
    %keep12 = arith.addi %c0, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %extra {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    %use = arith.addf %wide, %wide {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    %keep16 = arith.addi %c0, %c1 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"}
    %yielded = arith.addf %left, %left {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16>
    %after_use = arith.extf %yielded {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf16> to tensor<4xf32>
    "scf.yield"(%right, %yielded) {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<4xf16>, tensor<4xf16>) -> ()
  }
  return
}

// -----
// Check every result's actual users; inspecting only the selected result would
// incorrectly move a split whose other result is still consumed in group 15.
// CHECK-LABEL: func.func @all_results_must_have_private_users(
func.func @all_results_must_have_private_users(%input: tensor<4x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: tt.split {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %left, %right = tt.split %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x2xf32> -> tensor<4xf32>
    %left_use = arith.addf %left, %left {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    %right_use = arith.addf %right, %right {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
  }
  return
}

// -----
// Both input and output estimates exceed uint64_t. Saturation makes them equal;
// neither multiplication nor addition may wrap around and invent a saving.
// CHECK-LABEL: func.func @saturated_sizes_do_not_wrap(
func.func @saturated_sizes_do_not_wrap(%input: tensor<9223372036854775807xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %retained = arith.addi %iv, %c1 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    // CHECK: arith.extf {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %wide = arith.extf %input {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<9223372036854775807xf32> to tensor<9223372036854775807xf64>
    %use = arith.addf %wide, %wide {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<9223372036854775807xf64>
  }
  return
}
