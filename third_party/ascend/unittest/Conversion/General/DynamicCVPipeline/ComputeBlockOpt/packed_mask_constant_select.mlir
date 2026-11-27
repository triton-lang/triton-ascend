// RUN: triton-opt %s -sink-i1-producers-into-users -verify-each | FileCheck %s

module {
  // A packed constant select can follow a non-add, f16 consumer.
  // CHECK-LABEL: func.func @multiply_f16
  func.func @multiply_f16(%bytes: tensor<32x16xi8>, %input: tensor<32x128xf16>) -> tensor<32x128xf16> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf16>
    %one = arith.constant dense<1.0> : tensor<32x128xf16>
    // CHECK: %[[KEEP:.*]] = hfusion.packed_mask_view {{.*}} {ssbuffer.block_id = 20
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: %[[SCALE:.*]] = arith.select %[[KEEP]], {{.*}} {ssbuffer.block_id = 20
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf16>
    // CHECK-NEXT: arith.mulf {{.*}}, %[[SCALE]] {ssbuffer.block_id = 20
    %result = arith.mulf %input, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf16>
    return %result : tensor<32x128xf16>
  }

  // All consumers in one compute block share a single select. Move before
  // the earliest user, independently of SSA use-list order.
  // CHECK-LABEL: func.func @multiple_users
  func.func @multiple_users(%bytes: tensor<32x16xi8>, %input: tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    // CHECK: %[[KEEP:.*]] = hfusion.packed_mask_view {{.*}} {ssbuffer.block_id = 20
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: %[[SCALE:.*]] = arith.select %[[KEEP]], {{.*}} {ssbuffer.block_id = 20
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK-NEXT: arith.mulf {{.*}}, %[[SCALE]] {ssbuffer.block_id = 20
    %a = arith.mulf %input, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    // CHECK-NEXT: arith.addf {{.*}}, %[[SCALE]] {ssbuffer.block_id = 20
    %b = arith.addf %input, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    // CHECK-NOT: arith.select
    return %a, %b : tensor<32x128xf32>, tensor<32x128xf32>
  }

  // Repeated operands in a single consumer are also multiple uses.
  // CHECK-LABEL: func.func @repeated_operand
  func.func @repeated_operand(%bytes: tensor<32x16xi8>) -> tensor<32x128xf32> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: %[[SCALE:.*]] = arith.select {{.*}} {ssbuffer.block_id = 20
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK-NEXT: arith.addf %[[SCALE]], %[[SCALE]] {ssbuffer.block_id = 20
    %result = arith.addf %scale, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }

  // Do not duplicate a float select into different compute blocks.
  // CHECK-LABEL: func.func @different_compute_blocks
  func.func @different_compute_blocks(%bytes: tensor<32x16xi8>, %input: tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: %[[SCALE:.*]] = arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: arith.mulf {{.*}}, %[[SCALE]] {ssbuffer.block_id = 20
    %a = arith.mulf %input, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    // CHECK: arith.addf {{.*}}, %[[SCALE]] {ssbuffer.block_id = 30
    %b = arith.addf %input, %scale {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %a, %b : tensor<32x128xf32>, tensor<32x128xf32>
  }

  // A valid block id does not make a Cube consumer eligible.
  // CHECK-LABEL: func.func @cube_consumer
  func.func @cube_consumer(%bytes: tensor<32x16xi8>, %input: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 20
    %result = arith.mulf %input, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }

  // An unclassified consumer cannot provide a target compute block.
  // CHECK-LABEL: func.func @unclassified_consumer
  func.func @unclassified_consumer(%bytes: tensor<32x16xi8>, %input: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    %result = arith.mulf %input, %scale : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }

  // Do not move a select from an outer block into a conditional region.
  // CHECK-LABEL: func.func @nested_user
  func.func @nested_user(%bytes: tensor<32x16xi8>, %input: tensor<32x128xf32>, %condition: i1) -> tensor<32x128xf32> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    %result = scf.if %condition -> tensor<32x128xf32> {
      %value = arith.mulf %input, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
      scf.yield %value : tensor<32x128xf32>
    } else {
      scf.yield %input : tensor<32x128xf32>
    }
    return %result : tensor<32x128xf32>
  }

  // A region-based consumer is outside local elementwise sinking.
  // CHECK-LABEL: func.func @region_consumer
  func.func @region_consumer(%bytes: tensor<32x16xi8>, %upper: index) -> tensor<32x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    %result = scf.for %i = %c0 to %upper step %c1 iter_args(%value = %scale) -> tensor<32x128xf32> {
      scf.yield %value : tensor<32x128xf32>
    } {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"}
    return %result : tensor<32x128xf32>
  }

  // A pure tensor extraction is not an elementwise consumer.
  // CHECK-LABEL: func.func @extract_consumer
  func.func @extract_consumer(%bytes: tensor<32x16xi8>) -> f32 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    %result = tensor.extract %scale[%c0, %c0] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : f32
  }

  // A call with unknown memory effects is not a local arithmetic consumer.
  // CHECK-LABEL: func.func @call_consumer
  func.func @call_consumer(%bytes: tensor<32x16xi8>) {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: %[[SCALE:.*]] = arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: call @consume(%[[SCALE]])
    func.call @consume(%scale) {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<32x128xf32>) -> ()
    return
  }

  func.func private @consume(tensor<32x128xf32>)

  // Do not merge the producer and consumer across an external sync.
  // CHECK-LABEL: func.func @sync_boundary
  func.func @sync_boundary(%bytes: tensor<32x16xi8>, %input: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %one = arith.constant dense<1.0> : tensor<32x128xf32>
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %scale = arith.select %keep, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: hivm.hir.sync_block_wait
    hivm.hir.sync_block_wait {ssbuffer.external_sync = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 20
    %result = arith.mulf %input, %scale {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }
}
