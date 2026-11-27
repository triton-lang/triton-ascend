// RUN: triton-opt %s -sink-i1-producers-into-users | FileCheck %s

// Constant penalties are computed with the score; packed bytes stay in the
// memory block. This is independent of predicate-load sharing in NPU-IR.
module {
  // CHECK-LABEL: func.func @packed_penalty
  func.func @packed_penalty(%bytes: tensor<32x16xi8>, %score: tensor<32x128xf32>, %valid: tensor<32x128xi1>) -> tensor<32x128xf32> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %negative = arith.constant dense<0xFF800000> : tensor<32x128xf32>
    // CHECK: hfusion.packed_mask_view {{.*}} {ssbuffer.block_id = 20
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.andi {{.*}} {ssbuffer.block_id = 20
    %bounded = arith.andi %keep, %valid {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 20
    %penalty = arith.select %bounded, %zero, %negative {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20
    %result = arith.addf %score, %penalty {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }

  // A dynamically selected floating-point payload is not a constant penalty.
  // CHECK-LABEL: func.func @dynamic_payload
  func.func @dynamic_payload(%bytes: tensor<32x16xi8>, %score: tensor<32x128xf32>, %a: tensor<32x128xf32>, %b: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %selected = arith.select %keep, %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20
    %result = arith.addf %score, %selected {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }

  // Tail-column selection can become extract/insert slices before sinking.
  // The constant penalty must still follow its score consumer.
  // CHECK-LABEL: func.func @sliced_packed_penalty
  func.func @sliced_packed_penalty(%bytes: tensor<32x16xi8>, %score: tensor<32x128xf32>, %valid_columns: index) -> tensor<32x128xf32> {
    %zeros = arith.constant dense<false> : tensor<32x128xi1>
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %negative = arith.constant dense<0xFF800000> : tensor<32x128xf32>
    // CHECK: hfusion.packed_mask_view {{.*}} {ssbuffer.block_id = 20
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x16xi8> -> tensor<32x128xi1>
    // CHECK: tensor.extract_slice {{.*}} {ssbuffer.block_id = 20
    %slice = tensor.extract_slice %keep[0, 0] [32, %valid_columns] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1> to tensor<32x?xi1>
    // CHECK: tensor.insert_slice {{.*}} {ssbuffer.block_id = 20
    %padded = tensor.insert_slice %slice into %zeros[0, 0] [32, %valid_columns] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x?xi1> into tensor<32x128xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 20
    %penalty = arith.select %padded, %zero, %negative {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20
    %result = arith.addf %score, %penalty {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }

  // A plain i1 condition must not trigger packed-penalty sinking.
  // CHECK-LABEL: func.func @plain_condition
  func.func @plain_condition(%keep: tensor<32x128xi1>, %score: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %negative = arith.constant dense<0xFF800000> : tensor<32x128xf32>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %penalty = arith.select %keep, %zero, %negative {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20
    %result = arith.addf %score, %penalty {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }

  // An integer shift alone does not establish the packed-view contract.
  // CHECK-LABEL: func.func @shift_condition
  func.func @shift_condition(%values: tensor<32x128xi8>, %shifts: tensor<32x128xi8>, %score: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %zeros = arith.constant dense<0> : tensor<32x128xi8>
    %zero = arith.constant dense<0.0> : tensor<32x128xf32>
    %negative = arith.constant dense<0xFF800000> : tensor<32x128xf32>
    %shifted = arith.shrui %values, %shifts {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi8>
    %keep = arith.cmpi ne, %shifted, %zeros {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi8>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %penalty = arith.select %keep, %zero, %negative {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xi1>, tensor<32x128xf32>
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20
    %result = arith.addf %score, %penalty {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }
}
