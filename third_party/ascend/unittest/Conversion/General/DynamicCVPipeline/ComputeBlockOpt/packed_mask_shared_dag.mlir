// RUN: triton-opt %s -sink-i1-producers-into-users -verify-each | FileCheck %s

// Shared conditions must not cause exponential source traversal.
module {
  // CHECK-LABEL: func.func @plain_shared_dag
  func.func @plain_shared_dag(%arg: tensor<1x8xi1>, %input: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %zero = arith.constant dense<0.0> : tensor<1x8xf32>
    %one = arith.constant dense<1.0> : tensor<1x8xf32>
    %p0 = arith.andi %arg, %arg {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p1 = arith.andi %p0, %p0 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p2 = arith.andi %p1, %p1 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p3 = arith.andi %p2, %p2 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p4 = arith.andi %p3, %p3 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p5 = arith.andi %p4, %p4 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p6 = arith.andi %p5, %p5 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p7 = arith.andi %p6, %p6 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p8 = arith.andi %p7, %p7 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p9 = arith.andi %p8, %p8 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p10 = arith.andi %p9, %p9 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p11 = arith.andi %p10, %p10 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p12 = arith.andi %p11, %p11 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p13 = arith.andi %p12, %p12 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p14 = arith.andi %p13, %p13 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p15 = arith.andi %p14, %p14 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p16 = arith.andi %p15, %p15 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p17 = arith.andi %p16, %p16 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p18 = arith.andi %p17, %p17 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p19 = arith.andi %p18, %p18 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p20 = arith.andi %p19, %p19 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p21 = arith.andi %p20, %p20 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p22 = arith.andi %p21, %p21 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p23 = arith.andi %p22, %p22 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p24 = arith.andi %p23, %p23 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p25 = arith.andi %p24, %p24 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p26 = arith.andi %p25, %p25 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p27 = arith.andi %p26, %p26 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p28 = arith.andi %p27, %p27 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p29 = arith.andi %p28, %p28 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p30 = arith.andi %p29, %p29 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p31 = arith.andi %p30, %p30 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p32 = arith.andi %p31, %p31 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 10
    %selected = arith.select %p32, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>, tensor<1x8xf32>
    %result = arith.addf %input, %selected {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xf32>
    return %result : tensor<1x8xf32>
  }

  // CHECK-LABEL: func.func @packed_shared_dag
  func.func @packed_shared_dag(%bytes: tensor<1x1xi8>, %input: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %zero = arith.constant dense<0.0> : tensor<1x8xf32>
    %one = arith.constant dense<1.0> : tensor<1x8xf32>
    // CHECK: hfusion.packed_mask_view {{.*}} {ssbuffer.block_id = 20
    %keep = hfusion.packed_mask_view %bytes {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x1xi8> -> tensor<1x8xi1>
    %p0 = arith.andi %keep, %keep {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p1 = arith.andi %p0, %p0 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p2 = arith.andi %p1, %p1 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p3 = arith.andi %p2, %p2 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p4 = arith.andi %p3, %p3 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p5 = arith.andi %p4, %p4 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p6 = arith.andi %p5, %p5 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p7 = arith.andi %p6, %p6 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p8 = arith.andi %p7, %p7 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p9 = arith.andi %p8, %p8 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p10 = arith.andi %p9, %p9 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p11 = arith.andi %p10, %p10 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p12 = arith.andi %p11, %p11 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p13 = arith.andi %p12, %p12 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p14 = arith.andi %p13, %p13 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p15 = arith.andi %p14, %p14 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p16 = arith.andi %p15, %p15 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p17 = arith.andi %p16, %p16 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p18 = arith.andi %p17, %p17 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p19 = arith.andi %p18, %p18 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p20 = arith.andi %p19, %p19 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p21 = arith.andi %p20, %p20 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p22 = arith.andi %p21, %p21 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    %p23 = arith.andi %p22, %p22 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 20
    %selected = arith.select %p23, %one, %zero {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xi1>, tensor<1x8xf32>
    %result = arith.addf %input, %selected {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x8xf32>
    return %result : tensor<1x8xf32>
  }

}
