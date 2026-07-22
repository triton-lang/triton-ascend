// RUN: triton-opt --verify-each %s -graph-optimize -o - | FileCheck %s

// This is deliberately pre-layout TTIR. A direct tt.trans -> pointwise ->
// tt.dot chain is valid before GPU layouts are assigned; once GPU encodings
// exist, tt.dot requires DotOperandEncodingAttr while tt.trans inference only
// supports Blocked/Shared inputs. The rule keeps its encoded-input preflight
// fail-closed, but its structural rewrite is exercised here at the pipeline
  // stage where the chain can exist.
module {
  // CHECK-LABEL: tt.func @positive_a_truncf_bitcast(
  // CHECK-SAME: %[[A_SRC:arg[0-9]+]]: tensor<8x16xf32>
  // CHECK-SAME: %[[A_B:arg[0-9]+]]: tensor<8x16xi16>
  // CHECK-SAME: %[[A_C:arg[0-9]+]]: tensor<16x16xi32>
  // CHECK-NOT: tt.trans %[[A_SRC]]
  // CHECK: %[[A_TRUNC:.*]] = arith.truncf %[[A_SRC]] : tensor<8x16xf32> to tensor<8x16xf16>
  // CHECK-NEXT: %[[A_BITCAST:.*]] = arith.bitcast %[[A_TRUNC]] : tensor<8x16xf16> to tensor<8x16xi16>
  // CHECK-NEXT: %[[A_TRANS:.*]] = tt.trans %[[A_BITCAST]] {order = array<i32: 1, 0>} : tensor<8x16xi16> -> tensor<16x8xi16>
  // CHECK-NEXT: %[[A_DOT:.*]] = tt.dot %[[A_TRANS]], %[[A_B]], %[[A_C]] : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
  tt.func @positive_a_truncf_bitcast(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xi16>,
      %c: tensor<16x16xi32>) {
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %a_trunc = arith.truncf %a_trans : tensor<16x8xf32> to tensor<16x8xf16>
    %a_bitcast = arith.bitcast %a_trunc : tensor<16x8xf16> to tensor<16x8xi16>
    %dot = tt.dot %a_bitcast, %b, %c : tensor<16x8xi16> * tensor<8x16xi16> -> tensor<16x16xi32>
    tt.return
  }

  // CHECK-LABEL: tt.func @positive_b_extf(
  // CHECK-SAME: %[[B_A:arg[0-9]+]]: tensor<16x8xf32>
  // CHECK-SAME: %[[B_SRC:arg[0-9]+]]: tensor<16x8xf16>
  // CHECK-SAME: %[[B_C:arg[0-9]+]]: tensor<16x16xf32>
  // CHECK-NOT: tt.trans %[[B_SRC]]
  // CHECK: %[[B_EXT:.*]] = arith.extf %[[B_SRC]] : tensor<16x8xf16> to tensor<16x8xf32>
  // CHECK-NEXT: %[[B_TRANS:.*]] = tt.trans %[[B_EXT]] {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
  // CHECK-NEXT: %[[B_DOT:.*]] = tt.dot %[[B_A]], %[[B_TRANS]], %[[B_C]] : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
  tt.func @positive_b_extf(
      %a: tensor<16x8xf32>,
      %b_src: tensor<16x8xf16>,
      %c: tensor<16x16xf32>) {
    %b_trans = tt.trans %b_src {order = array<i32: 1, 0>} : tensor<16x8xf16> -> tensor<8x16xf16>
    %b_ext = arith.extf %b_trans : tensor<8x16xf16> to tensor<8x16xf32>
    %dot = tt.dot %a, %b_ext, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_external_trans_user(
  // CHECK: %[[EXTERNAL_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK-NEXT: %[[EXTERNAL_CAST:.*]] = arith.truncf %[[EXTERNAL_TRANS]] : tensor<16x8xf32> to tensor<16x8xf16>
  // CHECK-NEXT: %{{.*}} = tt.trans %[[EXTERNAL_TRANS]] {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
  // CHECK-NEXT: %[[EXTERNAL_DOT:.*]] = tt.dot %[[EXTERNAL_CAST]], %{{.*}}, %{{.*}} : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_external_trans_user(
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xf16>,
      %c: tensor<16x16xf32>) {
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %a_trunc = arith.truncf %a_trans : tensor<16x8xf32> to tensor<16x8xf16>
    %extra = tt.trans %a_trans {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
    %dot = tt.dot %a_trunc, %b, %c : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_addf(
  // CHECK: %[[ADD_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[ADD_TRANS]], %{{.*}} : tensor<16x8xf16>
  // CHECK-NEXT: %[[ADD_DOT:.*]] = tt.dot %[[ADD]], %{{.*}}, %{{.*}} : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_addf(
      %a_src: tensor<8x16xf16>,
      %b: tensor<8x16xf16>,
      %c: tensor<16x16xf32>) {
    %zero = arith.constant dense<0.0> : tensor<16x8xf16>
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>
    %a_add = arith.addf %a_trans, %zero : tensor<16x8xf16>
    %dot = tt.dot %a_add, %b, %c : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
    tt.return
  }

  // The accumulator has the matching shape but is deliberately not an A/B
  // candidate. Its original transpose/cast chain must stay untouched.
  // CHECK-LABEL: tt.func @reject_dot_accumulator(
  // CHECK: %[[C_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<16x16xf64> -> tensor<16x16xf64>
  // CHECK-NEXT: %[[C_TRUNC:.*]] = arith.truncf %[[C_TRANS]] : tensor<16x16xf64> to tensor<16x16xf32>
  // CHECK-NEXT: %[[C_DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[C_TRUNC]] : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_dot_accumulator(
      %a: tensor<16x8xf16>,
      %b: tensor<8x16xf16>,
      %c_src: tensor<16x16xf64>) {
    %c_trans = tt.trans %c_src {order = array<i32: 1, 0>} : tensor<16x16xf64> -> tensor<16x16xf64>
    %c_trunc = arith.truncf %c_trans : tensor<16x16xf64> to tensor<16x16xf32>
    %dot = tt.dot %a, %b, %c_trunc : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @reject_region_boundary(
  // CHECK: %[[REGION_TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  // CHECK: scf.if %{{.*}} -> (tensor<16x16xf32>) {
  // CHECK: %[[REGION_CAST:.*]] = arith.truncf %[[REGION_TRANS]] : tensor<16x8xf32> to tensor<16x8xf16>
  // CHECK-NEXT: %[[REGION_DOT:.*]] = tt.dot %[[REGION_CAST]], %{{.*}}, %{{.*}} : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
  tt.func @reject_region_boundary(
      %condition: i1,
      %a_src: tensor<8x16xf32>,
      %b: tensor<8x16xf16>,
      %c: tensor<16x16xf32>) {
    %a_trans = tt.trans %a_src {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    %result = scf.if %condition -> (tensor<16x16xf32>) {
      %a_trunc = arith.truncf %a_trans : tensor<16x8xf32> to tensor<16x8xf16>
      %dot = tt.dot %a_trunc, %b, %c : tensor<16x8xf16> * tensor<8x16xf16> -> tensor<16x16xf32>
      scf.yield %dot : tensor<16x16xf32>
    } else {
      scf.yield %c : tensor<16x16xf32>
    }
    tt.return
  }
}
