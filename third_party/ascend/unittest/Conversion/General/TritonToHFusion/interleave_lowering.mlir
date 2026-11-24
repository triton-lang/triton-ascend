// RUN: triton-opt %s -triton-to-hfusion --split-input-file | FileCheck %s

// CHECK-LABEL: tt.func @join_reshape_to_interleave
tt.func @join_reshape_to_interleave(%a: tensor<16x64xi8>, %b: tensor<16x64xi8>) -> tensor<16x128xi8> {
  // CHECK: %[[INTER:.*]] = hfusion.interleave %arg0, %arg1 : tensor<16x64xi8>, tensor<16x64xi8> -> tensor<16x128xi8>
  // CHECK-NOT: tt.join
  // CHECK: tt.return %[[INTER]] : tensor<16x128xi8>
  %joined = tt.join %a, %b : tensor<16x64xi8> -> tensor<16x64x2xi8>
  %res = tt.reshape %joined : tensor<16x64x2xi8> -> tensor<16x128xi8>
  tt.return %res : tensor<16x128xi8>
}

// -----

// CHECK-LABEL: tt.func @expand_broadcast_reshape_factor4
tt.func @expand_broadcast_reshape_factor4(%src: tensor<16x32xi8>) -> tensor<16x128xi8> {
  // CHECK: %[[I0:.*]] = hfusion.interleave %arg0, %arg0 : tensor<16x32xi8>, tensor<16x32xi8> -> tensor<16x64xi8>
  // CHECK: %[[I1:.*]] = hfusion.interleave %[[I0]], %[[I0]] : tensor<16x64xi8>, tensor<16x64xi8> -> tensor<16x128xi8>
  // CHECK-NOT: tt.broadcast
  // CHECK: tt.return %[[I1]] : tensor<16x128xi8>
  %exp = tt.expand_dims %src {axis = 2 : i32} : tensor<16x32xi8> -> tensor<16x32x1xi8>
  %bcast = tt.broadcast %exp : tensor<16x32x1xi8> -> tensor<16x32x4xi8>
  %res = tt.reshape %bcast : tensor<16x32x4xi8> -> tensor<16x128xi8>
  tt.return %res : tensor<16x128xi8>
}

// -----

// CHECK-LABEL: tt.func @reshape_broadcast_reshape_factor8
tt.func @reshape_broadcast_reshape_factor8(%src: tensor<256xf16>) -> tensor<16x128xf16> {
  // CHECK: %[[BASE:.*]] = tt.reshape %arg0 : tensor<256xf16> -> tensor<16x16xf16>
  // CHECK: %[[I0:.*]] = hfusion.interleave %[[BASE]], %[[BASE]] : tensor<16x16xf16>, tensor<16x16xf16> -> tensor<16x32xf16>
  // CHECK: %[[I1:.*]] = hfusion.interleave %[[I0]], %[[I0]] : tensor<16x32xf16>, tensor<16x32xf16> -> tensor<16x64xf16>
  // CHECK: %[[I2:.*]] = hfusion.interleave %[[I1]], %[[I1]] : tensor<16x64xf16>, tensor<16x64xf16> -> tensor<16x128xf16>
  // CHECK-NOT: tt.broadcast
  // CHECK: tt.return %[[I2]] : tensor<16x128xf16>
  %r = tt.reshape %src : tensor<256xf16> -> tensor<16x16x1xf16>
  %bcast = tt.broadcast %r : tensor<16x16x1xf16> -> tensor<16x16x8xf16>
  %res = tt.reshape %bcast : tensor<16x16x8xf16> -> tensor<16x128xf16>
  tt.return %res : tensor<16x128xf16>
}

// -----

// Non-power-of-2 inner broadcast factor (3) must remain untouched for standard Linalg lowering.
// CHECK-LABEL: tt.func @expand_broadcast_reshape_non_pow2
tt.func @expand_broadcast_reshape_non_pow2(%src: tensor<16x10xf16>) -> tensor<16x30xf16> {
  // CHECK: tt.broadcast
  // CHECK-NOT: hfusion.interleave
  %exp = tt.expand_dims %src {axis = 2 : i32} : tensor<16x10xf16> -> tensor<16x10x1xf16>
  %bcast = tt.broadcast %exp : tensor<16x10x1xf16> -> tensor<16x10x3xf16>
  %res = tt.reshape %bcast : tensor<16x10x3xf16> -> tensor<16x30xf16>
  tt.return %res : tensor<16x30xf16>
}

