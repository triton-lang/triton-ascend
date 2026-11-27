// RUN: triton-opt '--triton-to-structured' --split-input-file --verify-each %s | FileCheck %s --check-prefix=STRUCT
// RUN: triton-opt '--triton-to-structured' '--triton-to-unstructure' '--triton-to-hivm' '--triton-to-hfusion' '--triton-to-llvm' --split-input-file --verify-each %s | FileCheck %s --check-prefix=SURVIVE

// `tl.flip` arrives as a bit-reversal idiom: reshape the 2^k dim into k
// trailing 2s, then reduce-xor / expand_dims / broadcast / xori over each new
// axis, then reshape back. Rewrite the whole chain to the native ascend.flip
// before later passes see the inflated rank. hivm.vflip only flips the
// innermost dim, so leading-dim flips must keep the bit-reversal form.
// Non-integer element types are wrapped in a to(int)/to(dtype) bitcast
// round-trip; the pattern absorbs it and flips the original element type.

// STRUCT-LABEL: tt.func @flip_1d_i32
// STRUCT-NOT: tt.reduce
// STRUCT-NOT: tt.reshape
// STRUCT: ascend.flip %{{.*}} : tensor<8xi32> -> tensor<8xi32>
tt.func @flip_1d_i32(%x: tensor<8xi32>) -> tensor<8xi32> {
  %split = tt.reshape %x : tensor<8xi32> -> tensor<2x2x2xi32>
  %r0 = "tt.reduce"(%split) <{axis = 0 : i32}> ({
  ^bb0(%a0: i32, %b0: i32):
    %z0 = arith.xori %a0, %b0 : i32
    tt.reduce.return %z0 : i32
  }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
  %e0 = tt.expand_dims %r0 {axis = 0 : i32} : tensor<2x2xi32> -> tensor<1x2x2xi32>
  %ba0 = tt.broadcast %e0 : tensor<1x2x2xi32> -> tensor<2x2x2xi32>
  %xor0 = arith.xori %split, %ba0 : tensor<2x2x2xi32>
  %r1 = "tt.reduce"(%xor0) <{axis = 1 : i32}> ({
  ^bb0(%a1: i32, %b1: i32):
    %z1 = arith.xori %a1, %b1 : i32
    tt.reduce.return %z1 : i32
  }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
  %e1 = tt.expand_dims %r1 {axis = 1 : i32} : tensor<2x2xi32> -> tensor<2x1x2xi32>
  %ba1 = tt.broadcast %e1 : tensor<2x1x2xi32> -> tensor<2x2x2xi32>
  %xor1 = arith.xori %xor0, %ba1 : tensor<2x2x2xi32>
  %r2 = "tt.reduce"(%xor1) <{axis = 2 : i32}> ({
  ^bb0(%a2: i32, %b2: i32):
    %z2 = arith.xori %a2, %b2 : i32
    tt.reduce.return %z2 : i32
  }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
  %e2 = tt.expand_dims %r2 {axis = 2 : i32} : tensor<2x2xi32> -> tensor<2x2x1xi32>
  %ba2 = tt.broadcast %e2 : tensor<2x2x1xi32> -> tensor<2x2x2xi32>
  %xor2 = arith.xori %xor1, %ba2 : tensor<2x2x2xi32>
  %collapse = tt.reshape %xor2 : tensor<2x2x2xi32> -> tensor<8xi32>
  tt.return %collapse : tensor<8xi32>
}

// -----

// The flip is built on the pre-bitcast element type, so both the leading
// bitcast and the trailing bitcast disappear.
// STRUCT-LABEL: tt.func @flip_1d_f32_bitcast
// STRUCT-NOT: tt.bitcast
// STRUCT-NOT: tt.reduce
// STRUCT: ascend.flip %{{.*}} : tensor<8xf32> -> tensor<8xf32>
tt.func @flip_1d_f32_bitcast(%x: tensor<8xf32>) -> tensor<8xf32> {
  %xi = tt.bitcast %x : tensor<8xf32> -> tensor<8xi32>
  %split = tt.reshape %xi : tensor<8xi32> -> tensor<2x2x2xi32>
  %r0 = "tt.reduce"(%split) <{axis = 0 : i32}> ({
  ^bb0(%a0: i32, %b0: i32):
    %z0 = arith.xori %a0, %b0 : i32
    tt.reduce.return %z0 : i32
  }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
  %e0 = tt.expand_dims %r0 {axis = 0 : i32} : tensor<2x2xi32> -> tensor<1x2x2xi32>
  %ba0 = tt.broadcast %e0 : tensor<1x2x2xi32> -> tensor<2x2x2xi32>
  %xor0 = arith.xori %split, %ba0 : tensor<2x2x2xi32>
  %r1 = "tt.reduce"(%xor0) <{axis = 1 : i32}> ({
  ^bb0(%a1: i32, %b1: i32):
    %z1 = arith.xori %a1, %b1 : i32
    tt.reduce.return %z1 : i32
  }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
  %e1 = tt.expand_dims %r1 {axis = 1 : i32} : tensor<2x2xi32> -> tensor<2x1x2xi32>
  %ba1 = tt.broadcast %e1 : tensor<2x1x2xi32> -> tensor<2x2x2xi32>
  %xor1 = arith.xori %xor0, %ba1 : tensor<2x2x2xi32>
  %r2 = "tt.reduce"(%xor1) <{axis = 2 : i32}> ({
  ^bb0(%a2: i32, %b2: i32):
    %z2 = arith.xori %a2, %b2 : i32
    tt.reduce.return %z2 : i32
  }) : (tensor<2x2x2xi32>) -> tensor<2x2xi32>
  %e2 = tt.expand_dims %r2 {axis = 2 : i32} : tensor<2x2xi32> -> tensor<2x2x1xi32>
  %ba2 = tt.broadcast %e2 : tensor<2x2x1xi32> -> tensor<2x2x2xi32>
  %xor2 = arith.xori %xor1, %ba2 : tensor<2x2x2xi32>
  %collapse = tt.reshape %xor2 : tensor<2x2x2xi32> -> tensor<8xi32>
  %out = tt.bitcast %collapse : tensor<8xi32> -> tensor<8xf32>
  tt.return %out : tensor<8xf32>
}

// -----

// Leading-dim flip: hivm.vflip is last-dim only, so the idiom must stay.
// STRUCT-LABEL: tt.func @flip_2d_leading_dim
// STRUCT-NOT: ascend.flip
// STRUCT: tt.reduce
tt.func @flip_2d_leading_dim(%x: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %split = tt.reshape %x : tensor<4x8xi32> -> tensor<2x2x8xi32>
  %r0 = "tt.reduce"(%split) <{axis = 0 : i32}> ({
  ^bb0(%a0: i32, %b0: i32):
    %z0 = arith.xori %a0, %b0 : i32
    tt.reduce.return %z0 : i32
  }) : (tensor<2x2x8xi32>) -> tensor<2x8xi32>
  %e0 = tt.expand_dims %r0 {axis = 0 : i32} : tensor<2x8xi32> -> tensor<1x2x8xi32>
  %ba0 = tt.broadcast %e0 : tensor<1x2x8xi32> -> tensor<2x2x8xi32>
  %xor0 = arith.xori %split, %ba0 : tensor<2x2x8xi32>
  %r1 = "tt.reduce"(%xor0) <{axis = 1 : i32}> ({
  ^bb0(%a1: i32, %b1: i32):
    %z1 = arith.xori %a1, %b1 : i32
    tt.reduce.return %z1 : i32
  }) : (tensor<2x2x8xi32>) -> tensor<2x8xi32>
  %e1 = tt.expand_dims %r1 {axis = 1 : i32} : tensor<2x8xi32> -> tensor<2x1x8xi32>
  %ba1 = tt.broadcast %e1 : tensor<2x1x8xi32> -> tensor<2x2x8xi32>
  %xor1 = arith.xori %xor0, %ba1 : tensor<2x2x8xi32>
  %collapse = tt.reshape %xor1 : tensor<2x2x8xi32> -> tensor<4x8xi32>
  tt.return %collapse : tensor<4x8xi32>
}

// Every rewritten function must carry ascend.flip across the passes between
// triton-to-structured and triton-to-linalg.
// SURVIVE-LABEL: tt.func @flip_1d_i32
// SURVIVE: ascend.flip
// SURVIVE-LABEL: tt.func @flip_1d_f32_bitcast
// SURVIVE: ascend.flip
// SURVIVE-LABEL: tt.func @flip_2d_leading_dim
// SURVIVE-NOT: ascend.flip
// SURVIVE: tt.reduce
