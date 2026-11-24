// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=PASS
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=ROUTE

// Negative controls for the axis-discrete carve-out:
//   causal   - lower-triangular mask, not a rectangle
//   xori     - arith.xori is not understood, must stay conservative
//   c1       - per-row variable-length prefix, no constant axis

// PASS-LABEL: tt.func @causal_mask_negative
// PASS: tt.load {{.*}} {MixCompileDiscreteMask}
// PASS-LABEL: tt.func @xori_mask_negative
// PASS: tt.load {{.*}} {MixCompileDiscreteMask}
// PASS-LABEL: tt.func @c1_mask_negative
// PASS: tt.load {{.*}} {MixCompileDiscreteMask}

// ROUTE-LABEL: tt.func @causal_mask_negative
// ROUTE: ascend.indirect_load
// ROUTE-LABEL: tt.func @xori_mask_negative
// ROUTE: ascend.indirect_load
// ROUTE-LABEL: tt.func @c1_mask_negative
// ROUTE: ascend.indirect_load

tt.func @causal_mask_negative(%base: !tt.ptr<i32>) {
  %c0 = arith.constant dense<0> : tensor<4x8xi32>
  %c8 = arith.constant dense<8> : tensor<4x1xi32>

  %k = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %g = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %kk = tt.broadcast %k1 : tensor<4x1xi32> -> tensor<4x8xi32>
  %gg = tt.broadcast %g1 : tensor<1x8xi32> -> tensor<4x8xi32>

  // mask = k >= g (lower triangular)
  %mask = arith.cmpi sge, %kk, %gg : tensor<4x8xi32>

  %k2 = arith.muli %k1, %c8 : tensor<4x1xi32>
  %k3 = tt.broadcast %k2 : tensor<4x1xi32> -> tensor<4x8xi32>
  %off = arith.addi %k3, %gg : tensor<4x8xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>

  %v = tt.load %ptrs, %mask, %c0 : tensor<4x8x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<4x8x!tt.ptr<i32>>
  tt.return
}

// -----

tt.func @xori_mask_negative(%base: !tt.ptr<i32>) {
  %c0 = arith.constant dense<0> : tensor<4x8xi32>
  %c8 = arith.constant dense<8> : tensor<4x1xi32>
  %two = arith.constant dense<2> : tensor<4xi32>
  %zero = arith.constant dense<0> : tensor<4xi32>
  %true = arith.constant dense<true> : tensor<4x8xi1>

  %k = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %g = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>

  %mod = arith.remsi %k, %two : tensor<4xi32>
  %valid = arith.cmpi eq, %mod, %zero : tensor<4xi32>
  %v1 = tt.expand_dims %valid {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %base_mask = tt.broadcast %v1 : tensor<4x1xi1> -> tensor<4x8xi1>
  %mask = arith.xori %base_mask, %true : tensor<4x8xi1>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %k2 = arith.muli %k1, %c8 : tensor<4x1xi32>
  %k3 = tt.broadcast %k2 : tensor<4x1xi32> -> tensor<4x8xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %g2 = tt.broadcast %g1 : tensor<1x8xi32> -> tensor<4x8xi32>
  %off = arith.addi %k3, %g2 : tensor<4x8xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>

  %v = tt.load %ptrs, %mask, %c0 : tensor<4x8x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<4x8x!tt.ptr<i32>>
  tt.return
}

// -----

tt.func @c1_mask_negative(%base: !tt.ptr<i32>, %boundp: !tt.ptr<i32>) {
  %c0 = arith.constant dense<0> : tensor<4x8xi32>
  %c8 = arith.constant dense<8> : tensor<4x1xi32>

  %k = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %g = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>

  %sb = tt.splat %boundp : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %bk = tt.addptr %sb, %k : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %bound = tt.load %bk : tensor<4x!tt.ptr<i32>>
  %b1 = tt.expand_dims %bound {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %b2 = tt.broadcast %b1 : tensor<4x1xi32> -> tensor<4x8xi32>

  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %g2 = tt.broadcast %g1 : tensor<1x8xi32> -> tensor<4x8xi32>
  %mask = arith.cmpi slt, %g2, %b2 : tensor<4x8xi32>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %k2 = arith.muli %k1, %c8 : tensor<4x1xi32>
  %k3 = tt.broadcast %k2 : tensor<4x1xi32> -> tensor<4x8xi32>
  %off = arith.addi %k3, %g2 : tensor<4x8xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>

  %v = tt.load %ptrs, %mask, %c0 : tensor<4x8x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<4x8x!tt.ptr<i32>>
  tt.return
}
