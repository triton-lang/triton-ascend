// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=LABEL
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=ROUTE

// Positive axis-discrete load cases:
//   rank-3 pure broadcast + discrete & column prefix
//   rank-2 AND association (left/right) + rank-2 select form
// LABEL checks the routing tag; ROUTE checks the final SIMD form.

// LABEL-LABEL: tt.func @rank3_masks
// LABEL: tt.load {{.*}} {AxisDiscreteMask}
// LABEL: tt.load {{.*}} {AxisDiscreteMask}
// LABEL-LABEL: tt.func @and_left_assoc
// LABEL: tt.load {{.*}} {AxisDiscreteMask}
// LABEL-LABEL: tt.func @and_right_assoc
// LABEL: tt.load {{.*}} {AxisDiscreteMask}
// LABEL-LABEL: tt.func @select_mask
// LABEL: tt.load {{.*}} {AxisDiscreteMask}

// ROUTE-LABEL: tt.func @rank3_masks
// ROUTE: arith.select
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load
// ROUTE-LABEL: tt.func @and_left_assoc
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load
// ROUTE-LABEL: tt.func @and_right_assoc
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load
// ROUTE-LABEL: tt.func @select_mask
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load

tt.func @rank3_masks(%base: !tt.ptr<i32>, %klimit: i32) {
  %c0 = arith.constant dense<0> : tensor<2x2x4xi32>
  %two = arith.constant dense<2> : tensor<2xi32>
  %zero = arith.constant dense<0> : tensor<2xi32>
  %c8 = arith.constant dense<8> : tensor<2x1xi32>
  %c4 = arith.constant dense<4> : tensor<1x2xi32>

  %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %c = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %k = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>

  // disc: (c % 2) == 0, invariant along axes 0 and 2.
  %mod = arith.remsi %c, %two : tensor<2xi32>
  %valid = arith.cmpi eq, %mod, %zero : tensor<2xi32>
  %v1 = tt.expand_dims %valid {axis = 0 : i32} : tensor<2xi1> -> tensor<1x2xi1>
  %v2 = tt.broadcast %v1 : tensor<1x2xi1> -> tensor<2x2xi1>
  %v3 = tt.expand_dims %v2 {axis = 2 : i32} : tensor<2x2xi1> -> tensor<2x2x1xi1>
  %disc = tt.broadcast %v3 : tensor<2x2x1xi1> -> tensor<2x2x4xi1>

  // cont: k < klimit.
  %klim = tt.splat %klimit : i32 -> tensor<4xi32>
  %col = arith.cmpi slt, %k, %klim : tensor<4xi32>
  %col1 = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
  %col2 = tt.expand_dims %col1 {axis = 0 : i32} : tensor<1x4xi1> -> tensor<1x1x4xi1>
  %cont = tt.broadcast %col2 : tensor<1x1x4xi1> -> tensor<2x2x4xi1>
  %mask_and = arith.andi %disc, %cont : tensor<2x2x4xi1>

  %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %r2 = arith.muli %r1, %c8 : tensor<2x1xi32>
  %r3 = tt.expand_dims %r2 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %r4 = tt.broadcast %r3 : tensor<2x1x1xi32> -> tensor<2x2x4xi32>

  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %c2 = arith.muli %c1, %c4 : tensor<1x2xi32>
  %c3 = tt.expand_dims %c2 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
  %c5 = tt.broadcast %c3 : tensor<1x2x1xi32> -> tensor<2x2x4xi32>

  %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %k2 = tt.expand_dims %k1 {axis = 0 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
  %k3 = tt.broadcast %k2 : tensor<1x1x4xi32> -> tensor<2x2x4xi32>

  %off_rc = arith.addi %r4, %c5 : tensor<2x2x4xi32>
  %off = arith.addi %off_rc, %k3 : tensor<2x2x4xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<2x2x4x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<2x2x4x!tt.ptr<i32>>, tensor<2x2x4xi32>
  %v_pure = tt.load %ptrs, %disc, %c0 : tensor<2x2x4x!tt.ptr<i32>>
  %v_and = tt.load %ptrs, %mask_and, %c0 : tensor<2x2x4x!tt.ptr<i32>>
  tt.store %ptrs, %v_pure : tensor<2x2x4x!tt.ptr<i32>>
  tt.store %ptrs, %v_and : tensor<2x2x4x!tt.ptr<i32>>
  tt.return
}

tt.func @and_left_assoc(%base: !tt.ptr<i32>, %limit1: i32, %limit2: i32) {
  %c0 = arith.constant dense<0> : tensor<4x8xi32>
  %c8 = arith.constant dense<8> : tensor<4x1xi32>
  %two = arith.constant dense<2> : tensor<4xi32>
  %zero = arith.constant dense<0> : tensor<4xi32>

  %k = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %g = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>

  %mod = arith.remsi %k, %two : tensor<4xi32>
  %row = arith.cmpi eq, %mod, %zero : tensor<4xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %disc = tt.broadcast %row1 : tensor<4x1xi1> -> tensor<4x8xi1>

  %l1 = tt.splat %limit1 : i32 -> tensor<8xi32>
  %col1 = arith.cmpi slt, %g, %l1 : tensor<8xi32>
  %col1e = tt.expand_dims %col1 {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %cont1 = tt.broadcast %col1e : tensor<1x8xi1> -> tensor<4x8xi1>

  %l2 = tt.splat %limit2 : i32 -> tensor<8xi32>
  %col2 = arith.cmpi slt, %g, %l2 : tensor<8xi32>
  %col2e = tt.expand_dims %col2 {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %cont2 = tt.broadcast %col2e : tensor<1x8xi1> -> tensor<4x8xi1>

  %inner = arith.andi %disc, %cont1 : tensor<4x8xi1>
  %mask = arith.andi %inner, %cont2 : tensor<4x8xi1>

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

tt.func @and_right_assoc(%base: !tt.ptr<i32>, %limit1: i32, %limit2: i32) {
  %c0 = arith.constant dense<0> : tensor<4x8xi32>
  %c8 = arith.constant dense<8> : tensor<4x1xi32>
  %two = arith.constant dense<2> : tensor<4xi32>
  %zero = arith.constant dense<0> : tensor<4xi32>

  %k = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %g = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>

  %mod = arith.remsi %k, %two : tensor<4xi32>
  %row = arith.cmpi eq, %mod, %zero : tensor<4xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %disc = tt.broadcast %row1 : tensor<4x1xi1> -> tensor<4x8xi1>

  %l1 = tt.splat %limit1 : i32 -> tensor<8xi32>
  %col1 = arith.cmpi slt, %g, %l1 : tensor<8xi32>
  %col1e = tt.expand_dims %col1 {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %cont1 = tt.broadcast %col1e : tensor<1x8xi1> -> tensor<4x8xi1>

  %l2 = tt.splat %limit2 : i32 -> tensor<8xi32>
  %col2 = arith.cmpi slt, %g, %l2 : tensor<8xi32>
  %col2e = tt.expand_dims %col2 {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %cont2 = tt.broadcast %col2e : tensor<1x8xi1> -> tensor<4x8xi1>

  %inner = arith.andi %cont1, %cont2 : tensor<4x8xi1>
  %mask = arith.andi %disc, %inner : tensor<4x8xi1>

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

tt.func @select_mask(%base: !tt.ptr<i32>) {
  %c0 = arith.constant dense<0> : tensor<4x8xi32>
  %c8 = arith.constant dense<8> : tensor<4x1xi32>
  %c3 = arith.constant dense<3> : tensor<4xi32>
  %c2 = arith.constant dense<2> : tensor<4xi32>
  %zero = arith.constant dense<0> : tensor<4xi32>

  %k = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %g = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>

  %cond = arith.cmpi slt, %k, %c3 : tensor<4xi32>
  %first = arith.cmpi slt, %k, %c2 : tensor<4xi32>
  %second = arith.cmpi sge, %k, %zero : tensor<4xi32>

  %cond1 = tt.expand_dims %cond {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %condF = tt.broadcast %cond1 : tensor<4x1xi1> -> tensor<4x8xi1>
  %first1 = tt.expand_dims %first {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %firstF = tt.broadcast %first1 : tensor<4x1xi1> -> tensor<4x8xi1>
  %second1 = tt.expand_dims %second {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %secondF = tt.broadcast %second1 : tensor<4x1xi1> -> tensor<4x8xi1>

  %mask = arith.select %condF, %firstF, %secondF : tensor<4x8xi1>, tensor<4x8xi1>

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
