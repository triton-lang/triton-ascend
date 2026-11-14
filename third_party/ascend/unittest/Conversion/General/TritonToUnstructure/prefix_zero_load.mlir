// RUN: triton-opt %s --triton-to-unstructure --verify-each | FileCheck %s
// CHECK-LABEL: tt.func public @prefix_zero_load
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: tt.load %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}} {DiscreteMemAccess}
// CHECK-NOT: arith.select
// CHECK: tt.return
// Zero padding must be installed in the final loop destination. Retained
// axes still require a masked load with other=0, including empty prefixes.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @prefix_zero_load(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %rows: i32, %lower: i32, %upper: i32) {
    %x = tt.make_range {start = 0 : i32, end = 3 : i32} : tensor<3xi32>
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %two = arith.constant dense<2> : tensor<3xi32>
    %stride = arith.constant dense<16> : tensor<32xi32>
    %row = arith.remsi %x, %two : tensor<3xi32>
    %col = arith.muli %r, %stride : tensor<32xi32>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rowb = tt.broadcast %row2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %colb = tt.broadcast %col2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %offset = arith.addi %rowb, %colb : tensor<3x32xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    %rowsb = tt.splat %rows : i32 -> tensor<3xi32>
    %xm = arith.cmpi slt, %x, %rowsb : tensor<3xi32>
    %lo = tt.splat %lower : i32 -> tensor<32xi32>
    %hi = tt.splat %upper : i32 -> tensor<32xi32>
    %rmlo = arith.cmpi sge, %r, %lo : tensor<32xi32>
    %rmhi = arith.cmpi slt, %r, %hi : tensor<32xi32>
    %rm = arith.andi %rmhi, %rmhi : tensor<32xi1>
    %xm2d = tt.expand_dims %xm {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %rm2d = tt.expand_dims %rm {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %xmb = tt.broadcast %xm2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %rmb = tt.broadcast %rm2d : tensor<1x32xi1> -> tensor<3x32xi1>
    %mask = arith.andi %xmb, %rmb : tensor<3x32xi1>
    %other = arith.constant dense<0.0> : tensor<3x32xf32>
    %value = tt.load %ptr, %mask, %other : tensor<3x32x!tt.ptr<f32>>
    %outstride = arith.constant dense<32> : tensor<3xi32>
    %outrow = arith.muli %x, %outstride : tensor<3xi32>
    %outrow2d = tt.expand_dims %outrow {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %r2d = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %outrowb = tt.broadcast %outrow2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %rb = tt.broadcast %r2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %outoffset = arith.addi %outrowb, %rb : tensor<3x32xi32>
    %outbase = tt.splat %output : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %outptr = tt.addptr %outbase, %outoffset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    tt.store %outptr, %value : tensor<3x32x!tt.ptr<f32>>
    tt.return
  }
}
