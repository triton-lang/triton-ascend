// RUN: triton-opt %s --discrete-mask-access-conversion --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=MASK
// RUN: triton-opt %s --discrete-mask-access-conversion --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s

// The selected row determines both the address and whether it may be read.
// Each scalar access must also respect the independent column limit.
// MASK-LABEL: tt.func public @masked_indirect_gather(
// MASK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<4x8xi1>
// MASK: %[[ACTIVE:.*]] = tensor.extract %[[MASK]][
// MASK: tt.load %{{[^,]+}}, %[[ACTIVE]], %{{.*}} {{.*}}: !tt.ptr<f32>
// CHECK-LABEL: func.func @masked_indirect_gather(
// CHECK: %[[OTHER:.*]] = arith.constant -7.000000e+00 : f32
// CHECK-DAG: %[[ROW:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i32
// CHECK-DAG: %[[COL:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i32
// CHECK: %[[ACTIVE:.*]] = arith.andi %[[ROW]], %[[COL]] : i1
// CHECK: scf.if %[[ACTIVE]] -> (f32) {
// CHECK-NEXT: %[[VALUE:.*]] = memref.load
// CHECK-NEXT: scf.yield %[[VALUE]] : f32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[OTHER]] : f32
tt.func public @masked_indirect_gather(%src: !tt.ptr<f32>, %indices: !tt.ptr<i32>, %width: i32) -> tensor<4x8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %ibase = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %iptr = tt.addptr %ibase, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %ids = tt.load %iptr : tensor<4x!tt.ptr<i32>>
  %three = arith.constant dense<3> : tensor<4xi32>
  %rows = arith.andi %ids, %three : tensor<4xi32>
  %stride = arith.constant dense<8> : tensor<4xi32>
  %offset = arith.muli %rows, %stride : tensor<4xi32>
  %r2 = tt.expand_dims %offset {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %rb = tt.broadcast %r2 : tensor<4x1xi32> -> tensor<4x8xi32>
  %cb = tt.broadcast %c2 : tensor<1x8xi32> -> tensor<4x8xi32>
  %address = arith.addi %rb, %cb : tensor<4x8xi32>
  %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %address : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %validrow = arith.cmpi ne, %rows, %three : tensor<4xi32>
  %rpred = tt.expand_dims %validrow {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %rowmask = tt.broadcast %rpred : tensor<4x1xi1> -> tensor<4x8xi1>
  %widths = tt.splat %width : i32 -> tensor<8xi32>
  %validcol = arith.cmpi slt, %c, %widths : tensor<8xi32>
  %cpred = tt.expand_dims %validcol {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %colmask = tt.broadcast %cpred : tensor<1x8xi1> -> tensor<4x8xi1>
  %active = arith.andi %rowmask, %colmask : tensor<4x8xi1>
  %other = arith.constant dense<-7.0> : tensor<4x8xf32>
  %value = tt.load %ptr, %active, %other : tensor<4x8x!tt.ptr<f32>>
  tt.return %value : tensor<4x8xf32>
}
