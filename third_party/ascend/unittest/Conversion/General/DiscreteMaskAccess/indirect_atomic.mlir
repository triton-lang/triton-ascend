// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=True' --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func @structured_disc_mask_atomic_add_2d
// CHECK: tt.atomic_rmw add, acq_rel, gpu, {{.*}} {route_discrete_mask_to_simt} : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
tt.func @structured_disc_mask_atomic_add_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<16> : tensor<4x4xi32>
	%cst_0 = arith.constant dense<2> : tensor<4x4xi32>
	%cst_1 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_1 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = arith.muli %6, %cst_0 : tensor<4x4xi32>
	%8 = arith.cmpi slt, %7, %cst : tensor<4x4xi32>
	%9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%10 = tt.addptr %9, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%11 = tt.load %10 : tensor<4x4x!tt.ptr<i32>>
	%12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%13 = tt.addptr %12, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%14 = tt.atomic_rmw add, acq_rel, gpu, %13, %11, %8 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%16 = tt.addptr %15, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %16, %14, %8 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func @structured_disc_mask_atomic_and_2d
// CHECK: tt.atomic_rmw and, acq_rel, gpu, {{.*}} {route_discrete_mask_to_simt} : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
tt.func @structured_disc_mask_atomic_and_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<16> : tensor<4x4xi32>
	%cst_0 = arith.constant dense<2> : tensor<4x4xi32>
	%cst_1 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_1 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = arith.muli %6, %cst_0 : tensor<4x4xi32>
	%8 = arith.cmpi slt, %7, %cst : tensor<4x4xi32>
	%9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%10 = tt.addptr %9, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%11 = tt.load %10 : tensor<4x4x!tt.ptr<i32>>
	%12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%13 = tt.addptr %12, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%14 = tt.atomic_rmw and, acq_rel, gpu, %13, %11, %8 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%16 = tt.addptr %15, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %16, %14, %8 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func @structured_disc_mask_atomic_or_2d
// CHECK: tt.atomic_rmw or, acq_rel, gpu, {{.*}} {route_discrete_mask_to_simt} : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
tt.func @structured_disc_mask_atomic_or_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<16> : tensor<4x4xi32>
	%cst_0 = arith.constant dense<2> : tensor<4x4xi32>
	%cst_1 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_1 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = arith.muli %6, %cst_0 : tensor<4x4xi32>
	%8 = arith.cmpi slt, %7, %cst : tensor<4x4xi32>
	%9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%10 = tt.addptr %9, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%11 = tt.load %10 : tensor<4x4x!tt.ptr<i32>>
	%12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%13 = tt.addptr %12, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%14 = tt.atomic_rmw or, acq_rel, gpu, %13, %11, %8 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%16 = tt.addptr %15, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %16, %14, %8 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func @structured_disc_mask_atomic_xor_2d
// CHECK: tt.atomic_rmw xor, acq_rel, gpu, {{.*}} {route_discrete_mask_to_simt} : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
tt.func @structured_disc_mask_atomic_xor_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<16> : tensor<4x4xi32>
	%cst_0 = arith.constant dense<2> : tensor<4x4xi32>
	%cst_1 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_1 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = arith.muli %6, %cst_0 : tensor<4x4xi32>
	%8 = arith.cmpi slt, %7, %cst : tensor<4x4xi32>
	%9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%10 = tt.addptr %9, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%11 = tt.load %10 : tensor<4x4x!tt.ptr<i32>>
	%12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%13 = tt.addptr %12, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%14 = tt.atomic_rmw xor, acq_rel, gpu, %13, %11, %8 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%16 = tt.addptr %15, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %16, %14, %8 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func @structured_disc_mask_atomic_xchg_2d
// CHECK: tt.atomic_rmw exch, acq_rel, gpu, {{.*}} {route_discrete_mask_to_simt} : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
tt.func @structured_disc_mask_atomic_xchg_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<16> : tensor<4x4xi32>
	%cst_0 = arith.constant dense<2> : tensor<4x4xi32>
	%cst_1 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_1 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = arith.muli %6, %cst_0 : tensor<4x4xi32>
	%8 = arith.cmpi slt, %7, %cst : tensor<4x4xi32>
	%9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%10 = tt.addptr %9, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%11 = tt.load %10 : tensor<4x4x!tt.ptr<i32>>
	%12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%13 = tt.addptr %12, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%14 = tt.atomic_rmw exch, acq_rel, gpu, %13, %11, %8 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%16 = tt.addptr %15, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %16, %14, %8 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func @structured_disc_mask_atomic_max_2d
// CHECK: tt.atomic_rmw max, acq_rel, gpu, {{.*}} {route_discrete_mask_to_simt} : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
tt.func @structured_disc_mask_atomic_max_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<16> : tensor<4x4xi32>
	%cst_0 = arith.constant dense<2> : tensor<4x4xi32>
	%cst_1 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_1 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = arith.muli %6, %cst_0 : tensor<4x4xi32>
	%8 = arith.cmpi slt, %7, %cst : tensor<4x4xi32>
	%9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%10 = tt.addptr %9, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%11 = tt.load %10 : tensor<4x4x!tt.ptr<i32>>
	%12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%13 = tt.addptr %12, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%14 = tt.atomic_rmw max, acq_rel, gpu, %13, %11, %8 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%16 = tt.addptr %15, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %16, %14, %8 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func @structured_disc_mask_atomic_min_2d
// CHECK: tt.atomic_rmw min, acq_rel, gpu, {{.*}} {route_discrete_mask_to_simt} : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
tt.func @structured_disc_mask_atomic_min_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<16> : tensor<4x4xi32>
	%cst_0 = arith.constant dense<2> : tensor<4x4xi32>
	%cst_1 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_1 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = arith.muli %6, %cst_0 : tensor<4x4xi32>
	%8 = arith.cmpi slt, %7, %cst : tensor<4x4xi32>
	%9 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%10 = tt.addptr %9, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%11 = tt.load %10 : tensor<4x4x!tt.ptr<i32>>
	%12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%13 = tt.addptr %12, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%14 = tt.atomic_rmw min, acq_rel, gpu, %13, %11, %8 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%15 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%16 = tt.addptr %15, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %16, %14, %8 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}
