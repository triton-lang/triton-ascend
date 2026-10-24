// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// Volatility must survive the original tt.load -> copy -> tensor pipeline.

// -----

// CHECK-LABEL: func.func @direct_plain
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @direct_plain(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %Snapshot: !tt.ptr<f16>, %C: !tt.ptr<f32>) attributes {noinline = false} {
    %c = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst = arith.constant dense<64> : tensor<16x1xi32>
    %bp = arith.constant dense<64> : tensor<64x1xi32>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bp_0 = tt.expand_dims %n {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %bp_1 = arith.muli %bp_0, %bp : tensor<64x1xi32>
    %bp_2 = tt.splat %B : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
    %bp_3 = tt.addptr %bp_2, %bp_1 : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
    %bp_4 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %bp_5 = tt.broadcast %bp_3 : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
    %bp_6 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<64x64xi32>
    %bp_7 = tt.addptr %bp_5, %bp_6 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
    %b = tt.load %bp_7 : tensor<64x64x!tt.ptr<f16>>
    %a = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_8 = arith.muli %a, %cst : tensor<16x1xi32>
    %a_9 = tt.splat %A : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
    %a_10 = tt.addptr %a_9, %a_8 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
    %a_11 = tt.broadcast %a_10 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x64x!tt.ptr<f16>>
    %a_12 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<16x64xi32>
    %a_13 = tt.addptr %a_11, %a_12 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %a_14 = tt.load %a_13 : tensor<16x64x!tt.ptr<f16>>
    %c_15 = tt.dot_scaled %a_14, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %a_8 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %3 = tt.addptr %2, %a_12 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    tt.store %3, %c_15 cacheModifier = wt : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @direct_volatile
// CHECK: memref.alloc() : memref<64x64xf16>
// CHECK: memref.copy {{.*}} {isVolatile = true}
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// CHECK-NOT: memref.alloc() : memref<32x32xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @direct_volatile(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %Snapshot: !tt.ptr<f16>, %C: !tt.ptr<f32>) attributes {noinline = false} {
    %c = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst = arith.constant dense<64> : tensor<16x1xi32>
    %bp = arith.constant dense<64> : tensor<64x1xi32>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bp_0 = tt.expand_dims %n {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %bp_1 = arith.muli %bp_0, %bp : tensor<64x1xi32>
    %bp_2 = tt.splat %B : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
    %bp_3 = tt.addptr %bp_2, %bp_1 : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
    %bp_4 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %bp_5 = tt.broadcast %bp_3 : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
    %bp_6 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<64x64xi32>
    %bp_7 = tt.addptr %bp_5, %bp_6 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
    %b = tt.load %bp_7 {isVolatile = true} : tensor<64x64x!tt.ptr<f16>>
    %a = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_8 = arith.muli %a, %cst : tensor<16x1xi32>
    %a_9 = tt.splat %A : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
    %a_10 = tt.addptr %a_9, %a_8 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
    %a_11 = tt.broadcast %a_10 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x64x!tt.ptr<f16>>
    %a_12 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<16x64xi32>
    %a_13 = tt.addptr %a_11, %a_12 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %a_14 = tt.load %a_13 : tensor<16x64x!tt.ptr<f16>>
    %c_15 = tt.dot_scaled %a_14, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %a_8 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %3 = tt.addptr %2, %a_12 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    tt.store %3, %c_15 cacheModifier = wt : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @native_direct_plain
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<16x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>, tensor<16x2xi8>, tensor<64x2xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @native_direct_plain(%A: !tt.ptr<f8E4M3FN>, %B: !tt.ptr<f8E4M3FN>, %Snapshot: !tt.ptr<f8E4M3FN>, %C: !tt.ptr<f32>) attributes {noinline = false} {
    %c = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst = arith.constant dense<64> : tensor<16x1xi32>
    %bp = arith.constant dense<64> : tensor<64x1xi32>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bp_0 = tt.expand_dims %n {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %bp_1 = arith.muli %bp_0, %bp : tensor<64x1xi32>
    %bp_2 = tt.splat %B : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %bp_3 = tt.addptr %bp_2, %bp_1 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %bp_4 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %bp_5 = tt.broadcast %bp_3 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %bp_6 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<64x64xi32>
    %bp_7 = tt.addptr %bp_5, %bp_6 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %b = tt.load %bp_7 : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %a = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_8 = arith.muli %a, %cst : tensor<16x1xi32>
    %a_9 = tt.splat %A : !tt.ptr<f8E4M3FN> -> tensor<16x1x!tt.ptr<f8E4M3FN>>
    %a_10 = tt.addptr %a_9, %a_8 : tensor<16x1x!tt.ptr<f8E4M3FN>>, tensor<16x1xi32>
    %a_11 = tt.broadcast %a_10 : tensor<16x1x!tt.ptr<f8E4M3FN>> -> tensor<16x64x!tt.ptr<f8E4M3FN>>
    %a_12 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<16x64xi32>
    %a_13 = tt.addptr %a_11, %a_12 : tensor<16x64x!tt.ptr<f8E4M3FN>>, tensor<16x64xi32>
    %a_14 = tt.load %a_13 : tensor<16x64x!tt.ptr<f8E4M3FN>>
    %c_15 = tt.dot_scaled %a_14, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x64xf8E4M3FN> -> tensor<16x64xf32>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %a_8 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %3 = tt.addptr %2, %a_12 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    tt.store %3, %c_15 cacheModifier = wt : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @native_direct_volatile
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = true}
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<16x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>, tensor<16x2xi8>, tensor<64x2xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @native_direct_volatile(%A: !tt.ptr<f8E4M3FN>, %B: !tt.ptr<f8E4M3FN>, %Snapshot: !tt.ptr<f8E4M3FN>, %C: !tt.ptr<f32>) attributes {noinline = false} {
    %c = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst = arith.constant dense<64> : tensor<16x1xi32>
    %bp = arith.constant dense<64> : tensor<64x1xi32>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bp_0 = tt.expand_dims %n {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %bp_1 = arith.muli %bp_0, %bp : tensor<64x1xi32>
    %bp_2 = tt.splat %B : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %bp_3 = tt.addptr %bp_2, %bp_1 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %bp_4 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %bp_5 = tt.broadcast %bp_3 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %bp_6 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<64x64xi32>
    %bp_7 = tt.addptr %bp_5, %bp_6 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %b = tt.load %bp_7 {isVolatile = true} : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %a = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_8 = arith.muli %a, %cst : tensor<16x1xi32>
    %a_9 = tt.splat %A : !tt.ptr<f8E4M3FN> -> tensor<16x1x!tt.ptr<f8E4M3FN>>
    %a_10 = tt.addptr %a_9, %a_8 : tensor<16x1x!tt.ptr<f8E4M3FN>>, tensor<16x1xi32>
    %a_11 = tt.broadcast %a_10 : tensor<16x1x!tt.ptr<f8E4M3FN>> -> tensor<16x64x!tt.ptr<f8E4M3FN>>
    %a_12 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<16x64xi32>
    %a_13 = tt.addptr %a_11, %a_12 : tensor<16x64x!tt.ptr<f8E4M3FN>>, tensor<16x64xi32>
    %a_14 = tt.load %a_13 : tensor<16x64x!tt.ptr<f8E4M3FN>>
    %c_15 = tt.dot_scaled %a_14, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x64xf8E4M3FN> -> tensor<16x64xf32>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %a_8 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %3 = tt.addptr %2, %a_12 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    tt.store %3, %c_15 cacheModifier = wt : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @native_transpose_plain
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<16x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x64xf8E4M3FN>) outs({{.*}} : tensor<64x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>, tensor<16x2xi8>, tensor<64x2xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @native_transpose_plain(%A: !tt.ptr<f8E4M3FN>, %B: !tt.ptr<f8E4M3FN>, %Snapshot: !tt.ptr<f8E4M3FN>, %C: !tt.ptr<f32>) attributes {noinline = false} {
    %c = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst = arith.constant dense<64> : tensor<16x1xi32>
    %bp = arith.constant dense<64> : tensor<64x1xi32>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bp_0 = tt.expand_dims %n {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %bp_1 = arith.muli %bp_0, %bp : tensor<64x1xi32>
    %bp_2 = tt.splat %B : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %bp_3 = tt.addptr %bp_2, %bp_1 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %bp_4 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %bp_5 = tt.broadcast %bp_3 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %bp_6 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<64x64xi32>
    %bp_7 = tt.addptr %bp_5, %bp_6 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %b = tt.load %bp_7 : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %a = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_8 = arith.muli %a, %cst : tensor<16x1xi32>
    %a_9 = tt.splat %A : !tt.ptr<f8E4M3FN> -> tensor<16x1x!tt.ptr<f8E4M3FN>>
    %a_10 = tt.addptr %a_9, %a_8 : tensor<16x1x!tt.ptr<f8E4M3FN>>, tensor<16x1xi32>
    %a_11 = tt.broadcast %a_10 : tensor<16x1x!tt.ptr<f8E4M3FN>> -> tensor<16x64x!tt.ptr<f8E4M3FN>>
    %a_12 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<16x64xi32>
    %a_13 = tt.addptr %a_11, %a_12 : tensor<16x64x!tt.ptr<f8E4M3FN>>, tensor<16x64xi32>
    %a_14 = tt.load %a_13 : tensor<16x64x!tt.ptr<f8E4M3FN>>
    %bt = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf8E4M3FN> -> tensor<64x64xf8E4M3FN>
    %c_15 = tt.dot_scaled %a_14, %bt, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x64xf8E4M3FN> -> tensor<16x64xf32>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %a_8 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %3 = tt.addptr %2, %a_12 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    tt.store %3, %c_15 cacheModifier = wt : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @native_transpose_volatile
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<64x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = true}
// CHECK-NOT: scf.for
// CHECK: memref.alloc() : memref<16x64xf8E4M3FN>
// CHECK-NOT: scf.for
// CHECK: memref.copy {{.*}} {isVolatile = false}
// CHECK-NOT: scf.for
// CHECK: linalg.transpose ins({{.*}} : tensor<64x64xf8E4M3FN>) outs({{.*}} : tensor<64x64xf8E4M3FN>)
// CHECK-NOT: scf.for
// CHECK: hfusion.matmul_mx
// CHECK-SAME: : tensor<16x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>, tensor<16x2xi8>, tensor<64x2xi8>) outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: scf.for
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @native_transpose_volatile(%A: !tt.ptr<f8E4M3FN>, %B: !tt.ptr<f8E4M3FN>, %Snapshot: !tt.ptr<f8E4M3FN>, %C: !tt.ptr<f32>) attributes {noinline = false} {
    %c = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst = arith.constant dense<64> : tensor<16x1xi32>
    %bp = arith.constant dense<64> : tensor<64x1xi32>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bp_0 = tt.expand_dims %n {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %bp_1 = arith.muli %bp_0, %bp : tensor<64x1xi32>
    %bp_2 = tt.splat %B : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %bp_3 = tt.addptr %bp_2, %bp_1 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %bp_4 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %bp_5 = tt.broadcast %bp_3 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
    %bp_6 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<64x64xi32>
    %bp_7 = tt.addptr %bp_5, %bp_6 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
    %b = tt.load %bp_7 {isVolatile = true} : tensor<64x64x!tt.ptr<f8E4M3FN>>
    %a = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_8 = arith.muli %a, %cst : tensor<16x1xi32>
    %a_9 = tt.splat %A : !tt.ptr<f8E4M3FN> -> tensor<16x1x!tt.ptr<f8E4M3FN>>
    %a_10 = tt.addptr %a_9, %a_8 : tensor<16x1x!tt.ptr<f8E4M3FN>>, tensor<16x1xi32>
    %a_11 = tt.broadcast %a_10 : tensor<16x1x!tt.ptr<f8E4M3FN>> -> tensor<16x64x!tt.ptr<f8E4M3FN>>
    %a_12 = tt.broadcast %bp_4 : tensor<1x64xi32> -> tensor<16x64xi32>
    %a_13 = tt.addptr %a_11, %a_12 : tensor<16x64x!tt.ptr<f8E4M3FN>>, tensor<16x64xi32>
    %a_14 = tt.load %a_13 : tensor<16x64x!tt.ptr<f8E4M3FN>>
    %bt = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf8E4M3FN> -> tensor<64x64xf8E4M3FN>
    %c_15 = tt.dot_scaled %a_14, %bt, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x64xf8E4M3FN> -> tensor<16x64xf32>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %1 = tt.addptr %0, %a_8 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %3 = tt.addptr %2, %a_12 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    tt.store %3, %c_15 cacheModifier = wt : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}
