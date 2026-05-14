// RUN: triton-opt --pass-pipeline="builtin.module(triton-to-unstructure{compile-on-910-95=true force-simt-template=false},triton-to-linalg{compile-on-910-95=true enable-nd2nz-on-vector=false global-kernel=false named-ops=true})" --split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  tt.func public @dot_scale_fp8_kernel(%a_base: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %stride_a1: i32 {tt.divisibility = 16 : i32}, %a_scale: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %b_base: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %stride_b0: i32 {tt.divisibility = 16 : i32}, %out: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %out_ptr = arith.constant dense<32> : tensor<32x1xi32>
    %c = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %scale_a_ptr = arith.constant dense<4> : tensor<32x1xi32>
    %a_ptr = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %a_ptr_0 = tt.expand_dims %a_ptr {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %a_ptr_1 = tt.splat %a_base : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>>
    %a_ptr_2 = tt.addptr %a_ptr_1, %a_ptr_0 : tensor<32x1x!tt.ptr<i8>>, tensor<32x1xi32>
    %a_ptr_3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %a_ptr_4 = tt.expand_dims %a_ptr_3 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %a_ptr_5 = tt.splat %stride_a1 : i32 -> tensor<1x128xi32>
    %a_ptr_6 = arith.muli %a_ptr_4, %a_ptr_5 : tensor<1x128xi32>
    %a_ptr_7 = tt.broadcast %a_ptr_2 : tensor<32x1x!tt.ptr<i8>> -> tensor<32x128x!tt.ptr<i8>>
    %a_ptr_8 = tt.broadcast %a_ptr_6 : tensor<1x128xi32> -> tensor<32x128xi32>
    %a_ptr_9 = tt.addptr %a_ptr_7, %a_ptr_8 : tensor<32x128x!tt.ptr<i8>>, tensor<32x128xi32>
    %b_ptr = tt.expand_dims %a_ptr_3 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %b_ptr_10 = tt.splat %stride_b0 : i32 -> tensor<128x1xi32>
    %b_ptr_11 = arith.muli %b_ptr, %b_ptr_10 : tensor<128x1xi32>
    %b_ptr_12 = tt.splat %b_base : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>>
    %b_ptr_13 = tt.addptr %b_ptr_12, %b_ptr_11 : tensor<128x1x!tt.ptr<i8>>, tensor<128x1xi32>
    %b_ptr_14 = tt.expand_dims %a_ptr {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %b_ptr_15 = tt.broadcast %b_ptr_13 : tensor<128x1x!tt.ptr<i8>> -> tensor<128x32x!tt.ptr<i8>>
    %b_ptr_16 = tt.broadcast %b_ptr_14 : tensor<1x32xi32> -> tensor<128x32xi32>
    %b_ptr_17 = tt.addptr %b_ptr_15, %b_ptr_16 : tensor<128x32x!tt.ptr<i8>>, tensor<128x32xi32>
    %scale_a_ptr_18 = arith.muli %a_ptr_0, %scale_a_ptr : tensor<32x1xi32>
    %scale_a_ptr_19 = tt.splat %a_scale : !tt.ptr<i8> -> tensor<32x1x!tt.ptr<i8>>
    %scale_a_ptr_20 = tt.addptr %scale_a_ptr_19, %scale_a_ptr_18 : tensor<32x1x!tt.ptr<i8>>, tensor<32x1xi32>
    %scale_a_ptr_21 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %scale_a_ptr_22 = tt.expand_dims %scale_a_ptr_21 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %scale_a_ptr_23 = tt.broadcast %scale_a_ptr_20 : tensor<32x1x!tt.ptr<i8>> -> tensor<32x4x!tt.ptr<i8>>
    %scale_a_ptr_24 = tt.broadcast %scale_a_ptr_22 : tensor<1x4xi32> -> tensor<32x4xi32>
    %scale_a_ptr_25 = tt.addptr %scale_a_ptr_23, %scale_a_ptr_24 : tensor<32x4x!tt.ptr<i8>>, tensor<32x4xi32>
    %a = tt.load %a_ptr_9 : tensor<32x128x!tt.ptr<i8>>
    %b = tt.load %b_ptr_17 : tensor<128x32x!tt.ptr<i8>>
    %a_scale_26 = tt.load %scale_a_ptr_25 : tensor<32x4x!tt.ptr<i8>>
    %c_27 = tt.bitcast %a : tensor<32x128xi8> -> tensor<32x128xf8E5M2>
    %c_28 = tt.bitcast %b : tensor<128x32xi8> -> tensor<128x32xf8E4M3FN>
    %c_29 = tt.dot_scaled %c_27 scale %a_scale_26, %c_28, %c lhs = e5m2 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E5M2>, tensor<32x4xi8> * tensor<128x32xf8E4M3FN> -> tensor<32x32xf32>
    %out_ptr_30 = arith.muli %a_ptr_0, %out_ptr : tensor<32x1xi32>
    %out_ptr_31 = tt.splat %out : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %out_ptr_32 = tt.addptr %out_ptr_31, %out_ptr_30 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %out_ptr_33 = tt.broadcast %out_ptr_32 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %out_ptr_34 = tt.broadcast %b_ptr_14 : tensor<1x32xi32> -> tensor<32x32xi32>
    %out_ptr_35 = tt.addptr %out_ptr_33, %out_ptr_34 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %0 = arith.truncf %c_29 : tensor<32x32xf32> to tensor<32x32xbf16>
    tt.store %out_ptr_35, %0 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}


// CHECK-LABEL: func.func @dot_scale_fp8_kernel
// CHECK: hfusion.matmul_mx
// CHECK-DAG: tensor<32x128xf8E5M2>
// CHECK-DAG: tensor<128x32xf8E4M3FN>
// CHECK-DAG: tensor<32x4xi8>
// CHECK: outs([[OUT:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: arith.truncf {{.*}} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK: bufferization.materialize_in_destination
