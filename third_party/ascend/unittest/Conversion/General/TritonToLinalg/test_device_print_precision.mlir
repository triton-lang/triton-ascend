// RUN: triton-opt --pass-pipeline="builtin.module(auto-blockify{auto-blockify-size=1},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-on-910-95=false force-simt-template=true},triton-to-annotation,triton-to-unstructure{compile-on-910-95=false force-simt-template=true},triton-to-hivm,triton-to-hfusion,triton-to-llvm,bubble-up-operation,triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},triton-to-linalg{compile-on-910-95=false enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true})" --split-input-file %s | FileCheck %s


module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @kernel_max_2d_precision_fix(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %3 = arith.muli %2, %cst : tensor<16x1xi32>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %5 = tt.broadcast %3 : tensor<16x1xi32> -> tensor<16x32xi32>
    %6 = tt.broadcast %4 : tensor<1x32xi32> -> tensor<16x32xi32>
    %7 = arith.addi %5, %6 : tensor<16x32xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x32x!tt.ptr<f16>>
    %9 = tt.addptr %8, %7 : tensor<16x32x!tt.ptr<f16>>, tensor<16x32xi32>
    %10 = tt.load %9 : tensor<16x32x!tt.ptr<f16>>
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %10 : tensor<16x32xf16>
    %11 = arith.extf %10 : tensor<16x32xf16> to tensor<16x32xf32>
    %12 = "tt.reduce"(%11) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %16 = arith.maxnumf %arg2, %arg3 : f32
      tt.reduce.return %16 : f32
    }) : (tensor<16x32xf32>) -> tensor<16xf32>
    tt.print " ret: " {hex = false, isSigned = array<i32: 0>} : %12 : tensor<16xf32>
    %13 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %14 = tt.addptr %13, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    %15 = arith.truncf %12 : tensor<16xf32> to tensor<16xf16>
    tt.store %14, %15 : tensor<16x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func private @triton_print_0(tensor<16x32xf16>)
// CHECK-LABEL: func.func private @triton_print_1(tensor<16xf16>)

// CHECK:         %[[EXT:.+]] = arith.extf {{.+}} : tensor<16x32xf16> to tensor<16x32xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill {{.+}} -> tensor<16xf32>
// CHECK:         %[[REDUCED:.+]] = linalg.reduce ins(%[[EXT]] : tensor<16x32xf32>) outs(%[[FILL]] : tensor<16xf32>) dimensions = [1]
// CHECK:           arith.maxnumf
// CHECK:           linalg.yield
// CHECK:         %[[TRUNC:.+]] = arith.truncf %[[REDUCED]] : tensor<16xf32> to tensor<16xf16>
// CHECK:         call @triton_print_1(%[[TRUNC]]) : (tensor<16xf16>) -> ()
// CHECK:         bufferization.materialize_in_destination %[[TRUNC]]