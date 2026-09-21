// RUN: triton-opt %s --triton-to-structured '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt' | FileCheck %s

// A bool load with data-dependent offsets. The frontend bitcasts ptr<i1> to ptr<i8> for
// bool accesses. The indirect load has to keep the ptr<i1> base: UseAnalysis marks that
// operand MetaUse, so a bitcast left here is erased by MetaUseEraser and TritonToLinalg
// can no longer materialize a memref for the base.

// CHECK-LABEL: tt.func public @indirect_load_bool
// CHECK-NOT:   tt.bitcast %arg0
// CHECK:       ascend.unstructured_load %arg0 : <i1>, %{{[0-9]+}} : tensor<8xi64> unstructured_dims = [0]{{.*}}-> tensor<8xi8>
tt.func public @indirect_load_bool(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i8>) attributes {noinline = false} {
  %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %3 = tt.load %2 : tensor<8x!tt.ptr<i32>>
  %4 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<8x!tt.ptr<i1>>
  %5 = tt.addptr %4, %3 : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
  %6 = tt.bitcast %5 : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>
  %7 = tt.load %6 : tensor<8x!tt.ptr<i8>>
  %8 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
  %9 = tt.addptr %8, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
  tt.store %9, %7 : tensor<8x!tt.ptr<i8>>
  tt.return
}
