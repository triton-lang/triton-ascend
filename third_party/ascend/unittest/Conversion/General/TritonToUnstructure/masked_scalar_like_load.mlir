// RUN: triton-opt --pass-pipeline='any(triton-control-flow-opt,triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-mode=simd_simt_template compile-on-910-95=true},triton-to-annotation,triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true})' --verify-each %s | FileCheck %s
// CHECK-LABEL: tt.func public @block_batching_memory
// CHECK: arith.index_castui
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
// CHECK: scf.yield

module {
  tt.func public @block_batching_memory(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32, %k: i32) {
    %p = tt.get_program_id x : i32
    %one = arith.constant 1.0 : f32
    %zero_i = arith.constant 0 : i32
    %huge = arith.constant 1099511627776 : i64
    %invalid = tt.addptr %x, %huge : !tt.ptr<f32>, i64
    %xp = tt.splat %invalid : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %enabled = arith.cmpi slt, %k, %zero_i : i32
    %mask = tt.splat %enabled : i1 -> tensor<4xi1>
    %other = arith.constant dense<7.0> : tensor<4xf32>
    %value = tt.load %xp, %mask, %other : tensor<4x!tt.ptr<f32>>
    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bv = tt.splat %base : i32 -> tensor<4xi32>
    %offset = arith.addi %bv, %range : tensor<4xi32>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ya = tt.addptr %yp, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %ya, %value : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
