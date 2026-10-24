// RUN: triton-opt %s --triton-control-flow-opt --triton-to-structured --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure -verify-each | FileCheck %s --check-prefix=UNSTRUCTURE
// RUN: triton-opt %s --triton-control-flow-opt --triton-to-structured --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg -verify-each | FileCheck %s --check-prefix=LINALG --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast

// Follow the memory-pass order in compiler.py::ttir_to_linalg. The contiguous
// index load/output store can be handled by Structured, but the data-dependent
// gather must reach Unstructure and consume the public scalar base + full O.
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @public_gather_pipeline(%data: !tt.ptr<i32>, %indices: !tt.ptr<i32>, %output: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %ip = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ips = tt.addptr %ip, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %index = tt.load %ips : tensor<8x!tt.ptr<i32>>
    %dp = tt.splat %data : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptrs = tt.addptr %dp, %index : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %values = tt.load %ptrs : tensor<8x!tt.ptr<i32>>
    %op = tt.splat %output : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ops = tt.addptr %op, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %ops, %values : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
// UNSTRUCTURE-LABEL: @public_gather_pipeline
// UNSTRUCTURE: arith.extsi {{.*}} : tensor<8xi32> to tensor<8xi64>
// UNSTRUCTURE: scf.for
// UNSTRUCTURE: tt.load {{.*}} {DiscreteMemAccess}
// LINALG-LABEL: @public_gather_pipeline
// LINALG: scf.for
// LINALG: memref.load
