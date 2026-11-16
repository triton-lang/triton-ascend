// RUN: triton-opt %s --triton-control-flow-opt --verify-each | FileCheck %s --check-prefix=CFO
// RUN: triton-opt %s --triton-control-flow-opt --triton-to-unstructure --bubble-up-operation --triton-to-linalg --canonicalize --cse --verify-each | FileCheck %s --check-prefix=LINALG --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast

// Each lane can choose a different allocation. CFO's offset proof is valid
// even though the rebuilt pointer base is a tensor select, not a scalar splat.
// CFO-LABEL: tt.func public @opaque_tensor_ptr_loop_kernel
// CFO: %[[BASE:.*]] = arith.select {{.*}} : tensor<32xi1>, tensor<32x!tt.ptr<f32>>
// CFO: tt.addptr %[[BASE]], {{.*}} {PointerDescriptorRebuild, PointerDescriptorStructuredAxes = array<i32: 0>}
// LINALG-LABEL: func.func @opaque_tensor_ptr_loop_kernel
// LINALG: scf.for
// LINALG: memref.load
// LINALG: return

module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
  tt.func public @opaque_tensor_ptr_loop_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<i32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_0 = arith.constant dense<128> : tensor<32xi32>
    %cst_1 = arith.constant dense<0> : tensor<32xi32>
    %cst_2 = arith.constant dense<1> : tensor<32xi32>
    %0 = tt.load %arg3 : !tt.ptr<i32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = arith.andi %1, %cst_2 : tensor<32xi32>
    %3 = arith.cmpi eq, %2, %cst_1 : tensor<32xi32>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %5 = tt.addptr %4, %1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %7 = tt.addptr %6, %1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %8 = arith.select %3, %5, %7 : tensor<32xi1>, tensor<32x!tt.ptr<f32>>
    %9 = scf.for %arg4 = %c0_i32 to %0 step %c1_i32 iter_args(%arg5 = %8) -> (tensor<32x!tt.ptr<f32>>)  : i32 {
      %16 = tt.addptr %arg5, %cst_2 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      scf.yield %16 : tensor<32x!tt.ptr<f32>>
    }
    %10 = tt.splat %0 : i32 -> tensor<32xi32>
    %11 = arith.addi %1, %10 : tensor<32xi32>
    %12 = arith.cmpi slt, %11, %cst_0 : tensor<32xi32>
    %13 = tt.load %9, %12, %cst : tensor<32x!tt.ptr<f32>>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %15 = tt.addptr %14, %1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %15, %13 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
