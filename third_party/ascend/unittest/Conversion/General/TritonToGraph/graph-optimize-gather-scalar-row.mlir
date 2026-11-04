// RUN: triton-opt %s --verify-each -graph-optimize='target-arch=Ascend910B1 rule-mask=65536 ub-capacity-bytes=157286 compile-mode=simd' | FileCheck %s
// RUN: triton-opt %s --verify-each -graph-optimize='target-arch=Ascend910B1 rule-mask=65536 ub-capacity-bytes=157286 compile-mode=simd' --triton-to-structured | FileCheck %s --check-prefix=LOWERED
// RUN: triton-opt %s --verify-each -graph-optimize='target-arch=Ascend910B1 rule-mask=65536 ub-capacity-bytes=1 compile-mode=simd' | FileCheck %s --check-prefix=DISABLED
// DISABLED-NOT: tt.gather

// LOWERED: tt.gather
// The early Gather rewrite must remain valid through structured conversion.
// The input has no full-row source read: the indirect load must still rewrite.
// Covers a shape with no tt.expand_dims to anchor on: one row per scf.for
// iteration, so the source offset is a scalar splat instead of built up
// axis by axis. See findScalarAxisDimension in GatherOptimizationRule.cpp.

// CHECK: %[[OTHER:[^ ]+]] = arith.constant dense<-7.000000e+00> : tensor<1x4096xf32>
// CHECK: %[[INDICES:[^ ]+]] = arith.select %[[MASK:[^, ]+]],
// CHECK: arith.minsi
// CHECK: arith.maxsi
// CHECK: %[[MIN_ALLOWED:.*]] = arith.constant -1024 : i32
// CHECK: arith.cmpi sge, {{.*}}, %[[MIN_ALLOWED]]
// CHECK: scf.if
// CHECK: %[[SIGN:[^ ]+]] = arith.shrsi %[[INDICES]],
// CHECK: %[[ADJUSTMENT:[^ ]+]] = arith.andi {{.*}}, %[[SIGN]]
// CHECK: %[[NORMALIZED:[^ ]+]] = arith.addi %[[INDICES]], %[[ADJUSTMENT]]
// CHECK-NOT: tt.broadcast
// CHECK:   %[[SOURCE:[^ ]+]] = tt.load {{%[^, ]+}}, {{%[^, ]+}}, {{%[^ ]+}} {gather.optimised.load = "source"} : tensor<1x1024x!tt.ptr<f32>>
// CHECK:   %[[GATHER:[^ ]+]] = tt.gather %[[SOURCE]][%[[NORMALIZED]]] {axis = 1 : i32} : (tensor<1x1024xf32>, tensor<1x4096xi32>) -> tensor<1x4096xf32>
// CHECK:   arith.select %[[MASK]], %[[GATHER]], %[[OTHER]]
// CHECK:   scf.yield
// CHECK:   else
// CHECK:   tt.load {{%[^, ]+}}, %[[MASK]], %[[OTHER]] {gather.optimised.load = "fallback"} : tensor<1x4096x!tt.ptr<f32>>

module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  tt.func public @indirect_load_nd(%src_ptr: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %c4096_i32 = arith.constant 4096 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<-7.000000e+00> : tensor<1x4096xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x4096xi32>
    %c65536_i32 = arith.constant 65536 : i32
    %c1639_i32 = arith.constant 1639 : i32
    %row_begin = tt.get_program_id x : i32
    %row_begin_1 = arith.muli %row_begin, %c1639_i32 : i32
    %row_end = arith.addi %row_begin_1, %c1639_i32 : i32
    %row_end_2 = arith.minsi %row_end, %c65536_i32 : i32
    %idx_offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %idx_offsets_3 = tt.expand_dims %idx_offsets {axis = 0 : i32} : tensor<4096xi32> -> tensor<1x4096xi32>
    %idx = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<1x4096x!tt.ptr<i32>>
    %out = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<1x4096x!tt.ptr<f32>>
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1x4096x!tt.ptr<f32>>
    scf.for %rb = %c0_i32 to %c1639_i32 step %c1_i32  : i32 {
      %in_offsets_base = arith.addi %row_begin_1, %rb : i32
      %in_offsets_base_4 = arith.muli %in_offsets_base, %c1024_i32 : i32
      %mask = arith.cmpi slt, %in_offsets_base, %row_end_2 : i32
      %idx_offsets_5 = arith.muli %in_offsets_base, %c4096_i32 : i32
      %idx_offsets_6 = tt.splat %idx_offsets_5 : i32 -> tensor<1x4096xi32>
      %idx_offsets_7 = arith.addi %idx_offsets_6, %idx_offsets_3 : tensor<1x4096xi32>
      %idx_8 = tt.addptr %idx, %idx_offsets_7 : tensor<1x4096x!tt.ptr<i32>>, tensor<1x4096xi32>
      %idx_9 = tt.splat %mask : i1 -> tensor<1x4096xi1>
      %idx_10 = tt.load %idx_8, %idx_9, %cst_0 : tensor<1x4096x!tt.ptr<i32>>
      %in_offsets_splat = tt.splat %in_offsets_base_4 : i32 -> tensor<1x4096xi32>
      // Keep this identity broadcast: Gather must classify it without asserting.
      %in_offsets = tt.broadcast %in_offsets_splat : tensor<1x4096xi32> -> tensor<1x4096xi32>
      %in_offsets_11 = arith.addi %in_offsets, %idx_10 : tensor<1x4096xi32>
      %out_12 = tt.addptr %out, %in_offsets_11 : tensor<1x4096x!tt.ptr<f32>>, tensor<1x4096xi32>
      %out_13 = tt.load %out_12, %idx_9, %cst : tensor<1x4096x!tt.ptr<f32>>
      %1 = tt.addptr %0, %idx_offsets_7 : tensor<1x4096x!tt.ptr<f32>>, tensor<1x4096xi32>
      tt.store %1, %out_13, %idx_9 : tensor<1x4096x!tt.ptr<f32>>
    }
    tt.return
  }
}
