// RUN: triton-opt --triton-to-structured="enable-mask-fallback-conversion=false optimize-dynamic-offset=false" %s | FileCheck %s

// The pointer is the canonical form of a non-singleton zero-stride dimension:
// an addptr over the non-zero-stride axis, followed by a one-dimensional
// pointer broadcast. StoreConverter must defer this legal form to the later
// pointer-broadcast lowering instead of rebuilding a lower-rank store.
module {
  tt.func public @zero_stride_store_broadcast(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offsets_2d = tt.expand_dims %offsets {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr_small = tt.addptr %ptr_splat, %offsets_2d : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %ptr = tt.broadcast %ptr_small : tensor<1x64x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %zero = arith.constant 0.000000e+00 : f32
    %value = tt.splat %zero : f32 -> tensor<16x64xf32>
    %mask = arith.constant dense<true> : tensor<16x64xi1>
    tt.store %ptr, %value, %mask : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @zero_stride_store_broadcast
// CHECK: %[[VALUE:.*]] = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
// CHECK: %[[PTR:.*]] = tt.broadcast {{.*}} : tensor<1x64x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
// CHECK: tt.store %[[PTR]], %[[VALUE]] : tensor<16x64x!tt.ptr<f32>>
