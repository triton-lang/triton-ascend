// RUN: triton-opt %s --triton-to-linalg='compile-on-910-95=true' --split-input-file | FileCheck %s --check-prefixes=ON-CHECK-LABEL,ON-CHECK-SAME
// RUN: triton-opt %s --triton-to-linalg='compile-on-910-95=false' --split-input-file | FileCheck %s --check-prefixes=OFF-CHECK-LABEL,OFF-CHECK-SAME

// isSIMTOp() routes math.sin / math.cos on f16/f32 inputs to the SIMT template
// only in mix-mode kernels (module contains a cube op): downstream A5
// normalize rewrites them into Payne-Hanek range reduction with 2/pi table
// gathers.  Pure-AIV kernels keep sin/cos in the vector scope.  bf16 is the
// negative control: the high-precision trait only accepts f16/f32.

// Pure-AIV kernel: no cube op -> sin stays SIMD even with
// compile-on-910-95=true.
// ON-CHECK-LABEL: func.func @sin_f32
// ON-CHECK-SAME: parallel_mode = "simd"
// OFF-CHECK-LABEL: func.func @sin_f32
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @sin_f32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f32>>
    %result = math.sin %input : tensor<1024xf32>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Pure-AIV kernel: cos behaves the same as sin.
// ON-CHECK-LABEL: func.func @cos_f32
// ON-CHECK-SAME: parallel_mode = "simd"
// OFF-CHECK-LABEL: func.func @cos_f32
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @cos_f32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f32>>
    %result = math.cos %input : tensor<1024xf32>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Pure-AIV kernel: f16 input, still SIMD without a cube op.
// ON-CHECK-LABEL: func.func @sin_f16
// ON-CHECK-SAME: parallel_mode = "simd"
// OFF-CHECK-LABEL: func.func @sin_f16
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @sin_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f16>>
    %result = math.sin %input : tensor<1024xf16>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Pure-AIV kernel: f16 cos, still SIMD without a cube op.
// ON-CHECK-LABEL: func.func @cos_f16
// ON-CHECK-SAME: parallel_mode = "simd"
// OFF-CHECK-LABEL: func.func @cos_f16
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @cos_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f16>>
    %result = math.cos %input : tensor<1024xf16>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Mix-mode kernel: the module contains a tt.dot, so mix_mode = "mix" and the
// f32 sin routes to the SIMT template (parallel_mode = "mix_simd_simt").
// With compile-on-910-95=false the sin stays SIMD, but mix_mode is still
// "mix" because the dot is detected regardless of the flag.
// ON-CHECK-LABEL: func.func @sin_f32_mix
// ON-CHECK-SAME: mix_mode = "mix"
// ON-CHECK-SAME: parallel_mode = "mix_simd_simt"
// OFF-CHECK-LABEL: func.func @sin_f32_mix
// OFF-CHECK-SAME: mix_mode = "mix"
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @sin_f32_mix(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f32>>
    %result = math.sin %input : tensor<1024xf32>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f32>>
    %dot_a = arith.constant dense<0.0> : tensor<16x16xf16>
    %dot_acc = arith.constant dense<0.0> : tensor<16x16xf32>
    %dot = tt.dot %dot_a, %dot_a, %dot_acc : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %dot_ptrs = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %dot_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %dot_offsets_2d_r = tt.expand_dims %dot_offsets {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %dot_offsets_2d = tt.broadcast %dot_offsets_2d_r : tensor<16x1xi32> -> tensor<16x16xi32>
    %dot_addrs = tt.addptr %dot_ptrs, %dot_offsets_2d : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %dot_addrs, %dot : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Negative control: bf16 input does not match the f16/f32 filter, so sin
// stays SIMD (no downstream high-precision table gather for bf16), even in a
// mix-mode kernel.
// ON-CHECK-LABEL: func.func @sin_bf16
// ON-CHECK-SAME: parallel_mode = "simd"
// OFF-CHECK-LABEL: func.func @sin_bf16
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @sin_bf16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<bf16>>
    %result = math.sin %input : tensor<1024xbf16>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}
