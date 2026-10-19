// RUN: triton-opt %s --ssbuf-standardize-op-pattern-match --verify-each -o %t.base
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each -o %t.once
// RUN: diff %t.base %t.once
// RUN: triton-opt %t.once --ssbuf-standardize-op --verify-each -o %t.twice
// RUN: diff %t.once %t.twice

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_pv(%score: tensor<32x8x16x16xf16>, %m: tensor<128xf16>, %lut: tensor<256xi8>, %v: tensor<512x128xf8E4M3FN>, %out: memref<128x128xf32>, %state: memref<128xf16>, %sum: memref<128xf32>) {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 3 : i32
    %step = arith.constant 1 : i32
    %z = arith.constant 0.0 : f32
    %pe = tensor.empty() : tensor<16x8x16x32xf8E4M3FN>
    %me = tensor.empty() : tensor<128xf16>
    %se = tensor.empty() : tensor<128xf32>
    %ce = tensor.empty() : tensor<128x128xf32>
    %init = linalg.fill ins(%z : f32) outs(%ce : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.for %i = %lb to %ub step %step : i32 {
      %p:3 = hivm.hir.custom {symbol = "__builtin_online_softmax_nz", arg_attrs = [{}, {}, {}, {}], gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>) outs(%pe, %me, %se : tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>) -> (tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>)
      %nd = hivm.hir.convert_layout %p#0 output_shape [128, 512] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>} : (tensor<16x8x16x32xf8E4M3FN>) -> tensor<128x512xf8E4M3FN>
      bufferization.materialize_in_destination %p#1 in writable %state : (tensor<128xf16>, memref<128xf16>) -> ()
      bufferization.materialize_in_destination %p#2 in writable %sum : (tensor<128xf32>, memref<128xf32>) -> ()
      %pv = linalg.matmul ins(%nd, %v : tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
      bufferization.materialize_in_destination %pv in writable %out : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
    }
    return
  }

  func.func @ordinary_nd(%arg: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %e = math.exp %arg : tensor<128x128xf32>
    %s = arith.addf %arg, %e : tensor<128x128xf32>
    %result = arith.divf %e, %s : tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }

  func.func @standalone_m32(%score: tensor<32x2x16x16xf16>, %m: tensor<32xf16>, %lut: tensor<256xi8>, %pe: tensor<16x2x16x32xf8E4M3FN>, %me: tensor<32xf16>, %se: tensor<32xf32>) -> (tensor<16x2x16x32xf8E4M3FN>, tensor<32xf16>, tensor<32xf32>) {
    %p:3 = hivm.hir.custom {symbol = "__builtin_online_softmax_nz", arg_attrs = [{}, {}, {}, {}], gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : tensor<32x2x16x16xf16>, tensor<32xf16>, tensor<256xi8>) outs(%pe, %me, %se : tensor<16x2x16x32xf8E4M3FN>, tensor<32xf16>, tensor<32xf32>) -> (tensor<16x2x16x32xf8E4M3FN>, tensor<32xf16>, tensor<32xf32>)
    return %p#0, %p#1, %p#2 : tensor<16x2x16x32xf8E4M3FN>, tensor<32xf16>, tensor<32xf32>
  }

  func.func @standalone_m64(%score: tensor<32x4x16x16xf16>, %m: tensor<64xf16>, %lut: tensor<256xi8>, %pe: tensor<16x4x16x32xf8E4M3FN>, %me: tensor<64xf16>, %se: tensor<64xf32>) -> (tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>) {
    %p:3 = hivm.hir.custom {symbol = "__builtin_online_softmax_nz", arg_attrs = [{}, {}, {}, {}], gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : tensor<32x4x16x16xf16>, tensor<64xf16>, tensor<256xi8>) outs(%pe, %me, %se : tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>) -> (tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>)
    return %p#0, %p#1, %p#2 : tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>
  }

  func.func @standalone_m128(%score: tensor<32x8x16x16xf16>, %m: tensor<128xf16>, %lut: tensor<256xi8>, %pe: tensor<16x8x16x32xf8E4M3FN>, %me: tensor<128xf16>, %se: tensor<128xf32>) -> (tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>) {
    %p:3 = hivm.hir.custom {symbol = "__builtin_online_softmax_nz", arg_attrs = [{}, {}, {}, {}], gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>) outs(%pe, %me, %se : tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>) -> (tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>)
    return %p#0, %p#1, %p#2 : tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>
  }
}
