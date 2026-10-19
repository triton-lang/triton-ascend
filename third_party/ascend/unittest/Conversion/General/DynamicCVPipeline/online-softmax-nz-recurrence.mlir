// RUN: triton-opt %s --ssbuf-standardize-op --plan-compute-block --verify-each | FileCheck %s

// CHECK-LABEL: func.func @single_qk512
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: arith.extf {{.*}}ssbuffer.block_id = [[UPDATE:[0-9]+]]
// CHECK: hivm.hir.custom {{.*}}ssbuffer.block_id = [[UPDATE]]
// CHECK: linalg.matmul
// CHECK: scf.yield
// CHECK: arith.divf
// CHECK: bufferization.materialize_in_destination

#score = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#state = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 16 + d2)>
#lut = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
#prob = affine_map<(d0, d1, d2, d3, d4) -> (d0 floordiv 2, d1, d2, (d0 mod 2) * 16 + d3)>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @single_qk512(%q: tensor<128x128xf8E4M3FN>, %k: tensor<128x512xf8E4M3FN>, %v: tensor<512x128xf8E4M3FN>, %lut: tensor<256xi8>, %scale: tensor<128x128xf32>, %out: memref<128x128xf32>, %ub: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %neg_inf = arith.constant 0xFC00 : f16
    %qk_empty = tensor.empty() : tensor<128x512xf32>
    %qk_init = linalg.fill ins(%zero : f32) outs(%qk_empty : tensor<128x512xf32>) -> tensor<128x512xf32>
    %acc_empty = tensor.empty() : tensor<128x128xf32>
    %acc_init = linalg.fill ins(%zero : f32) outs(%acc_empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %m_empty = tensor.empty() : tensor<128xf16>
    %m_init = linalg.fill ins(%neg_inf : f16) outs(%m_empty : tensor<128xf16>) -> tensor<128xf16>
    %l_empty = tensor.empty() : tensor<128xf32>
    %l_init = linalg.fill ins(%one : f32) outs(%l_empty : tensor<128xf32>) -> tensor<128xf32>
    %p_empty = tensor.empty() : tensor<16x8x16x32xf8E4M3FN>

    %state:3 = scf.for %i = %c0 to %ub step %c1 iter_args(%m = %m_init, %l = %l_init, %acc = %acc_init) -> (tensor<128xf16>, tensor<128xf32>, tensor<128x128xf32>) {
      %qk = linalg.matmul ins(%q, %k : tensor<128x128xf8E4M3FN>, tensor<128x512xf8E4M3FN>) outs(%qk_init : tensor<128x512xf32>) -> tensor<128x512xf32>
      %qk_nz = hivm.hir.convert_layout %qk output_shape [32, 8, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<128x512xf32>) -> tensor<32x8x16x16xf32>
      %scores = arith.truncf %qk_nz : tensor<32x8x16x16xf32> to tensor<32x8x16x16xf16>
      %p:3 = hivm.hir.custom {symbol = "__builtin_online_softmax_nz", indexing_map = [#score, #state, #lut, #prob, #state, #state], iterator_types = [#hivm.iterator_type<opaque>, #hivm.iterator_type<parallel>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>], gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%scores, %m, %lut : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>) outs(%p_empty, %m_empty, %l_empty : tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>) -> (tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>)
      %old_m = arith.extf %m : tensor<128xf16> to tensor<128xf32>
      %new_m = arith.extf %p#1 : tensor<128xf16> to tensor<128xf32>
      %delta = arith.subf %old_m, %new_m : tensor<128xf32>
      %alpha = math.exp %delta : tensor<128xf32>
      %scaled_l = arith.mulf %l, %alpha : tensor<128xf32>
      %next_l = arith.addf %scaled_l, %p#2 : tensor<128xf32>

      %p_nd = hivm.hir.convert_layout %p#0 output_shape [128, 512] {srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, dstLayout = #hivm.data_layout<ND>} : (tensor<16x8x16x32xf8E4M3FN>) -> tensor<128x512xf8E4M3FN>
      %pv = linalg.matmul ins(%p_nd, %v : tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>) outs(%acc_init : tensor<128x128xf32>) -> tensor<128x128xf32>
      %scaled_pv = arith.mulf %pv, %scale : tensor<128x128xf32>
      %alpha_rows = linalg.broadcast ins(%alpha : tensor<128xf32>) outs(%acc_empty : tensor<128x128xf32>) dimensions = [1]
      %scaled_acc = arith.mulf %acc, %alpha_rows : tensor<128x128xf32>
      %next_acc = arith.addf %scaled_acc, %scaled_pv : tensor<128x128xf32>
      scf.yield %p#1, %next_l, %next_acc : tensor<128xf16>, tensor<128xf32>, tensor<128x128xf32>
    }

    %l_rows = linalg.broadcast ins(%state#1 : tensor<128xf32>) outs(%acc_empty : tensor<128x128xf32>) dimensions = [1]
    %normalized = arith.divf %state#2, %l_rows : tensor<128x128xf32>
    bufferization.materialize_in_destination %normalized in writable %out : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
    return
  }
}
