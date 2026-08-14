// RUN: triton-opt --pass-pipeline="builtin.module(triton-to-unstructure{compile-on-910-95=true force-simt-template=true},triton-to-linalg{compile-on-910-95=true enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true})" %s | FileCheck %s

// On A5 SIMT, a block-pointer load with a statically zero stride must not
// materialize a zero-strided memref. BishengIR rejects that layout. Route the
// load through per-element offsets instead; the zero stride naturally produces
// a zero offset increment while the logical boundary check remains intact.
// CHECK-LABEL: func.func private @triton_indirect_load
// CHECK-LABEL: func.func @zero_stride_block_ptr_indirect_a5
// CHECK-NOT: strided<[0
// CHECK: call @triton_indirect_load
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  tt.func public @zero_stride_block_ptr_indirect_a5(
      %src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %shape_m: i32, %shape_n: i32, %offset_m: i32, %offset_n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %shape_m_i64 = arith.extsi %shape_m : i32 to i64
    %shape_n_i64 = arith.extsi %shape_n : i32 to i64
    %src_block = tt.make_tensor_ptr %src, [%shape_m_i64, %shape_n_i64],
        [%c0_i64, %c0_i64], [%offset_m, %offset_n]
        {order = array<i32: 0, 1>} : <tensor<4x4xf32>>
    %value = tt.load %src_block {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
        : !tt.ptr<tensor<4x4xf32>>
    %dst_block = tt.make_tensor_ptr %dst, [%c4_i64, %c4_i64],
        [%c4_i64, %c1_i64], [%c0_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    tt.store %dst_block, %value : !tt.ptr<tensor<4x4xf32>>
    tt.return
  }
}
