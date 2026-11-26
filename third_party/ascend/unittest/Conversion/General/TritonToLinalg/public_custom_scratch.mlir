// RUN: triton-opt %s --triton-to-linalg --verify-each | FileCheck %s

// The scratch buffer is an address consumer even though it is neither an
// input nor an output. Its addptr stays pure until the custom op converts.
// CHECK-LABEL: func.func @custom_scratch
// CHECK: %[[SCRATCH:.*]] = memref.reinterpret_cast {{.*}}offset: [4], sizes: [1], strides: [1]
// CHECK: hivm.hir.custom
// CHECK-SAME: %[[SCRATCH]]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @custom_scratch(%workspace: !tt.ptr<f32>) {
    %four = arith.constant 4 : i32
    %scratch = tt.addptr %workspace, %four : !tt.ptr<f32>, i32
    "hivm.hir.custom"(%scratch) {name = "scratch_consumer", symbol = "scratch_impl", operandSegmentSizes = array<i32: 0, 0, 1>, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} : (!tt.ptr<f32>) -> ()
    tt.return
  }
}
