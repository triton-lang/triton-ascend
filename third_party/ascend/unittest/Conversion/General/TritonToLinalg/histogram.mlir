// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_histogram_i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[NUM_BINS:.*]] = arith.constant 256 : i64
// CHECK: hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>} "__builtin_histogram" ins(%{{.*}}, %[[NUM_BINS]] : tensor<1024xi32>, i64) outs(%[[EMPTY]] : tensor<256xi32>) -> tensor<256xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @test_histogram_i32(%arg0: tensor<1024xi32>) -> tensor<256xi32> attributes {noinline = false} {
    %0 = tt.histogram %arg0 : tensor<1024xi32> -> tensor<256xi32>
    tt.return %0 : tensor<256xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_histogram_i16
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xi16>
// CHECK: %[[NUM_BINS:.*]] = arith.constant 8 : i64
// CHECK: hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>} "__builtin_histogram" ins(%{{.*}}, %[[NUM_BINS]] : tensor<16xi16>, i64) outs(%[[EMPTY]] : tensor<8xi16>) -> tensor<8xi16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @test_histogram_i16(%arg0: tensor<16xi16>) -> tensor<8xi16> attributes {noinline = false} {
    %0 = tt.histogram %arg0 : tensor<16xi16> -> tensor<8xi16>
    tt.return %0 : tensor<8xi16>
  }
}
