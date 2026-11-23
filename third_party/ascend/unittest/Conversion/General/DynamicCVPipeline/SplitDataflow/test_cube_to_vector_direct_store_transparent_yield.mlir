// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization, ssbuffer.merge_small_block_first_run_done = true} {
  func.func @test_cube_to_vector_direct_store_transparent_yield(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} dense<256> : tensor<1xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} dense<16> : tensor<2xi64>
    %c1_i32 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_2 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %3 = arith.muli %arg8, %c1_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %4 = arith.index_cast %3 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %0 = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256xf32>
    %1 = linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_2 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %alloc = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<256xf32>
    %11 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<256xf32> to tensor<256xf32>

    %cst_4 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %12 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf32>
    %13 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_4 : f32) outs(%12 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %14 = scf.for %arg50 = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%arg49 = %11) -> (tensor<256xf32>)  : i32 {
      %19:2 = scf.while (%arg15 = %c0_i32, %arg16 = %arg49) : (i32, tensor<256xf32>) -> (i32, tensor<256xf32>) {
        %20 = arith.cmpi slt, %arg15, %c1_i32 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : i32
        scf.condition(%20) {ssbuffer.core_type = "VECTOR"} %arg15, %arg16 : i32, tensor<256xf32>
      } do {
      ^bb0(%arg15: i32, %arg16: tensor<256xf32>):
        %reshape = tensor.reshape %arg16(%cst_0) {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<256xf32>, tensor<2xi64>) -> tensor<16x16xf32>
        %20 = linalg.matmul {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%reshape, %reshape : tensor<16x16xf32>, tensor<16x16xf32>) outs(%13 : tensor<16x16xf32>) -> tensor<16x16xf32>

        %reshape_7 = tensor.reshape %20(%cst) {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<16x16xf32>, tensor<1xi64>) -> tensor<256xf32>

        %21:2 = scf.while (%arg17 = %c0_i32, %arg18 = %reshape_7) : (i32, tensor<256xf32>) -> (i32, tensor<256xf32>) {
          %23 = arith.cmpi slt, %arg17, %c1_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : i32
          scf.condition(%23) {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} %arg17, %arg18 : i32, tensor<256xf32>
        } do {
        ^bb0(%arg17: i32, %arg18: tensor<256xf32>):
          %23:2 = scf.while (%arg19 = %c0_i32, %arg20 = %arg18) : (i32, tensor<256xf32>) -> (i32, tensor<256xf32>) {
            %25 = arith.cmpi slt, %arg19, %c1_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : i32
            scf.condition(%25) {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} %arg19, %arg20 : i32, tensor<256xf32>
          } do {
          ^bb0(%arg19: i32, %arg20: tensor<256xf32>):
            %25 = arith.addf %arg20, %1 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256xf32>
            %26 = arith.addi %arg19, %c1_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : i32
            scf.yield {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR, VECTOR"} %26, %25 : i32, tensor<256xf32>
          } attributes {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR, VECTOR"}
          %24 = arith.addi %arg17, %c1_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : i32
          scf.yield {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR, VECTOR"} %24, %23#1 : i32, tensor<256xf32>
        } attributes {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR, VECTOR"}
        %22 = arith.addi %arg15, %c1_i32 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : i32
        scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"} %22, %21#1 : i32, tensor<256xf32>
      } attributes {ssbuffer.core_type = "VECTOR, VECTOR"}
      scf.yield {ssbuffer.core_type = "VECTOR"} %19#1 : tensor<256xf32>
    } {ssbuffer.core_type = "VECTOR"}

    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [256], strides: [1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%4] [1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %14[0] [%4] [1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256xf32> to tensor<?xf32>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


// CHECK-LABEL: func.func @test_cube_to_vector_direct_store_transparent_yield

// CHECK: hivm.hir.sync_block_set
// CHECK: hivm.hir.sync_block_set
// CHECK: scf.for

// CHECK: scf.while
// CHECK: tensor.reshape
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.copy
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.convert_layout
// CHECK: memref.memory_space_cast
// CHECK: bufferization.to_tensor
// CHECK: linalg.matmul
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.hir.sync_block_set
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast
// CHECK: bufferization.to_tensor
// CHECK: tensor.reshape
// CHECK: scf.while
// CHECK: scf.while
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: hivm.hir.sync_block_set
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: hivm.hir.sync_block_set{{.*}}[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>] flag = 2
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.sync_block_wait

// CHECK: memref.reinterpret_cast
// CHECK: memref.subview
// CHECK: tensor.extract_slice
// CHECK: hivm.hir.sync_block_wait{{.*}}[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 2
// CHECK-NEXT: bufferization.materialize_in_destination
// CHECK-NEXT: return
