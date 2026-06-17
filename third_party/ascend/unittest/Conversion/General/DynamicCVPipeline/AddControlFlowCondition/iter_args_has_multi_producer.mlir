// RUN: triton-opt --update-for-ops --debug %s 2>&1 | FileCheck %s

// Test for tensor iter_arg with two if producers - should fail with error
// CHECK: [Error]: tensor iter_arg <block argument> of type 'tensor<64x64xf32>' at index: 1 has multiple different producers!
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_iter_args_multi_producer(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 1} 0.0 : f32
    %c0 = arith.constant {ssbuffer.block_id = 1} 0 : i32
    %c1 = arith.constant {ssbuffer.block_id = 1} 1 : i32
    %c5 = arith.constant {ssbuffer.block_id = 1} 5 : i32
    %true = arith.constant true
    %false = arith.constant false
    
    scope.scope : () -> () {
      // Create initial tensor
      %tensor_init = tensor.empty() {ssbuffer.block_id = 1} : tensor<64x64xf32>
      %filled_tensor = linalg.fill {ssbuffer.block_id = 1} ins(%cst : f32) outs(%tensor_init : tensor<64x64xf32>) -> tensor<64x64xf32>
      
      // Loop with tensor iter_arg
      %result = scf.for %i = %c0 to %c5 step %c1 iter_args(%iter_arg = %filled_tensor) -> (tensor<64x64xf32>) : i32 {
        // First if that yields the tensor
        %val1 = scf.if %true -> (tensor<64x64xf32>) {
          %new_tensor1 = tensor.empty() {ssbuffer.block_id = 2} : tensor<64x64xf32>
          %filled1 = linalg.fill {ssbuffer.block_id = 2} ins(%cst : f32) outs(%new_tensor1 : tensor<64x64xf32>) -> tensor<64x64xf32>
          scf.yield %filled1 : tensor<64x64xf32>
        } else {
          scf.yield %iter_arg : tensor<64x64xf32>
        } {ssbuffer.if = 2 : i32}
        
        // Second if that also yields the tensor
        %val2 = scf.if %false -> (tensor<64x64xf32>) {
          %new_tensor2 = tensor.empty() {ssbuffer.block_id = 3} : tensor<64x64xf32>
          %filled2 = linalg.fill {ssbuffer.block_id = 3} ins(%cst : f32) outs(%new_tensor2 : tensor<64x64xf32>) -> tensor<64x64xf32>
          scf.yield %filled2 : tensor<64x64xf32>
        } else {
          scf.yield %iter_arg : tensor<64x64xf32>
        } {ssbuffer.if = 3 : i32}
        
        // Yield from loop
        scf.yield %val2 : tensor<64x64xf32>
      } {ssbuffer.block_id = 10, ssbuffer.main_loop = 0 : i32}
      
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    
    return {ssbuffer.core_type = "VECTOR"}
  }
}