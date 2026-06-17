// RUN: triton-opt --update-for-ops --debug %s 2>&1 | FileCheck %s

// Test for tensor iter_arg with only consumer if ops
// CHECK: tensor iter_arg <block argument> of type 'tensor<64x64xf32>' at index: 1 has only consumers, skipped
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_iter_args_only_consumer(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 1} 0.0 : f32
    %c0 = arith.constant {ssbuffer.block_id = 1} 0 : i32
    %c1 = arith.constant {ssbuffer.block_id = 1} 1 : i32
    %c5 = arith.constant {ssbuffer.block_id = 1} 5 : i32
    %true = arith.constant true
    %false = arith.constant false
    %empty_tensor = tensor.empty() : tensor<64x64xf32>

    scope.scope : () -> () {
      // Create initial tensor
      %tensor_init = tensor.empty() {ssbuffer.block_id = 1} : tensor<64x64xf32>
      %filled_tensor = linalg.fill {ssbuffer.block_id = 1} ins(%cst : f32) outs(%tensor_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // Loop with tensor iter_arg
      %result = scf.for %i = %c0 to %c5 step %c1 iter_args(%iter_arg = %filled_tensor) -> (tensor<64x64xf32>) : i32 {
        // Create a separate tensor for yield
        %yield_tensor = tensor.empty() {ssbuffer.block_id = 2} : tensor<64x64xf32>
        %filled_yield = linalg.fill {ssbuffer.block_id = 2} ins(%cst : f32) outs(%yield_tensor : tensor<64x64xf32>) -> tensor<64x64xf32>
        
        // Only consumer if op 1 - uses iter_arg but yields other value
        %val1 = scf.if %true -> (tensor<64x64xf32>) {
          // Use iter_arg in some operation (consumer) - use broadcast with empty dimensions
          %broadcasted = linalg.broadcast {ssbuffer.block_id = 2 : i32} ins(%iter_arg : tensor<64x64xf32>) outs(%yield_tensor : tensor<64x64xf32>) dimensions = []
          // Yield a different value
          scf.yield %filled_yield : tensor<64x64xf32>
        } else {
          // Also use iter_arg in another way
          %broadcasted2 = linalg.broadcast {ssbuffer.block_id = 2 : i32} ins(%iter_arg : tensor<64x64xf32>) outs(%yield_tensor : tensor<64x64xf32>) dimensions = []
          scf.yield %filled_yield : tensor<64x64xf32>
        } {ssbuffer.if = 2 : i32}

        // Yield from loop - yield the same value that came from if
        scf.yield %val1 : tensor<64x64xf32>
      } {ssbuffer.block_id = 10, ssbuffer.main_loop = 0 : i32}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    return {ssbuffer.core_type = "VECTOR"}
  }
}