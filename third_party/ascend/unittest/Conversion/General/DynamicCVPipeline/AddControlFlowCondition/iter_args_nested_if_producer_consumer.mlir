// RUN: triton-opt --update-for-ops --debug %s 2>&1 | FileCheck %s

// Nested if yielding a tensor iter_arg must not mark the enclosing ssbuffer.if
// as a producer unless that ssbuffer.if itself yields the iter_arg.
//
// IR has:
//   1) a true producer (else directly yields %iter_arg; then forwards via nested if)
//   2) a consumer that only nest-yields %iter_arg, and yields another tensor itself
//
// If the nested yield in (2) were wrongly treated as a producer, analysis would
// hit: [Error]: ... has multiple different producers!
// Correct behavior: one producer + one consumer, and a condition iter_arg is added.
//
// CHECK-NOT: has multiple different producers
// CHECK: Recorded tensor iter_arg dependency
// CHECK-SAME: has 1 producer, 1 consumers
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_nested_if_iter_arg_producer_consumer(
      %arg0: memref<?xi8>, %arg1: memref<?xi8>,
      %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32})
      attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                  global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 1} 0.0 : f32
    %c0 = arith.constant {ssbuffer.block_id = 1} 0 : i32
    %c1 = arith.constant {ssbuffer.block_id = 1} 1 : i32
    %c5 = arith.constant {ssbuffer.block_id = 1} 5 : i32
    %true = arith.constant true

    scope.scope : () -> () {
      %init_empty = tensor.empty() {ssbuffer.block_id = 1} : tensor<64x64xf32>
      %init = linalg.fill {ssbuffer.block_id = 1}
          ins(%cst : f32) outs(%init_empty : tensor<64x64xf32>) -> tensor<64x64xf32>
      %other_empty = tensor.empty() {ssbuffer.block_id = 1} : tensor<64x64xf32>
      %other = linalg.fill {ssbuffer.block_id = 1}
          ins(%cst : f32) outs(%other_empty : tensor<64x64xf32>) -> tensor<64x64xf32>

      %result = scf.for %i = %c0 to %c5 step %c1
          iter_args(%iter_arg = %init) -> (tensor<64x64xf32>) : i32 {
        // True producer: nested yield forwards iter_arg; else yields it directly.
        %prod = scf.if %true -> (tensor<64x64xf32>) {
          %inner = scf.if %true -> (tensor<64x64xf32>) {
            scf.yield %iter_arg : tensor<64x64xf32>
          } else {
            scf.yield %iter_arg : tensor<64x64xf32>
          }
          scf.yield %inner : tensor<64x64xf32>
        } else {
          scf.yield %iter_arg : tensor<64x64xf32>
        } {ssbuffer.if = 1 : i32}

        // Consumer: nested if yields iter_arg, but this ssbuffer.if does not.
        %cons = scf.if %true -> (tensor<64x64xf32>) {
          %inner = scf.if %true -> (tensor<64x64xf32>) {
            scf.yield %iter_arg : tensor<64x64xf32>
          } else {
            scf.yield %iter_arg : tensor<64x64xf32>
          }
          scf.yield %other : tensor<64x64xf32>
        } else {
          scf.yield %other : tensor<64x64xf32>
        } {ssbuffer.if = 2 : i32}

        // Keep %cons live so the consumer ssbuffer.if is not trivially unused.
        %out = arith.addf %prod, %cons : tensor<64x64xf32>
        scf.yield %out : tensor<64x64xf32>
      } {ssbuffer.block_id = 10, ssbuffer.main_loop = 0 : i32}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    return {ssbuffer.core_type = "VECTOR"}
  }
}
