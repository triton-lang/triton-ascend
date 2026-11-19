// RUN: triton-opt --remove-ssbuf-attr %s | FileCheck %s

// The block dependency graph recorded by DataDependencyAnalysis sits on the
// module rather than on an operation inside it, so it needs a check of its
// own. It has to go: hivmc rejects module attributes it does not recognise.
// CHECK-NOT: ssbuffer.blockDeps

// The variant search's seed and the record of the seed that was applied sit on
// the module too, and go the same way for the same reason. The second is the
// one worth checking: it is written late, by ReorderOpsByBlockId, specifically
// so that something downstream can still read the seed -- which makes
// forgetting to strip it an easy mistake and hivmc the place it shows up.
// CHECK-NOT: ssbuffer.reorder_seed
// CHECK-NOT: ssbuffer.reorder_seed_applied

module attributes {ssbuffer.outer_main_loop,
    ssbuffer.blockDeps = [
    {producer = 1 : i32, consumer = 2 : i32, kind = "v2c"},
    {producer = 2 : i32, consumer = 3 : i32, kind = "mem"}],
    ssbuffer.reorder_seed = -1 : i64,
    ssbuffer.reorder_seed_applied = 7 : i64} {
  // CHECK-LABEL: func.func @remove_explicit_main_loop
  func.func @remove_explicit_main_loop(%lb: index, %ub: index, %step: index) {
    // CHECK: scf.for
    // CHECK-NOT: ssbuffer.explicit_main_loop
    // CHECK-NOT: ssbuffer.main_loop
    // CHECK: return
    scf.for %i = %lb to %ub step %step {
    } {ssbuffer.explicit_main_loop, ssbuffer.main_loop = 0 : i32}
    return
  }

  // CHECK-LABEL: func.func @test_remove_core_type_and_block_id
  func.func @test_remove_core_type_and_block_id(%arg0: memref<1024x1024xf32>) {
    // CHECK: memref.alloc() : memref<1024x1024xf32>
    // CHECK-NOT: ssbuffer.core_type
    // CHECK-NOT: ssbuffer.block_id
    %memref = memref.alloc() {ssbuffer.core_type = "CUBE", ssbuffer.block_id = 1 : i32} : memref<1024x1024xf32>

    // CHECK: bufferization.to_tensor %{{.*}} : memref<1024x1024xf32>
    // CHECK-NOT: ssbuffer.core_type
    %tensor = bufferization.to_tensor %memref {ssbuffer.core_type = "CUBE"} : memref<1024x1024xf32> to tensor<1024x1024xf32>
    return
  }

  // CHECK-LABEL: func.func @test_remove_transfer_and_order_attrs
  func.func @test_remove_transfer_and_order_attrs(%arg0: tensor<128x128xf32>) {
    // CHECK: linalg.matmul
    // CHECK-NOT: ssbuffer.transfer_id
    // CHECK-NOT: ssbuffer.cube_first
    // CHECK-NOT: ssbuffer.vector_first
    %init = tensor.empty() : tensor<128x128xf32>
    %result = linalg.matmul {
      ssbuffer.transfer_id = 42 : i32,
      ssbuffer.cube_first,
      ssbuffer.vector_first
    } ins(%arg0, %arg0 : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>

    return
  }

  // CHECK-LABEL: func.func @test_preserve_other_attributes
  func.func @test_preserve_other_attributes(%arg0: tensor<64x64xf32>) {
    // CHECK: linalg.transpose
    // CHECK-SAME: permutation = [1, 0]
    // CHECK-NOT: ssbuffer.core_type
    %out = tensor.empty() : tensor<64x64xf32>
    %transposed = linalg.transpose
      ins(%arg0 : tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>)
      permutation = [1, 0]
      {ssbuffer.core_type = "VECTOR"}

    return
  }

  // CHECK-LABEL: func.func @test_nested_ops
  func.func @test_nested_ops(%arg0: i1, %arg1: tensor<128xf32>) {
    // CHECK: scf.if
    %res = scf.if %arg0 -> (tensor<128xf32>) {
      // CHECK: arith.addf
      // CHECK-NOT: ssbuffer.core_type
      %add = arith.addf %arg1, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
      scf.yield %add : tensor<128xf32>
    } else {
      scf.yield %arg1 : tensor<128xf32>
    }
    return
  }

  // The loop extension factors recorded by AddControlFlowCondition, which the
  // cost model reads to undo the extension. They must not survive either.
  // CHECK-LABEL: func.func @test_remove_iter_extension
  func.func @test_remove_iter_extension(%lb: index, %ub: index, %st: index) {
    // CHECK: scf.for
    // CHECK-NOT: ssbuffer.iter_extension
    scf.for %i = %lb to %ub step %st {
      scf.yield
    } {ssbuffer.iter_extension = [3 : i32, 1 : i32, 4 : i32]}
    return
  }
}
