// RUN: triton-opt --plan-cube-block %s | FileCheck %s

module {

  // S208. Test cycle detection across sub-regions (e.g., scf.if).
  // The memory dependence graph (memGraph) only tracks memory dependencies within the same block/region.
  // Thus, directly tracing from an operation inside scf.if (linalg.fill) cannot reach its main-block consumer (linalg.matmul).
  //
  // By resolving the inner-region operation to its main-block ancestor (scf.if), the cycle detector
  // correctly identifies the cyclic dependency: group (%alloc) -> scf.if (external) -> group (%result).
  // This prevents invalid grouping that would otherwise cause a topological scheduling deadlock.

  // CHECK-LABEL: func.func @test_subregion_dependency_cycle(
  // CHECK:       [[ALLOC:%[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[TC_CUBE0:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK:       scf.if %{{.*}} {
  // CHECK:         linalg.fill {{.*}} {ssbuffer.block_id = [[TC_CUBE1:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK:       }
  // CHECK:       memref.copy {{.*}} {ssbuffer.block_id = [[TC_CUBE2:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK:       linalg.matmul {input_precision = "ieee", ssbuffer.block_id = [[TC_CUBE3:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @test_subregion_dependency_cycle(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: tensor<32x32xf32>, %cond: i1) -> tensor<32x32xf32> {
    %alloc = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<32x32xf32>

    scf.if %cond {
      %cst = arith.constant 0.0 : f32
      linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
    }

    memref.copy %arg0, %alloc {ssbuffer.core_type = "CUBE"} : memref<32x32xf32> to memref<32x32xf32>
    %tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "CUBE"} : memref<32x32xf32>
    %out = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %result = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
      ins(%tensor, %arg2 : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>

    return %result : tensor<32x32xf32>
  }
}
