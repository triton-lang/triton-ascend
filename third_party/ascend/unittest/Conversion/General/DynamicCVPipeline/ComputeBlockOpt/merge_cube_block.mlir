// RUN: triton-opt --merge-cube-block %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // ============================================
  // Test Case 1: @test_merge_cube_blocks_with_vector
  // ============================================
  // Scenario: Two cube blocks with same vector predecessors and successors
  // - Vector block 1 (block_id=1) produces vec_tensor1
  // - Cube block 1 (block_id=2) uses vec_tensor1, produces cube_tensor1
  // - Cube block 2 (block_id=3) uses vec_tensor1, produces cube_tensor2
  // - Vector block 2 (block_id=4) uses cube_tensor1 and cube_tensor2
  // Expected: The two cube blocks should be merged (block_id=3 -> block_id=2)
  // Wrapped in two-layer for loop (only innermost should be processed)
  // ============================================
  // CHECK-LABEL: func.func @test_merge_cube_blocks_with_vector
  func.func @test_merge_cube_blocks_with_vector(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        %vec_alloc1 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec_cond1 = arith.constant 1 : i1
        scf.if %vec_cond1 {
          linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 1 : i32}
        %vec_tensor1 = bufferization.to_tensor %vec_alloc1 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // CHECK: %{{.*}} = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_alloc1 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond1 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond1 {
          linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 2 : i32}
        %cube_tensor1 = bufferization.to_tensor %cube_alloc1 restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out1 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        %cube_matmul1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

        // CHECK: %{{.*}} = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_alloc2 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond2 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond2 {
          linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc2 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 3 : i32}
        %cube_tensor2 = bufferization.to_tensor %cube_alloc2 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out2 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        %cube_matmul2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor2 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out2 : tensor<128x128xf32>) -> tensor<128x128xf32>

        %vec_alloc2 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
        %vec_cond2 = arith.constant 1 : i1
        scf.if %vec_cond2 {
          linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec_alloc2 : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        // Uses both cube_matmul1 and cube_matmul2
        %vec_add = arith.addf %cube_matmul1, %cube_matmul2 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %vec_tensor2 = bufferization.to_tensor %vec_alloc2 restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32> to tensor<128x128xf32>

        scf.yield
      }
    }

    return
  }

  // ============================================
  // Test Case 2: @test_no_merge_with_extra_vector_dependency
  // ============================================
  // Scenario: Based on scenario 1, add an extra vector node pointing to one cube block
  // - Vector block 5 (block_id=5) produces vec_tensor1
  // - Vector block 9 (block_id=9) produces vec_tensor3
  // - Cube block 6 (block_id=6) uses vec_tensor1, produces cube_matmul1
  // - Cube block 7 (block_id=7) uses vec_tensor3, produces cube_matmul2
  // - Vector block 8 (block_id=8) uses cube_matmul1 and cube_matmul2
  // Expected: The two cube blocks should NOT be merged (different source nodes)
  // Wrapped in two-layer for loop
  // ============================================
  // CHECK-LABEL: func.func @test_no_merge_with_extra_vector_dependency
  func.func @test_no_merge_with_extra_vector_dependency(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    // Outer for loop
    scf.for %i = %c0 to %c128 step %c1 {
      // Inner for loop (innermost, should be processed)
      scf.for %j = %c0 to %c128 step %c1 {
        // Vector block 1 (predecessor for cube block 1)
        %vec_alloc1 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec_cond1 = arith.constant 1 : i1
        scf.if %vec_cond1 {
          linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
        %vec_tensor1 = bufferization.to_tensor %vec_alloc1 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // Cube block 1 (block_id=2) - uses vec_tensor1
        // CHECK: memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        %cube_alloc1 = memref.alloc() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond1 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond1 {
          linalg.fill {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 6 : i32}
        %cube_tensor1 = bufferization.to_tensor %cube_alloc1 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out1 = tensor.empty() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        %cube_matmul1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Extra vector block (block_id=5) - only used by cube block 2
        %vec_alloc3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec_cond3 = arith.constant 1 : i1
        scf.if %vec_cond3 {
          linalg.fill {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc3 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 9 : i32}
        %vec_tensor3 = bufferization.to_tensor %vec_alloc3 restrict writable {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // Cube block 2 (block_id=3) - should NOT be merged (different source nodes)
        // Uses vec_tensor3
        // CHECK: memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        %cube_alloc2 = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond2 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond2 {
          linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc2 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
        %cube_tensor2 = bufferization.to_tensor %cube_alloc2 restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out2 = tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        // Uses vec_tensor3 as one of the inputs
        %cube_matmul2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor3, %cube_tensor2 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out2 : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Vector block 2 (successor for both cube blocks)
        // Uses both cube_matmul1 and cube_matmul2
        %vec_alloc2 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
        %vec_cond2 = arith.constant 1 : i1
        scf.if %vec_cond2 {
          linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec_alloc2 : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        %vec_add = arith.addf %cube_matmul1, %cube_matmul2 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %vec_tensor2 = bufferization.to_tensor %vec_alloc2 restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32> to tensor<128x128xf32>

        scf.yield
      }
    }

    return
  }

  // ============================================
  // Test Case 3: @test_no_merge_single_layer_for
  // ============================================
  // Scenario: Same as scenario 1, but only one layer of for loop
  // - Vector block 1 (block_id=1) produces vec_tensor1
  // - Cube block 1 (block_id=2) uses vec_tensor1, produces cube_matmul1
  // - Cube block 2 (block_id=3) uses vec_tensor1, produces cube_matmul2
  // - Vector block 2 (block_id=4) uses cube_matmul1 and cube_matmul2
  // Expected: The two cube blocks should NOT be merged (not innermost loop)
  // Only one layer of for loop
  // ============================================
  // CHECK-LABEL: func.func @test_no_merge_single_layer_for
  func.func @test_no_merge_single_layer_for(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    // Single layer for loop (not innermost, should NOT be processed)
    scf.for %i = %c0 to %c128 step %c1 {
      // Vector block 1 (predecessor for both cube blocks)
      %vec_alloc1 = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
      %vec_cond1 = arith.constant 1 : i1
      scf.if %vec_cond1 {
        linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc1 : memref<128x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
      %vec_tensor1 = bufferization.to_tensor %vec_alloc1 restrict writable {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

      // Cube block 1 (block_id=2) - should remain unchanged
      // CHECK: memref.alloc() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
      %cube_alloc1 = memref.alloc() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
      %cube_cond1 = arith.constant 1 : i1
      // CHECK: scf.if %{{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
      scf.if %cube_cond1 {
        linalg.fill {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc1 : memref<128x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 16 : i32}
      %cube_tensor1 = bufferization.to_tensor %cube_alloc1 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
      %cube_out1 = tensor.empty() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
      %cube_matmul1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

      // Cube block 2 (block_id=3) - should remain unchanged
      // CHECK: memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"}
      %cube_alloc2 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
      %cube_cond2 = arith.constant 1 : i1
      // CHECK: scf.if %{{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"}
      scf.if %cube_cond2 {
        linalg.fill {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc2 : memref<128x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 17 : i32}
      %cube_tensor2 = bufferization.to_tensor %cube_alloc2 restrict writable {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
      %cube_out2 = tensor.empty() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"}
      %cube_matmul2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor2 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out2 : tensor<128x128xf32>) -> tensor<128x128xf32>

      // Vector block 2 (successor for both cube blocks)
      // Uses both cube_matmul1 and cube_matmul2
      %vec_alloc2 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
      %vec_cond2 = arith.constant 1 : i1
      scf.if %vec_cond2 {
        linalg.fill {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec_alloc2 : memref<128x128xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 18 : i32}
      %vec_add = arith.addf %cube_matmul1, %cube_matmul2 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
      %vec_tensor2 = bufferization.to_tensor %vec_alloc2 restrict writable {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32> to tensor<128x128xf32>


      %cube_alloc3 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32>
      %cube_cond3 = arith.constant 1 : i1
      scf.if %cube_cond3 {
        linalg.fill {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc3 : memref<128x128xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 19 : i32}
      %cube_tensor3 = bufferization.to_tensor %cube_alloc3 restrict writable {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32> to tensor<128x128xf32>
      %cube_out3 = tensor.empty() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      %cube_matmul3 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor2, %cube_tensor3 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%cube_out3 : tensor<128x128xf32>) -> tensor<128x128xf32>


      %vec_add3 = arith.addf %cube_matmul2, %cube_matmul3 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
      scf.yield
    }

    return
  }

  // ============================================
  // Test Case 4: @test_no_merge_dead_end_downstream_cube
  // ============================================
  // Scenario: Two cube blocks with a common vector successor, where that
  // vector block is followed by another cube block that has NO vector
  // successor downstream. The downstream pattern is
  //   C2/C3 -> V2 -> C5
  // and C5 has no vector successor. The merge pass must refuse to merge
  // C2/C3 because both candidates sit on a cube->vector->cube chain whose
  // terminal cube block is a dead-end (no vector hop after it).
  // Expected: C2 and C3 are NOT merged (both retain their original ids).
  // ============================================
  // CHECK-LABEL: func.func @test_no_merge_dead_end_downstream_cube
  func.func @test_no_merge_dead_end_downstream_cube(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xbf16>, %arg3: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        // Vector block 1 (block_id=21): common vector predecessor of both cubes.
        %vec1_alloc = memref.alloc() {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec1_cond = arith.constant 1 : i1
        scf.if %vec1_cond {
          linalg.fill {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec1_alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 21 : i32}
        %vec1_tensor = bufferization.to_tensor %vec1_alloc restrict writable {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // Cube block C2 (block_id=22) - shares vec1_tensor with C3.
        // CHECK: memref.alloc() {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"}
        %c2_alloc = memref.alloc() {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %c2_cond = arith.constant 1 : i1
        scf.if %c2_cond {
          linalg.fill {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%c2_alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 22 : i32}
        %c2_tensor = bufferization.to_tensor %c2_alloc restrict writable {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %c2_out = tensor.empty() {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"}
        %c2_matmul = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "CUBE"} ins(%vec1_tensor, %c2_tensor : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%c2_out : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Cube block C3 (block_id=23) - shares vec1_tensor with C2.
        // Must remain at block_id 23 (dead-end downstream cube detected).
        // CHECK: memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"}
        %c3_alloc = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %c3_cond = arith.constant 1 : i1
        scf.if %c3_cond {
          linalg.fill {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%c3_alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 23 : i32}
        %c3_tensor = bufferization.to_tensor %c3_alloc restrict writable {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %c3_out = tensor.empty() {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"}
        %c3_matmul = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "CUBE"} ins(%vec1_tensor, %c3_tensor : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%c3_out : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Vector block 2 (block_id=24): common vector successor of C2 and C3.
        // Materialize the result tensor into a memref so the next cube can
        // build a memory dependence on it.
        %vec2_dst = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
        %vec2_cond = arith.constant 1 : i1
        scf.if %vec2_cond {
          linalg.fill {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec2_dst : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 24 : i32}
        %vec2_sum = arith.addf %c2_matmul, %c3_matmul {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        bufferization.materialize_in_destination %vec2_sum in writable %vec2_dst {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x128xf32>, memref<128x128xf32>) -> ()

        // Cube block C5 (block_id=25): the downstream cube reached via V2.
        // It has NO vector successor (dead-end), which triggers the
        // refuse-to-merge rule for the C2/C3 pair.
        // CHECK: memref.alloc() {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"}
        %c5_in = memref.reinterpret_cast %vec2_dst to offset: [%c0], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
        %c5_alloc = memref.alloc() {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32>
        %c5_cond = arith.constant 1 : i1
        scf.if %c5_cond {
          linalg.fill {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_f32 : f32) outs(%c5_alloc : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 25 : i32}
        memref.copy %c5_in, %c5_alloc {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32, strided<[128, 1], offset: ?>> to memref<128x128xf32>
        %c5_tensor = bufferization.to_tensor %c5_alloc restrict writable {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32> to tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"}
        %c5_matmul = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "CUBE"} ins(%c2_matmul, %c5_tensor : tensor<128x128xf32>, tensor<128x128xf32>) outs(%c3_out : tensor<128x128xf32>) -> tensor<128x128xf32>

        scf.yield
      }
    }

    return
  }

  // ============================================
  // Test Case 5: @test_merge_with_vector_after_downstream_cube
  // ============================================
  // Scenario: Same shape as Test Case 4 (C27/C28 -> V29 -> C30), but the
  // downstream cube C30 has a vector successor V31, so the chain still
  // exposes a vector hop and the refuse-to-merge rule does not fire.
  // Expected: C27 and C28 ARE merged (C28 -> C27).
  // ============================================
  // CHECK-LABEL: func.func @test_merge_with_vector_after_downstream_cube
  func.func @test_merge_with_vector_after_downstream_cube(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xbf16>, %arg3: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        // Vector block 1 (block_id=26): shared vector predecessor.
        %vec1_alloc = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec1_cond = arith.constant 1 : i1
        scf.if %vec1_cond {
          linalg.fill {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec1_alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 26 : i32}
        %vec1_tensor = bufferization.to_tensor %vec1_alloc restrict writable {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // Cube block C27 (block_id=27) - the merge target.
        // CHECK: memref.alloc() {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"}
        %c27_alloc = memref.alloc() {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %c27_cond = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %c27_cond {
          linalg.fill {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%c27_alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 27 : i32}
        %c27_tensor = bufferization.to_tensor %c27_alloc restrict writable {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %c27_out = tensor.empty() {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"}
        %c27_matmul = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} ins(%vec1_tensor, %c27_tensor : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%c27_out : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Cube block C28 (block_id=28) - the merge source. Must collapse into C27.
        // CHECK: memref.alloc() {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"}
        %c28_alloc = memref.alloc() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %c28_cond = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %c28_cond {
          linalg.fill {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%c28_alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 28 : i32}
        %c28_tensor = bufferization.to_tensor %c28_alloc restrict writable {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %c28_out = tensor.empty() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"}
        %c28_matmul = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} ins(%vec1_tensor, %c28_tensor : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%c28_out : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Vector block 2 (block_id=29): shared vector successor of the cubes.
        %vec2_dst = memref.alloc() {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
        %vec2_cond = arith.constant 1 : i1
        scf.if %vec2_cond {
          linalg.fill {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec2_dst : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 29 : i32}
        %vec2_sum = arith.addf %c27_matmul, %c28_matmul {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        bufferization.materialize_in_destination %vec2_sum in writable %vec2_dst {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x128xf32>, memref<128x128xf32>) -> ()

        // Cube block C30 (block_id=30): downstream cube reached via V29.
        // Unlike Test Case 4, C30 still has a vector successor V31 below.
        %c30_in = memref.reinterpret_cast %vec2_dst to offset: [%c0], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
        %c30_alloc = memref.alloc() {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32>
        %c30_cond = arith.constant 1 : i1
        scf.if %c30_cond {
          linalg.fill {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_f32 : f32) outs(%c30_alloc : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 30 : i32}
        memref.copy %c30_in, %c30_alloc {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32, strided<[128, 1], offset: ?>> to memref<128x128xf32>
        %c30_tensor = bufferization.to_tensor %c30_alloc restrict writable {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32> to tensor<128x128xf32>
        %c30_out = tensor.empty() {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        %c30_matmul = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} ins(%c27_matmul, %c30_tensor : tensor<128x128xf32>, tensor<128x128xf32>) outs(%c30_out : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Vector block 3 (block_id=31): the vector successor of C30.
        // Its presence is what makes the chain C27/C28 -> V29 -> C30 -> V31
        // still contain a vector hop after C30, so the merge rule does not
        // fire and C27/C28 are merged as usual.
        %vec3_dst = memref.alloc() {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
        %vec3_cond = arith.constant 1 : i1
        scf.if %vec3_cond {
          linalg.fill {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec3_dst : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 31 : i32}
        %vec3_sum = arith.addf %c30_matmul, %c30_matmul {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        bufferization.materialize_in_destination %vec3_sum in writable %vec3_dst {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x128xf32>, memref<128x128xf32>) -> ()

        scf.yield
      }
    }

    return
  }
}
