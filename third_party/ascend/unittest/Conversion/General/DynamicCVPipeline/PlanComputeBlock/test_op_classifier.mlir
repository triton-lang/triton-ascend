// RUN: triton-opt --op-classifier %s | FileCheck %s

module {
  // CHECK: func.func @test_matmul_to_tensor
  // Test: bufferization.to_tensor upstream pattern -> CUBE
  // CHECK-DAG: memref.alloc{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.materialize_in_destination{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_matmul_to_tensor(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>) {
    %memref = memref.alloc() : memref<1024x1024xf32>
    %tensor = bufferization.to_tensor %memref : memref<1024x1024xf32> to tensor<1024x1024xf32>
    %init = tensor.empty() : tensor<1024x1024xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %result = linalg.matmul ins(%tensor, %tensor : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%filled : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    bufferization.materialize_in_destination %result in writable %memref : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    return
  }

  // CHECK: func.func @test_matmul_to_tensor
  // Test: bufferization.to_tensor upstream pattern -> CUBE
  // CHECK-DAG: memref.alloc{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.materialize_in_destination{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_matmul_to_tensor1(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>) {
    %memref = memref.alloc() : memref<1024x1024xf32>
    %tensor = bufferization.to_tensor %memref : memref<1024x1024xf32> to tensor<1024x1024xf32>
    %init = tensor.empty() : tensor<1024x1024xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %init1 = tensor.empty() : tensor<1024x1024xf32>
    %1 = arith.constant 0.0 : f32
    %filled1 = linalg.fill ins(%1 : f32) outs(%init1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %add = arith.addf %filled, %filled1 : tensor<1024x1024xf32>
    %result = linalg.matmul ins(%tensor, %tensor : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%add : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    bufferization.materialize_in_destination %result in writable %memref : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    return
  }

  // CHECK: func.func @test_transpose_only
  // Test: linalg.transpose -> linalg.matmul (transpose feeds matmul) -> transpose is CUBE
  // Also tests: memref.alloc -> memref.copy -> bufferization.to_tensor -> linalg.transpose -> linalg.matmul
  // CHECK-DAG: memref.alloc{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: memref.copy{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.transpose{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  func.func @test_transpose_only(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
    // Allocate temp buffer for copy
    %temp = memref.alloc() : memref<64x64xf32>
    // Copy data from input to temp buffer
    memref.copy %arg0, %temp : memref<64x64xf32> to memref<64x64xf32>
    // Convert to tensor
    %tensor = bufferization.to_tensor %temp : memref<64x64xf32> to tensor<64x64xf32>
    // Transpose: 64x64 -> 64x64 (permutation [1,0])
    %trans_out = tensor.empty() : tensor<64x64xf32>
    %transposed = linalg.transpose ins(%tensor : tensor<64x64xf32>) outs(%trans_out : tensor<64x64xf32>) permutation = [1, 0]
    // Fill output for matmul
    %fill_out = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%fill_out : tensor<64x64xf32>) -> tensor<64x64xf32>
    // Matmul using transposed input -> transpose should be CUBE
    %result = linalg.matmul ins(%transposed, %tensor : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_transpose_only_add
  // Test: linalg.transpose -> linalg.matmul (transpose feeds matmul) -> transpose is CUBE
  // Also tests: memref.alloc -> memref.copy -> bufferization.to_tensor -> linalg.transpose -> linalg.matmul
  // CHECK-DAG: memref.alloc{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-NOT: memref.alloc
  // CHECK-DAG: memref.copy{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.transpose{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_transpose_only_add(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
    // Allocate temp buffer for copy
    %temp = memref.alloc() : memref<64x64xf32>
    // Copy data from input to temp buffer
    memref.copy %arg0, %temp : memref<64x64xf32> to memref<64x64xf32>
    // Convert to tensor
    %tensor = bufferization.to_tensor %temp : memref<64x64xf32> to tensor<64x64xf32>
    // Transpose: 64x64 -> 64x64 (permutation [1,0])
    %trans_out = tensor.empty() : tensor<64x64xf32>
    %transposed = linalg.transpose ins(%tensor : tensor<64x64xf32>) outs(%trans_out : tensor<64x64xf32>) permutation = [1, 0]
    %result = arith.addf %transposed, %tensor : tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_alloc_is_cube_and_vector
  // Test: linalg.transpose -> linalg.matmul (transpose feeds matmul) -> transpose is CUBE
  // Also tests: memref.alloc -> memref.copy -> bufferization.to_tensor -> linalg.transpose -> linalg.matmul
  // CHECK-DAG: memref.alloc{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: memref.copy{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.transpose{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.transpose{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_alloc_is_cube_and_vector(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
    // Allocate temp buffer for copy
    %temp = memref.alloc() : memref<64x64xf32>
    // Copy data from input to temp buffer
    memref.copy %arg0, %temp : memref<64x64xf32> to memref<64x64xf32>
    // Convert to tensor
    %tensor = bufferization.to_tensor %temp : memref<64x64xf32> to tensor<64x64xf32>
    // Transpose: 64x64 -> 64x64 (permutation [1,0])
    %trans_out = tensor.empty() : tensor<64x64xf32>
    %transposed = linalg.transpose ins(%tensor : tensor<64x64xf32>) outs(%trans_out : tensor<64x64xf32>) permutation = [1, 0]
    // Fill output for matmul
    %fill_out = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%fill_out : tensor<64x64xf32>) -> tensor<64x64xf32>
    // Matmul using transposed input -> transpose should be CUBE
    %result = linalg.matmul ins(%transposed, %tensor : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %result2 = arith.addf %transposed, %tensor : tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_matmul_store
  // Test: tensor.extract_slice and hivm.hir.store downstream patterns -> CUBE
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: tensor.extract_slice{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: hivm.hir.store{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_matmul_store(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>) {
    %memref_a = bufferization.to_tensor %arg0 : memref<256x256xf32> to tensor<256x256xf32>
    %memref_b = bufferization.to_tensor %arg1 : memref<256x256xf32> to tensor<256x256xf32>
    %init = tensor.empty() : tensor<256x256xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
    %result = linalg.matmul ins(%memref_a, %memref_b : tensor<256x256xf32>, tensor<256x256xf32>) outs(%filled : tensor<256x256xf32>) -> tensor<256x256xf32>
    %slice = tensor.extract_slice %result[0, 0] [128, 128] [1, 1] : tensor<256x256xf32> to tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg2[%c0, %c0] [128, 128] [1, 1] : memref<256x256xf32> to memref<128x128xf32, strided<[256, 1], offset: ?>>
    hivm.hir.store ins(%slice : tensor<128x128xf32>) outs(%subview : memref<128x128xf32, strided<[256, 1], offset: ?>>)
    return
  }

  // CHECK: func.func @test_vector_ops
  // Test: element-wise operations -> VECTOR
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: arith.mulf{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: arith.subf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_vector_ops(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) {
    %a = arith.addf %arg0, %arg1 : tensor<1024xf32>
    %b = arith.mulf %a, %arg0 : tensor<1024xf32>
    %c = arith.subf %b, %arg1 : tensor<1024xf32>
    return
  }

  // CHECK: func.func @test_scf_if
  // Test: scf.if with matmul inside -> matmul is CUBE
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: scf.yield{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_scf_if(%arg0: i1, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>) {
    %init = tensor.empty() : tensor<512x512xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
    %result = scf.if %arg0 -> (tensor<512x512xf32>) {
      %matmul_result = linalg.matmul ins(%arg1, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%filled : tensor<512x512xf32>) -> tensor<512x512xf32>
      scf.yield %matmul_result : tensor<512x512xf32>
    } else {
      scf.yield %filled : tensor<512x512xf32>
    }
    return
  }

  // CHECK: func.func @test_scf_for_loop
  // Test: scf.for loop with operations inside -> operations are VECTOR
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_scf_for_loop(%arg0: tensor<512x512xf32>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index
    %init = tensor.empty() : tensor<512x512xf32>
    %zero = arith.constant 0.0 : f32
    scf.for %i = %lb to %ub step %step {
      %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
      %add = arith.addf %filled, %arg0 : tensor<512x512xf32>
    }
    return
  }

  // CHECK: func.func @test_memref_matmul_and_vector
  // Test: memref input with alloc/copy/fill, matmul and element-wise both use the same tensor
  // Note: bufferization.to_tensor bridges memref<->tensor, gets CUBE for matmul, VECTOR for add
  // CHECK-DAG: memref.alloc{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: memref.copy{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor %{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor %{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: tensor.empty{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.constant{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_memref_matmul_and_vector(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %temp = memref.alloc() : memref<64x64xf32>
    memref.copy %arg0, %temp : memref<64x64xf32> to memref<64x64xf32>
    %tensor = bufferization.to_tensor %temp : memref<64x64xf32> to tensor<64x64xf32>
    %init = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %result = linalg.matmul ins(%filled, %tensor : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %add = arith.addf %result, %tensor : tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_scf_if_result_mixed_cube_vector_users
  // Test: scf.if result used by both CUBE (matmul) and VECTOR (arith.addf) operations.
  // This triggers hasMixedUsers=true -> CUBE_AND_VECTOR conflict handling.
  // The scf.yield and scf.if should be annotated with the mixed core_type.
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: scf.yield{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_scf_if_result_mixed_cube_vector_users(%arg0: i1, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) {
    %init = tensor.empty() : tensor<128x128xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    %cond_result = scf.if %arg0 -> (tensor<128x128xf32>) {
      // then branch: matmul (CUBE) - matmul feeds into both matmul and add outside
      %matmul_result = linalg.matmul ins(%arg1, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%filled : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %matmul_result : tensor<128x128xf32>
    } else {
      // else branch: fill (CUBE)
      scf.yield %filled : tensor<128x128xf32>
    }
    // cond_result is used by both matmul (CUBE) and add (VECTOR) -> mixed users
    %matmul2_result = linalg.matmul ins(%cond_result, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%filled : tensor<128x128xf32>) -> tensor<128x128xf32>
    %add_result = arith.addf %cond_result, %arg2 : tensor<128x128xf32>
    return
  }

  // CHECK: func.func @test_scf_for_iter_arg_mixed_users
  // Test: scf.for iter_arg is used directly by both matmul (CUBE) inside loop and
  // arith.addf (VECTOR) outside loop -> triggers hasMixedUsers on iter_arg.
  // handleSCFYieldAndIterArgConflicts collects dependency chain, marks shared ops
  // as CUBE_AND_VECTOR, propagates core types.
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_scf_for_iter_arg_mixed_users(%arg0: tensor<64x64xf32>) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    %init = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %initial = linalg.fill ins(%zero : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // iter_arg carries %initial; inside loop, matmul uses the iter_arg (CUBE).
    // outside loop, arith.addf uses the loop result which depends on iter_arg (VECTOR).
    %loop_result = scf.for %i = %lb to %ub step %step iter_args(%initial_iter = %initial) -> tensor<64x64xf32> {
      %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
      %matmul_result = linalg.matmul ins(%initial_iter, %initial_iter : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %matmul_result : tensor<64x64xf32>
    }
    %add_result = arith.addf %loop_result, %initial : tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_cube_and_vector_split_simple
  // Test: A single shared operation (linalg.fill) is used by both matmul (CUBE)
  // and arith.addf (VECTOR). After classification, fill should be CUBE_AND_VECTOR,
  // triggering handleCubeAndVector to split it into CUBE_ONLY (for matmul) and
  // VECTOR_ONLY (for add) versions.
  // Both original and cloned operations should pass CHECK-DAG.
  // CHECK-DAG: tensor.empty{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: tensor.empty{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: arith.constant{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.constant{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_cube_and_vector_split_simple(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) {
    %init = tensor.empty() : tensor<128x128xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    // %filled feeds matmul (CUBE) and is also used by add (VECTOR)
    %matmul_result = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%filled : tensor<128x128xf32>) -> tensor<128x128xf32>
    %add_result = arith.addf %filled, %matmul_result : tensor<128x128xf32>
    return
  }

  // CHECK: func.func @test_cube_and_vector_split_chain
  // Test: A chain of operations where the middle op is shared between CUBE and VECTOR.
  //   fill → matmul (CUBE)     fill → add (VECTOR)
  // After split, the fill that feeds matmul is CUBE_ONLY, and a cloned fill for add is VECTOR_ONLY.
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_cube_and_vector_split_chain(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) {
    %init = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %matmul_result = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %add_result = arith.addf %filled, %arg0 : tensor<64x64xf32>
    %mul_result = arith.mulf %add_result, %matmul_result : tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_cube_and_vector_split_multiple_users
  // Test: Multiple matmul (CUBE) and multiple arith (VECTOR) users share the same fill.
  // handleCubeAndVector should split the shared fill into one CUBE version and one VECTOR
  // version, with both matmul ops using the CUBE version and all arith ops using the VECTOR clone.
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: arith.mulf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_cube_and_vector_split_multiple_users(%arg0: tensor<64x64xf32>) {
    %init = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mat0 = linalg.matmul ins(%arg0, %arg0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mat1 = linalg.matmul ins(%mat0, %arg0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %add = arith.addf %filled, %arg0 : tensor<64x64xf32>
    %mul = arith.mulf %add, %mat1 : tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_cube_and_vector_split_with_alloc
  // Test: memref.alloc -> to_tensor -> fill -> {matmul(CUBE), add(VECTOR)}.
  // The fill is shared between CUBE and VECTOR paths and will be split.
  // CHECK-DAG: memref.alloc{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: bufferization.to_tensor{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_cube_and_vector_split_with_alloc(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
    %alloc = memref.alloc() : memref<64x64xf32>
    %tensor = bufferization.to_tensor %alloc : memref<64x64xf32> to tensor<64x64xf32>
    %init = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %matmul_result = linalg.matmul ins(%tensor, %tensor : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %add_result = arith.addf %filled, %tensor : tensor<64x64xf32>
    return
  }

  // CHECK: func.func @test_scf_scalar_with_cube_and_vector_users
  // Test: scf.for carrying a tensor through iterations with only scalar arithmetic
  // inside. The tensor iter_arg is used by both matmul (CUBE) and addf (VECTOR)
  // inside the loop. Since there's no matmul seed inside the loop, the scalar
  // body (arith.addi/muli) propagates VECTOR classification from downstream.
  // The fill outside the loop feeds the VECTOR path only (through iter_arg) and
  // is classified VECTOR. The matmul and add inside the loop are correctly CUBE
  // and VECTOR respectively.
  // CHECK-DAG: arith.addi{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: arith.muli{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.fill{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-DAG: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK-DAG: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_scf_scalar_with_cube_and_vector_users(%arg0: i32, %arg1: i32) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x32xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    // Loop carries %filled tensor through iterations with scalar ops alongside.
    // Inside the loop, %filled is used by both matmul (CUBE) and addf (VECTOR).
    // Since no matmul seed is inside the loop, scalar body propagates VECTOR.
    %loop_result:2 = scf.for %i = %lb to %ub step %step iter_args(%iter = %arg0, %fill_iter = %filled) -> (i32, tensor<32x32xf32>) {
      %sum = arith.addi %iter, %arg1 : i32
      %prod = arith.muli %sum, %arg1 : i32
      %mat1 = linalg.matmul ins(%fill_iter, %fill_iter : tensor<32x32xf32>, tensor<32x32xf32>) outs(%fill_iter : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %prod, %mat1 : i32, tensor<32x32xf32>
    }
    %mat = linalg.matmul ins(%loop_result#1, %loop_result#1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%loop_result#1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %add = arith.addf %loop_result#1, %loop_result#1 : tensor<32x32xf32>
    return
  }

  // CHECK: func.func @test_linalg_generic_with_block
  // Test: linalg.generic with an explicit block body. The generic has a block
  // containing arith.mulf and arith.addf. These inner ops should NOT receive
  // ssbuffer.core_type attributes (they inherit from the parent generic only).
  // The generic's output uses the VECTOR version of the split fill, so generic
  // is classified VECTOR. Its result feeds both matmul (CUBE) and addf (VECTOR).
  //
  // The linalg.generic block body is verified with CHECK/CHECK-NOT (sequential).
  // Since CHECK-NOT for inner ops consumes the stream past linalg.yield, outer
  // ops that appear earlier (linalg.fill) are verified with CHECK-DAG before
  // the linalg.generic check.
  // CHECK: linalg.fill{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK: linalg.generic{{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK: ^bb0
  // CHECK: arith.mulf
  // CHECK-NOT: ssbuffer.core_type
  // CHECK: arith.addf
  // CHECK-NOT: ssbuffer.core_type
  // CHECK: linalg.yield
  // CHECK: linalg.fill{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK: linalg.matmul{{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK: arith.addf{{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @test_linalg_generic_with_block(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x64xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    // linalg.generic with explicit block: computes element-wise mul + add
    %generic_result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%filled : tensor<64x64xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %mul, %c : f32
      linalg.yield %add : f32
    } -> tensor<64x64xf32>
    %mat = linalg.matmul ins(%generic_result, %arg0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    %add = arith.addf %generic_result, %mat : tensor<64x64xf32>
    return
  }
}
