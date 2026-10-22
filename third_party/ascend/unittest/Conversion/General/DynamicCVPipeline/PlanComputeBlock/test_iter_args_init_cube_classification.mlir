// RUN: triton-opt --op-classifier %s | FileCheck %s

module {

  // ==========================================================================
  // — nested scf.for: the outer loop's per-iter_arg classification must
  // mirror the inner loop's, since each outer iter_arg flows directly into one
  // of the inner iter_args.
  //
  // Pattern:
  //   %outer = scf.for ... iter_args(%a = fill, %b = fill) -> (...) {
  //     %inner = scf.for ... iter_args(%x = %a, %y = %b) -> (...) {
  //       %vx = arith.addf %x, %x          // consumes %x -> VECTOR body user
  //       %mm = linalg.matmul ... %y=%q,%r=%y  // consumes %y -> CUBE body user
  //       scf.yield %vx, %mm
  //     }
  //     scf.yield %inner#0, %inner#1
  //   }
  //
  // Inner-loop classification: %x (VECTOR), %y (CUBE)  -> "VECTOR, CUBE"
  // Outer-loop classification: %a (= %x) VECTOR, %b (= %y) CUBE  -> "VECTOR, CUBE"
  //
  // Before the fix the outer loop was mis-classified as "CUBE, CUBE" because
  // processYieldOperation used the op-level `opCoreTypes[innerFor]` (simplified
  // to OP_CUBE_ONLY since any CUBE -> CUBE_ONLY) instead of the per-operand
  // "VECTOR, CUBE" attribute that Step 8 stamps on the inner for.
  // ==========================================================================
  func.func @nested_for_mixed_iter_arg(
      %a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %q: tensor<64x64xf32>,
      %c0: i32, %c3: i32, %c1: i32) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    // CHECK-LABEL: func.func @nested_for_mixed_iter_arg
    // CHECK:       scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:         %{{.*}} = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:           linalg.matmul {{.*}} outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK:           arith.addf %{{.*}}, %{{.*}} {{.*}} : tensor<64x64xf32>
    // CHECK:           scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:         } {ssbuffer.core_type = "VECTOR, CUBE"}
    // CHECK:         scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:       } {ssbuffer.core_type = "VECTOR, CUBE"}

    %outer:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg_x = %a, %arg_y = %b) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
      %inner:2 = scf.for %j = %c0 to %c3 step %c1 iter_args(%x = %arg_x, %y = %arg_y) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
        // %y feeds matmul (CUBE).
        %mm = linalg.matmul ins(%q, %q : tensor<64x64xf32>, tensor<64x64xf32>)
                            outs(%y : tensor<64x64xf32>) -> tensor<64x64xf32>
        // %x feeds an arith tensor op (VECTOR).
        %vx = arith.addf %x, %x : tensor<64x64xf32>
        scf.yield %vx, %mm : tensor<64x64xf32>, tensor<64x64xf32>
      }
      scf.yield %inner#0, %inner#1 : tensor<64x64xf32>, tensor<64x64xf32>
    }
    return %outer#0, %outer#1 : tensor<64x64xf32>, tensor<64x64xf32>
  }

  // -----

  // Same nested shape but with the inner results yielded in the opposite
  // order, proving the per-operand lookup is positional rather than
  // "any CUBE -> CUBE" at the op level: the outer loop must come out
  // "CUBE, VECTOR" while the inner one stays "VECTOR, CUBE".
  func.func @nested_for_swapped_iter_arg(
      %a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %q: tensor<64x64xf32>,
      %c0: i32, %c3: i32, %c1: i32) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    // CHECK-LABEL: func.func @nested_for_swapped_iter_arg
    // CHECK:       scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:         %{{.*}} = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:           linalg.matmul {{.*}} outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK:           arith.addf %{{.*}}, %{{.*}} {{.*}} : tensor<64x64xf32>
    // CHECK:           scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:         } {ssbuffer.core_type = "VECTOR, CUBE"}
    // CHECK:         scf.yield {ssbuffer.core_type = "CUBE, VECTOR"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:       } {ssbuffer.core_type = "CUBE, VECTOR"}

    %outer:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg_x = %a, %arg_y = %b) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
      %inner:2 = scf.for %j = %c0 to %c3 step %c1 iter_args(%x = %arg_x, %y = %arg_y) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
        // %y feeds matmul (CUBE).
        %mm = linalg.matmul ins(%q, %q : tensor<64x64xf32>, tensor<64x64xf32>)
                            outs(%y : tensor<64x64xf32>) -> tensor<64x64xf32>
        // %x feeds an arith tensor op (VECTOR).
        %vx = arith.addf %x, %x : tensor<64x64xf32>
        scf.yield %vx, %mm : tensor<64x64xf32>, tensor<64x64xf32>
      }
      scf.yield %inner#1, %inner#0 : tensor<64x64xf32>, tensor<64x64xf32>
    }
    return %outer#0, %outer#1 : tensor<64x64xf32>, tensor<64x64xf32>
  }
}
