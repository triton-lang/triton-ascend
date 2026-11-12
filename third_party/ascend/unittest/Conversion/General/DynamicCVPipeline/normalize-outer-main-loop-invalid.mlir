// RUN: triton-opt --normalize-outer-main-loop --split-input-file --verify-diagnostics %s

func.func @three_inner_loops(%n: index, %a: tensor<4x4xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  // expected-error @+1 {{outer mainloop supports at most two direct inner scf.for loops and no deeper loops}}
  scf.for %task = %zero to %n step %one {
    scf.for %j = %zero to %n step %one {
      %product = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) -> tensor<4x4xf32>
      %sum = arith.addf %product, %a : tensor<4x4xf32>
    }
    scf.for %j = %zero to %n step %one {}
    scf.for %j = %zero to %n step %one {}
  } {ssbuffer.outer_main_loop}
  return
}

// -----

func.func @deeper_inner_loop(%n: index, %a: tensor<4x4xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  // expected-error @+1 {{outer mainloop supports at most two direct inner scf.for loops and no deeper loops}}
  scf.for %task = %zero to %n step %one {
    scf.for %j = %zero to %n step %one {
      scf.for %k = %zero to %n step %one {
        %product = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) -> tensor<4x4xf32>
        %sum = arith.addf %product, %a : tensor<4x4xf32>
      }
    }
  } {ssbuffer.outer_main_loop}
  return
}

// -----

func.func @inner_inside_if(%n: index, %condition: i1, %a: tensor<4x4xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  // expected-error @+1 {{outer mainloop supports at most two direct inner scf.for loops and no deeper loops}}
  scf.for %task = %zero to %n step %one {
    scf.if %condition {
      scf.for %j = %zero to %n step %one {
        %product = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) -> tensor<4x4xf32>
        %sum = arith.addf %product, %a : tensor<4x4xf32>
      }
    }
  } {ssbuffer.outer_main_loop}
  return
}

// -----

func.func @mutable_event_metadata(%n: index, %counts: memref<?xindex>, %a: tensor<4x4xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %task = %zero to %n step %one {
    %count = memref.load %counts[%task] : memref<?xindex>
    scf.for %j = %zero to %count step %one {
      %product = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) -> tensor<4x4xf32>
      %sum = arith.addf %product, %a : tensor<4x4xf32>
    }
    // expected-error @+1 {{outer event metadata must stay invariant across tasks}}
    memref.store %one, %counts[%task] : memref<?xindex>
  } {ssbuffer.outer_main_loop}
  return
}
