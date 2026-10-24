// RUN: triton-opt %s --ssbuf-split-matmul --verify-each | FileCheck %s
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each > /dev/null

// Mark the nearest SCF ancestor of each matmul as limited in CUBE.
// Only the outer residency boundary receives the may-not-execute attribute.
// CHECK-LABEL: func.func @nested_optional_chain
// CHECK: scf.for
// CHECK: scf.if
// CHECK: linalg.matmul
// CHECK: scf.yield
// CHECK-NEXT: } else {
// CHECK: scf.yield
// CHECK-NEXT: } {hivm.matmul_limited_in_cube}{{$}}
// CHECK-NEXT: scf.yield
// CHECK-NEXT: } {may_not_exec = true}{{$}}
// CHECK: memref.load
// CHECK: scf.if
// CHECK-NOT: may_not_exec
// CHECK: arith.addf
// CHECK: return
func.func @nested_optional_chain(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1, %n: index) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> tensor<4x4xf32> {
    %v = scf.if %go -> tensor<4x4xf32> {
      %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %m : tensor<4x4xf32>
    } else {
      scf.yield %acc : tensor<4x4xf32>
    }
    scf.yield %v : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// Per-self MNE is retained even when another self in the same link must run,
// making the link as a whole non-MNE. No link counter is needed in this case.
// CHECK-LABEL: func.func @mixed_chain
// CHECK-NOT: memref.store
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: } {hivm.matmul_limited_in_cube}
// CHECK-NOT: memref.store
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: } {hivm.matmul_limited_in_cube, may_not_exec = true}
// CHECK-NOT: memref.load
// CHECK: arith.addf
// CHECK: return
func.func @mixed_chain(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %n: index) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %c) -> tensor<4x4xf32> {
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %x) -> tensor<4x4xf32> {
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// A local matmul without an SCF ancestor receives no control-flow attributes.
// CHECK-LABEL: func.func @plain_chain
// CHECK: linalg.matmul
// CHECK-NOT: hivm.matmul_limited_in_cube
// CHECK-NOT: may_not_exec
// CHECK: return
func.func @plain_chain(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// A while residency boundary receives the same BoolAttr after guard insertion.
// CHECK-LABEL: func.func @while_optional_chain
// CHECK: scf.while
// CHECK: scf.condition
// CHECK: linalg.matmul
// CHECK: } attributes {hivm.matmul_limited_in_cube, may_not_exec = true}
// CHECK: memref.load
// CHECK: scf.if
// CHECK-NOT: may_not_exec
// CHECK: arith.addf
// CHECK: return
func.func @while_optional_chain(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1) -> tensor<4x4xf32> {
  %r = scf.while (%acc = %c) : (tensor<4x4xf32>) -> tensor<4x4xf32> {
    scf.condition(%go) %acc : tensor<4x4xf32>
  } do {
  ^bb0(%arg: tensor<4x4xf32>):
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}
