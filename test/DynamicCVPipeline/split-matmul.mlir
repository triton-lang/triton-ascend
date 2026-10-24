// RUN: triton-opt %s --ssbuf-split-matmul --verify-each | FileCheck %s
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each > /dev/null

// CHECK-LABEL: func.func @plain
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[MM:.*]] = linalg.matmul
// CHECK: %[[ADD:.*]] = arith.addf %[[MM]], %arg2
// CHECK: return %[[ADD]]
func.func @plain(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @loop_chain
// CHECK: memref.store {{.*}}, %[[COUNTER0:.*]][]
// CHECK: memref.store {{.*}}, %[[COUNTER1:.*]][]
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK: linalg.matmul
// CHECK: memref.store {{.*}}, %[[COUNTER0]][]
// CHECK: linalg.matmul
// CHECK: memref.store {{.*}}, %[[COUNTER1]][]
// CHECK: memref.load %[[COUNTER0]][]
// CHECK: %[[SKIP0:.*]] = arith.cmpi eq
// CHECK: memref.load %[[COUNTER1]][]
// CHECK: %[[SKIP1:.*]] = arith.cmpi eq
// CHECK: %[[ALLSKIPPED:.*]] = arith.andi %[[SKIP0]], %[[SKIP1]]
// CHECK: %[[GUARD:.*]] = scf.if %[[ALLSKIPPED]]
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield %[[LOOP]]
// CHECK: arith.addf %[[GUARD]], %arg2
func.func @loop_chain(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %n: index) -> tensor<4x4xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %n step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %m0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    %m1 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%m0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m1 : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @cascade
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK-NOT: arith.addf
// CHECK: return
func.func @cascade(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %x = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  %r = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%x : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// The predicate reaches the matmul through a one-way Control edge. It is
// outside the tensor's directed SCC, so the C residency ring stays single-entry.
// CHECK-LABEL: func.func @while_signatures
// CHECK: %[[LOOP:.*]] = scf.while
// CHECK: scf.condition
// CHECK: linalg.matmul
// CHECK: memref.store
// CHECK: memref.load
// CHECK: %[[GUARD:.*]] = scf.if
// CHECK: } else {
// CHECK: scf.yield %[[LOOP]]
// CHECK: arith.addf %[[GUARD]], %arg2

func.func @while_signatures(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1) -> tensor<4x4xf32> {
  %r = scf.while (%acc = %c, %p = %go) : (tensor<4x4xf32>, i1) -> tensor<4x4xf32> {
    scf.condition(%p) %acc : tensor<4x4xf32>
  } do {
  ^bb0(%arg: tensor<4x4xf32>):
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg : tensor<4x4xf32>) -> tensor<4x4xf32>
    %stop = arith.constant false
    scf.yield %m, %stop : tensor<4x4xf32>, i1
  }
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @if_passthrough
// CHECK: scf.for
// CHECK: scf.if
// CHECK: linalg.matmul
// CHECK: memref.load
// CHECK: scf.if
// CHECK: arith.addf
func.func @if_passthrough(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %go: i1) -> tensor<4x4xf32> {
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %x = scf.if %go -> tensor<4x4xf32> {
      %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %m : tensor<4x4xf32>
    } else {
      scf.yield %acc : tensor<4x4xf32>
    }
    scf.yield %x : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @dynamic
// CHECK: tensor.dim %arg2
// CHECK: tensor.empty(%
// CHECK: linalg.matmul
// CHECK: arith.addf
func.func @dynamic(%a: tensor<?x4xf32>, %b: tensor<4x4xf32>, %c: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %m = linalg.matmul ins(%a, %b : tensor<?x4xf32>, tensor<4x4xf32>) outs(%c : tensor<?x4xf32>) -> tensor<?x4xf32>
  return %m : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @alias_operand
// CHECK: linalg.matmul {{.*}}ins(%arg0, %arg1
// CHECK: arith.addf {{.*}}, %arg0
func.func @alias_operand(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @fanout
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK: linalg.matmul
// CHECK: arith.addf
func.func @fanout(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %x = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  %y = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%x : tensor<4x4xf32>) -> tensor<4x4xf32>
  %z = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%x : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %y, %z : tensor<4x4xf32>, tensor<4x4xf32>
}

// CHECK-LABEL: func.func @while_before
// CHECK: scf.while
// CHECK: linalg.matmul
// CHECK-NOT: memref.load
// CHECK: arith.addf
func.func @while_before(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = scf.while (%acc = %c) : (tensor<4x4xf32>) -> tensor<4x4xf32> {
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    %stop = arith.constant false
    scf.condition(%stop) %m : tensor<4x4xf32>
  } do {
  ^bb0(%arg: tensor<4x4xf32>):
    scf.yield %arg : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @zero_bias
// CHECK: linalg.matmul
// CHECK-NOT: arith.addf
// CHECK: return
func.func @zero_bias(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %z = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %c = linalg.fill ins(%z : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// First loop can run while second loop executes zero times. One link must
// record BOTH matmuls, then guard the final contribution once.
// CHECK-LABEL: func.func @cascade_loops
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: memref.store
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: memref.store
// CHECK: memref.load
// CHECK: arith.cmpi eq
// CHECK: memref.load
// CHECK: arith.cmpi eq
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: arith.addf
// CHECK-NOT: arith.addf
// CHECK: return
func.func @cascade_loops(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %n: index, %k: index) -> tensor<4x4xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %x = scf.for %i = %lb to %n step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  %r = scf.for %i = %lb to %k step %step iter_args(%acc = %x) -> tensor<4x4xf32> {
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %m : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}

// Intermediate accumulator is observed: keep the observer correct by splitting
// at the matmul itself, inside the loop, instead of hoisting bias past it.
// CHECK-LABEL: func.func @observed_accumulator
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: %[[LOCAL:.*]] = arith.addf
// CHECK: arith.addf %[[LOCAL]], %arg2
// CHECK: scf.yield
func.func @observed_accumulator(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>, %n: index) -> tensor<4x4xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r = scf.for %i = %lb to %n step %step iter_args(%acc = %c) -> tensor<4x4xf32> {
    %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
    %sum = arith.addf %m, %c : tensor<4x4xf32>
    scf.yield %sum : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}
