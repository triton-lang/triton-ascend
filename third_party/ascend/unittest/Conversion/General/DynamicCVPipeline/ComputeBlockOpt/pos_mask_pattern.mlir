// RUN: triton-opt --split-input-file --pos-mask-pattern-opt %s | FileCheck %s

// CHECK-LABEL: func.func @legacy
func.func @legacy(%x: tensor<2xi32>, %y: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %e = tensor.empty() : tensor<2x2xi32>
  // CHECK: linalg.broadcast {{.*}}ssbuffer.block_id = 2 : i32
  %v = linalg.broadcast ins(%x : tensor<2xi32>) outs(%e : tensor<2x2xi32>) dimensions = [0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: arith.cmpi eq, {{.*}}ssbuffer.block_id = 2 : i32
  %eq = arith.cmpi eq, %v, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.cmpi sle, {{.*}}ssbuffer.block_id = 2 : i32
  %le = arith.cmpi sle, %v, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.extui {{.*}}ssbuffer.block_id = 2 : i32
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.extui {{.*}}ssbuffer.block_id = 2 : i32
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %r = arith.addi %i, %j {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  return %r : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func.func @compare_mask
func.func @compare_mask(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 2 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 2 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @bitmask
func.func @bitmask(%x: tensor<2x2xi8>, %y: tensor<2x2xi8>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi8>
  %one = arith.constant dense<1> : tensor<2x2xi8>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %s = arith.shrui %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8>
  // CHECK: arith.andi {{.*}}ssbuffer.block_id = 2 : i32
  %mask = arith.andi %s, %one {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8>
  %empty = tensor.empty() : tensor<2x1xi8>
  // CHECK: tensor.extract_slice {{.*}}ssbuffer.block_id = 2 : i32
  %slice = tensor.extract_slice %mask[0, 0] [2, 1] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8> to tensor<2x1xi8>
  // CHECK: tensor.insert_slice {{.*}}ssbuffer.block_id = 2 : i32
  %padded = tensor.insert_slice %slice into %zero[0, 0] [2, 1] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x1xi8> into tensor<2x2xi8>
  %p = arith.cmpi ne, %padded, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 2 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @swapped_bias
func.func @swapped_bias(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 2 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 2 : i32
  %bias = arith.select %p, %neg, %z {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %bias, %scaled {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @shared_input
func.func @shared_input(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.extui {{.*}}ssbuffer.block_id = 1 : i32
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 2 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %keep = arith.addi %i, %one {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 2 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r, %keep : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func.func @positive_bias_rejected
func.func @positive_bias_rejected(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<1.0> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 1 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 1 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @non_matmul_rejected
func.func @non_matmul_rejected(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 1 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 1 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = arith.addf %q, %k : tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @multiple_consumer_blocks_rejected
func.func @multiple_consumer_blocks_rejected(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 1 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 1 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>
  %other = arith.addf %bias, %z {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r, %other : tensor<2x2xf32>, tensor<2x2xf32>
}

// -----


// CHECK-LABEL: func.func @fill_bias_constants
func.func @fill_bias_constants(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %zero_scalar = arith.constant 0.0 : f32
  %neg_scalar = arith.constant -1.073741824E+9 : f32
  %empty = tensor.empty() : tensor<2x2xf32>
  %z = linalg.fill ins(%zero_scalar : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %neg = linalg.fill ins(%neg_scalar : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 2 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 2 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}


// -----


// CHECK-LABEL: func.func @sync_rejected
func.func @sync_rejected(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 1 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 1 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  gpu.barrier {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.external_sync = 1 : i32}
  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}


// -----


// CHECK-LABEL: func.func @non_bit_extraction_rejected
func.func @non_bit_extraction_rejected(%x: tensor<2x2xi8>, %y: tensor<2x2xi8>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi8>
  %one = arith.constant dense<3> : tensor<2x2xi8>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %s = arith.shrui %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8>
  // CHECK: arith.andi {{.*}}ssbuffer.block_id = 1 : i32
  %mask = arith.andi %s, %one {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8>
  %empty = tensor.empty() : tensor<2x1xi8>
  // CHECK: tensor.extract_slice {{.*}}ssbuffer.block_id = 1 : i32
  %slice = tensor.extract_slice %mask[0, 0] [2, 1] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8> to tensor<2x1xi8>
  // CHECK: tensor.insert_slice {{.*}}ssbuffer.block_id = 1 : i32
  %padded = tensor.insert_slice %slice into %zero[0, 0] [2, 1] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x1xi8> into tensor<2x2xi8>
  %p = arith.cmpi ne, %padded, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi8>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 1 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %q {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}


// -----


// CHECK-LABEL: func.func @cross_block_dependency_rejected
func.func @cross_block_dependency_rejected(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>, %q: tensor<2x2xf32>, %k: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0> : tensor<2x2xi32>
  %one = arith.constant dense<1> : tensor<2x2xi32>
  %z = arith.constant dense<0.0> : tensor<2x2xf32>
  %neg = arith.constant dense<-1.073741824E+9> : tensor<2x2xf32>
  %eq = arith.cmpi eq, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %le = arith.cmpi sle, %x, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.ori {{.*}}ssbuffer.block_id = 1 : i32
  %mask = arith.ori %i, %j {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  %p = arith.cmpi ne, %mask, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.select {{.*}}ssbuffer.block_id = 1 : i32
  %bias = arith.select %p, %z, %neg {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1>, tensor<2x2xf32>

  %bridge = arith.sitofp %mask {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32> to tensor<2x2xf32>
  %mm = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<2x2xf32>, tensor<2x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
  %scaled = arith.mulf %mm, %bridge {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  %r = arith.addf %scaled, %bias {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}


// -----
// CHECK-LABEL: func.func @legacy_predicate_rejected
func.func @legacy_predicate_rejected(%x: tensor<2xi32>, %y: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %e = tensor.empty() : tensor<2x2xi32>
  // CHECK: linalg.broadcast {{.*}}ssbuffer.block_id = 1 : i32
  %v = linalg.broadcast ins(%x : tensor<2xi32>) outs(%e : tensor<2x2xi32>) dimensions = [0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: arith.cmpi eq, {{.*}}ssbuffer.block_id = 1 : i32
  %eq = arith.cmpi eq, %v, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.cmpi sge, {{.*}}ssbuffer.block_id = 1 : i32
  %le = arith.cmpi sge, %v, %y {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  // CHECK: arith.extui {{.*}}ssbuffer.block_id = 1 : i32
  %i = arith.extui %eq {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  // CHECK: arith.extui {{.*}}ssbuffer.block_id = 1 : i32
  %j = arith.extui %le {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi1> to tensor<2x2xi32>
  %r = arith.addi %i, %j {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xi32>
  return %r : tensor<2x2xi32>
}

