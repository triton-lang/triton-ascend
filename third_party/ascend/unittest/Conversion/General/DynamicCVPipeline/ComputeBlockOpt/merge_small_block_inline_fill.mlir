// RUN: triton-opt %s --split-input-file --merge-small-block --verify-each | FileCheck %s --enable-var-scope

// A runtime scalar fill is inlined by direct tensor arith and must not count.
// CHECK-LABEL: func.func @runtime_fill_add(
func.func @runtime_fill_add(%scalar: f32, %input: tensor<8xf32>) -> (tensor<8xf32>) {
  %empty = tensor.empty() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %x = arith.addf %fill, %input {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %y = arith.addf %x, %x {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %z = arith.addf %y, %y {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d0 = arith.addf %z, %z {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d1 = arith.addf %d0, %d0 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d2 = arith.addf %d1, %d1 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d3 = arith.addf %d2, %d2 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xf32>
}

// -----

// Constant scalar fills retain the same direct-inline behavior.
// CHECK-LABEL: func.func @constant_fill_add(
func.func @constant_fill_add(%scalar: f32, %input: tensor<8xf32>) -> (tensor<8xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 120 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"} ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %x = arith.addf %fill, %input {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %y = arith.addf %x, %x {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %z = arith.addf %y, %y {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d0 = arith.addf %z, %z {ssbuffer.block_id = 120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d1 = arith.addf %d0, %d0 {ssbuffer.block_id = 120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d2 = arith.addf %d1, %d1 {ssbuffer.block_id = 120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d3 = arith.addf %d2, %d2 {ssbuffer.block_id = 120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 120 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xf32>
}

// -----

// Constant fills are excluded from the compute count regardless of their users.
// The three remaining compute ops can merge into block 220.
// CHECK-LABEL: func.func @constant_fill_math(
func.func @constant_fill_math(%scalar: f32, %input: tensor<8xf32>) -> (tensor<8xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: %[[FILL:[a-zA-Z0-9_]+]] = linalg.fill {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: math.exp %[[FILL]] {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %x = math.exp %fill {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %y = arith.addf %x, %x {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %z = arith.addf %y, %y {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d0 = arith.addf %z, %z {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d1 = arith.addf %d0, %d0 {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d2 = arith.addf %d1, %d1 {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d3 = arith.addf %d2, %d2 {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xf32>
}

// -----

// Runtime scalar plus unsupported user must also count.
// CHECK-LABEL: func.func @runtime_fill_math(
func.func @runtime_fill_math(%scalar: f32, %input: tensor<8xf32>) -> (tensor<8xf32>) {
  %empty = tensor.empty() {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %x = math.exp %fill {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %y = arith.addf %x, %x {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %z = arith.addf %y, %y {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d0 = arith.addf %z, %z {ssbuffer.block_id = 320 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d1 = arith.addf %d0, %d0 {ssbuffer.block_id = 320 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d2 = arith.addf %d1, %d1 {ssbuffer.block_id = 320 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d3 = arith.addf %d2, %d2 {ssbuffer.block_id = 320 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 320 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xf32>
}

// -----

// Every fill use must be eliminable; returning the fill prevents exclusion.
// CHECK-LABEL: func.func @fill_mixed_users(
func.func @fill_mixed_users(%scalar: f32, %input: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %empty = tensor.empty() {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %x = arith.addf %fill, %input {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %y = arith.addf %x, %x {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %z = arith.addf %y, %y {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d0 = arith.addf %z, %z {ssbuffer.block_id = 420 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d1 = arith.addf %d0, %d0 {ssbuffer.block_id = 420 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d2 = arith.addf %d1, %d1 {ssbuffer.block_id = 420 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d3 = arith.addf %d2, %d2 {ssbuffer.block_id = 420 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 420 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3, %fill : tensor<8xf32>, tensor<8xf32>
}

// -----

// Dynamic arith destinations cannot promise a fresh static tensor.empty.
// CHECK-LABEL: func.func @fill_dynamic_arith(
func.func @fill_dynamic_arith(%scalar: f32, %input: tensor<?xf32>, %n: index) -> (tensor<?xf32>) {
  %empty = tensor.empty(%n) {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %x = arith.addf %fill, %input {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  %y = arith.addf %x, %x {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  %z = arith.addf %y, %y {ssbuffer.block_id = 510 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  %d0 = arith.addf %z, %z {ssbuffer.block_id = 520 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  %d1 = arith.addf %d0, %d0 {ssbuffer.block_id = 520 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  %d2 = arith.addf %d1, %d1 {ssbuffer.block_id = 520 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  %d3 = arith.addf %d2, %d2 {ssbuffer.block_id = 520 : i32, ssbuffer.core_type = "VECTOR"} : tensor<?xf32>
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 520 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<?xf32>
}

// -----

// Supported ordered floating comparison directly lowers and may inline fill.
// CHECK-LABEL: func.func @fill_cmpf_ogt(
func.func @fill_cmpf_ogt(%scalar: f32, %input: tensor<8xf32>) -> (tensor<8xi1>) {
  %empty = tensor.empty() {ssbuffer.block_id = 610 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 620 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 610 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %x = arith.cmpf ogt, %fill, %input {ssbuffer.block_id = 610 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %y = arith.xori %x, %x {ssbuffer.block_id = 610 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %z = arith.xori %y, %y {ssbuffer.block_id = 610 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d0 = arith.xori %z, %z {ssbuffer.block_id = 620 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d1 = arith.xori %d0, %d0 {ssbuffer.block_id = 620 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d2 = arith.xori %d1, %d1 {ssbuffer.block_id = 620 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d3 = arith.xori %d2, %d2 {ssbuffer.block_id = 620 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  // CHECK: arith.xori {{.*}} {ssbuffer.block_id = 620 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xi1>
}

// -----

// ord is outside the direct-lowering predicate whitelist.
// CHECK-LABEL: func.func @fill_cmpf_ord(
func.func @fill_cmpf_ord(%scalar: f32, %input: tensor<8xf32>) -> (tensor<8xi1>) {
  %empty = tensor.empty() {ssbuffer.block_id = 710 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 710 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 710 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %x = arith.cmpf ord, %fill, %input {ssbuffer.block_id = 710 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %y = arith.xori %x, %x {ssbuffer.block_id = 710 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %z = arith.xori %y, %y {ssbuffer.block_id = 710 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d0 = arith.xori %z, %z {ssbuffer.block_id = 720 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d1 = arith.xori %d0, %d0 {ssbuffer.block_id = 720 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d2 = arith.xori %d1, %d1 {ssbuffer.block_id = 720 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d3 = arith.xori %d2, %d2 {ssbuffer.block_id = 720 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  // CHECK: arith.xori {{.*}} {ssbuffer.block_id = 720 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xi1>
}

// -----

// i1 equality does not introduce an ordered-comparison cast.
// CHECK-LABEL: func.func @fill_cmpi_eq_i1(
func.func @fill_cmpi_eq_i1(%scalar: i1, %input: tensor<8xi1>) -> (tensor<8xi1>) {
  %empty = tensor.empty() {ssbuffer.block_id = 810 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  // CHECK: linalg.fill {ssbuffer.block_id = 820 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 810 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : i1) outs(%empty : tensor<8xi1>) -> tensor<8xi1>
  %x = arith.cmpi eq, %fill, %input {ssbuffer.block_id = 810 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %y = arith.xori %x, %x {ssbuffer.block_id = 810 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %z = arith.xori %y, %y {ssbuffer.block_id = 810 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d0 = arith.xori %z, %z {ssbuffer.block_id = 820 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d1 = arith.xori %d0, %d0 {ssbuffer.block_id = 820 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d2 = arith.xori %d1, %d1 {ssbuffer.block_id = 820 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d3 = arith.xori %d2, %d2 {ssbuffer.block_id = 820 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  // CHECK: arith.xori {{.*}} {ssbuffer.block_id = 820 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xi1>
}

// -----

// Ordered i1 comparisons introduce casts and cannot inline the fill.
// CHECK-LABEL: func.func @fill_cmpi_slt_i1(
func.func @fill_cmpi_slt_i1(%scalar: i1, %input: tensor<8xi1>) -> (tensor<8xi1>) {
  %empty = tensor.empty() {ssbuffer.block_id = 910 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  // CHECK: linalg.fill {ssbuffer.block_id = 910 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 910 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : i1) outs(%empty : tensor<8xi1>) -> tensor<8xi1>
  %x = arith.cmpi slt, %fill, %input {ssbuffer.block_id = 910 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %y = arith.xori %x, %x {ssbuffer.block_id = 910 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %z = arith.xori %y, %y {ssbuffer.block_id = 910 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d0 = arith.xori %z, %z {ssbuffer.block_id = 920 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d1 = arith.xori %d0, %d0 {ssbuffer.block_id = 920 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d2 = arith.xori %d1, %d1 {ssbuffer.block_id = 920 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d3 = arith.xori %d2, %d2 {ssbuffer.block_id = 920 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  // CHECK: arith.xori {{.*}} {ssbuffer.block_id = 920 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xi1>
}

// -----

// Ordered comparisons of wider integers remain supported.
// CHECK-LABEL: func.func @fill_cmpi_slt_i32(
func.func @fill_cmpi_slt_i32(%scalar: i32, %input: tensor<8xi32>) -> (tensor<8xi1>) {
  %empty = tensor.empty() {ssbuffer.block_id = 1010 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi32>
  // CHECK: linalg.fill {ssbuffer.block_id = 1020 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 1010 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : i32) outs(%empty : tensor<8xi32>) -> tensor<8xi32>
  %x = arith.cmpi slt, %fill, %input {ssbuffer.block_id = 1010 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi32>
  %y = arith.xori %x, %x {ssbuffer.block_id = 1010 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %z = arith.xori %y, %y {ssbuffer.block_id = 1010 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d0 = arith.xori %z, %z {ssbuffer.block_id = 1020 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d1 = arith.xori %d0, %d0 {ssbuffer.block_id = 1020 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d2 = arith.xori %d1, %d1 {ssbuffer.block_id = 1020 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  %d3 = arith.xori %d2, %d2 {ssbuffer.block_id = 1020 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
  // CHECK: arith.xori {{.*}} {ssbuffer.block_id = 1020 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xi1>
}

// -----

// Do not look through a cast to an otherwise supported arith user.
// CHECK-LABEL: func.func @fill_cast_user(
func.func @fill_cast_user(%scalar: f16, %input: tensor<8xf16>) -> (tensor<8xf32>) {
  %empty = tensor.empty() {ssbuffer.block_id = 1110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf16>
  // CHECK: linalg.fill {ssbuffer.block_id = 1110 : i32, ssbuffer.core_type = "VECTOR"}
  %fill = linalg.fill {ssbuffer.block_id = 1110 : i32, ssbuffer.core_type = "VECTOR"} ins(%scalar : f16) outs(%empty : tensor<8xf16>) -> tensor<8xf16>
  %x = arith.extf %fill {ssbuffer.block_id = 1110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf16> to tensor<8xf32>
  %y = arith.addf %x, %x {ssbuffer.block_id = 1110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %z = arith.addf %y, %y {ssbuffer.block_id = 1110 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d0 = arith.addf %z, %z {ssbuffer.block_id = 1120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d1 = arith.addf %d0, %d0 {ssbuffer.block_id = 1120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d2 = arith.addf %d1, %d1 {ssbuffer.block_id = 1120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  %d3 = arith.addf %d2, %d2 {ssbuffer.block_id = 1120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 1120 : i32, ssbuffer.core_type = "VECTOR"}
  return %d3 : tensor<8xf32>
}
