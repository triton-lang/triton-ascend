// RUN: triton-opt --triton-to-linalg %s | FileCheck %s --enable-var-scope
// RUN: triton-opt --triton-to-linalg="enable-select-analysis=false" %s | FileCheck %s --enable-var-scope

module {

// CHECK-LABEL: func.func @dynamic(
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: tensor.insert %
// CHECK: scf.yield %arg0
func.func @dynamic(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @reversed(
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: tensor.insert %
// CHECK: scf.yield %arg0
func.func @reversed(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %p, %r : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @scalar_value(
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: tensor.insert %
// CHECK: scf.yield %arg0
func.func @scalar_value(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = tt.splat %value : f32 -> tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @nan_payload(
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: tensor.insert %
// CHECK: scf.yield %arg0
func.func @nan_payload(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0x7FC12345> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @single_lane(
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: tensor.insert %
// CHECK: scf.yield %arg0
func.func @single_lane(%old: tensor<1xf32>, %position: i32, %value: f32, %other: tensor<1xf32>) -> tensor<1xf32> {
  %r = tt.make_range {start = 0 : i32, end = 1 : i32} : tensor<1xi32>
  %p = tt.splat %position : i32 -> tensor<1xi32>
  %m = arith.cmpi eq, %r, %p : tensor<1xi32>
  %v = arith.constant dense<0xFF800000> : tensor<1xf32>
  %updated = arith.select %m, %v, %old : tensor<1xi1>, tensor<1xf32>
  return %updated : tensor<1xf32>
}

// CHECK-LABEL: func.func @first_lane(
// CHECK: tensor.insert %
func.func @first_lane(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %fixed = arith.constant 0 : i32
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %fixed : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @last_lane(
// CHECK: tensor.insert %
func.func @last_lane(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %fixed = arith.constant 7 : i32
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %fixed : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @negative(
// CHECK-NOT: tensor.insert %
// CHECK: return %arg0
func.func @negative(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %fixed = arith.constant -1 : i32
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %fixed : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @equal_n(
// CHECK-NOT: tensor.insert %
// CHECK: return %arg0
func.func @equal_n(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %fixed = arith.constant 8 : i32
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %fixed : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @past_end(
// CHECK-NOT: tensor.insert %
// CHECK: return %arg0
func.func @past_end(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %fixed = arith.constant 9 : i32
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %fixed : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @non_eq(
// CHECK-NOT: tensor.insert %
// CHECK: return
func.func @non_eq(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi slt, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @shifted_range(
// CHECK-NOT: tensor.insert %
// CHECK: return
func.func @shifted_range(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %r = tt.make_range {start = 1 : i32, end = 9 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @non_splat_value(
// CHECK-NOT: tensor.insert %
// CHECK: return
func.func @non_splat_value(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> tensor<8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %updated = arith.select %m, %other, %old : tensor<8xi1>, tensor<8xf32>
  return %updated : tensor<8xf32>
}

// CHECK-LABEL: func.func @preserve_old(
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: tensor.insert %
// CHECK: scf.yield %arg0
// CHECK: return %{{.*}}, %arg0
func.func @preserve_old(%old: tensor<8xf32>, %position: i32, %value: f32, %other: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %position : i32 -> tensor<8xi32>
  %m = arith.cmpi eq, %r, %p : tensor<8xi32>
  %v = arith.constant dense<0xFF800000> : tensor<8xf32>
  %updated = arith.select %m, %v, %old : tensor<8xi1>, tensor<8xf32>
  return %updated, %old : tensor<8xf32>, tensor<8xf32>
}
}
