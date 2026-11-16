// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --mlir-print-op-generic | FileCheck %s

// An explicit volatile load must retain that property in indirect lowering.
// CHECK-LABEL: sym_name = "volatile_indirect_load"
// CHECK: "ascend.indirect_load"{{.*}}isVolatile = true
tt.func @volatile_indirect_load(%src: !tt.ptr<i8>, %n: i32, %stride: i32) -> tensor<4x8xi8> {
  %rows = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %nvec = tt.splat %n : i32 -> tensor<8xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<8xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<4x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<4x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %brows = tt.broadcast %scaled : tensor<4x1xi32> -> tensor<4x8xi32>
  %bcols = tt.broadcast %col2d : tensor<1x8xi32> -> tensor<4x8xi32>
  %offset = arith.addi %brows, %bcols : tensor<4x8xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<4x8x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<4x8x!tt.ptr<i8>>, tensor<4x8xi32>
  %value = tt.load %ptr {isVolatile = true} : tensor<4x8x!tt.ptr<i8>>
  tt.return %value : tensor<4x8xi8>
}

// A read-only load without an explicit volatile flag retains its normal form.
// CHECK-LABEL: sym_name = "ordinary_indirect_load"
// CHECK: "ascend.indirect_load"{{.*}}isVolatile = false
tt.func @ordinary_indirect_load(%src: !tt.ptr<i8>, %n: i32, %stride: i32) -> tensor<4x8xi8> {
  %rows = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %nvec = tt.splat %n : i32 -> tensor<8xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<8xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<4x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<4x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %brows = tt.broadcast %scaled : tensor<4x1xi32> -> tensor<4x8xi32>
  %bcols = tt.broadcast %col2d : tensor<1x8xi32> -> tensor<4x8xi32>
  %offset = arith.addi %brows, %bcols : tensor<4x8xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<4x8x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<4x8x!tt.ptr<i8>>, tensor<4x8xi32>
  %value = tt.load %ptr : tensor<4x8x!tt.ptr<i8>>
  tt.return %value : tensor<4x8xi8>
}
