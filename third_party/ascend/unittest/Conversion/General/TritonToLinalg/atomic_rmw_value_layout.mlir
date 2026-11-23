// RUN: triton-opt %s --triton-to-linalg="compile-on-910-95=false" --verify-each --mlir-print-op-generic | FileCheck %s

// Each four-element GM row has pitch 8 and a dynamic origin. The local
// buffer packs the rows at pitch 4. Atomic conversion must reuse this buffer
// through the A2/A3 XCHG view and preserve the GM destination descriptor.
// The view prevents backend dead-store elimination from erasing XCHG when
// its input is a directly allocated local buffer.
// CHECK-LABEL: sym_name = "atomic_xchg_local_value"
// CHECK: %[[LOCAL:.*]] = "memref.alloc"() {{.*}} : () -> memref<2x1x4xf32>
// CHECK-NEXT: "memref.copy"(%{{.*}}, %[[LOCAL]])
// CHECK-NOT: "memref.alloc"
// CHECK-NOT: "memref.copy"
// CHECK: %[[DEST:.*]] = "memref.reinterpret_cast"({{.*}}) {{.*}} -> memref<2x1x4xf32, strided<[8, 4, 1], offset: ?>>
// CHECK-NOT: "memref.alloc"
// CHECK-NOT: "memref.copy"
// CHECK: %[[VIEW:.*]] = "memref.cast"(%[[LOCAL]]) : (memref<2x1x4xf32>) -> memref<2x1x4xf32, strided<[4, 4, 1], offset: ?>>
// CHECK-NOT: "memref.alloc"
// CHECK-NOT: "memref.copy"
// CHECK: "hfusion.atomic_xchg"(%[[VIEW]], %[[DEST]])
// CHECK-NOT: "memref.alloc"
// CHECK-NOT: "memref.copy"
// CHECK: "func.return"

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @atomic_xchg_local_value(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %origin: i32) {
    %eight = arith.constant dense<8> : tensor<2x1x1xi32>
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %three = arith.constant 3 : i32
    %all = arith.constant dense<true> : tensor<2x1x4xi1>
    %rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %rows_3d = tt.expand_dims %rows_2d {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %row_offsets = arith.muli %rows_3d, %eight : tensor<2x1x1xi32>
    %cols = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %cols_3d = tt.expand_dims %cols_2d {axis = 0 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %row_tile = tt.broadcast %row_offsets : tensor<2x1x1xi32> -> tensor<2x1x4xi32>
    %col_tile = tt.broadcast %cols_3d : tensor<1x1x4xi32> -> tensor<2x1x4xi32>
    %relative = arith.addi %row_tile, %col_tile : tensor<2x1x4xi32>
    %origins = tt.splat %origin : i32 -> tensor<2x1x4xi32>
    %offsets = arith.addi %relative, %origins : tensor<2x1x4xi32>
    %inputs = tt.splat %input : !tt.ptr<f32> -> tensor<2x1x4x!tt.ptr<f32>>
    %input_ptrs = tt.addptr %inputs, %offsets : tensor<2x1x4x!tt.ptr<f32>>, tensor<2x1x4xi32>
    %values = tt.load %input_ptrs : tensor<2x1x4x!tt.ptr<f32>>
    %outputs = tt.splat %output : !tt.ptr<f32> -> tensor<2x1x4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %outputs, %offsets : tensor<2x1x4x!tt.ptr<f32>>, tensor<2x1x4xi32>
    scf.for %i = %zero to %three step %one : i32 {
      %old = tt.atomic_rmw exch, acq_rel, gpu, %output_ptrs, %values, %all : (tensor<2x1x4x!tt.ptr<f32>>, tensor<2x1x4xf32>, tensor<2x1x4xi1>) -> tensor<2x1x4xf32>
    }
    tt.return
  }
}
