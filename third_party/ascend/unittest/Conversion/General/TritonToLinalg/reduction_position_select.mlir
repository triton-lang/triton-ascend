// RUN: triton-opt --triton-to-linalg="compile-on-910-95=true" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s --check-prefix=A2

// A vector-computed position must not become scalar-controlled dynamic copies.
// Keep the reduction combiner and elementwise equality/select unchanged.
// CHECK-LABEL: func.func @reduction_position_eq
// A2-LABEL: func.func @reduction_position_eq
// A2: scf.if
// A2: tensor.insert_slice
// CHECK: linalg.reduce
// CHECK: arith.minsi
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.if
// CHECK: linalg.broadcast
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.if
// CHECK: return
tt.func public @reduction_position_eq(%input: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %zero = arith.constant dense<0> : tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %input_base = tt.splat %input : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %input_ptrs = tt.addptr %input_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %values = tt.load %input_ptrs : tensor<16x!tt.ptr<i32>>
  %position = "tt.reduce"(%values) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %min = arith.minsi %a, %b : i32
    tt.reduce.return %min : i32
  }) : (tensor<16xi32>) -> i32
  %positions = tt.splat %position : i32 -> tensor<16xi32>
  %mask = arith.cmpi eq, %range, %positions : tensor<16xi32>
  %updated = arith.select %mask, %zero, %values : tensor<16xi1>, tensor<16xi32>
  %base = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  tt.store %ptrs, %updated : tensor<16x!tt.ptr<i32>>
  tt.return
}

// -----

// Reversed operands and the complementary update have the same vector origin.
// CHECK-LABEL: func.func @reduction_position_ne_reversed
// CHECK: linalg.reduce
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.if
// CHECK: linalg.broadcast
// CHECK: arith.cmpi ne
// CHECK: arith.select
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: return
tt.func public @reduction_position_ne_reversed(%input: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %zero = arith.constant dense<0> : tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %input_base = tt.splat %input : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %input_ptrs = tt.addptr %input_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %values = tt.load %input_ptrs : tensor<16x!tt.ptr<i32>>
  %position = "tt.reduce"(%values) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %max = arith.maxsi %a, %b : i32
    tt.reduce.return %max : i32
  }) : (tensor<16xi32>) -> i32
  %positions = tt.splat %position : i32 -> tensor<16xi32>
  %mask = arith.cmpi ne, %positions, %range : tensor<16xi32>
  %updated = arith.select %mask, %values, %zero : tensor<16xi1>, tensor<16xi32>
  %base = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  tt.store %ptrs, %updated : tensor<16x!tt.ptr<i32>>
  tt.return
}

// -----

// A scalar runtime index is not a vector feedback decision. The existing
// continuous-mask slice optimization and its out-of-bounds fallback remain.
// CHECK-LABEL: func.func @runtime_position_control
// CHECK: scf.if
// CHECK: tensor.insert_slice
// CHECK: return
tt.func public @runtime_position_control(%input: !tt.ptr<i32>, %position: i32, %out: !tt.ptr<i32>) {
  %zero = arith.constant dense<0> : tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %input_base = tt.splat %input : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %input_ptrs = tt.addptr %input_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %values = tt.load %input_ptrs : tensor<16x!tt.ptr<i32>>
  %positions = tt.splat %position : i32 -> tensor<16xi32>
  %mask = arith.cmpi eq, %range, %positions : tensor<16xi32>
  %updated = arith.select %mask, %zero, %values : tensor<16xi1>, tensor<16xi32>
  %base = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  tt.store %ptrs, %updated : tensor<16x!tt.ptr<i32>>
  tt.return
}

// -----

// A prefix mask still benefits from contiguous-slice lowering.
// CHECK-LABEL: func.func @runtime_prefix_control
// CHECK: tensor.insert_slice
// CHECK: return
tt.func public @runtime_prefix_control(%input: !tt.ptr<i32>, %bound: i32, %out: !tt.ptr<i32>) {
  %zero = arith.constant dense<0> : tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %input_base = tt.splat %input : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %input_ptrs = tt.addptr %input_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %values = tt.load %input_ptrs : tensor<16x!tt.ptr<i32>>
  %bounds = tt.splat %bound : i32 -> tensor<16xi32>
  %mask = arith.cmpi slt, %range, %bounds : tensor<16xi32>
  %updated = arith.select %mask, %values, %zero : tensor<16xi1>, tensor<16xi32>
  %base = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  tt.store %ptrs, %updated : tensor<16x!tt.ptr<i32>>
  tt.return
}

// -----

// A scalar store is canonicalized into a one-element tensor store. Reuse the
// reduced tensor by reshaping it; do not read a scalar and insert it into UB.
// CHECK-LABEL: func.func @reduced_scalar_store
// CHECK: linalg.reduce
// CHECK: tensor.expand_shape
// CHECK-NOT: tensor.insert
// CHECK: return
tt.func public @reduced_scalar_store(%input: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %base = tt.splat %input : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %values = tt.load %ptrs : tensor<16x!tt.ptr<i32>>
  %reduced = "tt.reduce"(%values) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %max = arith.maxsi %a, %b : i32
    tt.reduce.return %max : i32
  }) : (tensor<16xi32>) -> i32
  tt.store %out, %reduced : !tt.ptr<i32>
  tt.return
}

// -----

// A real scalar-control consumer must still receive a scalar. Only the splat
// consumer reuses the rank-zero tensor; external/control-flow types stay intact.
// CHECK-LABEL: func.func @scalar_control_boundary
// CHECK: linalg.reduce
// CHECK: tensor.extract
// CHECK: arith.cmpi sgt
// CHECK: scf.if
// CHECK: arith.select
// CHECK: return
tt.func public @scalar_control_boundary(%input: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %zero = arith.constant 0 : i32
  %zeros = arith.constant dense<0> : tensor<16xi32>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %base = tt.splat %input : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %values = tt.load %ptrs : tensor<16x!tt.ptr<i32>>
  %position = "tt.reduce"(%values) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %min = arith.minsi %a, %b : i32
    tt.reduce.return %min : i32
  }) : (tensor<16xi32>) -> i32
  %condition = arith.cmpi sgt, %position, %zero : i32
  scf.if %condition {
    %positions = tt.splat %position : i32 -> tensor<16xi32>
    %mask = arith.cmpi eq, %range, %positions : tensor<16xi32>
    %selected = arith.select %mask, %values, %zeros : tensor<16xi1>, tensor<16xi32>
    %output_base = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %output_ptrs = tt.addptr %output_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %output_ptrs, %selected : tensor<16x!tt.ptr<i32>>
  }
  tt.return
}

// -----

// Tensor reuse does not replace a floating-point combiner with another max
// flavor: maximumf's NaN behavior remains explicit in the reduction region.
// CHECK-LABEL: func.func @floating_reduction_broadcast
// CHECK: linalg.reduce
// CHECK: arith.maximumf
// CHECK: linalg.broadcast
// CHECK: return
tt.func public @floating_reduction_broadcast(%input: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %base = tt.splat %input : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %values = tt.load %ptrs : tensor<16x!tt.ptr<f32>>
  %maximum = "tt.reduce"(%values) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %max = arith.maximumf %a, %b : f32
    tt.reduce.return %max : f32
  }) : (tensor<16xf32>) -> f32
  %broadcast = tt.splat %maximum : f32 -> tensor<16xf32>
  %output_base = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %output_ptrs = tt.addptr %output_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %output_ptrs, %broadcast : tensor<16x!tt.ptr<f32>>
  tt.return
}
