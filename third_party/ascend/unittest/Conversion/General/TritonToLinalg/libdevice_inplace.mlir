// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=true" %s | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=false" %s | FileCheck %s --check-prefix=A2A3 --implicit-check-not=inplace_outs

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  // CHECK-LABEL: func.func @tanh_single_use
  // CHECK: hivm.hir.custom
  // CHECK-SAME: ins(%[[INPUT:[^ ]+]] : tensor<256xf32> {inplace_outs = 0 : i32})
  // CHECK-SAME: outs(%[[INPUT]] : tensor<256xf32>)
  // A2A3-LABEL: func.func @tanh_single_use
  // A2A3: %[[A3_INPUT:.*]] = arith.addf
  // A2A3: %[[A3_ALLOC:.*]] = memref.alloc
  // A2A3: %[[A3_OUTPUT:.*]] = bufferization.to_tensor %[[A3_ALLOC]]
  // A2A3: hivm.hir.custom
  // A2A3-SAME: ins(%[[A3_INPUT]] : tensor<256xf32>)
  // A2A3-SAME: outs(%[[A3_OUTPUT]] : tensor<256xf32>)
  tt.func @tanh_single_use(%arg: tensor<256xf32>, %out: !tt.ptr<f32>) {
    %tmp = arith.addf %arg, %arg : tensor<256xf32>
    %res = tt.extern_elementwise %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_tanh_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @tanh_rank2
  // CHECK: %[[FLAT:.*]] = tensor.collapse_shape
  // CHECK: hivm.hir.custom
  // CHECK-SAME: ins(%[[FLAT]] : tensor<256xf32> {inplace_outs = 0 : i32})
  // CHECK-SAME: outs(%[[FLAT]] : tensor<256xf32>)
  // CHECK: tensor.expand_shape
  tt.func @tanh_rank2(%arg: tensor<16x16xf32>, %out: !tt.ptr<f32>) {
    %tmp = arith.addf %arg, %arg : tensor<16x16xf32>
    %res = tt.extern_elementwise %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_tanh_fp32"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @tanh_live_input
  // CHECK: hivm.hir.custom
  // CHECK-NOT: inplace_outs
  // CHECK-SAME: outs(
  tt.func @tanh_live_input(%arg: tensor<256xf32>, %out: !tt.ptr<f32>) {
    %tmp = arith.addf %arg, %arg : tensor<256xf32>
    %res = tt.extern_elementwise %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_tanh_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
    %sum = arith.addf %tmp, %res : tensor<256xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %ptrs, %sum : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @tanh_argument
  // CHECK: hivm.hir.custom
  // CHECK-NOT: inplace_outs
  // CHECK-SAME: outs(
  tt.func @tanh_argument(%arg: tensor<256xf32>, %out: !tt.ptr<f32>) {
    %res = tt.extern_elementwise %arg {libname = "", libpath = "", pure = true, symbol = "__hmf_tanh_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @other_libdevice
  // CHECK: hivm.hir.custom
  // CHECK-SAME: ins(%[[SIN_INPUT:[^ ]+]] : tensor<256xf32> {inplace_outs = 0 : i32})
  // CHECK-SAME: outs(%[[SIN_INPUT]] : tensor<256xf32>)
  tt.func @other_libdevice(%arg: tensor<256xf32>, %out: !tt.ptr<f32>) {
    %tmp = arith.addf %arg, %arg : tensor<256xf32>
    %res = tt.extern_elementwise %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_sin_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @binary_second_input
  // CHECK: hivm.hir.custom
  // CHECK-SAME: ins(%{{[^,]+}}, %[[RHS:[^ ]+]] : tensor<256xf32>, tensor<256xf32> {inplace_outs = 0 : i32})
  // CHECK-SAME: outs(%[[RHS]] : tensor<256xf32>)
  // A2A3-LABEL: func.func @binary_second_input
  // A2A3: %[[A3_RHS:.*]] = arith.addf
  // A2A3: %[[A3_BINARY_ALLOC:.*]] = memref.alloc
  // A2A3: %[[A3_BINARY_OUTPUT:.*]] = bufferization.to_tensor %[[A3_BINARY_ALLOC]]
  // A2A3: hivm.hir.custom
  // A2A3-SAME: ins(%{{[^,]+}}, %[[A3_RHS]] : tensor<256xf32>, tensor<256xf32>)
  // A2A3-SAME: outs(%[[A3_BINARY_OUTPUT]] : tensor<256xf32>)
  tt.func @binary_second_input(%lhs: tensor<256xf32>, %rhs: tensor<256xf32>, %out: !tt.ptr<f32>) {
    %tmp = arith.addf %rhs, %rhs : tensor<256xf32>
    %res = tt.extern_elementwise %lhs, %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_pow_fp32"} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @different_output_type
  // CHECK: hivm.hir.custom
  // CHECK-NOT: inplace_outs
  // CHECK-SAME: outs(
  tt.func @different_output_type(%arg: tensor<256xf32>, %out: !tt.ptr<i32>) {
    %tmp = arith.addf %arg, %arg : tensor<256xf32>
    %res = tt.extern_elementwise %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_float2int_rn_fp32"} : (tensor<256xf32>) -> tensor<256xi32>
    %ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    tt.store %ptrs, %res : tensor<256x!tt.ptr<i32>>
    tt.return
  }

  // CHECK-LABEL: func.func @scalar_input_output
  // CHECK: %[[SCALAR_TENSOR:.*]] = tensor.from_elements
  // CHECK: hivm.hir.custom
  // CHECK-NOT: inplace_outs
  // CHECK-SAME: ins(%[[SCALAR_TENSOR]] : tensor<1xf32>)
  // CHECK-SAME: outs(%{{[^ ]+}} : tensor<1xf32>)
  // CHECK: tensor.extract
  tt.func @scalar_input_output(%arg: f32, %out: !tt.ptr<f32>) {
    %res = tt.extern_elementwise %arg {libname = "", libpath = "", pure = true, symbol = "__hmf_tanh_fp32"} : (f32) -> f32
    tt.store %out, %res : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: func.func @single_element_tensor
  // CHECK: hivm.hir.custom
  // CHECK-NOT: inplace_outs
  // CHECK-SAME: outs(
  tt.func @single_element_tensor(%arg: tensor<1xf32>, %out: !tt.ptr<f32>) {
    %tmp = arith.addf %arg, %arg : tensor<1xf32>
    %res = tt.extern_elementwise %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_rsqrt_fp32"} : (tensor<1xf32>) -> tensor<1xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<1x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @region_result
  // CHECK: %[[REGION_RESULT:.*]] = scf.if
  // CHECK: hivm.hir.custom
  // CHECK-SAME: ins(%[[REGION_RESULT]] : tensor<256xf32> {inplace_outs = 0 : i32})
  // CHECK-SAME: outs(%[[REGION_RESULT]] : tensor<256xf32>)
  tt.func @region_result(%condition: i1, %arg: tensor<256xf32>, %out: !tt.ptr<f32>) {
    %tmp = scf.if %condition -> (tensor<256xf32>) {
      %v = arith.addf %arg, %arg : tensor<256xf32>
      scf.yield %v : tensor<256xf32>
    } else {
      %v = arith.mulf %arg, %arg : tensor<256xf32>
      scf.yield %v : tensor<256xf32>
    }
    %res = tt.extern_elementwise %tmp {libname = "", libpath = "", pure = true, symbol = "__hmf_tanh_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @uniform_exponent
  // CHECK: hivm.hir.custom
  // CHECK-SAME: ins(%{{[^,]+}}, %[[UNIFORM:[^ ]+]] : tensor<256xf32>, tensor<256xf32> {inplace_outs = 0 : i32})
  // CHECK-SAME: outs(%[[UNIFORM]] : tensor<256xf32>)
  tt.func @uniform_exponent(%arg: tensor<256xf32>, %exponent: f32, %out: !tt.ptr<f32>) {
    %uniform = tt.splat %exponent : f32 -> tensor<256xf32>
    %derived = arith.addf %uniform, %uniform : tensor<256xf32>
    %res = tt.extern_elementwise %arg, %derived {libname = "", libpath = "", pure = true, symbol = "__hmf_pow_fp32"} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: func.func @outside_loop
  // CHECK: scf.for
  // CHECK: hivm.hir.custom
  // CHECK-NOT: inplace_outs
  // CHECK-SAME: outs(
  tt.func @outside_loop(%arg: tensor<256xf32>, %out: !tt.ptr<f32>, %count: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %outside = arith.addf %arg, %arg : tensor<256xf32>
    scf.for %i = %c0 to %count step %c1 {
      %res = tt.extern_elementwise %outside {libname = "", libpath = "", pure = true, symbol = "__hmf_sin_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
      %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK-LABEL: func.func @hoistable_inside_loop
  // CHECK: scf.for
  // CHECK: hivm.hir.custom
  // CHECK-NOT: inplace_outs
  // CHECK-SAME: outs(
  tt.func @hoistable_inside_loop(%arg: tensor<256xf32>, %out: !tt.ptr<f32>, %count: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %count step %c1 {
      %invariant = arith.addf %arg, %arg : tensor<256xf32>
      %res = tt.extern_elementwise %invariant {libname = "", libpath = "", pure = true, symbol = "__hmf_sin_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
      %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK-LABEL: func.func @loop_variant_input
  // CHECK: scf.for
  // CHECK: hivm.hir.custom
  // CHECK-SAME: ins(%[[VARIANT:[^ ]+]] : tensor<256xf32> {inplace_outs = 0 : i32})
  // CHECK-SAME: outs(%[[VARIANT]] : tensor<256xf32>)
  tt.func @loop_variant_input(%arg: tensor<256xf32>, %out: !tt.ptr<f32>, %count: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %count step %c1 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      %splat = tt.splat %f : f32 -> tensor<256xf32>
      %variant = arith.addf %arg, %splat : tensor<256xf32>
      %res = tt.extern_elementwise %variant {libname = "", libpath = "", pure = true, symbol = "__hmf_sin_fp32"} : (tensor<256xf32>) -> tensor<256xf32>
      %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      tt.store %ptrs, %res : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }
}
