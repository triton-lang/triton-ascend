// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=false" %s | FileCheck %s --check-prefix=A2A3
// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=false" --mlir-print-op-generic %s | FileCheck %s --check-prefix=SEGMENTS

// SEGMENTS-LABEL: "builtin.module"
// SEGMENTS: operandSegmentSizes = array<i32: 2, 1, 0>

// Workspace attributes are produced by the Python frontend. This fixture
// supplies them explicitly to check conversion passthrough for every A2/A3
// bitcode-only operator.
// Store every pure extern result so it survives DCE; the functions intentionally
// return no tensors to keep bufferized return types valid.
// Operation-sized rows use the 2048-element result extent. Fixed rows contain
// 64 lanes regardless of the tensor extent.

// A2A3-LABEL: func.func @test_fast_pow_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [512]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "fast_pow_fp32"

// A2A3-LABEL: func.func @test_float2ull_rd_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_rd_fp32"

// A2A3-LABEL: func.func @test_float2ull_rn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_rn_fp32"

// A2A3-LABEL: func.func @test_float2ull_ru_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_ru_fp32"

// A2A3-LABEL: func.func @test_float2ull_rz_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_rz_fp32"

// A2A3-LABEL: func.func @test_tgamma_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [576]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "tgamma_fp32"

// A2A3-LABEL: func.func @test_logb_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [6144]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "logb_fp32"

// A2A3-LABEL: func.func @test_clz_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [18432]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "clz_i32"

// A2A3-LABEL: func.func @test_popc_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [4096]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "popc_i32"

// A2A3-LABEL: func.func @test_byte_perm_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [10240]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "byte_perm_i32"

// A2A3-LABEL: func.func @test_brev_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [8192]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "brev_i32"

// A2A3-LABEL: func.func @test_fdim_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "fdim_fp32"

// A2A3-LABEL: func.func @test_rsqrt_rn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "rsqrt_rn_fp32"

// A2A3-LABEL: func.func @test_fast_sinf_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [8192]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "fast_sin_fp32"

// A2A3-LABEL: func.func @test_fast_cosf_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [8192]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "fast_cos_fp32"

// A2A3-LABEL: func.func @test_fast_tanf_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [16384]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "fast_tan_fp32"

// A2A3-LABEL: func.func @test_fast_tanhf_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [6144]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "fast_tanh_fp32"

// A2A3-LABEL: func.func @test_remainder_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [16384]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "remainder_fp32"

// A2A3-LABEL: func.func @test_float2uint_rn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [4096]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2uint_rn_fp32"

// A2A3-LABEL: func.func @test_float2uint_rz_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [4096]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2uint_rz_fp32"

// A2A3-LABEL: func.func @test_float2uint_rd_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [4096]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2uint_rd_fp32"

// A2A3-LABEL: func.func @test_float2uint_ru_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [4096]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2uint_ru_fp32"

// A2A3-LABEL: func.func @test_uint2float_rn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "uint2float_rn_fp32"

// A2A3-LABEL: func.func @test_uint2float_rz_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "uint2float_rz_fp32"

// A2A3-LABEL: func.func @test_uint2float_rd_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "uint2float_rd_fp32"

// A2A3-LABEL: func.func @test_uint2float_ru_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "uint2float_ru_fp32"

// A2A3-LABEL: func.func @test_ull2float_rn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "ull2float_rn_fp32"

// A2A3-LABEL: func.func @test_ull2float_rz_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "ull2float_rz_fp32"

// A2A3-LABEL: func.func @test_ull2float_rd_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "ull2float_rd_fp32"

// A2A3-LABEL: func.func @test_ull2float_ru_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "ull2float_ru_fp32"

// A2A3-LABEL: func.func @test_rcbrt_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "rcbrt_fp32"

// A2A3-LABEL: func.func @test_rhypot_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [4096]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "rhypot_fp32"

// A2A3-LABEL: func.func @test_norm3d_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "norm3d_fp32"

// A2A3-LABEL: func.func @test_rnorm3d_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "rnorm3d_fp32"

// A2A3-LABEL: func.func @test_norm4d_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "norm4d_fp32"

// A2A3-LABEL: func.func @test_rnorm4d_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "rnorm4d_fp32"

// A2A3-LABEL: func.func @test_j0_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [8192]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "j0_fp32"

// A2A3-LABEL: func.func @test_j1_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [8192]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "j1_fp32"

// A2A3-LABEL: func.func @test_jn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [14336]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "jn_fp32"

// A2A3-LABEL: func.func @test_y0_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [24576]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "y0_fp32"

// A2A3-LABEL: func.func @test_y1_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [24576]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "y1_fp32"

// A2A3-LABEL: func.func @test_yn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [28672]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "yn_fp32"

// A2A3-LABEL: func.func @test_cyl_bessel_i1_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [16384]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "cyl_bessel_i1_fp32"

// A2A3-LABEL: func.func @test_erfc_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [8192]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "erfc_fp32"

// A2A3-LABEL: func.func @test_erfcx_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [10240]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "erfcx_fp32"

// A2A3-LABEL: func.func @test_erfcinv_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [10240]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "erfcinv_fp32"

// A2A3-LABEL: func.func @test_normcdf_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [6144]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "normcdf_fp32"

// A2A3-LABEL: func.func @test_normcdfinv_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [20480]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "normcdfinv_fp32"

// A2A3-LABEL: func.func @test_sinpi_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "sinpi_fp32"

// A2A3-LABEL: func.func @test_cospi_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "cospi_fp32"

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func @test_fast_pow_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_fast_pow_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [512]
    } : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_float2ull_rd_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i64>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2ull_rd_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [320]
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    %out_ptrs = tt.splat %out : !tt.ptr<i64> -> tensor<2048x!tt.ptr<i64>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i64>>
    tt.return
  }

  tt.func @test_float2ull_rn_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i64>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2ull_rn_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [320]
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    %out_ptrs = tt.splat %out : !tt.ptr<i64> -> tensor<2048x!tt.ptr<i64>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i64>>
    tt.return
  }

  tt.func @test_float2ull_ru_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i64>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2ull_ru_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [320]
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    %out_ptrs = tt.splat %out : !tt.ptr<i64> -> tensor<2048x!tt.ptr<i64>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i64>>
    tt.return
  }

  tt.func @test_float2ull_rz_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i64>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2ull_rz_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [320]
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    %out_ptrs = tt.splat %out : !tt.ptr<i64> -> tensor<2048x!tt.ptr<i64>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i64>>
    tt.return
  }

  tt.func @test_tgamma_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_tgamma_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [576]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_logb_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_logb_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [6144]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_clz_2048(%arg0: tensor<2048xi32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_clz_i32",
      extra_buffers_types = [i32], extra_buffers_sizes = [18432]
    } : (tensor<2048xi32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_popc_2048(%arg0: tensor<2048xi32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_popc_i32",
      extra_buffers_types = [i32], extra_buffers_sizes = [4096]
    } : (tensor<2048xi32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_byte_perm_2048(%arg0: tensor<2048xi32>, %arg1: tensor<2048xi32>, %arg2: tensor<2048xi32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0, %arg1, %arg2 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_byte_perm_i32",
      extra_buffers_types = [i32], extra_buffers_sizes = [10240]
    } : (tensor<2048xi32>, tensor<2048xi32>, tensor<2048xi32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_brev_2048(%arg0: tensor<2048xi32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_brev_i32",
      extra_buffers_types = [i32], extra_buffers_sizes = [8192]
    } : (tensor<2048xi32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_fdim_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_fdim_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_rsqrt_rn_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_rsqrt_rn_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_fast_sinf_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_fast_sin_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [8192]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_fast_cosf_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_fast_cos_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [8192]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_fast_tanf_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_fast_tan_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [16384]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_fast_tanhf_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_fast_tanh_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [6144]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_remainder_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_remainder_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [16384]
    } : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_float2uint_rn_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2uint_rn_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [4096]
    } : (tensor<2048xf32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_float2uint_rz_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2uint_rz_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [4096]
    } : (tensor<2048xf32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_float2uint_rd_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2uint_rd_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [4096]
    } : (tensor<2048xf32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_float2uint_ru_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<i32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_float2uint_ru_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [4096]
    } : (tensor<2048xf32>) -> tensor<2048xi32>
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<i32>>
    tt.return
  }

  tt.func @test_uint2float_rn_2048(%arg0: tensor<2048xi32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_uint2float_rn_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_uint2float_rz_2048(%arg0: tensor<2048xi32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_uint2float_rz_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_uint2float_rd_2048(%arg0: tensor<2048xi32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_uint2float_rd_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_uint2float_ru_2048(%arg0: tensor<2048xi32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_uint2float_ru_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_ull2float_rn_2048(%arg0: tensor<2048xi64>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_ull2float_rn_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi64>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_ull2float_rz_2048(%arg0: tensor<2048xi64>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_ull2float_rz_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi64>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_ull2float_rd_2048(%arg0: tensor<2048xi64>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_ull2float_rd_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi64>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_ull2float_ru_2048(%arg0: tensor<2048xi64>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_ull2float_ru_fp32",
      extra_buffers_types = [i32], extra_buffers_sizes = [2048]
    } : (tensor<2048xi64>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_rcbrt_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_rcbrt_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_rhypot_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_rhypot_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [4096]
    } : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_norm3d_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1, %arg2 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_norm3d_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_rnorm3d_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1, %arg2 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_rnorm3d_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_norm4d_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1, %arg2, %arg3 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_norm4d_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_rnorm4d_2048(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1, %arg2, %arg3 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_rnorm4d_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_j0_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_j0_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [8192]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_j1_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_j1_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [8192]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_jn_2048(%arg0: tensor<2048xi32>, %arg1: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_jn_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [14336]
    } : (tensor<2048xi32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_y0_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_y0_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [24576]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_y1_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_y1_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [24576]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_yn_2048(%arg0: tensor<2048xi32>, %arg1: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_yn_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [28672]
    } : (tensor<2048xi32>, tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_cyl_bessel_i1_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_cyl_bessel_i1_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [16384]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_erfc_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_erfc_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [8192]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_erfcx_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_erfcx_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [10240]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_erfcinv_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_erfcinv_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [10240]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_normcdf_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_normcdf_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [6144]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_normcdfinv_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_normcdfinv_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [20480]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_sinpi_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_sinpi_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

  tt.func @test_cospi_2048(%arg0: tensor<2048xf32>, %out: !tt.ptr<f32>) {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_cospi_fp32",
      extra_buffers_types = [f32], extra_buffers_sizes = [2048]
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    tt.store %out_ptrs, %0 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }

}
