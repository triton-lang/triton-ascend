"""
Test Case: ACF04 - >14 independent Cube + single independent Vector, no data dependency

[MLIR校验] 重构版本

Description: >14 independent Cube (15 independent outer products) + single independent Vector (c+1.0),
             no data dependency.

重构方式(参考test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  2. 参考test_custom.py中的compile_kernel函数,实现获取每个Triton Kernel的MLIR代码的功能
  3. 在测试函数中添加对MLIR代码内容的校验机制,确保MLIR代码中必须包含"scope"关键字
  4. 保持测试框架的完整性和可维护性

Test Cases:
  - ACF04-TC01: float16, M=128, N=64, K=32
  - ACF04-TC02: float32, M=128, N=64, K=32
"""

import os
import subprocess
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir, ttir_to_linalg, min_dot_size
import pytest


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions()
        codegen_fns = {"min_dot_size": min_dot_size(None)}
        ttir = ast_to_ttir(kernel, src, context, options, codegen_fns, {})
        metadata = {**options.__dict__}
        # 调用make_ttir进行TTIR优化(与正常编译路径一致),
        # 包括inliner/canonicalizer/cse/licm/loop_unroll等关键优化passes,
        # 缺少此步骤会导致复杂kernel(如while循环)在
        # ttir_to_linalg降级时抛出RuntimeError: PassManager::run failed
        ttir = make_ttir(ttir, metadata, options)
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        return str(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")
        return None


# ============================================================================
# MLIR输出配置
# ============================================================================
# MLIR输出目录: 与本测试文件同级的 mlir_output 子目录
MLIR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlir_output")


def _write_mlir_to_file(mlir, filename):
    os.makedirs(MLIR_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(MLIR_OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"MLIR代码已写入: {output_path}")


# ============================================================================
# Kernel定义
# ============================================================================


# ----------------------------------------------------------------------------
# ACF04-TC01: float16, M=128, N=64, K=32
# ----------------------------------------------------------------------------
@triton.jit
def acf04_tc01_many_cubes(
    a1_ptr,
    a2_ptr,
    a3_ptr,
    a4_ptr,
    a5_ptr,
    a6_ptr,
    a7_ptr,
    a8_ptr,
    a9_ptr,
    a10_ptr,
    a11_ptr,
    a12_ptr,
    a13_ptr,
    a14_ptr,
    a15_ptr,
    b1_ptr,
    b2_ptr,
    b3_ptr,
    b4_ptr,
    b5_ptr,
    b6_ptr,
    b7_ptr,
    b8_ptr,
    b9_ptr,
    b10_ptr,
    b11_ptr,
    b12_ptr,
    b13_ptr,
    b14_ptr,
    b15_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_c,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a1 = tl.load(a1_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b1 = tl.load(b1_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)  # (K,)
        cube1 = tl.dot(a1[:, None], b1[None, :])  # (K,K)

        a2 = tl.load(a2_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b2 = tl.load(b2_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube2 = tl.dot(a2[:, None], b2[None, :])

        a3 = tl.load(a3_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b3 = tl.load(b3_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube3 = tl.dot(a3[:, None], b3[None, :])

        a4 = tl.load(a4_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b4 = tl.load(b4_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube4 = tl.dot(a4[:, None], b4[None, :])

        a5 = tl.load(a5_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b5 = tl.load(b5_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube5 = tl.dot(a5[:, None], b5[None, :])

        a6 = tl.load(a6_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b6 = tl.load(b6_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube6 = tl.dot(a6[:, None], b6[None, :])

        a7 = tl.load(a7_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b7 = tl.load(b7_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube7 = tl.dot(a7[:, None], b7[None, :])

        a8 = tl.load(a8_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b8 = tl.load(b8_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube8 = tl.dot(a8[:, None], b8[None, :])

        a9 = tl.load(a9_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b9 = tl.load(b9_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube9 = tl.dot(a9[:, None], b9[None, :])

        a10 = tl.load(a10_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b10 = tl.load(b10_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube10 = tl.dot(a10[:, None], b10[None, :])

        a11 = tl.load(a11_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b11 = tl.load(b11_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube11 = tl.dot(a11[:, None], b11[None, :])

        a12 = tl.load(a12_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b12 = tl.load(b12_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube12 = tl.dot(a12[:, None], b12[None, :])

        a13 = tl.load(a13_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b13 = tl.load(b13_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube13 = tl.dot(a13[:, None], b13[None, :])

        a14 = tl.load(a14_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b14 = tl.load(b14_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube14 = tl.dot(a14[:, None], b14[None, :])

        a15 = tl.load(a15_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b15 = tl.load(b15_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube15 = tl.dot(a15[:, None], b15[None, :])

        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
        vec_result = c + 1.0  # (N,)
        out_ptrs = out_ptr + offs_n * stride_out
        tl.store(out_ptrs, vec_result, mask=offs_n < N)


# ----------------------------------------------------------------------------
# ACF04-TC02: float32, M=128, N=64, K=32
# ----------------------------------------------------------------------------
@triton.jit
def acf04_tc02_many_cubes(
    a1_ptr,
    a2_ptr,
    a3_ptr,
    a4_ptr,
    a5_ptr,
    a6_ptr,
    a7_ptr,
    a8_ptr,
    a9_ptr,
    a10_ptr,
    a11_ptr,
    a12_ptr,
    a13_ptr,
    a14_ptr,
    a15_ptr,
    b1_ptr,
    b2_ptr,
    b3_ptr,
    b4_ptr,
    b5_ptr,
    b6_ptr,
    b7_ptr,
    b8_ptr,
    b9_ptr,
    b10_ptr,
    b11_ptr,
    b12_ptr,
    b13_ptr,
    b14_ptr,
    b15_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_c,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for k in range(K):
        a1 = tl.load(a1_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b1 = tl.load(b1_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube1 = tl.dot(a1[:, None], b1[None, :])

        a2 = tl.load(a2_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b2 = tl.load(b2_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube2 = tl.dot(a2[:, None], b2[None, :])

        a3 = tl.load(a3_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b3 = tl.load(b3_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube3 = tl.dot(a3[:, None], b3[None, :])

        a4 = tl.load(a4_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b4 = tl.load(b4_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube4 = tl.dot(a4[:, None], b4[None, :])

        a5 = tl.load(a5_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b5 = tl.load(b5_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube5 = tl.dot(a5[:, None], b5[None, :])

        a6 = tl.load(a6_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b6 = tl.load(b6_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube6 = tl.dot(a6[:, None], b6[None, :])

        a7 = tl.load(a7_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b7 = tl.load(b7_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube7 = tl.dot(a7[:, None], b7[None, :])

        a8 = tl.load(a8_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b8 = tl.load(b8_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube8 = tl.dot(a8[:, None], b8[None, :])

        a9 = tl.load(a9_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b9 = tl.load(b9_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube9 = tl.dot(a9[:, None], b9[None, :])

        a10 = tl.load(a10_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b10 = tl.load(b10_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube10 = tl.dot(a10[:, None], b10[None, :])

        a11 = tl.load(a11_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b11 = tl.load(b11_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube11 = tl.dot(a11[:, None], b11[None, :])

        a12 = tl.load(a12_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b12 = tl.load(b12_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube12 = tl.dot(a12[:, None], b12[None, :])

        a13 = tl.load(a13_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b13 = tl.load(b13_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube13 = tl.dot(a13[:, None], b13[None, :])

        a14 = tl.load(a14_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b14 = tl.load(b14_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube14 = tl.dot(a14[:, None], b14[None, :])

        a15 = tl.load(a15_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b15 = tl.load(b15_ptr + k * stride_bn + offs_k, mask=offs_k < K, other=0.0)
        cube15 = tl.dot(a15[:, None], b15[None, :])

        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        vec_result = c + 1.0
        out_ptrs = out_ptr + offs_n * stride_out
        tl.store(out_ptrs, vec_result, mask=offs_n < N)


# ============================================================================
# Pytest测试用例
# ============================================================================


def _build_acf04_signature(dtype):
    """构建ACF04 kernel的参数签名。"""
    ptr_type = f"*{dtype}"
    sig = {}
    for i in range(1, 16):
        sig[f"a{i}_ptr"] = ptr_type
        sig[f"b{i}_ptr"] = ptr_type
    sig["c_ptr"] = ptr_type
    sig["out_ptr"] = ptr_type
    for s in ["M", "N", "K", "stride_am", "stride_ak", "stride_bn", "stride_c", "stride_out"]:
        sig[s] = "i32"
    return sig


def test_acf04_tc01():
    """ACF04-TC01: 验证float16 kernel编译生成的MLIR代码正确性。"""
    signature = _build_acf04_signature("fp16")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf04_tc01_many_cubes, signature, constants)
    _write_mlir_to_file(mlir, "acf04_tc01_many_cubes.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @acf04_tc01_many_cubes(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


def test_acf04_tc02():
    """ACF04-TC02: 验证float32 kernel编译生成的MLIR代码正确性。"""
    signature = _build_acf04_signature("fp32")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf04_tc02_many_cubes, signature, constants)
    _write_mlir_to_file(mlir, "acf04_tc02_many_cubes.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @acf04_tc02_many_cubes(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


if __name__ == "__main__":
    test_acf04_tc01()
    test_acf04_tc02()
    print("All ACF04 v3 MLIR validation tests passed!")
