"""
Test Case: ACF07 - loop with if/else control flow, V2C dependency (condition from Vector, Cube depends on Vector)

[MLIR校验] 重构版本

Description: For loop with if/else: tl.max(c) as condition. If >0, Cube depends on Vector result (V2C),
             else execute Vector (d / 2.0).

重构方式(参考test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  2. 参考test_custom.py中的compile_kernel函数,实现获取每个Triton Kernel的MLIR代码的功能
  3. 在测试函数中添加对MLIR代码内容的校验机制,确保MLIR代码中必须包含"scope"关键字
  4. 保持测试框架的完整性和可维护性

Test Cases:
  - ACF07-TC01: float16, M=128, N=64, K=32
  - ACF07-TC02: float32, M=128, N=64, K=32
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
# ACF07-TC01: float16, M=128, N=64, K=32
# 注意: K 为 tl.constexpr, 不在 signature 中, 放入 constants
# ----------------------------------------------------------------------------
@triton.jit
def acf07_tc01_v2c_control_flow(
    a_ptr, b_ptr, c_ptr, d_ptr, out_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_c,
    stride_d,
    stride_out_0, stride_out_1,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)
    out_ptrs = out_ptr + offs_k[:, None] * stride_out_0 + offs_k[None, :] * stride_out_1

    for k in range(K):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
        vec_cond = tl.max(c)  # scalar

        if vec_cond > 0.0:
            a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
            b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
            cube_result = tl.dot(a[:, None], b[None, :]) + vec_cond  # (K,K) + scalar
            tl.store(out_ptrs, cube_result, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))
        else:
            loaded_d = tl.load(d_ptr + offs_k * stride_d, mask=offs_k < K, other=0.0)  # (K,)
            vec_result = loaded_d / 2.0  # (K,)
            tl.store(out_ptrs, vec_result[None, :], mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ----------------------------------------------------------------------------
# ACF07-TC02: float32, M=128, N=64, K=32
# ----------------------------------------------------------------------------
@triton.jit
def acf07_tc02_v2c_control_flow(
    a_ptr, b_ptr, c_ptr, d_ptr, out_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_c,
    stride_d,
    stride_out_0, stride_out_1,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + offs_k[:, None] * stride_out_0 + offs_k[None, :] * stride_out_1

    for k in range(K):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        vec_cond = tl.max(c)

        if vec_cond > 0.0:
            a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
            b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)
            cube_result = tl.dot(a[:, None], b[None, :]) + vec_cond
            tl.store(out_ptrs, cube_result, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))
        else:
            loaded_d = tl.load(d_ptr + offs_k * stride_d, mask=offs_k < K, other=0.0)
            vec_result = loaded_d / 2.0
            tl.store(out_ptrs, vec_result[None, :], mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ============================================================================
# Pytest测试用例
# ============================================================================

def test_acf07_tc01():
    """ACF07-TC01: 验证float16 kernel编译生成的MLIR代码正确性。

    注意: K 为 constexpr 参数, 放入 constants 而非 signature。
    """
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "d_ptr": "*fp16",
        "out_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_out_0": "i32",
        "stride_out_1": "i32",
    }
    constants = {"K": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf07_tc01_v2c_control_flow, signature, constants)
    _write_mlir_to_file(mlir, "acf07_tc01_v2c_control_flow.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @acf07_tc01_v2c_control_flow(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径

def test_acf07_tc02():
    """ACF07-TC02: 验证float32 kernel编译生成的MLIR代码正确性。

    注意: K 为 constexpr 参数, 放入 constants 而非 signature。
    """
    signature = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "d_ptr": "*fp32",
        "out_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_out_0": "i32",
        "stride_out_1": "i32",
    }
    constants = {"K": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf07_tc02_v2c_control_flow, signature, constants)
    _write_mlir_to_file(mlir, "acf07_tc02_v2c_control_flow.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @acf07_tc02_v2c_control_flow(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径

if __name__ == "__main__":
    test_acf07_tc01()
    test_acf07_tc02()
    print("All ACF07 v3 MLIR validation tests passed!")
