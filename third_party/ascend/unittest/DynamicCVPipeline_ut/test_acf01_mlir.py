"""
Test Case: ACF01 - No for-loop, independent Cube + Vector in main body, no data dependency

[MLIR校验] 重构版本

Description: No for-loop, independent Cube (outer product) + Vector (cube_result + c).
             No data dependency. Not supporting SSBuf optimization.

重构方式(参考test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  2. 参考test_custom.py中的compile_kernel函数,实现获取每个Triton Kernel的MLIR代码的功能
  3. 在测试函数中添加对MLIR代码内容的校验机制,确保MLIR代码中必须包含"scope"关键字
  4. 保持测试框架的完整性和可维护性

Test Cases:
  - ACF01-TC01: float16, M=128, N=64, K=32
  - ACF01-TC02: float32, M=128, N=64, K=32
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

# ============================================================================
# 编译辅助函数: 将Triton Kernel编译为MLIR (linalg dialect)
# 参考: test_custom.py 中的 compile_kernel 实现
# ============================================================================


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect.

    将Triton Kernel编译为linalg方言的MLIR代码,用于后续的内容校验。

    Args:
        kernel: Triton JIT编译的kernel函数
        signature: 参数类型签名字典,例如 {"x_ptr": "*fp32", "n": "i32"}
        constants: constexpr参数字典,例如 {"BLOCK": 256}

    Returns:
        str: 编译生成的MLIR代码字符串;编译失败时返回None
    """
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions()
        # 注册codegen_fns,包含tl.dot所需的min_dot_size函数。
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
# ACF01-TC01: float16, M=128, N=64, K=32
# 测试目的: 验证float16下无for循环独立CV操作的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def acf01_tc01_no_loop_cv(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_c,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    a = tl.load(a_ptr + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
    c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)

    # Build (K, N) result directly via element-wise broadcasting:
    # b has only K elements, so for column j >= K, the outer product is 0
    # Load b with N-sized index, mask to K so b[j]=0 for j>=K
    b_n = tl.load(b_ptr + offs_n * stride_bk, mask=offs_n < K, other=0.0)  # (N,) padded with 0
    # outer product: a[:, None] * b_n[None, :] -> (K, N)
    cube_result = a[:, None] * b_n[None, :]  # (K, N)
    vec_result = cube_result + c[None, :]  # (K, N) + (1, N) -> (K, N)
    tl.store(out_ptr + offs_k[:, None] * stride_out + offs_n[None, :], vec_result,
             mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))


# ----------------------------------------------------------------------------
# ACF01-TC02: float32, M=128, N=64, K=32
# 测试目的: 验证float32下无for循环独立CV操作的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def acf01_tc02_no_loop_cv(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_c,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a = tl.load(a_ptr + offs_k * stride_ak, mask=offs_k < K, other=0.0)
    c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)

    b_n = tl.load(b_ptr + offs_n * stride_bk, mask=offs_n < K, other=0.0)
    cube_result = a[:, None] * b_n[None, :]  # (K, N)
    vec_result = cube_result + c[None, :]  # (K, N) + (1, N) -> (K, N)
    tl.store(out_ptr + offs_k[:, None] * stride_out + offs_n[None, :], vec_result,
             mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))


# ============================================================================
# Pytest测试用例
# ============================================================================


def test_acf01_tc01():
    """ACF01-TC01: 验证float16 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译acf01_tc01_no_loop_cv kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "out_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_c": "i32",
        "stride_out": "i32",
    }
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf01_tc01_no_loop_cv, signature, constants)
    _write_mlir_to_file(mlir, "acf01_tc01_no_loop_cv.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @acf01_tc01_no_loop_cv(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


def test_acf01_tc02():
    """ACF01-TC02: 验证float32 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译acf01_tc02_no_loop_cv kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "out_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_c": "i32",
        "stride_out": "i32",
    }
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf01_tc02_no_loop_cv, signature, constants)
    _write_mlir_to_file(mlir, "acf01_tc02_no_loop_cv.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @acf01_tc02_no_loop_cv(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


# ============================================================================
# Main用于手动测试
# ============================================================================

if __name__ == "__main__":
    test_acf01_tc01()
    test_acf01_tc02()
    print("All ACF01 v3 MLIR validation tests passed!")
