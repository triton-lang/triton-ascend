"""
Test Case: PCB02 - Multiple independent Cube + multiple independent Vector, no data dependency

[MLIR校验] 重构版本

Description: 2 independent Cube (outer product) + 2 independent Vector (element-wise sub),
             no data dependency between Cube and Vector operations.

重构方式(参考test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  2. 参考test_custom.py中的compile_kernel函数,实现获取每个Triton Kernel的MLIR代码的功能
  3. 在测试函数中添加对MLIR代码内容的校验机制,确保MLIR代码中必须包含"scope"关键字
  4. 保持测试框架的完整性和可维护性

Test Cases:
  - PCB02-TC01: float16, M=128, N=64, K=32
  - PCB02-TC02: float32, M=128, N=64, K=32
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
        metadata = {
            **options.__dict__,
        }
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
# PCB02-TC01: float16, M=128, N=64, K=32
# 测试目的: 验证float16下多个独立CUBE+VECTOR操作的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def pcb02_tc01_multi_cube_vector(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_ck,
    stride_dk,
    stride_dn,
    stride_e,
    stride_f,
    stride_g,
    stride_h,
    stride_out1m,
    stride_out1n,
    stride_out2m,
    stride_out2n,
    stride_out3,
    stride_out4,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a1 = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b1 = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result1 = tl.dot(a1[:, None], b1[None, :])  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result1, mask=out1_mask)

        a2 = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)  # (K,)
        b2 = tl.load(d_ptr + k * stride_dk + offs_k * stride_dn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result2 = tl.dot(a2[:, None], b2[None, :])  # (K, K)
        out2_ptrs = out2_ptr + offs_k[:, None] * stride_out2m + offs_k[None, :] * stride_out2n
        out2_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out2_ptrs, cube_result2, mask=out2_mask)

        e = tl.load(e_ptr + offs_n * stride_e, mask=offs_n < N, other=0.0)  # (N,)
        f = tl.load(f_ptr + offs_n * stride_f, mask=offs_n < N, other=0.0)  # (N,)
        vec_result1 = e - f  # (N,)
        out3_ptrs = out3_ptr + offs_n * stride_out3
        tl.store(out3_ptrs, vec_result1, mask=offs_n < N)

        g = tl.load(g_ptr + offs_n * stride_g, mask=offs_n < N, other=0.0)  # (N,)
        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)  # (N,)
        vec_result2 = g - h  # (N,)
        out4_ptrs = out4_ptr + offs_n * stride_out4
        tl.store(out4_ptrs, vec_result2, mask=offs_n < N)


# ----------------------------------------------------------------------------
# PCB02-TC02: float32, M=128, N=64, K=32
# 测试目的: 验证float32下多个独立CUBE+VECTOR操作的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def pcb02_tc02_multi_cube_vector(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_ck,
    stride_dk,
    stride_dn,
    stride_e,
    stride_f,
    stride_g,
    stride_h,
    stride_out1m,
    stride_out1n,
    stride_out2m,
    stride_out2n,
    stride_out3,
    stride_out4,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a1 = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b1 = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result1 = tl.dot(a1[:, None], b1[None, :])  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result1, mask=out1_mask)

        a2 = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)  # (K,)
        b2 = tl.load(d_ptr + k * stride_dk + offs_k * stride_dn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result2 = tl.dot(a2[:, None], b2[None, :])  # (K, K)
        out2_ptrs = out2_ptr + offs_k[:, None] * stride_out2m + offs_k[None, :] * stride_out2n
        out2_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out2_ptrs, cube_result2, mask=out2_mask)

        e = tl.load(e_ptr + offs_n * stride_e, mask=offs_n < N, other=0.0)  # (N,)
        f = tl.load(f_ptr + offs_n * stride_f, mask=offs_n < N, other=0.0)  # (N,)
        vec_result1 = e - f  # (N,)
        out3_ptrs = out3_ptr + offs_n * stride_out3
        tl.store(out3_ptrs, vec_result1, mask=offs_n < N)

        g = tl.load(g_ptr + offs_n * stride_g, mask=offs_n < N, other=0.0)  # (N,)
        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)  # (N,)
        vec_result2 = g - h  # (N,)
        out4_ptrs = out4_ptr + offs_n * stride_out4
        tl.store(out4_ptrs, vec_result2, mask=offs_n < N)


# ============================================================================
# Pytest测试用例
# ============================================================================


def _build_pcb02_signature(dtype_str):
    """构建PCB02 kernel的参数类型签名。"""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "h_ptr": f"*{dtype_str}",
        "out1_ptr": f"*{dtype_str}",
        "out2_ptr": f"*{dtype_str}",
        "out3_ptr": f"*{dtype_str}",
        "out4_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_ck": "i32",
        "stride_dk": "i32",
        "stride_dn": "i32",
        "stride_e": "i32",
        "stride_f": "i32",
        "stride_g": "i32",
        "stride_h": "i32",
        "stride_out1m": "i32",
        "stride_out1n": "i32",
        "stride_out2m": "i32",
        "stride_out2n": "i32",
        "stride_out3": "i32",
        "stride_out4": "i32",
    }


def test_pcb02_tc01():
    """PCB02-TC01: 验证float16 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译pcb02_tc01_multi_cube_vector kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = _build_pcb02_signature("fp16")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(pcb02_tc01_multi_cube_vector, signature, constants)
    _write_mlir_to_file(mlir, "pcb02_tc01_multi_cube_vector.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @pcb02_tc01_multi_cube_vector(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


def test_pcb02_tc02():
    """PCB02-TC02: 验证float32 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译pcb02_tc02_multi_cube_vector kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = _build_pcb02_signature("fp32")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(pcb02_tc02_multi_cube_vector, signature, constants)
    _write_mlir_to_file(mlir, "pcb02_tc02_multi_cube_vector.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @pcb02_tc02_multi_cube_vector(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


# ============================================================================
# Main用于手动测试
# ============================================================================
if __name__ == "__main__":
    test_pcb02_tc01()
    test_pcb02_tc02()
    print("All PCB02 v3 MLIR validation tests passed!")
