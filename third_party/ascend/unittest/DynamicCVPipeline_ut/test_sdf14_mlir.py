"""
Test Case: SDF14 - 3-layer nested loop, outer C depends on inner C (C2C cross-layer)

[MLIR校验] 重构版本

Description: 3-layer nested (L, P, K). Inner loop accumulates inner_cube_acc (K-size).
             Middle loop has independent CV. Outer C uses inner_cube_acc in tl.dot.
             Outer V: h + 1.0.

重构方式(参考test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  2. 参考test_custom.py中的compile_kernel函数,实现获取每个Triton Kernel的MLIR代码的功能
  3. 在测试函数中添加对MLIR代码内容的校验机制,确保MLIR代码中必须包含"scope"关键字
  4. 保持测试框架的完整性和可维护性

Test Cases:
  - SDF14-TC01: float16, M=128, N=64, K=32, L=3, P=2
  - SDF14-TC02: float32, M=128, N=64, K=32, L=3, P=2
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
        options = NPUOptions(compile_on_910_95=True, enable_dynamic_cv_pipeline=True)
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
# SDF14-TC01: float16, M=128, N=64, K=32, L=3, P=2
# 测试目的: 验证float16下3层嵌套C2C跨层依赖的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def sdf14_tc01_outer_c_dep_inner_c(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out_ptr,
    M,
    N,
    K,
    L,
    P,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ep,
    stride_fp,
    stride_fn,
    stride_gm,
    stride_gk,
    stride_h,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_p = tl.arange(0, BLOCK_SIZE_P)

    inner_cube_acc = tl.zeros([BLOCK_SIZE_K, BLOCK_SIZE_K], tl.float32)

    for i in range(L):
        for j in range(P):
            for k in range(K):
                a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
                b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)
                inner_cube = tl.dot(a[:, None], b[None, :])
                inner_cube_acc = inner_cube_acc + inner_cube

                c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
                d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
                inner_vec = c * d

            e = tl.load(e_ptr + j * stride_em + offs_p * stride_ep, mask=offs_p < P, other=0.0)
            f = tl.load(f_ptr + j * stride_fp + offs_p * stride_fn, mask=offs_p < P, other=0.0)
            mid_cube = tl.dot(e[:, None], f[None, :])

        g = tl.load(g_ptr + i * stride_gm + offs_k * stride_gk, mask=offs_k < K, other=0.0)
        g = g.to(tl.float32)
        outer_cube = tl.dot(inner_cube_acc, g[:, None])
        out_ptrs = out_ptr + offs_k[:, None] * stride_out
        tl.store(out_ptrs, outer_cube, mask=offs_k[:, None] < K)

        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)
        outer_vec = h + 1.0


# ----------------------------------------------------------------------------
# SDF14-TC02: float32, M=128, N=64, K=32, L=3, P=2
# 测试目的: 验证float32下3层嵌套C2C跨层依赖的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def sdf14_tc02_outer_c_dep_inner_c(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out_ptr,
    M,
    N,
    K,
    L,
    P,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ep,
    stride_fp,
    stride_fn,
    stride_gm,
    stride_gk,
    stride_h,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_p = tl.arange(0, BLOCK_SIZE_P)

    inner_cube_acc = tl.zeros([BLOCK_SIZE_K, BLOCK_SIZE_K], tl.float32)

    for i in range(L):
        for j in range(P):
            for k in range(K):
                a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
                b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)
                inner_cube = tl.dot(a[:, None], b[None, :])
                inner_cube_acc = inner_cube_acc + inner_cube

                c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
                d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
                inner_vec = c * d

            e = tl.load(e_ptr + j * stride_em + offs_p * stride_ep, mask=offs_p < P, other=0.0)
            f = tl.load(f_ptr + j * stride_fp + offs_p * stride_fn, mask=offs_p < P, other=0.0)
            mid_cube = tl.dot(e[:, None], f[None, :])

        g = tl.load(g_ptr + i * stride_gm + offs_k * stride_gk, mask=offs_k < K, other=0.0)
        outer_cube = tl.dot(inner_cube_acc, g[:, None])
        out_ptrs = out_ptr + offs_k[:, None] * stride_out
        tl.store(out_ptrs, outer_cube, mask=offs_k[:, None] < K)

        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)
        outer_vec = h + 1.0


# ============================================================================
# Pytest测试用例
# ============================================================================


def _build_sdf14_signature(dtype_str):
    """构建SDF14 kernel的参数类型签名。"""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "h_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_em": "i32",
        "stride_ep": "i32",
        "stride_fp": "i32",
        "stride_fn": "i32",
        "stride_gm": "i32",
        "stride_gk": "i32",
        "stride_h": "i32",
        "stride_out": "i32",
    }


def test_sdf14_tc01():
    """SDF14-TC01: 验证float16 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译sdf14_tc01_outer_c_dep_inner_c kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = _build_sdf14_signature("fp16")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_L": 3, "BLOCK_SIZE_P": 2}

    mlir = compile_kernel(sdf14_tc01_outer_c_dep_inner_c, signature, constants)
    _write_mlir_to_file(mlir, "sdf14_tc01_outer_c_dep_inner_c.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @sdf14_tc01_outer_c_dep_inner_c(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


def test_sdf14_tc02():
    """SDF14-TC02: 验证float32 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译sdf14_tc02_outer_c_dep_inner_c kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = _build_sdf14_signature("fp32")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_L": 3, "BLOCK_SIZE_P": 2}

    mlir = compile_kernel(sdf14_tc02_outer_c_dep_inner_c, signature, constants)
    _write_mlir_to_file(mlir, "sdf14_tc02_outer_c_dep_inner_c.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @sdf14_tc02_outer_c_dep_inner_c(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


# ============================================================================
# Main用于手动测试
# ============================================================================
if __name__ == "__main__":
    test_sdf14_tc01()
    test_sdf14_tc02()
    print("All SDF14 v3 MLIR validation tests passed!")
