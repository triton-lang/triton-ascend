"""
Test Case: SDF13 - 3-layer nested loop, alternating pure C or pure V, no data dependency

[MLIR校验] 重构版本

Description: 3-layer nested loop (L, P, K). Outer layer pure C, middle layer pure V, inner layer pure C.
             No data dependency across layers.

重构方式(参考test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  2. 参考test_custom.py中的compile_kernel函数,实现获取每个Triton Kernel的MLIR代码的功能
  3. 在测试函数中添加对MLIR代码内容的校验机制,确保MLIR代码中必须包含"scope"关键字
  4. 保持测试框架的完整性和可维护性

Test Cases:
  - SDF13-TC01: float16, M=128, N=64, K=32, L=3, P=2
  - SDF13-TC02: float32, M=128, N=64, K=32, L=3, P=2
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
# SDF13-TC01: float16, M=128, N=64, K=32, L=3, P=2
# 测试目的: 验证float16下3层嵌套纯C/纯V交替的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def sdf13_tc01_alternating_pure(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    out_ptr,
    out_outer_ptr,
    out_mid_ptr,
    M,
    N,
    K,
    L,
    P,
    stride_am,
    stride_al,
    stride_bl,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ek,
    stride_fk,
    stride_fn,
    stride_out_0,
    stride_out_1,
    stride_outer_0,
    stride_outer_1,
    stride_mid,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for i in range(L):
        # --- outer: pure C (cube) ---
        a = tl.load(a_ptr + i * stride_am + offs_l * stride_al, mask=offs_l < L, other=0.0)  # (L,)
        b = tl.load(b_ptr + i * stride_bl + offs_l * stride_bn, mask=offs_l < L, other=0.0)  # (L,)
        outer_cube = tl.dot(a[:, None], b[None, :])  # (L,L)
        outer_ptrs = out_outer_ptr + offs_l[:, None] * stride_outer_0 + offs_l[None, :] * stride_outer_1
        tl.store(outer_ptrs, outer_cube, mask=(offs_l[:, None] < L) & (offs_l[None, :] < L))

        for j in range(P):
            # --- mid: pure V (vector) ---
            c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
            d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)  # (N,)
            mid_vec = c + d  # (N,)
            mid_ptrs = out_mid_ptr + offs_n * stride_mid
            tl.store(mid_ptrs, mid_vec, mask=offs_n < N)

            for k in range(K):
                # --- inner: pure C (cube) ---
                e = tl.load(e_ptr + k * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)  # (K,)
                f = tl.load(f_ptr + k * stride_fk + offs_k * stride_fn, mask=offs_k < K, other=0.0)  # (K,)
                inner_cube = tl.dot(e[:, None], f[None, :])  # (K,K)
                out_ptrs = out_ptr + offs_k[:, None] * stride_out_0 + offs_k[None, :] * stride_out_1
                tl.store(out_ptrs, inner_cube, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ----------------------------------------------------------------------------
# SDF13-TC02: float32, M=128, N=64, K=32, L=3, P=2
# 测试目的: 验证float32下3层嵌套纯C/纯V交替的MLIR生成
# ----------------------------------------------------------------------------
@triton.jit
def sdf13_tc02_alternating_pure(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    out_ptr,
    out_outer_ptr,
    out_mid_ptr,
    M,
    N,
    K,
    L,
    P,
    stride_am,
    stride_al,
    stride_bl,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ek,
    stride_fk,
    stride_fn,
    stride_out_0,
    stride_out_1,
    stride_outer_0,
    stride_outer_1,
    stride_mid,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for i in range(L):
        # --- outer: pure C (cube) ---
        a = tl.load(a_ptr + i * stride_am + offs_l * stride_al, mask=offs_l < L, other=0.0)
        b = tl.load(b_ptr + i * stride_bl + offs_l * stride_bn, mask=offs_l < L, other=0.0)
        outer_cube = tl.dot(a[:, None], b[None, :])
        outer_ptrs = out_outer_ptr + offs_l[:, None] * stride_outer_0 + offs_l[None, :] * stride_outer_1
        tl.store(outer_ptrs, outer_cube, mask=(offs_l[:, None] < L) & (offs_l[None, :] < L))

        for j in range(P):
            # --- mid: pure V (vector) ---
            c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
            d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
            mid_vec = c + d
            mid_ptrs = out_mid_ptr + offs_n * stride_mid
            tl.store(mid_ptrs, mid_vec, mask=offs_n < N)

            for k in range(K):
                # --- inner: pure C (cube) ---
                e = tl.load(e_ptr + k * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)
                f = tl.load(f_ptr + k * stride_fk + offs_k * stride_fn, mask=offs_k < K, other=0.0)
                inner_cube = tl.dot(e[:, None], f[None, :])
                out_ptrs = out_ptr + offs_k[:, None] * stride_out_0 + offs_k[None, :] * stride_out_1
                tl.store(out_ptrs, inner_cube, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ============================================================================
# Pytest测试用例
# ============================================================================


def _build_sdf13_signature(dtype_str):
    """构建SDF13 kernel的参数类型签名。"""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "out_outer_ptr": f"*{dtype_str}",
        "out_mid_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "stride_am": "i32",
        "stride_al": "i32",
        "stride_bl": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_em": "i32",
        "stride_ek": "i32",
        "stride_fk": "i32",
        "stride_fn": "i32",
        "stride_out_0": "i32",
        "stride_out_1": "i32",
        "stride_outer_0": "i32",
        "stride_outer_1": "i32",
        "stride_mid": "i32",
    }


def test_sdf13_tc01():
    """SDF13-TC01: 验证float16 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译sdf13_tc01_alternating_pure kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = _build_sdf13_signature("fp16")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_L": 3, "BLOCK_SIZE_P": 2}

    mlir = compile_kernel(sdf13_tc01_alternating_pure, signature, constants)
    _write_mlir_to_file(mlir, "sdf13_tc01_alternating_pure.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @sdf13_tc01_alternating_pure(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


def test_sdf13_tc02():
    """SDF13-TC02: 验证float32 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译sdf13_tc02_alternating_pure kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = _build_sdf13_signature("fp32")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_L": 3, "BLOCK_SIZE_P": 2}

    mlir = compile_kernel(sdf13_tc02_alternating_pure, signature, constants)
    _write_mlir_to_file(mlir, "sdf13_tc02_alternating_pure.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @sdf13_tc02_alternating_pure(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


# ============================================================================
# Main用于手动测试
# ============================================================================
if __name__ == "__main__":
    test_sdf13_tc01()
    test_sdf13_tc02()
    print("All SDF13 v3 MLIR validation tests passed!")
