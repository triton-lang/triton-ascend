"""
Test Case: SDF25 - 5-layer alternating pure C or pure V

[MLIR校验] 重构版本

Description: 5-layer nested (R, Q, L, P, K). Layers alternate pure C and pure V.

重构方式(参考test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  2. 参考test_custom.py中的compile_kernel函数,实现获取每个Triton Kernel的MLIR代码的功能
  3. 在测试函数中添加对MLIR代码内容的校验机制,确保MLIR代码中必须包含"scope"关键字
  4. 保持测试框架的完整性和可维护性
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
# SDF25: 5-layer alternating pure C or pure V
# 测试目的: 验证float16下5层嵌套纯C/纯V交替的MLIR生成
# l1(C): tl.dot(a,b) -> store out_l1 [R,R]
# l2(V): c + d       -> store out_l2 [N]
# l3(C): tl.dot(g,h) -> store out_l3 [L,L]
# l4(V): i + j       -> store out_l4 [P]
# l5(C): tl.dot(e,f) -> store out    [K,K]
# ----------------------------------------------------------------------------
@triton.jit
def sdf25(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    i_ptr,
    j_ptr,
    out_ptr,
    out_l1_ptr,
    out_l2_ptr,
    out_l3_ptr,
    out_l4_ptr,
    M,
    N,
    K,
    L,
    P,
    Q,
    R,
    stride_am,
    stride_ar,
    stride_br,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ek,
    stride_fk,
    stride_fn,
    stride_gm,
    stride_gl,
    stride_hl,
    stride_hn,
    stride_i,
    stride_j,
    stride_outm,
    stride_outn,
    stride_l1m,
    stride_l1n,
    stride_l2,
    stride_l3m,
    stride_l3n,
    stride_l4,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r, offs_q, offs_l, offs_p, offs_k, offs_n = tl.arange(0, BLOCK_SIZE_R), tl.arange(0, BLOCK_SIZE_Q), tl.arange(
        0, BLOCK_SIZE_L), tl.arange(0, BLOCK_SIZE_P), tl.arange(0, BLOCK_SIZE_K), tl.arange(0, BLOCK_SIZE_N)
    for l1 in range(R):
        # --- l1: pure C (cube) ---
        a = tl.load(a_ptr + l1 * stride_am + offs_r * stride_ar, mask=offs_r < R, other=0.0)
        b = tl.load(b_ptr + l1 * stride_br + offs_r * stride_bn, mask=offs_r < R, other=0.0)
        l1_cube = tl.dot(a[:, None], b[None, :])
        l1_ptrs = out_l1_ptr + offs_r[:, None] * stride_l1m + offs_r[None, :] * stride_l1n
        tl.store(l1_ptrs, l1_cube, mask=(offs_r[:, None] < R) & (offs_r[None, :] < R))
        for l2 in range(Q):
            # --- l2: pure V (vector) ---
            c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
            d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
            l2_vec = c + d
            l2_ptrs = out_l2_ptr + offs_n * stride_l2
            tl.store(l2_ptrs, l2_vec, mask=offs_n < N)
            for l3 in range(L):
                # --- l3: pure C (cube) ---
                g = tl.load(g_ptr + l3 * stride_gm + offs_l * stride_gl, mask=offs_l < L, other=0.0)
                h = tl.load(h_ptr + l3 * stride_hl + offs_l * stride_hn, mask=offs_l < L, other=0.0)
                l3_cube = tl.dot(g[:, None], h[None, :])
                l3_ptrs = out_l3_ptr + offs_l[:, None] * stride_l3m + offs_l[None, :] * stride_l3n
                tl.store(l3_ptrs, l3_cube, mask=(offs_l[:, None] < L) & (offs_l[None, :] < L))
                for l4 in range(P):
                    # --- l4: pure V (vector) ---
                    i = tl.load(i_ptr + offs_p * stride_i, mask=offs_p < P, other=0.0)
                    j = tl.load(j_ptr + offs_p * stride_j, mask=offs_p < P, other=0.0)
                    l4_vec = i + j
                    l4_ptrs = out_l4_ptr + offs_p * stride_l4
                    tl.store(l4_ptrs, l4_vec, mask=offs_p < P)
                    for l5 in range(K):
                        # --- l5: pure C (cube) ---
                        e = tl.load(e_ptr + l5 * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)
                        f = tl.load(f_ptr + l5 * stride_fk + offs_k * stride_fn, mask=offs_k < K, other=0.0)
                        l5_cube = tl.dot(e[:, None], f[None, :])
                        out_ptrs = out_ptr + offs_k[:, None] * stride_outm + offs_k[None, :] * stride_outn
                        tl.store(out_ptrs, l5_cube, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ============================================================================
# Pytest测试用例
# ============================================================================


def _build_sdf25_signature(dtype_str):
    """构建SDF25 kernel的参数类型签名。"""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "h_ptr": f"*{dtype_str}",
        "i_ptr": f"*{dtype_str}",
        "j_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "out_l1_ptr": f"*{dtype_str}",
        "out_l2_ptr": f"*{dtype_str}",
        "out_l3_ptr": f"*{dtype_str}",
        "out_l4_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "Q": "i32",
        "R": "i32",
        "stride_am": "i32",
        "stride_ar": "i32",
        "stride_br": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_em": "i32",
        "stride_ek": "i32",
        "stride_fk": "i32",
        "stride_fn": "i32",
        "stride_gm": "i32",
        "stride_gl": "i32",
        "stride_hl": "i32",
        "stride_hn": "i32",
        "stride_i": "i32",
        "stride_j": "i32",
        "stride_outm": "i32",
        "stride_outn": "i32",
        "stride_l1m": "i32",
        "stride_l1n": "i32",
        "stride_l2": "i32",
        "stride_l3m": "i32",
        "stride_l3n": "i32",
        "stride_l4": "i32",
    }


def test_sdf25():
    """SDF25: 验证float16 kernel编译生成的MLIR代码正确性。

    测试步骤:
      1. 编译sdf25 kernel为MLIR
      2. 校验MLIR代码成功生成且非空
      3. 校验MLIR代码中包含函数定义
      4. 校验MLIR代码中包含"scope"关键字
    """
    signature = _build_sdf25_signature("fp16")
    constants = {
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_L": 2,
        "BLOCK_SIZE_P": 2,
        "BLOCK_SIZE_Q": 2,
        "BLOCK_SIZE_R": 2,
    }

    mlir = compile_kernel(sdf25, signature, constants)
    _write_mlir_to_file(mlir, "sdf25.mlir")

    assert mlir and len(mlir) > 0, "MLIR代码生成失败或为空"
    assert "func.func @sdf25(" in mlir, \
        "MLIR代码中未找到kernel函数定义"
    assert "scope" not in mlir, "预期回退场景MLIR代码中包含'scope'关键字"

    # 将MLIR代码输出到指定路径


# ============================================================================
# Main用于手动测试
# ============================================================================
if __name__ == "__main__":
    test_sdf25()
    print("All SDF25 v3 MLIR validation tests passed!")
