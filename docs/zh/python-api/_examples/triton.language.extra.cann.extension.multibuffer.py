import os
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton._C.libtriton import ir, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


@triton.jit
def multibuffer(XBLOCK: tl.constexpr):
    buf = bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    al.multibuffer(buf, 2)


def test_multibuffer():
    print("=" * 60)
    print("Test 1: test_alloc_ub_multibuffer")
    print("=" * 60)
    mlir = compile_kernel(multibuffer, {}, {"XBLOCK": 256})
    print(f"Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)


if __name__ == "__main__":
    test_multibuffer()
