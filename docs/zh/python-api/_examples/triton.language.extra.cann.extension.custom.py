import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, get_libdevice
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource

# This runnable test compiles a CustomOp call to Triton IR without launching an NPU.
# It reuses the real add_rn_fp32 implementation in Triton-Ascend's bundled library;
# no additional bitcode or C++ files are needed. Device linking and numerical
# execution are outside this example; the bundled implementation targets A5.
# From the repository root:
# python3 -m pytest --import-mode=importlib -q \
#     docs/zh/python-api/_examples/triton.language.extra.cann.extension.custom.py


@al.register_custom_op
class custom_example_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    symbol = "add_rn_fp32"
    bitcode = get_libdevice()


@triton.jit
def custom_example_kernel(x_ptr, y_ptr, out_ptr):
    offsets = tl.arange(0, 64)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = al.custom("custom_example_op", x, y, out=tl.full((64, ), 0, tl.float32))
    tl.store(out_ptr + offsets, result)


def test_custom_frontend():
    # Explicit types let the frontend compile without allocating device tensors.
    source = ASTSource(custom_example_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"})
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    # Match the bundled implementation without querying the machine's device.
    options = NPUOptions(arch="Ascend950")
    module = ast_to_ttir(custom_example_kernel, source, context, options, {}, {})
    ttir = str(module)

    custom_lines = [line for line in ttir.splitlines() if "hivm.hir.custom" in line]
    assert len(custom_lines) == 1
    custom_call = custom_lines[0]
    assert '"custom_example_op"' in custom_call
    assert 'symbol = "add_rn_fp32"' in custom_call
    assert f'bitcode = "{get_libdevice()}"' in custom_call
    assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in custom_call
    assert "hivm.pipe = #hivm.pipe<PIPE_V>" in custom_call
    assert "hivm.vf_mode = #hivm.vf_mode<SIMD>" in custom_call
    assert "tensor<64xf32>" in custom_call
    assert "tt.load" in ttir
    assert "tt.store" in ttir


if __name__ == "__main__":
    test_custom_frontend()
