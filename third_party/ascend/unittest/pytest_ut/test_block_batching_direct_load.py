"""Masked indirect slices must keep padding without a second outer copy."""
from pathlib import Path
import re
import subprocess

import pytest
import triton

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def lower(tmp_path, *, other="0.0", prefix=True, mode="simd", to_linalg=False):
    fixture = (Path(__file__).parents[1] / "Conversion/General/"
               "TritonToUnstructure/prefix_zero_load.mlir")
    text = fixture.read_text().replace("dense<0.0>", f"dense<{other}>")
    if not prefix:
        text = text.replace("%rmhi, %rmhi :", "%rmlo, %rmhi :")
    path = tmp_path / "load.mlir"
    path.write_text(text)
    tool = Path(triton.__file__).parent / "_C/triton-opt"
    command = [str(tool),
               str(path), "--triton-to-unstructure=compile-on-910-95=true "
               f"compile-mode={mode}"]
    if to_linalg:
        command += [
            "--bubble-up-operation", "--triton-to-linalg=compile-on-910-95=true "
            f"compile-mode={mode}", "--canonicalize"
        ]
    return subprocess.check_output([*command, "--verify-each"], text=True)


@pytest.mark.parametrize("mode", ["simd", "simd_simt_template"])
def test_prefix_zero_uses_initialized_destination_and_masked_slice(tmp_path, mode):
    text = lower(tmp_path, mode=mode)
    # No unconditional read may replace the residual column mask. The initial
    # destination handles skipped rows, including a zero-iteration row loop.
    assert re.search(r"tt.load %[^,]+, %[^,]+, %[^ ]+ \{DiscreteMemAccess\}", text)
    assert text.index("linalg.fill") < text.index("scf.for")
    assert "arith.select" not in text


def test_dma_destination_retains_initialization_for_skipped_lanes(tmp_path):
    text = lower(tmp_path, to_linalg=True)
    # The loop's replacement allocation, not a dead tensor temporary, must be
    # initialized. Otherwise masked-out rows have undefined values after T2L.
    alloc = re.search(r"(%\w+) = memref.alloc\(\) : memref<3x32xf32>", text)
    assert alloc, text
    assert re.search(r"linalg.fill.*outs\(" + re.escape(alloc[1]) + r" : memref<3x32xf32>\)", text)
    assert text.index("linalg.fill") < text.index("scf.for")
    assert "arith.select" not in text


@pytest.mark.parametrize("other,prefix", [("-7.0", True), ("0.0", False)])
def test_nonzero_padding_and_nonprefix_masks_keep_existing_selection(tmp_path, other, prefix):
    text = lower(tmp_path, other=other, prefix=prefix)
    assert "arith.select" in text
    assert re.search(r"tt.load %[^,]+, %[^ ]+ \{DiscreteMemAccess\}", text)


@pytest.mark.parametrize("dtype,precision,hf32", [
    ("f32", "ieee", False),
    ("f32", "hf32", True),
    ("f16", "hf32", False),
    ("bf16", "ieee", False),
])
def test_batch_dot_keeps_full_logical_mkn_and_precision(tmp_path, dtype, precision, hf32):
    # Non-square operands make accidental use of batch, M, K or N as another
    # extent observable. These are padded tensor extents; masks need not be true.
    text = lower_dot(tmp_path, dtype=dtype, precision=precision, marked=True)
    assert "linalg.batch_matmul" not in text
    line, = [line for line in text.splitlines() if "hivm.hir.batchMmadL1" in line]
    assert "already_set_real_mkn" in line
    assert ("enable_HF32" in line) is hf32
    constants = dict(re.findall(r"(%\w+) = arith.constant (\d+) : index", text))
    operands = line.split("ins(", 1)[1].split(" : ", 1)[0].split(", ")
    assert [int(constants[v]) for v in operands[3:6]] == [16, 32, 64]


def test_unmarked_batch_dot_keeps_existing_linalg_lowering(tmp_path):
    text = lower_dot(tmp_path, dtype="f32", precision="ieee", marked=False)
    assert "linalg.batch_matmul" in text
    assert "hivm.hir.batchMmadL1" not in text


def lower_dot(tmp_path, *, dtype, precision, marked):
    a, b, c = f"tensor<2x16x32x{dtype}>", f"tensor<2x32x64x{dtype}>", "tensor<2x16x64xf32>"
    attrs = " {tt.cv_batch_full_mkn}" if marked else ""
    source = f"""module {{
  func.func @batch_dot(%a: {a}, %b: {b}, %c: {c}) -> {c} {{
    %result = tt.dot %a, %b, %c, inputPrecision = {precision}{attrs} : {a} * {b} -> {c}
    func.return %result : {c}
  }}
}}"""
    path = tmp_path / "batch-dot.mlir"
    path.write_text(source)
    tool = Path(triton.__file__).parent / "_C/triton-opt"
    return subprocess.check_output([
        str(tool),
        str(path), "--triton-to-linalg=global-kernel=false compile-on-910-95=true", "--canonicalize", "--verify-each"
    ], text=True)
