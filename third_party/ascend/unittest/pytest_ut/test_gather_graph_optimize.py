# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native-pass, early-TTIR and NPU regressions for automatic Gather optimization."""

import os
from pathlib import Path
import re

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource

pytestmark = pytest.mark.backend("none")


@triton.jit
def indirect_rows_kernel(src_ptr, idx_ptr, out_ptr, n_rows, WIDTH: tl.constexpr, K: tl.constexpr, ROW_BLK: tl.constexpr,
                         ROW_STEP: tl.constexpr, PER_LANE: tl.constexpr, VOLATILE: tl.constexpr,
                         BASE_ROW: tl.constexpr):
    for rb in range(0, ROW_BLK, ROW_STEP):
        rows = tl.program_id(0) * ROW_BLK + rb + tl.arange(0, ROW_STEP)
        cols = tl.arange(0, K)
        row_mask = rows < n_rows
        positions = rows[:, None] * K + cols[None, :]
        # Deliberately out of range: inactive indices must be sanitized before
        # both the bounds reductions and the actual gather.
        indices = tl.load(idx_ptr + positions, row_mask[:, None], other=-123)
        mask = row_mask[:, None]
        if PER_LANE:
            mask = mask & (cols[None, :] % 2 == 0)
        # Leave physical prefix rows so the original-load baseline and fallback
        # can read negative offsets. Wrapped indices deliberately differ.
        base = (rows + BASE_ROW)[:, None] * WIDTH
        # Keep the integer offset addition before the pointer addition. Nested
        # addptr operations with dynamic i32 offsets cannot generally be folded
        # together (the combined i32 sum could overflow), and are outside the
        # gather rule's direct-pointer matcher. Test addresses fit in i32.
        offsets = base + indices
        value = tl.load(src_ptr + offsets, mask, other=-7.0, volatile=VOLATILE)
        # Output includes padding rows: observe `other` rather than hiding it
        # behind a masked store. The host allocates ROW_BLK rows per program.
        tl.store(out_ptr + positions, value)


def make_indirect_ttir(rule_mask=None, *, per_lane=False, volatile=False, index_type="i32", value_type="fp32",
                       compile_mode="simd", arch="Ascend910B1"):
    mode_options = {} if compile_mode is None else {"compile_mode": compile_mode}
    rule_options = {} if rule_mask is None else {"rule_mask": rule_mask}
    options = NPUOptions(arch=arch, **rule_options, **mode_options)
    source = ASTSource(
        indirect_rows_kernel,
        {"src_ptr": f"*{value_type}", "idx_ptr": f"*{index_type}", "out_ptr": f"*{value_type}", "n_rows": "i32"},
        {"WIDTH": 16, "K": 8, "ROW_BLK": 8, "ROW_STEP": 2, "PER_LANE": per_lane, "VOLATILE": volatile, "BASE_ROW": 1},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(indirect_rows_kernel, source, context, options, {}, {})
    return make_ttir(module, {}, options)


@triton.jit
def mapping_gather_kernel(x_ptr, src_ptr, idx_ptr, out_ptr, H: tl.constexpr, K: tl.constexpr, W: tl.constexpr,
                          R: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, K)
    # Separate axis strides remain analyzable across IAT -> PTSM.
    x = tl.load(x_ptr + (token * (H * K) + head * K + cols))
    normalized = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + 1.0e-5)
    rows = tl.arange(0, R)
    indices = tl.load(idx_ptr + rows[:, None] * K + cols[None, :])
    offsets = rows[:, None] * W + indices
    selected = tl.load(src_ptr + offsets)
    out_offsets = token * (H * R * K) + head * (R * K) + rows[:, None] * K + cols[None, :]
    tl.store(out_ptr + out_offsets, normalized[None, :] + selected)


@pytest.mark.parametrize("rule_mask,mapping,gather", [(None, True, False), (3071, True, False), (66047, False, True)],
                         ids=["default", "gather-off", "mapping-off"])
def test_gather_skips_program_mapping(tmp_path, rule_mask, mapping, gather):
    options = NPUOptions(arch="Ascend910B1", sanitize_overflow=False,
                         **({} if rule_mask is None else {"rule_mask": rule_mask}))
    source = ASTSource(mapping_gather_kernel,
                       {"x_ptr": "*fp32", "src_ptr": "*fp32", "idx_ptr": "*i32", "out_ptr": "*fp32"},
                       {"H": 17, "K": 32, "W": 8192, "R": 2})
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(mapping_gather_kernel, source, context, options, {}, {})
    text = str(make_ttir(module, {}, options))
    assert ("hacc.independent_axis_tensorize" in text) == mapping
    assert ("hacc.persistent_task_strip_mining" in text) == mapping
    assert ("tt.gather" in text) == gather
    assert ("gather.optimised.load" in text) == gather
    path = tmp_path / "mapping-gather.ttir.mlir"
    path.write_text(text)
    ir.parse_mlir_module(str(path), context)


def gather_fixture():
    path = Path(__file__).parents[1] / "Conversion/General/TritonToGraph/graph-optimize-gather-safety.mlir"
    return re.search(r"tt.func @row_mask_other\(.*?\n}", path.read_text(), re.S).group()


def run_gather_pass(tmp_path, text, **options):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = tmp_path / "gather-input.mlir"
    path.write_text("module {\n" + text + "\n}")
    module = ir.parse_mlir_module(str(path), context)
    # Parsed modules do not carry the Python context attribute attached by
    # ast_to_ttir. Keep using the context that owns this parsed module.
    pm = ir.pass_manager(context)
    options.setdefault("target_arch", "Ascend910B1")
    ascend.passes.ttir.add_graph_optimize(pm, rule_mask=65536, ub_capacity_bytes=192 * 1024 * 80 // 100,
                                          compile_mode="simd", **options)
    pm.run(module, "")
    result = str(module)
    path.write_text(result)
    ir.parse_mlir_module(str(path), context)
    return result


@pytest.mark.parametrize("rule_mask,per_lane,volatile,index_type,expected", [
    (None, False, False, "i32", True),  # Omit rule_mask: production default.
    (0, False, False, "i32", False),
    (3071, False, False, "i32", False),
    (512, False, False, "i32", False),  # IAT owns bit 512 on main-dev.
    (65536, False, False, "i32", True),
    (68607, False, False, "i32", True),
    (65536, True, False, "i32", False),
    (65536, False, True, "i32", False),
    (65536, False, False, "i64", False),
])
def test_gather_rule_in_make_ttir(tmp_path, rule_mask, per_lane, volatile, index_type, expected):
    module = make_indirect_ttir(rule_mask, per_lane=per_lane, volatile=volatile, index_type=index_type)
    text = str(module)
    assert ("tt.gather" in text) == expected
    if rule_mask == 3071:
        # The input needs only an index load and an indirect source load.
        assert text.count("tt.load") == 2
    if expected:
        assert 'gather.optimised.load = "source"' in text
        assert 'gather.optimised.load = "fallback"' in text
        assert "arith.shrsi" in text
    # The real early pipeline must emit valid, reparsable IR.
    path = tmp_path / "gather.ttir.mlir"
    path.write_text(text)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ir.parse_mlir_module(str(path), context)
    # A second pass must leave the transformed source/fallback alone.
    pm = ir.pass_manager(module.context)
    ascend.passes.ttir.add_graph_optimize(pm, rule_mask=65536, ub_capacity_bytes=192 * 1024 * 80 // 100,
                                          compile_mode="simd", target_arch="Ascend910B1")
    pm.run(module, "")
    if expected:
        assert str(module).count("tt.gather") == text.count("tt.gather")
        assert str(module).count("scf.if") == text.count("scf.if")


def test_gather_rule_mask_changes_cache_key():
    assert NPUOptions(rule_mask=3071).hash() != NPUOptions().hash()


@pytest.mark.parametrize("rule_mask", [3071, 65536, None])
@pytest.mark.parametrize("arch,compile_mode", [
    ("Ascend910B1", None),
    ("Ascend910B1", "simd"),
    ("Ascend910B1", "simd_simt_template"),
    ("Ascend910B1", "unstructured_in_simt"),
    ("Ascend910_9589", None),
    ("Ascend910_9589", "simd"),
    ("Ascend910_9589", "simd_simt_template"),
    ("Ascend910_9589", "unstructured_in_simt"),
    ("Ascend910_9589", "simt_only"),
    ("Ascend950", "simd"),
    ("Ascend950", "simt_only"),
    ("Ascend910_9391", None),
    ("Ascend910_9391", "simd"),
    ("Ascend910_9391", "simd_simt_template"),
    ("Ascend910_9391", "unstructured_in_simt"),
])
def test_gather_rule_target_and_mode(rule_mask, arch, compile_mode):
    text = str(make_indirect_ttir(rule_mask, arch=arch, compile_mode=compile_mode))
    expected = (rule_mask is None or bool(rule_mask & 65536)) and arch in ("Ascend910B1", "Ascend910_9391")
    assert ("tt.gather" in text) == expected
    assert ("gather.optimised.load" in text) == expected


@pytest.mark.parametrize("arch,expected", [
    ("Ascend910B1", True),
    ("Ascend910_9391", True),
    ("Ascend910A", False),
    ("Ascend910D", False),
    ("Ascend310B1", False),
    ("Ascend910_9589", False),
    ("Ascend950", False),
    ("Ascend950PR_9599", False),
    ("", False),
    ("unknown", False),
])
def test_gather_target_eligibility(tmp_path, arch, expected):
    result = run_gather_pass(tmp_path, gather_fixture(), target_arch=arch)
    assert ("tt.gather" in result) == expected
    assert ("gather.optimised.load" in result) == expected


@pytest.mark.parametrize("lower,upper,step,dependency,expected", [
    (0, 4, 2, "row", False),
    (128, 132, 2, "row", False),  # two iterations, previously estimated as 66
    (128, 161, 2, "row", True),  # ceil(33 / 2) = 17, not 16
    (128, 160, 2, "row", False),  # exactly 16 iterations
    (128, 260, 2, "row", True),
    (-128, -124, 2, "row", False),
    (-128, 4, 2, "row", True),
    (132, 128, 2, "row", False),
    (128, 128, 2, "row", False),
    (None, 260, 2, "row", False),
    (0, None, 2, "row", False),
    (0, 256, 4, "row", False),  # loop step differs from the two-row tile
    (0, 256, 2, "unrelated", False),
    (0, 256, 2, "cancelled", False),
    (0, 256, 2, "nonlinear", False),
])
def test_gather_row_loop_iterations(tmp_path, lower, upper, step, dependency, expected):
    # FP32 source width 16 and index width 2: thresholds are 4 for <= 4
    # iterations, 3 for <= 16, and 2 for >= 17. The input contains only
    # indirect source reads, so it also exercises the original Gather pattern.
    text = gather_fixture().replace("2x8x", "2x2x")
    header, body = text.split("{\n", 1)
    text = header.replace("%src:", "%dynamic_lower: i32, %dynamic_upper: i32, %src:") + "{\n"
    lower_name, upper_name = "%dynamic_lower", "%dynamic_upper"
    if lower is not None:
        text += f"  %lower = arith.constant {lower} : i32\n"
        lower_name = "%lower"
    if upper is not None:
        text += f"  %upper = arith.constant {upper} : i32\n"
        upper_name = "%upper"
    text += f"  %step = arith.constant {step} : i32\n"
    text += "  %init = arith.constant dense<0.0> : tensor<2x2xf32>\n"
    text += (f"  %result = scf.for %iv = {lower_name} to {upper_name} step %step "
             "iter_args(%previous = %init) "
             "-> (tensor<2x2xf32>) : i32 {\n")
    if dependency != "unrelated":
        carrier = "%iv"
        if dependency == "cancelled":
            text += "  %carrier = arith.subi %iv, %iv : i32\n"
            carrier = "%carrier"
        elif dependency == "nonlinear":
            text += "  %carrier = arith.muli %iv, %iv : i32\n"
            carrier = "%carrier"
        text += f"  %row_bias = arith.addi %bias, {carrier} : i32\n"
        body = body.replace("tt.splat %bias :", "tt.splat %row_bias :")
    body = body.replace("tt.return %value", "scf.yield %value")
    text += body + "\n  tt.return %result : tensor<2x2xf32>\n}"
    result = run_gather_pass(tmp_path, text)
    assert ("tt.gather" in result) == expected
    assert ("gather.optimised.load" in result) == expected


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("case", [
    "tail",
    "negative",
    "negative_below_bound",
    "negative_upper_bound",
    "upper_bound",
    "all_masked",
    "per_lane",
    "volatile",
    "i64",
])
def test_gather_rule_npu_semantics(monkeypatch, dtype, case):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    width, k, row_blk = 16, 8, 8
    n_rows = 0 if case == "all_masked" else 7
    # Keep raw negative accesses valid even when the index is below -width.
    base_row = 2 if case == "negative_below_bound" else 1
    physical_rows = n_rows + base_row + (1 if case == "upper_bound" else 0)
    source = torch.arange(physical_rows * width, dtype=torch.float32).to(getattr(torch, dtype))
    index_dtype = torch.int64 if case == "i64" else torch.int32
    indices = torch.arange(max(n_rows, 1) * k, dtype=index_dtype).reshape(-1, k) % width
    if case.startswith("negative"):
        indices[0, 0] = -1
        indices[1, 1] = -width
        # The last partial tile must still wrap: inactive -123 lanes are masked.
        indices[-1, 0] = -1
    if case == "negative_below_bound":
        indices[0, 0] = -width - 1
    if case == "negative_upper_bound":
        # One upper-bound index must force the whole tile, including -1, back
        # to the original pointer-offset semantics.
        indices[1, 1] = width
    if case == "upper_bound":
        indices[-1, 0] = width
    reference = torch.full((row_blk, k), -7.0, dtype=getattr(torch, dtype))
    for row in range(n_rows):
        reference[row] = source[(row + base_row) * width + indices[row].long()]
    # The requested v2 / PR #1027 contract wraps only an in-range tile when
    # the rule is active. Do not claim equality with raw negative pointer loads.
    gather_reference = reference.clone()
    for begin in range(0, n_rows, 2):
        tile = indices[begin:min(begin + 2, n_rows)].long()
        if bool(((tile >= -width) & (tile < width)).all()):
            normalized = torch.where(tile < 0, tile + width, tile)
            for lane in range(tile.shape[0]):
                row = begin + lane
                gather_reference[row] = source[(row + base_row) * width + normalized[lane]]
    if case == "per_lane":
        reference[:, 1::2] = -7.0
    source_npu, indices_npu = source.npu(), indices.npu()
    arch = triton.runtime.driver.active.get_current_target().arch
    gather_target = arch.startswith(("Ascend910B", "Ascend910_93"))
    outputs = []
    # Exercise the public default-on / explicit-off launch contract.
    for gather_enabled in (False, True):
        rule_options = {} if gather_enabled else {"rule_mask": 3071}
        indirect_rows_kernel.device_caches.clear()
        output = torch.empty((row_blk, k), dtype=getattr(torch, dtype), device="npu")
        compiled = indirect_rows_kernel[(1, )](
            source_npu,
            indices_npu,
            output,
            n_rows,
            WIDTH=width,
            K=k,
            ROW_BLK=row_blk,
            ROW_STEP=2,
            PER_LANE=case == "per_lane",
            VOLATILE=case == "volatile",
            BASE_ROW=base_row,
            **rule_options,
        )
        torch.npu.synchronize()
        expected_rewrite = gather_target and gather_enabled and case not in ("per_lane", "volatile", "i64")
        assert ("tt.gather" in compiled.asm["ttir"]) == expected_rewrite
        outputs.append(output.cpu())
        expected_output = gather_reference if expected_rewrite else reference
        torch.testing.assert_close(outputs[-1], expected_output, rtol=0, atol=0)
    if case.startswith("negative") and gather_target:
        # Confirm the intended wraparound behavior actually differs from the
        # disabled-rule baseline. Distinct source values expose wrong addresses.
        assert not torch.equal(outputs[0], outputs[1])
        if case != "negative":
            # Only the out-of-range tile falls back; the final tile still wraps.
            torch.testing.assert_close(outputs[0][:2], outputs[1][:2], rtol=0, atol=0)
    else:
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("compile_mode", [None, "simd"])
def test_gather_rule_npu_full_row_offset_view(monkeypatch, compile_mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    # Both complete source rows must be readable from the passed view.
    # Keep a nonzero storage offset to verify the original pointer is preserved.
    storage = torch.arange(33, dtype=torch.float32, device="npu")
    source = storage[1:]
    indices = torch.arange(8, dtype=torch.int32, device="npu").repeat(2, 1)
    reference = torch.stack((storage[1:9], storage[17:25])).cpu()
    arch = triton.runtime.driver.active.get_current_target().arch
    gather_target = arch.startswith(("Ascend910B", "Ascend910_93"))
    mode_options = {} if compile_mode is None else {"compile_mode": compile_mode}
    # Exercise the public default-on / explicit-off launch contract.
    for gather_enabled in (False, True):
        rule_options = {} if gather_enabled else {"rule_mask": 3071}
        indirect_rows_kernel.device_caches.clear()
        output = torch.empty((2, 8), dtype=torch.float32, device="npu")
        compiled = indirect_rows_kernel[(1, )](source, indices, output, 2, WIDTH=16, K=8, ROW_BLK=2, ROW_STEP=2,
                                               PER_LANE=False, VOLATILE=False, BASE_ROW=0, **rule_options,
                                               **mode_options)
        torch.npu.synchronize()
        expected_rewrite = gather_target and gather_enabled
        assert ("tt.gather" in compiled.asm["ttir"]) == expected_rewrite
        assert ("gather.optimised.load" in compiled.asm["ttir"]) == expected_rewrite
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


def test_gather_rule_npu_negative_debug(tmp_path, monkeypatch):
    """Verbose, deterministic negative-index diagnostic; run with pytest -s -vv.

    Save IR before launching and stop immediately on a device error. References
    use CPU indexing: the raw-load baseline and v2-style wraparound intentionally
    differ. All active raw addresses and all complete source rows are allocated.
    """
    import json
    import sys
    import traceback

    artifacts = tmp_path / "gather-negative-debug"
    artifacts.mkdir()
    log_path = artifacts / "debug.log"

    def log(message):
        text = f"[gather-negative-debug] {message}"
        print(text, flush=True)
        with log_path.open("a") as stream:
            stream.write(text + "\n")
            stream.flush()

    stage = "import and initialize NPU"
    log(f"START {stage}; artifacts={artifacts}")
    try:
        torch = pytest.importorskip("torch")
        torch_npu = pytest.importorskip("torch_npu", exc_type=ImportError)
        log(f"python={sys.version}; torch={torch.__version__}; torch_npu={torch_npu.__version__}")
        log(f"triton={triton.__version__}; triton_file={triton.__file__}")
        native_module = sys.modules.get("triton._C.libtriton")
        log(f"native_extension={getattr(native_module, '__file__', '<unknown>')}")
        for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING", "TRITON_ALWAYS_COMPILE", "TRITON_CACHE_DIR",
                    "TRITON_DUMP_DIR", "TRITON_OVERRIDE_DIR"):
            log(f"environment {key}={os.environ.get(key)!r}")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            log("SKIP: Ascend NPU is unavailable")
            pytest.skip("Ascend NPU is unavailable")
        device = torch.npu.current_device()
        target = triton.runtime.driver.active.get_current_target()
        log(f"device={device}; name={torch.npu.get_device_name(device)}; target={target}")
        if not target.arch.startswith(("Ascend910B", "Ascend910_93")):
            log("SKIP: this target does not enable automatic Gather in v4")
            pytest.skip(f"Automatic Gather is disabled for {target.arch}")

        # Force fresh native compilation, then reuse the JIT entries for launches.
        monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
        indirect_rows_kernel.device_caches.clear()
        log("debug overrides: TRITON_ALWAYS_COMPILE=1; kernel JIT cache cleared")
        width, k, n_rows, row_blk, row_step, base_row = 16, 8, 3, 4, 2, 2
        dtype = torch.float32
        variants = {"disabled": {"rule_mask": 3071}, "default": {}}
        options = dict(WIDTH=width, K=k, ROW_BLK=row_blk, ROW_STEP=row_step, PER_LANE=False, VOLATILE=False,
                       BASE_ROW=base_row, compile_mode="simd")
        log(f"source_dtype={dtype}; index_dtype=int32; grid=(1,); n_rows={n_rows}; options={options}; variants={variants}"
            )
        log("disabled: rule_mask=3071 (raw-load baseline); default: rule_mask omitted (Gather enabled)")
        log("auto_hit below denotes a compiled rewrite; expected_branch is inferred from the CPU input bounds")

        # Two prefix rows protect the baseline's -W-1 access; a suffix row also
        # keeps the upper-bound fallback readable. Source values equal offsets.
        source = torch.arange((base_row + n_rows + 1) * width, dtype=dtype)
        log(f"source storage ({source.numel()} elements, values equal physical offsets):\n{source.reshape(-1, width).tolist()}"
            )
        log(f"logical source row bases={[(base_row + row) * width for row in range(n_rows)]}; padded output row={n_rows}, other=-7"
            )
        mixed = torch.tensor([
            [-1, -16, 0, 1, 7, 8, 14, 15],
            [-2, -15, 2, 3, 4, 5, 6, 7],
            [-1, -16, -8, 0, 1, 2, 7, 15],
        ], dtype=torch.int32)
        below = mixed.clone()
        below[0, 0] = -width - 1
        upper = mixed.clone()
        upper[1, 1] = width
        cases = {
            "positive_control": torch.arange(n_rows * k, dtype=torch.int32).reshape(n_rows, k) % width,
            "mixed_negative": mixed,
            "all_negative": -(torch.arange(n_rows * k, dtype=torch.int32).reshape(n_rows, k) % width) - 1,
            "below_negative_bound": below,
            "at_upper_bound": upper,
        }

        stage = "allocate and transfer initial tensors"
        log(f"START {stage}")
        source_npu = source.npu(device)
        indices_npu = cases["positive_control"].npu(device)
        output_npu = torch.empty((row_blk, k), dtype=dtype, device=source_npu.device)
        torch.npu.synchronize()
        for name, value in (("source", source_npu), ("indices", indices_npu), ("output", output_npu)):
            log(f"{name}: shape={tuple(value.shape)}, stride={value.stride()}, dtype={value.dtype}, device={value.device}, storage_offset={value.storage_offset()}, data_ptr={value.data_ptr():#x}"
                )

        # warmup compiles without launching. Persist every textual IR stage now,
        # so a later launch/synchronization failure does not lose compiler output.
        compiled_ir = {}
        for variant, rule_options in variants.items():
            stage = f"compile without launch: variant={variant}"
            log(f"START {stage}")
            compiled = indirect_rows_kernel.warmup(source_npu, indices_npu, output_npu, n_rows, grid=(1, ),
                                                   **rule_options, **options)
            log(f"compiled variant={variant}; metadata={compiled.metadata}; asm_keys={list(compiled.asm)}")
            for key, value in compiled.asm.items():
                if isinstance(value, str):
                    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", key)
                    path = artifacts / f"{variant}.{name}"
                    path.write_text(value)
                    log(f"saved {path}")
            ttir = compiled.asm["ttir"]
            compiled_ir[variant] = ttir
            gather = "tt.gather" in ttir
            source_marker = 'gather.optimised.load = "source"' in ttir
            fallback_marker = 'gather.optimised.load = "fallback"' in ttir
            normalization = "arith.shrsi" in ttir
            log(f"variant={variant}: gather={gather}, source_marker={source_marker}, fallback_marker={fallback_marker}, negative_normalization={normalization}"
                )
            lines = ttir.splitlines()
            selected = set()
            for line, text in enumerate(lines):
                if any(token in text
                       for token in ("tt.gather", "gather.optimised.load", "arith.shrsi", "arith.cmpi", "scf.if")):
                    selected.update(range(max(0, line - 2), min(len(lines), line + 3)))
            log("TTIR excerpts:\n" + "\n".join(f"{line + 1}: {lines[line]}" for line in sorted(selected)))
            if variant == "disabled":
                assert not any((gather, source_marker, fallback_marker)), "Baseline unexpectedly rewrote"
            else:
                assert all((gather, source_marker, fallback_marker, normalization)), (
                    f"variant={variant}: expected negative-index Gather rewrite; inspect saved TTIR and installed compiler"
                )

        for case, indices in cases.items():
            stage = f"CPU reference: {case}"
            log(f"START {stage}; indices={indices.tolist()}")
            padded = torch.full((row_blk, k), -123, dtype=torch.int32)
            padded[:n_rows] = indices
            active = torch.arange(row_blk)[:, None] < n_rows
            sanitized = torch.where(active, padded, 0).long()
            raw_addresses = (torch.arange(n_rows)[:, None] + base_row) * width + indices.long()
            assert bool(((raw_addresses >= 0) & (raw_addresses < source.numel())).all())
            raw = torch.full((row_blk, k), -7.0, dtype=dtype)
            raw[:n_rows] = source[raw_addresses]
            expected = raw.clone()
            branches = []
            for begin in range(0, row_blk, row_step):
                tile = sanitized[begin:begin + row_step]
                minimum, maximum = int(tile.min()), int(tile.max())
                fast = minimum >= -width and maximum < width
                branch = "gather + wrap" if fast else "original-load fallback"
                branches.append(dict(row_begin=begin, min=minimum, max=maximum, expected_branch=branch))
                log(f"case={case}, rows={begin}:{begin + row_step}, active={active[begin:begin + row_step].flatten().tolist()}, sanitized_indices={tile.tolist()}, min={minimum}, max={maximum}, expected_branch={branch}"
                    )
                if fast:
                    normalized = torch.where(tile < 0, tile + width, tile)
                    log(f"normalized indices={normalized.tolist()}")
                    for lane in range(row_step):
                        row = begin + lane
                        if row < n_rows:
                            addresses = (row + base_row) * width + normalized[lane]
                            log(f"row={row}: Gather physical addresses={addresses.tolist()}")
                            expected[row] = source[addresses]
            log(f"raw-load physical addresses={raw_addresses.tolist()}")
            log(f"expected disabled (raw offsets)={raw.tolist()}")
            log(f"expected default (tile-wise wrap/fallback)={expected.tolist()}")
            if case != "positive_control":
                assert not torch.equal(raw, expected), "Diagnostic data must distinguish wrapping from raw offsets"
            record = dict(case=case, indices=indices.tolist(), branches=branches, raw_addresses=raw_addresses.tolist(),
                          expected_raw=raw.tolist(), expected_gather=expected.tolist(), actual={})
            result_path = artifacts / f"{case}.json"
            result_path.write_text(json.dumps(record, indent=2) + "\n")
            stage = f"copy indices and synchronize: {case}"
            log(f"START {stage}")
            indices_npu.copy_(indices)
            torch.npu.synchronize()
            for variant, rule_options in variants.items():
                stage = f"poison output and synchronize: {case}, variant={variant}"
                log(f"START {stage}")
                output_npu.fill_(float("nan"))
                torch.npu.synchronize()
                stage = f"kernel launch: {case}, variant={variant}"
                log(f"START {stage}")
                launched = indirect_rows_kernel[(1, )](source_npu, indices_npu, output_npu, n_rows, **rule_options,
                                                       **options)
                stage = f"post-kernel synchronize: {case}, variant={variant}"
                log(f"START {stage}")
                torch.npu.synchronize()
                stage = f"copy output to CPU and compare: {case}, variant={variant}"
                log(f"START {stage}")
                actual = output_npu.cpu()
                log(f"actual={actual.tolist()}")
                record["actual"][variant] = actual.tolist()
                result_path.write_text(json.dumps(record, indent=2) + "\n")
                assert launched.asm["ttir"] == compiled_ir[variant], "Launch differs from saved compilation"
                reference = raw if variant == "disabled" else expected
                bad = (actual != reference).nonzero().tolist()
                log(f"mismatches={len(bad)}; max_abs_error={float((actual - reference).abs().max())}")
                for row, col in bad:
                    index = int(indices[row, col]) if row < n_rows else "masked"
                    log(f"mismatch [{row}, {col}]: index={index}, actual={float(actual[row, col])}, expected={float(reference[row, col])}"
                        )
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                log(f"PASS case={case}, variant={variant}")
        log(f"PASS all five inputs with Gather disabled and default options; artifacts={artifacts}")
    except Exception:
        # Do not issue further NPU calls after an asynchronous device failure.
        log(f"FAILED during {stage}; artifacts={artifacts}\n{traceback.format_exc()}")
        raise
