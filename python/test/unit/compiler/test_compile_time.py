import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_compile_time_module():
    module_path = Path(__file__).resolve().parents[3] / "triton" / "compiler" / "compile_time.py"
    assert module_path.exists(), f"{module_path} is missing"
    spec = importlib.util.spec_from_file_location("triton_compile_time", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_ta_compile_time_log_splits_npuir_from_total_compile_time():
    compile_time = load_compile_time_module()

    line = compile_time.build_ta_compile_time_log(
        "triton_actual_kernel",
        {
            "constants": {"BLOCK": 128},
            "multibuffer": True,
        },
        {"stage_ttir": 0.10, "stage_npubin": 2.25, "metadata_writeback": 0.05},
        total_compile_time=3.0,
    )

    assert line == (
        '[TA][triton_actual_kernel] '
        'config={"constants":{"BLOCK":128},"multibuffer":true} '
        "npuir_compile_time=2.250000s ta_compile_time=0.750000s"
    )


def test_build_ta_compile_time_log_skips_non_npu_compile_without_npubin_stage():
    compile_time = load_compile_time_module()

    line = compile_time.build_ta_compile_time_log(
        "cuda_kernel",
        {"constants": {}, "multibuffer": False},
        {"stage_ttir": 0.10, "stage_ttgir": 0.20},
        total_compile_time=0.30,
    )

    assert line is None


def test_collect_compile_config_keeps_only_constants_and_multibuffer():
    compile_time = load_compile_time_module()
    src = SimpleNamespace(
        signature={"x": "*fp32", "n": "i32"},
        constants={"BLOCK": 128},
        attrs=SimpleNamespace(divisibility=(16,), equal_to=()),
    )
    options = SimpleNamespace(num_warps=1, num_stages=2, debug=False, multibuffer=False)

    config = compile_time.collect_compile_config(src, options)

    assert config == {"constants": {"BLOCK": 128}, "multibuffer": False}


def test_emit_ta_compile_time_log_prefers_actual_metadata_kernel_name():
    compile_time = load_compile_time_module()
    src = SimpleNamespace(
        name="python_kernel_name",
        signature={"x": "*fp32"},
        constants={"BLOCK": 128},
        attrs=None,
    )
    options = SimpleNamespace(num_warps=1, multibuffer=True)
    printed = []

    line = compile_time.emit_ta_compile_time_log(
        src,
        {"kernel_name": "triton_actual_kernel"},
        options,
        {"stage_npubin": 1.5},
        total_compile_time=2.0,
        sink=printed.append,
    )

    assert line.startswith("[TA][triton_actual_kernel] ")
    assert 'config={"constants":{"BLOCK":128},"multibuffer":true}' in line
    assert "npuir_compile_time=1.500000s" in line
    assert "ta_compile_time=0.500000s" in line
    assert printed == [line]
