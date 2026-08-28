from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from triton._C.libtriton import ascend as ascend_capi
from triton.backends.ascend.runtime import costmodel_runtime


@triton.jit
def _dynamic_loop_kernel(output, n_elements, BLOCK_SIZE: tl.constexpr):
    for block_start in tl.range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        tl.store(output + offsets, 0.0, mask=offsets < n_elements)


def _make_dynamic_loop_ttir():
    n_elements = 257
    output = torch.empty(n_elements, device="npu", dtype=torch.float32)
    config = triton.Config({"BLOCK_SIZE": 128})
    grid = (1, )
    ttir, bound_args, signature, materialized_grid = costmodel_runtime._materialize_ttir_for_config(
        _dynamic_loop_kernel,
        config,
        {"output": output, "n_elements": n_elements},
        {"grid": grid, "warmup": False},
    )
    bindings = costmodel_runtime._build_costmodel_arg_bindings(ttir, bound_args, signature, materialized_grid)
    assert "arg1=257" in bindings
    return ttir, bindings


def _option_forms(bindings, hardware_config):
    return [
        [
            "-ascend-perf-model",
            f"arg-bindings={bindings}",
            f"hardware-config={hardware_config}",
        ],
        [
            "-ascend-perf-model="
            f"arg-bindings={bindings},hardware-config={hardware_config}",
        ],
        [
            "-ascend-perf-model="
            f"arg-bindings={bindings} hardware-config={hardware_config}",
        ],
    ]


def test_inproc_costmodel_parses_separate_and_legacy_option_forms():
    ttir, bindings = _make_dynamic_loop_ttir()
    hardware_config = costmodel_runtime._resolve_hardware_config(target_arch="Ascend910_9362")

    outputs = []
    for args in _option_forms(bindings, hardware_config):
        outputs.append(ascend_capi.run_costmodel_inproc(ttir, [*args, "-allow-unregistered-dialect"]))

    assert all("Estimated Time:" in output for output in outputs)
    assert len(set(outputs)) == 1


def test_inproc_costmodel_option_forms_do_not_drop_hardware_config():
    ttir, bindings = _make_dynamic_loop_ttir()
    missing_config = Path("/costmodel-test/missing-hardware-profile.json")

    for args in _option_forms(bindings, missing_config):
        # The native pass emits the detailed path error to diagnostics, while
        # the in-process binding intentionally exposes a stable generic error.
        with pytest.raises(RuntimeError, match="in-process costmodel pass pipeline failed"):
            ascend_capi.run_costmodel_inproc(ttir, [*args, "-allow-unregistered-dialect"])
