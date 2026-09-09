"""Opt-in full-runtime correctness checks for task_0005 fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_0005_harness.contracts import LOGITS_PRIMARY, MERGE_PRIMARY_CASES, NormRopeCase  # noqa: E402
from task_0005_harness.reference import (  # noqa: E402
    bf16_peak_ulp, max_abs_error, ref_indexer_logits, ref_indexer_norm_rope, ref_merge_split_states,
)
from task_0005_harness.grid_specialization import (  # noqa: E402
    run_unchanged_dsl_grid_baseline, )
from task_0005_harness.runtime import (  # noqa: E402
    compiler_identity, launch_indexer_logits, launch_merge_split, launch_norm_rope, make_logits_inputs,
    make_merge_inputs, make_norm_rope_inputs, npu_available,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK0005_ENABLE_E2E") != "1" or not npu_available(),
    reason="set TASK0005_ENABLE_E2E=1 on an initialized Ascend NPU runtime",
)


def test_external_bisheng_compiler_is_resolved_by_this_test_process():
    identity = compiler_identity()
    assert identity.get("matches_expected"), identity


@pytest.mark.parametrize("case", MERGE_PRIMARY_CASES)
def test_merge_split_primary_baseline_and_graph_control(case):
    inputs = make_merge_inputs(case, device=torch.device("npu"))
    expected = ref_merge_split_states(inputs["partial_out"], inputs["partial_lse"])
    baseline = launch_merge_split(inputs, case, graph_optimize=True)
    control = launch_merge_split(inputs, case, graph_optimize=False)
    torch.npu.synchronize()
    for result in (baseline, control):
        assert torch.allclose(result.outputs[0], expected[0], rtol=1.0e-2, atol=4.0e-3)
        assert torch.allclose(result.outputs[1], expected[1], rtol=1.0e-3, atol=1.0e-3)
        assert torch.isfinite(result.outputs[0]).all()
        assert torch.isfinite(result.outputs[1]).all()


@pytest.mark.parametrize(
    "tokens",
    tuple(int(value) for value in os.environ.get("TASK0005_NORM_TOKENS", "1,16").split(",")),
)
def test_norm_rope_primary_baseline_and_graph_control(tokens):
    case = NormRopeCase(tokens=tokens)
    inputs = make_norm_rope_inputs(case, device=torch.device("npu"))
    expected = ref_indexer_norm_rope(
        inputs["q"],
        inputs["k"],
        inputs["z"],
        inputs["q_weight"],
        inputs["k_weight"],
        inputs["k_bias"],
        inputs["cos"],
        inputs["sin"],
        inputs["positions"],
    )
    baseline = launch_norm_rope(inputs, case, graph_optimize=True)
    control = launch_norm_rope(inputs, case, graph_optimize=False)
    torch.npu.synchronize()
    for result in (baseline, control):
        for actual, reference in zip(result.outputs[:2], expected[:2]):
            assert max_abs_error(actual, reference) <= 2.0 * bf16_peak_ulp(reference)
        assert torch.equal(result.outputs[2], expected[2])


def test_logits_independent_runner_freezes_shape_and_oracle():
    inputs = make_logits_inputs(LOGITS_PRIMARY, device=torch.device("npu"))
    expected = ref_indexer_logits(inputs["q"], inputs["weights"], inputs["k"])
    baseline = launch_indexer_logits(inputs, LOGITS_PRIMARY, graph_optimize=True)
    control = launch_indexer_logits(inputs, LOGITS_PRIMARY, graph_optimize=False)
    torch.npu.synchronize()
    for result in (baseline, control):
        assert torch.allclose(result.outputs[0], expected, rtol=1.0e-2, atol=5.0e-2)
        assert torch.isfinite(result.outputs[0]).all()


@pytest.mark.skipif(
    os.environ.get("TASK0005_ENABLE_GRID_E2E") != "1",
    reason="set TASK0005_ENABLE_GRID_E2E=1 to run the original-wrapper grid contract",
)
def test_unchanged_dsl_grid_specialization_default_off_contract():
    report = run_unchanged_dsl_grid_baseline()
    assert report["passed"], report
    assert report["mode"]["effective_rule_mask"] == 511
    assert all(event["actual_launch_grid_observation"] == "generated_launcher_intercept"
               for event in report["events"]
               if event["case"].startswith("observe:"))
