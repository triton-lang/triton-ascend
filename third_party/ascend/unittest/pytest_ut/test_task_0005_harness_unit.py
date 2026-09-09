"""Pure-Python contracts for the standalone task_0005 test harness."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_0005_harness.contracts import (  # noqa: E402
    COMPILE_MODE, LOGITS_PRIMARY, MERGE_PRIMARY_CASES, MINIMAL_NEGATIVE_VARIANTS, NON_ORACLE_AFTER_DIFFERENCES,
    NORM_PRIMARY_TOKENS, SMOKE_SUITE, TARGET_STRUCTURES, artifact_schema, assert_fixture_integrity, expected_grid,
    expected_program_count, fixture_path,
)
from task_0005_harness.reference import (  # noqa: E402
    ref_indexer_logits, ref_indexer_norm_rope, ref_merge_split_states,
)
from task_0005_harness.run_baselines import (  # noqa: E402
    _target_samples, ingest_profile,
)
from task_0005_harness.runtime import (  # noqa: E402
    _compile_kwargs, collect_cache_records, launch_merge_split, make_merge_inputs, validate_primary_metadata,
)
from task_0102_static_program_axis_fusion_runner import _cache_evidence  # noqa: E402

pytestmark = pytest.mark.backend("none")


def test_frozen_dsl_fixtures_have_the_recorded_hashes():
    assert_fixture_integrity()


def test_primary_program_contracts_match_the_before_grids():
    merge = MERGE_PRIMARY_CASES[0]
    assert expected_grid("merge_split", merge) == (8, 64)
    assert expected_program_count("merge_split", merge) == 512

    norm = __import__("task_0005_harness.contracts", fromlist=["NormRopeCase"]).NormRopeCase(tokens=16)
    assert expected_grid("norm_rope", norm, specialization="q") == (16, 16)
    assert expected_grid("norm_rope", norm, specialization="k") == (16, 1)
    assert expected_program_count("norm_rope", norm, specialization="q") == 256
    assert expected_program_count("norm_rope", norm, specialization="k") == 16

    assert expected_grid("indexer_logits", LOGITS_PRIMARY) == (1, 1, 4)
    assert expected_program_count("indexer_logits", LOGITS_PRIMARY) == 4


def test_primary_contract_is_template_simd_not_explicit_simt_only():
    assert COMPILE_MODE == "simd_simt_template"
    assert "simt_only" not in COMPILE_MODE


def test_logits_runner_compile_options_keep_legacy_default_and_allow_spaf():
    assert _compile_kwargs(True) == {
        "compile_mode": COMPILE_MODE,
        "enable_graph_optimize": True,
    }
    assert _compile_kwargs(True, program_mapping_rule_mask=1024) == {
        "compile_mode": COMPILE_MODE,
        "enable_graph_optimize": True,
        "program_mapping_rule_mask": 1024,
    }


def test_merge_launcher_forwards_an_explicit_program_mapping_mask(monkeypatch):
    captured: dict[str, object] = {}

    class FakeKernel:

        def __getitem__(self, _grid):

            def launch(*_args, **kwargs):
                captured.update(kwargs)

            return launch

    class FakeModule:
        _merge_split_states_kernel = FakeKernel()

    monkeypatch.setattr("task_0005_harness.runtime.load_fixture", lambda _operator: FakeModule())
    case = MERGE_PRIMARY_CASES[0]
    launch_merge_split(
        make_merge_inputs(case, device=torch.device("cpu")),
        case,
        graph_optimize=True,
        program_mapping_rule_mask=512,
    )
    assert captured["program_mapping_rule_mask"] == 512


def test_spaf_cache_evidence_requires_a_real_transform_contract(tmp_path):
    kernel = "_indexer_logits_kernel"
    legacy = tmp_path / "legacy"
    fused = tmp_path / "fused"
    legacy.mkdir()
    fused.mkdir()
    (legacy / f"{kernel}.json").write_text(
        json.dumps({
            "program_mapping_rule_mask": None,
            "program_grid_transforms": None,
        }))
    (fused / f"{kernel}.json").write_text(
        json.dumps({
            "program_mapping_rule_mask": 1024,
            "program_grid_transforms": {
                "version": 1,
                "transforms": [{"axis": 2, "factor": 4}],
            },
        }))

    records = _cache_evidence(tmp_path, kernel)
    by_parent = {Path(record["manifest"]).parent.name: record for record in records}
    assert by_parent["legacy"]["program_grid_transform_seen"] is False
    assert by_parent["legacy"]["spaf_rule_mask_seen"] is False
    assert by_parent["fused"]["program_grid_transform_seen"] is True
    assert by_parent["fused"]["spaf_rule_mask_seen"] is True


def test_after_variants_are_explicitly_non_oracles():
    assert set(TARGET_STRUCTURES) == {
        "merge_split",
        "norm_rope",
        "indexer_logits",
    }
    assert set(NON_ORACLE_AFTER_DIFFERENCES) == set(TARGET_STRUCTURES)
    assert set(MINIMAL_NEGATIVE_VARIANTS) == set(TARGET_STRUCTURES)
    assert "BLOCK_D=head_dim" in NON_ORACLE_AFTER_DIFFERENCES["merge_split"]
    assert "overwrites" in NON_ORACLE_AFTER_DIFFERENCES["norm_rope"]
    assert "num_stages=2" in NON_ORACLE_AFTER_DIFFERENCES["indexer_logits"]


def test_fixture_sources_do_not_depend_on_a_task_book_absolute_import():
    for operator in TARGET_STRUCTURES:
        for variant in ("before", "after"):
            source = fixture_path(operator, variant).read_text()
            assert "/home/w00609825/triton_workspace/.tasks/" not in source
            assert "from operator_parity" not in source


def test_reference_merge_matches_a_hand_computed_weighted_mean():
    partial_out = torch.tensor([[[[2.0, 4.0]]], [[[6.0, 8.0]]]])
    partial_lse = torch.tensor([[[0.0]], [[0.0]]])
    output, lse = ref_merge_split_states(partial_out, partial_lse)
    assert torch.equal(output, torch.tensor([[[4.0, 6.0]]]))
    assert torch.equal(lse, torch.tensor([[torch.log(torch.tensor(2.0))]]))


def test_reference_norm_rope_keeps_z_bitwise_and_rejects_bad_span():
    q = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)
    k = torch.arange(8, 16, dtype=torch.bfloat16).reshape(1, 8)
    z = torch.tensor([[1.0, -0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.bfloat16)
    weight = torch.ones(4, dtype=torch.bfloat16)
    bias = torch.zeros(4, dtype=torch.bfloat16)
    cos = torch.ones((2, 1), dtype=torch.bfloat16)
    sin = torch.zeros((2, 1), dtype=torch.bfloat16)
    positions = torch.zeros(1, dtype=torch.int32)
    _, _, z_out = ref_indexer_norm_rope(
        q,
        k,
        z,
        weight,
        weight,
        bias,
        cos,
        sin,
        positions,
        head_dim=4,
        num_q_heads=2,
        num_k_heads=2,
        rotary_dim=1,
    )
    assert torch.equal(z_out, z)
    with pytest.raises(ValueError, match="rotary span"):
        ref_indexer_norm_rope(
            q,
            k,
            z,
            weight,
            weight,
            bias,
            cos,
            sin,
            positions,
            head_dim=4,
            num_q_heads=2,
            num_k_heads=2,
            rotary_dim=3,
        )


def test_reference_logits_has_group_major_shape():
    q = torch.ones((2, 2, 1, 4), dtype=torch.bfloat16)
    weights = torch.ones((2, 2, 1), dtype=torch.bfloat16)
    k = torch.ones((3, 1, 4), dtype=torch.bfloat16)
    output = ref_indexer_logits(q, weights, k)
    assert output.shape == (4, 3)
    assert torch.all(output >= 0)


def test_profiler_parser_requires_an_exact_target_name(tmp_path):
    profiler = tmp_path / "profile"
    profiler.mkdir()
    path = profiler / "op_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Op Name", "Task Duration(us)"))
        writer.writeheader()
        writer.writerow({"Op Name": "not_target", "Task Duration(us)": "1.0"})
        writer.writerow({"Op Name": "_merge_split_states_kernel", "Task Duration(us)": "2.0"})
        writer.writerow({"Op Name": "_merge_split_states_kernel", "Task Duration(us)": "3.0"})
    samples = _target_samples(profiler, "_merge_split_states_kernel")
    assert [sample[2] for sample in samples] == [2.0, 3.0]


def test_cache_artifact_discovery_supports_flat_ascend_manifests(tmp_path):
    entry = tmp_path / "flat-entry"
    entry.mkdir()
    kernel = "_merge_split_states_kernel"
    (entry / f"{kernel}.json").write_text(
        json.dumps({
            "target": {"backend": "npu", "arch": "Ascend950PR_9579"},
            "compile_mode": COMPILE_MODE,
            "parallel_mode": "simd",
            "mix_mode": "aiv",
            "is_pure_simt": False,
        }))
    (entry / f"{kernel}.ttadapter").write_text('module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {}\n'
                                               'func.func @kernel() attributes {parallel_mode = "simd"}\n')
    (entry / f"{kernel}.ttir").write_text("module {}\n")

    records = collect_cache_records(tmp_path, (kernel, ))
    assert len(records) == 1
    assert records[0]["target_arch"] == "Ascend950PR_9579"
    assert records[0]["parallel_mode"] == "simd"
    assert records[0]["compile_mode"] == COMPILE_MODE
    assert f"{kernel}.ttadapter" in records[0]["files"]
    assert validate_primary_metadata(records) == []


def test_profile_ingestion_keeps_both_norm_kernel_invocations(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "runs" / "norm-profile"
    run_dir.mkdir(parents=True)
    run_record = run_dir / "run_record.json"
    run_record.write_text(
        json.dumps({
            "run_label":
            "norm-profile",
            "variant":
            "baseline",
            "cases": [{
                "operator": "norm_rope",
                "case": "norm-t16-q16-k1-d256",
                "kernel_names": ["_indexer_norm_rope_kernel"],
                "kernel_invocations_per_launch": {"_indexer_norm_rope_kernel": 2},
                "warmup_samples": 5,
                "active_samples": 20,
                "cache_records": [{
                    "kernel_name": "_indexer_norm_rope_kernel",
                    "parallel_mode": "simd",
                }],
            }],
        }))
    profiler = tmp_path / "profiler"
    profiler.mkdir()
    with (profiler / "op_summary_fixture.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Op Name", "Task Duration(us)"))
        writer.writeheader()
        for value in range(40):
            writer.writerow({
                "Op Name": "_indexer_norm_rope_kernel",
                "Task Duration(us)": str(value + 1),
            })

    assert (ingest_profile(
        SimpleNamespace(
            artifact_root=artifact_root,
            run_record=run_record,
            profiler_root=profiler,
            round=1,
        )) == 0)
    with (artifact_root / "performance.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert {row["active_samples"] for row in rows} == {"40"}


def test_artifact_schema_and_smoke_suite_cover_required_outputs():
    schema = artifact_schema()
    assert set(schema["required_files"]) == {
        "environment.md",
        "correctness.json",
        "program_count.json",
        "performance.csv",
    }
    assert "kernel_name" in schema["performance_columns"]
    assert len(SMOKE_SUITE) == 3
    assert "test_task_0005_grid_specialization.py" in SMOKE_SUITE[0]
    assert NORM_PRIMARY_TOKENS == (1, 16, 512, 4096, 8192, 32768)
