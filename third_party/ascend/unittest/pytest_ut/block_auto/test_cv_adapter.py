"""CPU adapter checks against the original CV ranking and fallback policies."""
from collections import Counter
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[3] / "backend"
PACKAGE = "_block_auto_cv_test_backend"
package = ModuleType(PACKAGE)
package.__path__ = [str(BACKEND)]
sys.modules[PACKAGE] = package
adapter = importlib.import_module(PACKAGE + ".block_auto_cv")


def candidate(factor=1):
    axis = -1 if factor == 1 else 0
    metadata = dict(name="kernel", mix_mode="aic", parallel_mode="simd", auto_blockify_enabled=False)
    if factor > 1:
        metadata.update(
            block_batching=dict(compile_status="COMPILED_GUARDED", requested_factor=factor, selected_axis=axis,
                                tail_policy="full_groups_only", cv_mode="independent_batch_v1"),
            program_grid_transforms=dict(transforms=[dict(kind="ceil_div", factor=factor, axis=axis)]))
    return dict(
        key="unmerged" if factor == 1 else f"a0_f{factor}", factor=factor, axis=axis, metadata=metadata,
        resource_checked=True, ub_usage=dict(address_extent_bits=1024, capacity_bits=8192),
        binary_sha256="binary" + str(factor), compiler_sha256="same compiler",
        ttir="tt.func public @kernel(%arg0: !tt.ptr<f32>) {}", final_ir='"CUBE_CORE_COUNT", 28 : i32\n'
        'func.func @kernel() attributes {hacc.entry, '
        'hivm.func_core_type = #hivm.func_core_type<AIC>} {}', llvm='define void @kernel() #1 {\n  ret void\n}\n'
        'attributes #1 = { "cube-only-kernel" }')


def synthetic_trace(modules, ttir, final_ir, metadata, contract, axis, factor, table, **kwargs):
    # This fake backend tracer supplies exact whole-launch work ledgers. Real
    # feature extraction, ABI adaptation and CV ranking remain under test.
    assert list(contract["arguments"]) == ["arg0"]
    groups = 8 // factor
    records, total = [], Counter()
    for block in range(groups):
        counts = {
            "scalar_steps": 100, "mma_calls": 1, f"mma_shape:(16, {8 * factor}, 16)": 1, "useful_flops": 4096 * factor,
            "padded_flops": 4096 * factor, "l1_to_l0_bytes": 128 * factor
        }
        total.update(counts)
        records.append(dict(block=block, counts=counts, entry_block_visits={"aic": {"entry": 1}}))
    score = 100 if factor == 1 else 120
    return dict(status="STATIC_COST_ESTIMATED", launched_groups=groups, cube_core_count=28, group_traces=records,
                counts=dict(total), auto_blockify=False, logical_grid=[groups, 1, 1], mode="Cube", dma_read_bytes=1024,
                dma_write_bytes=512, score=score, critical_group_cycles=score, global_memory_bound=3,
                group_cycles=[score] * groups)


class CVAdapterTests(unittest.TestCase):

    def contract(self):
        return dict(grid=[8, 1, 1], arguments={"x": dict(kind="tensor", dtype="torch.float32")}, argument_order=["x"])

    def test_published_short_batch_preference_remains_default(self):
        with patch.object(adapter, "analyze_candidate", side_effect=synthetic_trace):
            result = adapter.rank_cv([candidate(), candidate(2)], self.contract(), {}, "model")
        self.assertEqual(result["selected_key"], "a0_f2")
        self.assertEqual(result["reason"], "EMPIRICAL_SINGLE_TILE_SHORT_BATCH_X2")
        self.assertTrue(result["empirical_short_batch"])
        self.assertEqual(result["decision"]["prior_latency_choice"], "unmerged")

    def test_optional_latency_only_policy_is_explicit(self):
        contract = dict(self.contract(), cv_empirical_short_batch=False)
        with patch.object(adapter, "analyze_candidate", side_effect=synthetic_trace):
            result = adapter.rank_cv([candidate(), candidate(2)], contract, {}, "model")
        self.assertEqual(result["selected_key"], "unmerged")
        self.assertTrue(result["model_complete"])

    def test_unknown_third_candidate_keeps_baseline(self):
        missing = dict(key="a0_f4", factor=4, axis=0, resource_checked=False, status="RESOURCE_UNCONFIRMED",
                       error="missing final allocation")
        with patch.object(adapter, "analyze_candidate", side_effect=synthetic_trace):
            result = adapter.rank_cv([candidate(), candidate(2), missing], self.contract(), {}, "model")
        self.assertEqual(result["selected_key"], "unmerged")
        self.assertFalse(result["model_complete"])
        self.assertEqual(result["incomplete_candidates"], ["a0_f4"])

    def test_compile_rejection_is_an_exclusion_not_a_zero_score(self):
        rejected = dict(key="a0_f4", factor=4, axis=0, resource_checked=False, status="COMPILE_REJECTED",
                        error="UB overflow")
        with patch.object(adapter, "analyze_candidate", side_effect=synthetic_trace):
            result = adapter.rank_cv([candidate(), candidate(2), rejected], self.contract(), {}, "model")
        self.assertEqual(result["selected_key"], "a0_f2")
        self.assertNotIn("estimated_cycles", result["candidates"]["a0_f4"])

    def test_compiler_axis_disagreement_keeps_baseline(self):
        merged = candidate(2)
        merged["metadata"]["program_grid_transforms"]["transforms"][0]["axis"] = 1
        with patch.object(adapter, "analyze_candidate", side_effect=synthetic_trace):
            result = adapter.rank_cv([candidate(), merged], self.contract(), {}, "model")
        self.assertEqual(result["selected_key"], "unmerged")
        self.assertFalse(result["model_complete"])

    def test_fp16_does_not_use_fp32_matrix_model(self):
        contract = self.contract()
        contract["arguments"]["x"]["dtype"] = "torch.float16"
        result = adapter.rank_cv([candidate(), candidate(2)], contract, {}, "model")
        self.assertEqual(result["reason"], "CV_DTYPE_NOT_COVERED")
        self.assertEqual(result["selected_key"], "unmerged")

    def test_final_cube_only_evidence_overrides_early_mix_classification(self):
        item = candidate()
        item["metadata"]["mix_mode"] = "mix"
        self.assertEqual(adapter._llvm_modules(item), {"aic": item["llvm"]})
        self.assertEqual(item["metadata"]["mix_mode"], "mix")

    def test_real_vector_function_cannot_be_ignored_when_aiv_llvm_is_missing(self):
        item = candidate()
        item["metadata"]["mix_mode"] = "mix"
        item["final_ir"] += '\nfunc.func @kernel_mix_aiv() attributes {' \
                            'hivm.func_core_type = #hivm.func_core_type<AIV>} {}'
        with self.assertRaisesRegex(ValueError, "does not cover final core roles"):
            adapter._llvm_modules(item)
        item["llvm_aic"] = item.pop("llvm")
        with self.assertRaisesRegex(ValueError, "do not cover final core roles"):
            adapter._llvm_modules(item)

    def test_cube_attribute_on_another_function_does_not_prove_entry_role(self):
        item = candidate()
        item["llvm"] = 'define void @kernel() #0 {\n  ret void\n}\n' \
                       'attributes #0 = { nounwind }\nattributes #1 = { "cube-only-kernel" }'
        with self.assertRaisesRegex(ValueError, "entry is not explicitly Cube"):
            adapter._llvm_modules(item)


class CVCandidatePlanTests(unittest.TestCase):

    def plan(self, grid, roles, axis=-1):

        def probe(backend, frozen, options, selected_axis, factor):
            role = roles[selected_axis]
            return dict(compile_status="REWRITTEN", requested_factor=factor, selected_axis=selected_axis,
                        cv_mode="independent_batch_v1" if role == "batch" else "spatial_concat_v1",
                        cv_lowering="" if role == "batch" else "spatial_" + role.lower() + "_v1")

        baseline = SimpleNamespace(metadata=SimpleNamespace(_asdict=lambda: dict(mix_mode="mix", parallel_mode="simd")))
        with patch.object(adapter, "_probe_candidate", side_effect=probe) as probes:
            result = adapter.candidate_plan(dict(grid=grid), baseline, backend=object(), frozen=object(), options={},
                                            axis=axis)
        return result, probes.call_args_list

    def test_both_spatial_directions_use_f4_without_batch_candidate(self):
        result, probes = self.plan([128, 4, 4], {0: "batch", 1: "M", 2: "N"})
        self.assertEqual([item["key"] for item in result["candidates"]], ["a1_f4", "a2_f4"])
        self.assertEqual(result["front_end_probe_count"], 6)
        self.assertEqual(len(probes), 6)

    def test_one_f4_spatial_direction_uses_two_factors_on_that_axis(self):
        result, _ = self.plan([128, 4, 2], {0: "batch", 1: "M", 2: "N"})
        self.assertEqual([item["key"] for item in result["candidates"]], ["a1_f2", "a1_f4"])
        self.assertEqual(result["front_end_probe_count"], 5)

    def test_single_spatial_tile_uses_batch_f2_f4(self):
        result, _ = self.plan([8, 1, 1], {0: "batch"})
        self.assertEqual([item["key"] for item in result["candidates"]], ["a0_f2", "a0_f4"])
        self.assertEqual(result["front_end_probe_count"], 2)

    def test_axis_roles_come_from_frontend_instead_of_axis_number(self):
        result, _ = self.plan([4, 8, 4], {0: "N", 1: "batch", 2: "M"})
        self.assertEqual([item["key"] for item in result["candidates"]], ["a2_f4", "a0_f4"])

    def test_explicit_axis_is_the_only_probed_axis(self):
        result, probes = self.plan([128, 4, 4], {2: "N"}, axis=2)
        self.assertEqual([item["key"] for item in result["candidates"]], ["a2_f2", "a2_f4"])
        self.assertTrue(all(call.args[-2] == 2 for call in probes))
        self.assertEqual(result["front_end_probe_count"], 2)

    def test_unknown_frontend_role_preserves_baseline(self):
        result, _ = self.plan([8, 1, 1], {0: "unknown"})
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["reason"], "CV_NO_ADMITTED_MERGE")
        self.assertTrue(all(probe["status"] == "FRONTEND_REJECTED" for probe in result["probes"]))

    def test_internal_frontend_failure_is_not_reported_as_admission_rejection(self):
        baseline = SimpleNamespace(metadata=SimpleNamespace(_asdict=lambda: dict(mix_mode="mix", parallel_mode="simd")))
        with patch.object(adapter, "_probe_candidate", side_effect=RuntimeError("MLIR verifier failed")):
            with self.assertRaisesRegex(RuntimeError, "MLIR verifier failed"):
                adapter.candidate_plan(dict(grid=[8, 1, 1]), baseline, backend=object(), frozen=object(), options={})

    def test_nondivisible_grid_does_not_probe_or_compile(self):
        result, probes = self.plan([3, 1, 1], {0: "batch"})
        self.assertEqual(result["candidates"], [])
        self.assertEqual(probes, [])


if __name__ == "__main__":
    unittest.main()
