# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU-only checks; no Triton driver or device is initialized."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[2] / "backend"
PACKAGE = "_block_auto_entry_test"


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class TestHostEntry(unittest.TestCase):

    def setUp(self):
        self.events = []
        self.fn, self.kernel = object(), object()
        self.args, self.kwargs, self.grid = (object(), ), {"BLOCK": 128}, (17, )
        self.backend = SimpleNamespace()
        self.values = (self.args, self.kwargs, self.grid, False, 0, 42)
        package = module(PACKAGE, __path__=[str(BACKEND)])
        config = module(PACKAGE + ".reuse.config", enabled=lambda: False)
        self.modules = {
            PACKAGE: package,
            PACKAGE + ".reuse": module(PACKAGE + ".reuse", __path__=[]),
            PACKAGE + ".reuse.config": config,
        }
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.dispatch = importlib.import_module(PACKAGE + ".block_auto_dispatch")
        self.handler = self.dispatch.get_launch_handler()

    def tail(self, *args, **kwargs):
        self.events.append((args, kwargs))
        return self.kernel

    def run_handler(self, values=None):
        return self.handler.run(self.fn, *(values or self.values), self.backend, self.tail)

    def test_absent_provider_preserves_original_arguments_and_result(self):
        self.assertIs(self.run_handler(), self.kernel)
        self.assertEqual(len(self.events), 1)
        args, kwargs = self.events[0]
        self.assertFalse(kwargs)
        for actual, expected in zip(args, self.values):
            self.assertIs(actual, expected)
        self.assertNotIn(PACKAGE + ".block_auto_model", sys.modules)

    def test_absent_provider_does_not_bind_or_evaluate_grid(self):

        def grid(_):
            raise AssertionError("must remain owned by the normal JIT")

        self.values = (self.args, self.kwargs, grid, False, 0, 42)
        self.run_handler()
        self.assertIs(self.events[0][0][2], grid)

    def test_existing_reuse_path_is_preserved(self):
        self.modules[PACKAGE + ".reuse.config"].enabled = lambda: True
        calls = []

        def reuse(fn, *arguments):
            calls.append((fn, arguments))
            return arguments[-1](*arguments[:-2])

        bridge = module(PACKAGE + ".reuse.bridge", get_handler=lambda: SimpleNamespace(run=reuse))
        with patch.dict(sys.modules, {PACKAGE + ".reuse.bridge": bridge}):
            self.assertIs(self.run_handler(), self.kernel)
        self.assertIs(calls[0][0], self.fn)
        self.assertIs(calls[0][1][-2], self.backend)
        self.assertEqual(len(self.events), 1)

    def test_warmup_does_not_invoke_candidate_provider(self):

        def fail(*args, **kwargs):
            raise AssertionError("warmup cannot create merged candidates")

        self.backend.run_block_auto = fail
        self.assertIs(self.run_handler((*self.values[:3], True, *self.values[4:])), self.kernel)

    def test_provider_receives_cpu_selector_and_single_continuation(self):
        select = object()
        calls = []

        def provider(fn, *arguments, select_candidates, continuation):
            calls.append((fn, select_candidates))
            return continuation(*arguments)

        self.backend.run_block_auto = provider
        with patch.dict(sys.modules, {PACKAGE + ".block_auto_model": module("model", select_candidates=select)}):
            self.assertIs(self.run_handler(), self.kernel)
        self.assertEqual(calls, [(self.fn, select)])
        self.assertEqual(len(self.events), 1)

    def test_callback_failure_is_not_replayed(self):

        def fail(*args, **kwargs):
            raise RuntimeError("launch callback failed")

        with self.assertRaisesRegex(RuntimeError, "launch callback failed"):
            self.handler.run(self.fn, *self.values, self.backend, fail)
        self.assertFalse(self.events)


class TestModelContracts(unittest.TestCase):

    def setUp(self):
        self.patcher = patch.dict(sys.modules, {PACKAGE: module(PACKAGE, __path__=[str(BACKEND)])})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.cv = importlib.import_module(PACKAGE + ".block_auto_cv")
        self.native = importlib.import_module(PACKAGE + ".block_auto_models.native_model")

    def probe(self, axis, factor):
        return dict(compile_status="REWRITTEN", selected_axis=axis, requested_factor=factor,
                    cv_mode="spatial_concat_v1", cv_lowering="spatial_m_v1" if axis == 0 else "spatial_n_v1")

    def test_cv_shortlist_contains_two_proved_axes(self):
        result = self.cv.candidate_plan({"grid": [8, 8, 1]}, probe=self.probe)
        self.assertEqual(result["reason"], "CV_M4_N4")
        self.assertEqual([(r["axis"], r["factor"]) for r in result["candidates"]], [(0, 4), (1, 4)])

    def test_cv_small_grid_never_invents_a_merge(self):
        result = self.cv.candidate_plan({"grid": [1, 1, 1]}, probe=self.probe)
        self.assertEqual(result["candidates"], [])

    def test_cv_rejected_frontend_roles_are_excluded(self):

        def reject(axis, factor):
            raise ValueError("not transformable")

        result = self.cv.candidate_plan({"grid": [8, 8, 1]}, probe=reject)
        self.assertEqual(result["candidates"], [])

    def test_packaged_model_assets_match_their_recorded_hashes(self):
        model = importlib.import_module(PACKAGE + ".block_auto_model")
        self.assertTrue(model._instruction_model())
        import hashlib
        import json
        data = BACKEND / "block_auto_models/vv_library_data"
        proof = json.loads((data / "provenance.json").read_text())
        self.assertEqual(hashlib.sha256((data / "pow_fp32.ll").read_bytes()).hexdigest(), proof["llvm_sha256"])

    def test_missing_candidates_keep_baseline_without_loading_a_driver(self):
        model = importlib.import_module(PACKAGE + ".block_auto_model")
        factor, report = model.select_candidates([], {})
        self.assertEqual(factor, 1)
        self.assertFalse(report["model_complete"])

    def test_unmerged_grid_does_not_need_batching_package(self):
        self.assertFalse(self.native.preserves_grid({}, 1))

    def test_iteration_schedule_requires_matching_loop_proof(self):
        loop = dict(id=0, bits=32, lower=0, upper=16, step=1)
        schedule = dict(version=1, kind="sequential_iterations_v1", grid_policy="preserve", factor=2, loop=loop)
        metadata = dict(
            block_batching=dict(vv_lowering="sequential_iterations_v1", lane_width=2, cv_mode="disabled",
                                guard=dict(loop_schedule=schedule, accesses=[dict(loops=[loop])])),
            program_grid_mapping_applied=False, row_coalescing_applied=False, ptsm_cap_authorized=False)
        self.assertTrue(self.native.preserves_grid(metadata, 2))
        metadata["block_batching"]["guard"]["accesses"] = []
        with self.assertRaisesRegex(ValueError, "VV_LOOP_SCHEDULE_MISMATCH"):
            self.native.preserves_grid(metadata, 2)


if __name__ == "__main__":
    unittest.main()
