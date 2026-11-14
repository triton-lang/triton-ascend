# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Automatic batching host integration, without loading device code."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

BACKEND = Path(__file__).resolve().parents[2] / "backend"
PACKAGE = "_block_auto_private_test"


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class Tensor:
    dtype = "torch.float32"
    shape = (256, 512)

    def stride(self):
        return (512, 1)

    def storage_offset(self):
        return 0


class TestHostIntegration(unittest.TestCase):

    def setUp(self):
        self.calls = []
        self.config = module(PACKAGE + ".reuse.config", enabled=lambda: False)
        self.patcher = patch.dict(
            sys.modules, {
                PACKAGE: module(PACKAGE, __path__=[str(BACKEND)]),
                PACKAGE + ".reuse": module(PACKAGE + ".reuse", __path__=[]),
                PACKAGE + ".reuse.config": self.config,
            })
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batching = importlib.import_module(PACKAGE + ".block_batching")
        self.dispatch = importlib.import_module(PACKAGE + ".block_auto_dispatch")
        capture = patch.object(self.batching, "_capture_state", return_value=False)
        self.capture = capture.start()
        self.addCleanup(capture.stop)
        self.backend = SimpleNamespace(target=SimpleNamespace(arch="Ascend950"))
        self.fn = SimpleNamespace(arg_names=["x", "size", "BLOCK"], device_caches={0: [None] * 4 + [self.bind]})
        self.kwargs = dict(block_batching_factor="auto", BLOCK=512)
        self.result = object()

    def bind(self, *args, **kwargs):
        self.calls.append("bind")
        return dict(x=args[0], size=args[1], BLOCK=kwargs["BLOCK"]), [("*fp32", None), ("i32", None),
                                                                      ("constexpr", 512)], kwargs

    def tail(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    def run_handler(self, grid=(256, ), warmup=False):
        return self.batching.get_launch_handler().run(self.fn, (Tensor(), 131072), self.kwargs, grid, warmup, 0, 42,
                                                      self.backend, self.tail)

    def test_auto_contract_resolves_grid_once_and_excludes_constexpr(self):

        def grid(bound):
            self.calls.append("grid")
            return (bound["size"] // bound["BLOCK"], )

        self.assertIs(self.run_handler(grid), self.result)
        self.assertEqual(self.calls[:2], ["bind", "grid"])
        self.assertEqual(len(self.calls), 3)
        args, kwargs = self.calls[-1]
        self.assertEqual(args[2], (256, ))
        contract = json.loads(args[1]["block_auto_contract"])
        self.assertEqual(contract["argument_order"], ["x", "size"])
        self.assertEqual(contract["grid"], [256, 1, 1])
        self.assertEqual(contract["arguments"]["x"]["shape"], [256, 512])
        self.assertNotIn("block_auto_contract", self.kwargs)

    def test_omitted_factor_matches_explicit_auto(self):
        explicit, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 131072), self.kwargs, (256, ), False, 0,
                                                   self.backend)
        self.kwargs.pop("block_batching_factor")
        self.calls.clear()
        self.run_handler()
        prepared = self.calls[-1][0][1]
        self.assertEqual(prepared, explicit)
        self.assertEqual(list(prepared), list(explicit))
        self.assertEqual(prepared["block_batching_factor"], "auto")
        self.assertNotIn("block_batching_factor", self.kwargs)

    def test_incomplete_tensor_metadata_does_not_break_default_launch(self):
        wrapper = SimpleNamespace(dtype="fp32", shape=(256, 512), stride=lambda: (512, 1))
        self.assertEqual(self.dispatch._argument(wrapper), dict(kind="unsupported"))

    def test_explicit_factors_keep_manual_dispatch(self):
        for factor in (1, 2, 4):
            self.kwargs = dict(BLOCK=512, block_batching_axis=0)
            self.kwargs["block_batching_factor"] = factor
            self.calls.clear()
            self.assertIs(self.run_handler(), self.result)
            self.assertEqual(len(self.calls), 1)
            self.assertIs(self.calls[0][0][1], self.kwargs)

    def test_warmup_and_capture_use_baseline_without_binding_or_grid_evaluation(self):
        self.kwargs.pop("block_batching_factor")

        def unavailable(_):
            self.fail("grid must stay lazy")

        for warmup, capture in ((True, False), (False, True), (False, None)):
            self.calls.clear()
            self.capture.return_value = capture
            self.run_handler(unavailable, warmup)
            self.assertEqual(len(self.calls), 1)
            values, _ = self.calls[0]
            self.assertEqual(values[1]["block_batching_factor"], 1)
            self.assertIs(values[2], unavailable)

    def test_non_a5_and_invalid_grid_keep_baseline(self):
        self.kwargs.pop("block_batching_factor")
        for target, grid in (("Ascend910B", (256, )), ("Ascend950", (0, ))):
            self.backend.target.arch = target
            self.calls.clear()
            self.run_handler(grid)
            self.assertEqual(self.calls[-1][0][1]["block_batching_factor"], 1)

    def test_a3_enters_auto_without_a5_empirical_override(self):
        self.backend.target.arch = "Ascend910_9391"
        self.run_handler()
        options = self.calls[-1][0][1]
        self.assertEqual(options["block_batching_factor"], "auto")
        self.assertFalse(json.loads(options["block_auto_contract"])["cv_empirical_short_batch"])

    def test_single_program_keeps_normal_cache_without_contract(self):
        for grid, axis in (((1, ), -1), ((1, 1, 1), -1), ((8, 1), 1)):
            self.kwargs["block_batching_axis"] = axis
            self.calls.clear()
            self.run_handler(grid)
            options = self.calls[-1][0][1]
            self.assertEqual(options["block_batching_factor"], 1)
            self.assertNotIn("block_auto_contract", options)

    def test_other_frontends_keep_their_original_compilation_path(self):
        self.fn.ASTSource = SimpleNamespace(frontend_kind="tla")
        self.kwargs.pop("block_batching_factor")
        self.run_handler()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0][1]["block_batching_factor"], 1)

    def test_shape_and_scalar_values_are_part_of_jit_option_identity(self):
        first, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 131072), self.kwargs, (256, ), False, 0,
                                                self.backend)
        second, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 65536), self.kwargs, (128, ), False, 0,
                                                 self.backend)
        self.assertNotEqual(first["block_auto_contract"], second["block_auto_contract"])

    def test_reuse_receives_prepared_contract_and_continues_once(self):
        self.config.enabled = lambda: True

        def reuse(fn, args, kwargs, grid, warmup, device, stream, backend, tail):
            self.assertIs(fn, self.fn)
            self.assertIn("block_auto_contract", kwargs)
            return tail(args, kwargs, grid, warmup, device, stream)

        with patch.dict(sys.modules,
                        {PACKAGE + ".reuse.bridge": module("bridge", get_handler=lambda: SimpleNamespace(run=reuse))}):
            self.assertIs(self.run_handler(), self.result)
        self.assertEqual(len([item for item in self.calls if isinstance(item, tuple)]), 1)

    def test_launch_exception_is_not_replayed(self):

        def fail(*args, **kwargs):
            self.calls.append("launch")
            raise RuntimeError("launch error")

        self.tail = fail
        with self.assertRaisesRegex(RuntimeError, "launch error"):
            self.run_handler()
        self.assertEqual(self.calls.count("launch"), 1)


class TestCompileIntegration(unittest.TestCase):

    def setUp(self):
        self.events = []
        self.baseline = SimpleNamespace(metadata=SimpleNamespace(mix_mode="aiv"))
        self.frozen = object()
        self.backend = SimpleNamespace(parse_options=lambda _: SimpleNamespace(block_batching_backend_revision="r"))
        self.options = dict(block_batching_factor="auto", block_auto_contract=json.dumps(dict(grid=[256, 1, 1])))
        self.bundle = SimpleNamespace(candidate=object(), block_batching=dict(selected_axis=0))
        self.manual = module("batching", _freeze_source=lambda *args: self.frozen,
                             _freeze_refusal=lambda normalized: None, _candidate_refusal=lambda *args: None,
                             _core_floor=lambda: 0, _runtime_positions=lambda src: [],
                             compile_bundle=self.manual_compile, BatchRequest=lambda: "F1",
                             BatchKernel=lambda *args: SimpleNamespace(baseline=args[0], candidate=None))
        self.model = module(
            "model", select_candidates=self.select,
            candidate_plan=lambda *args, **kw: dict(candidates=[dict(axis=1, factor=4),
                                                                dict(axis=2, factor=4)]))
        self.patcher = patch.dict(
            sys.modules, {
                PACKAGE: module(PACKAGE, __path__=[str(BACKEND)]),
                PACKAGE + ".block_batching": self.manual,
                PACKAGE + ".block_auto": self.model,
            })
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.dispatch = importlib.import_module(PACKAGE + ".block_auto_dispatch")
        self.worker = importlib.import_module(PACKAGE + ".block_batching_worker")
        environment = patch.object(self.worker, "_baseline_environment", return_value={})
        self.environment = environment.start()
        self.addCleanup(environment.stop)

    def compile(self, options, frozen=None):
        self.events.append((options, frozen))
        self.assertEqual(options["block_batching_factor"], 1)
        self.assertNotIn("block_auto_contract", options)
        return self.baseline

    def manual_compile(self, backend, frozen, options, compile_variant):
        self.assertIs(frozen, self.frozen)
        self.assertIs(compile_variant(dict(options, block_batching_factor=1), frozen), self.baseline)
        self.events.append((options["block_batching_axis"], options["block_batching_factor"]))
        if options["block_batching_factor"] == 2:
            return SimpleNamespace(candidate=None, block_batching=dict(reason="RESOURCE_LIMIT"))
        return self.bundle

    def select(self, candidates, contract):
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[1]["status"], "COMPILE_REJECTED")
        self.assertIs(candidates[2]["kernel"], self.bundle.candidate)
        return 4, dict(selected_key="a0_f4", selected_axis=0, reason="LOWER_ESTIMATED_LATENCY")

    def test_selection_returns_existing_manual_bundle_and_compiles_baseline_once(self):
        result = self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertIs(result, self.bundle)
        self.assertEqual(result.block_auto_selection["selected_factor"], 4)
        self.assertEqual(self.events[1:], [(-1, 2), (-1, 4)])

    def test_without_launch_contract_only_baseline_is_compiled(self):
        self.assertIs(
            self.dispatch.compile_auto_bundle(self.backend, object(), dict(block_batching_factor="auto"), self.compile),
            self.baseline)
        self.assertEqual(len(self.events), 1)

    def test_vv_below_core_floor_skips_candidates_and_model(self):
        self.manual._core_floor = lambda: 512
        self.model.select_candidates = Mock(side_effect=AssertionError("no candidate to score"))
        result = self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertEqual(len(self.events), 1)
        self.assertEqual(result.block_auto_selection["reason"], "GRID_BELOW_CORE_FLOOR")

    def test_all_candidates_declined_does_not_score_the_baseline(self):
        self.manual.compile_bundle = lambda *args: SimpleNamespace(candidate=None, block_batching=
                                                                   {"reason": "NO_PROGRAM_AXIS"})
        self.model.select_candidates = Mock(side_effect=AssertionError("no candidate to score"))
        result = self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertEqual(result.block_auto_selection["reason"], "NO_COMPILED_CANDIDATE")

    def test_cv_uses_native_shortlist_and_preserves_selected_axis(self):
        self.baseline.metadata.mix_mode = "mix"
        bundle1 = SimpleNamespace(candidate=object(), block_batching=dict(selected_axis=1))
        bundle2 = SimpleNamespace(candidate=object(), block_batching=dict(selected_axis=2))
        self.manual.compile_bundle = lambda backend, frozen, options, compile: bundle1 if options["block_batching_axis"
                                                                                                  ] == 1 else bundle2
        self.model.select_candidates = lambda candidates, contract: (4, dict(selected_key="a2_f4", selected_axis=2))
        result = self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertIs(result, bundle2)
        self.assertEqual(len(result.block_auto_selection["candidate_plan"]["candidates"]), 2)

    def test_modeled_freeze_decline_compiles_f1_from_original_source(self):
        src = object()
        self.manual._freeze_source = lambda *args: self.worker.BatchDecline("CUSTOM_CACHE_NOT_SUPPORTED", "freeze")
        self.model.select_candidates = lambda candidates, contract: (1, dict(reason="BASELINE_ONLY"))
        result = self.dispatch.compile_auto_bundle(self.backend, src, self.options, self.compile)
        self.assertEqual(result.block_auto_selection["candidate_plan"],
                         dict(candidates=[], reason="CUSTOM_CACHE_NOT_SUPPORTED"))
        self.assertEqual(len(self.events), 1)
        self.assertIs(self.events[0][1], src)

    def test_cv_with_unproven_baseline_environment_does_not_probe(self):
        self.baseline.metadata.mix_mode = "aic"
        self.environment.return_value = None
        self.model.candidate_plan = Mock(side_effect=AssertionError("unproven environment cannot probe"))
        self.model.select_candidates = lambda candidates, contract: (1, dict(reason="BASELINE_ONLY"))
        result = self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertEqual(result.block_auto_selection["candidate_plan"],
                         dict(candidates=[], reason="BASELINE_ENVIRONMENT_NOT_PROVEN"))
        self.model.candidate_plan.assert_not_called()
        self.assertIs(result.baseline, self.baseline)

    def test_cv_probe_passes_the_verified_snapshot_and_overrides_only_probe_options(self):
        self.baseline.metadata.mix_mode = "mix"
        env = {"SAVED_FLAG": "submitted"}
        self.environment.return_value = env
        probe = Mock(return_value={"selected_axis": 1, "requested_factor": 2})

        def planner(contract, axis, probe):
            self.assertEqual(probe(1, 2), {"selected_axis": 1, "requested_factor": 2})
            return dict(candidates=[], reason="NO_CANDIDATES")

        self.model.candidate_plan = planner
        self.model.select_candidates = lambda candidates, contract: (1, dict(reason="BASELINE_ONLY"))
        with patch.object(self.worker, "selection_probe", probe):
            self.dispatch.compile_auto_bundle(self.backend, object(), dict(self.options, num_stages=3), self.compile)
        probe.assert_called_once_with(
            self.backend, self.frozen,
            dict(num_stages=3, block_batching_factor=2, block_batching_axis=1, block_auto_collect=False,
                 enable_graph_optimize=False, debug=False), env)
        self.assertEqual(len(self.events), 1)

    def test_cv_probe_distinguishes_worker_refusal_from_internal_errors(self):
        args = (self.backend, self.frozen, {}, {}, 0, 2)
        with patch.object(self.worker, "selection_probe",
                          return_value=self.worker.BatchDecline("UNUSED_BATCH_AXIS", "ttir")):
            with self.assertRaisesRegex(ValueError, "UNUSED_BATCH_AXIS"):
                self.dispatch._probe(*args)
        failure = self.worker.BatchCompileError("native failure", kind="CompilerCrash", stage="ttir")
        with patch.object(self.worker, "selection_probe", side_effect=failure):
            with self.assertRaises(self.worker.BatchCompileError) as raised:
                self.dispatch._probe(*args)
            self.assertIs(raised.exception, failure)
        with patch.object(self.worker, "selection_probe", side_effect=ValueError("invalid serialized input")):
            with self.assertRaisesRegex(self.worker.BatchCompileError, "invalid serialized input"):
                self.dispatch._probe(*args)

    def test_model_fallback_cannot_launch_a_rejected_merge(self):
        self.model.select_candidates = lambda candidates, contract: (
            1, dict(selected_key="unmerged", reason="MODEL_INCOMPLETE"))
        result = self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertIs(result.baseline, self.baseline)
        self.assertIsNone(result.candidate)

    def test_candidate_internal_failure_is_not_silently_downgraded(self):
        worker = importlib.import_module(PACKAGE + ".block_batching_worker")

        def fail(*args):
            raise worker.BatchCompileError("broken protocol", kind="ProtocolError")

        self.manual.compile_bundle = fail
        with self.assertRaises(worker.BatchCompileError):
            self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertEqual(len(self.events), 1)


if __name__ == "__main__":
    unittest.main()
