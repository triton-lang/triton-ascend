# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Automatic batching host integration, without loading device code."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

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

    def test_default_and_explicit_factors_keep_manual_dispatch(self):
        for factor in (None, 1, 2, 4):
            self.kwargs = dict(BLOCK=512, block_batching_axis=0)
            if factor is not None:
                self.kwargs["block_batching_factor"] = factor
            self.calls.clear()
            self.assertIs(self.run_handler(), self.result)
            self.assertEqual(len(self.calls), 1)
            self.assertIs(self.calls[0][0][1], self.kwargs)

    def test_warmup_and_capture_use_baseline_without_binding_or_grid_evaluation(self):

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
        for target, grid in (("Ascend910B", (256, )), ("Ascend950", (0, ))):
            self.backend.target.arch = target
            self.calls.clear()
            self.run_handler(grid)
            self.assertEqual(self.calls[-1][0][1]["block_batching_factor"], 1)

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
        self.manual = module("batching", _freeze_source=lambda *args: self.frozen, BatchingRejected=ValueError,
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

    def test_model_fallback_cannot_launch_a_rejected_merge(self):
        self.model.select_candidates = lambda candidates, contract: (
            1, dict(selected_key="unmerged", reason="MODEL_INCOMPLETE"))
        result = self.dispatch.compile_auto_bundle(self.backend, object(), self.options, self.compile)
        self.assertIs(result.baseline, self.baseline)
        self.assertIsNone(result.candidate)


if __name__ == "__main__":
    unittest.main()
