# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU-only checks; no Triton driver or device is initialized."""
import importlib
from pathlib import Path
import sys
import tempfile
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
        self.assertNotIn(PACKAGE + ".block_auto", sys.modules)

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
        with patch.dict(sys.modules, {PACKAGE + ".block_auto": module("model", select_candidates=select)}):
            self.assertIs(self.run_handler(), self.kernel)
        self.assertEqual(calls, [(self.fn, select)])
        self.assertEqual(len(self.events), 1)

    def test_callback_failure_is_not_replayed(self):

        def fail(*args, **kwargs):
            raise RuntimeError("launch callback failed")

        with self.assertRaisesRegex(RuntimeError, "launch callback failed"):
            self.handler.run(self.fn, *self.values, self.backend, fail)
        self.assertFalse(self.events)


class TestModelBridge(unittest.TestCase):

    def setUp(self):
        patcher = patch.dict(sys.modules, {PACKAGE: module(PACKAGE, __path__=[str(BACKEND)])})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bridge = importlib.import_module(PACKAGE + ".block_auto")

    def test_probe_failure_reaches_native_planner_without_inventing_a_candidate(self):
        requested = [dict(axis=0, factor=2), dict(axis=1, factor=4)]
        reports = []

        def probe(axis, factor):
            reports.append((axis, factor))
            if axis == 1:
                raise ValueError("unsupported rewrite")
            return dict(selected_axis=axis, requested_factor=factor)

        result = dict(candidates=[requested[0]], reason="native decision")
        with patch.object(self.bridge, "_native", side_effect=[dict(probes=requested), result]) as native:
            self.assertEqual(self.bridge.candidate_plan(dict(grid=[8, 8, 1]), probe=probe), result)
        self.assertEqual(reports, [(0, 2), (1, 4)])
        self.assertEqual(len(native.call_args.kwargs["reports"]), 1)

    def test_native_json_keeps_argument_order(self):
        import json
        binding = SimpleNamespace(select_block_candidates=lambda request: request)
        with patch.dict(sys.modules, {"triton._C.libtriton": module("libtriton", ascend=binding)}):
            result = self.bridge._native("prepare", dict(arguments={"z": {}, "a": {}}), candidates=[])
        self.assertEqual(result["contract"]["argument_order"], ["z", "a"])
        self.assertEqual(json.dumps(result["candidates"]), "[]")


class LocalCache:

    def __init__(self, directory):
        self.cache_dir = str(directory)
        directory.mkdir(parents=True, exist_ok=True)

    def put(self, data, name, binary=False):
        path = Path(self.cache_dir) / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return str(path)

    def get_file(self, name):
        path = Path(self.cache_dir) / name
        return str(path) if path.is_file() else None


class TestModelCache(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compiler = self.root / "bisheng"
        self.compiler.touch()
        self.compiler.with_name("llvm-link").touch()
        self.keys = []
        self.cache_module = module("cache", get_cache_manager=self.cache)
        modules = {
            PACKAGE: module(PACKAGE, __path__=[str(BACKEND)]), PACKAGE + ".utils":
            module("utils", _get_bisheng_path=lambda: str(self.compiler)), "triton.runtime.cache": self.cache_module
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.artifacts = importlib.import_module(PACKAGE + ".block_auto_artifacts")
        self.optimizer = importlib.import_module(PACKAGE + ".block_auto_models.vv_compilation")

    def cache(self, key):
        self.keys.append(key)
        return LocalCache(self.root / "cache" / key)

    def collect(self, key, text):
        source = self.root / "temps"
        source.mkdir(exist_ok=True)
        (source / "module.hivm.opt.mlir").write_text("module {}")
        (source / "kernel.ll").write_text(text)
        metadata = {"hash": key}
        evidence = self.artifacts.collect(source, b"\x7fELF", self.compiler, metadata, [], cache_factory=self.cache)
        self.assertEqual(evidence["status"], "COLLECTED")
        metadata["block_auto_evidence"] = evidence
        return SimpleNamespace(metadata=SimpleNamespace(_asdict=lambda: metadata),
                               asm=dict(ttir="ttir", ttadapter="ir"))

    def test_artifacts_use_candidate_cache_and_need_no_file_records(self):
        first = self.collect("first", "first module")
        second = self.collect("second", "second module")
        self.compiler.write_text("new compiler installation")
        for kernel, expected in ((first, "first module"), (second, "second module")):
            result = self.artifacts.load_evidence(kernel, cache_factory=self.cache)
            self.assertEqual(result["llvm"], expected)
            self.assertEqual(result["files"]["compiler"], dict(path=str(self.compiler)))
        self.assertEqual(self.keys, ["first", "second", "first", "second"])

    def test_missing_cached_input_remains_unavailable(self):
        kernel = self.collect("missing", "module")
        (self.root / "cache/missing/block_auto_llvm.ll").unlink()
        with self.assertRaisesRegex(ValueError, "cached model input unavailable"):
            self.artifacts.load_evidence(kernel, cache_factory=self.cache)

    def test_optimizer_publishes_once_under_the_normal_candidate_key(self):
        item = dict(metadata={"hash": "candidate"},
                    files={"compiler": {"path": str(self.compiler)}, "llvm": {"path":
                                                                              "candidate.ll"}}, meta_library='meta.bc')

        def compile_cpu(command, **kwargs):
            output = Path(command[command.index("-o") + 1])
            output.write_text("optimized module")
            self.assertEqual(kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "")

        with patch.object(self.optimizer.subprocess, "run", side_effect=compile_cpu) as run:
            self.assertEqual(self.optimizer.optimized_llvm(item), "optimized module")
            self.assertEqual(self.optimizer.optimized_llvm(item), "optimized module")
        self.assertEqual(run.call_count, 2)
        self.assertIn("-emit-llvm", run.call_args_list[0].args[0])
        self.assertEqual(self.keys, ["candidate", "candidate"])

    def test_optimizer_failure_does_not_publish_a_cache_entry(self):
        item = dict(metadata={"hash": "failed"},
                    files={"compiler": {"path": str(self.compiler)}, "llvm": {"path":
                                                                              "candidate.ll"}}, meta_library='meta.bc')
        with patch.object(self.optimizer.subprocess, "run", side_effect=OSError("optimizer failed")):
            with self.assertRaisesRegex(OSError, "optimizer failed"):
                self.optimizer.optimized_llvm(item)
        self.assertFalse((self.root / "cache/failed/block_auto_optimized.ll").exists())


if __name__ == "__main__":
    unittest.main()
