# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU-only checks; no Triton driver or device is initialized."""
import importlib
from pathlib import Path
import subprocess
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

    def test_selection_deadline_falls_back_but_protocol_errors_propagate(self):
        worker = importlib.import_module(PACKAGE + ".block_batching_worker")
        artifacts = importlib.import_module(PACKAGE + ".block_auto_artifacts")
        for kind in ("Timeout", "ProtocolError"):
            with patch.object(artifacts, "load_resources", return_value={}), \
                 patch.object(self.bridge, "_native", return_value=dict(candidates=[dict(needs_llvm=False)])), \
                 patch.object(worker, "_attempt", side_effect=worker.BatchCompileError("test", kind=kind)) as attempt:
                candidates = [dict(factor=1, axis=-1, kernel=object())]
                if kind == "Timeout":
                    factor, report = self.bridge.select_candidates(candidates, {})
                    self.assertEqual(factor, 1)
                    self.assertIn("budget", report["error"])
                else:
                    with self.assertRaises(worker.BatchCompileError):
                        self.bridge.select_candidates(candidates, {})
                self.assertEqual(attempt.call_args.kwargs["timeout"], 30)

    def test_incomplete_resource_response_cannot_drop_a_candidate(self):
        artifacts = importlib.import_module(PACKAGE + ".block_auto_artifacts")
        with patch.object(artifacts, "load_resources", return_value={}), \
             patch.object(self.bridge, "_native", return_value=dict(candidates=[])), \
             patch.object(artifacts, "load_analysis") as analysis:
            factor, report = self.bridge.select_candidates([dict(factor=1, axis=-1)], {})
        self.assertEqual(factor, 1)
        self.assertFalse(report["model_complete"])
        self.assertIn("one state per candidate", report["error"])
        analysis.assert_not_called()


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

    def cache(self, key):
        self.keys.append(key)
        return LocalCache(self.root / "cache" / key)

    def collect(self, key, text):
        source = self.root / "temps"
        source.mkdir(exist_ok=True)
        (source / "module.hivm.opt.mlir").write_text("module {}")
        (source / "kernel.ll").write_text(text)
        metadata = {"hash": key}
        evidence = self.artifacts.collect(source, self.compiler, metadata, cache_factory=self.cache)
        self.assertEqual(evidence["status"], "COLLECTED")
        metadata["block_auto_evidence"] = evidence
        return SimpleNamespace(metadata=SimpleNamespace(_asdict=lambda: metadata),
                               asm=dict(ttir="ttir", ttadapter="ir"))

    def test_artifacts_use_candidate_cache_and_need_no_file_records(self):
        first = self.collect("first", "first module")
        second = self.collect("second", "second module")
        self.compiler.write_text("new compiler installation")
        for kernel, expected in ((first, "first module"), (second, "second module")):
            resources = self.artifacts.load_resources(dict(factor=2, axis=0, kernel=kernel))
            self.assertNotIn("llvm", resources)
            self.assertEqual(resources["files"]["compiler"], dict(path=str(self.compiler)))
            result = self.artifacts.load_analysis(resources, dict(vv=False, needs_llvm=False))
            self.assertEqual(result["llvm"], expected)
            self.assertFalse({"metadata", "files", "final_ir", "meta_library"}.intersection(result))
            self.assertNotIn("llvm", resources)
        self.assertEqual(self.keys, ["first", "second", "first", "second"])

    def test_missing_cached_input_remains_unavailable(self):
        kernel = self.collect("missing", "module")
        (self.root / "cache/missing/block_auto_llvm.ll").unlink()
        result = self.artifacts.load_resources(dict(factor=4, axis=1, kernel=kernel))
        self.assertEqual(result["status"], "RESOURCE_UNCONFIRMED")
        self.assertIn("cached model input unavailable", result["error"])
        self.assertNotIn("metadata", result)

    def test_optimizer_publishes_once_under_the_normal_candidate_key(self):
        item = dict(metadata={"hash": "candidate"},
                    files={"compiler": {"path": str(self.compiler)}, "llvm": {"path":
                                                                              "candidate.ll"}}, meta_library='meta.bc')

        def compile_cpu(command, **kwargs):
            output = Path(command[command.index("-o") + 1])
            output.write_text("optimized module")
            self.assertEqual(kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "")

        state = dict(vv=True, needs_llvm=True)
        with patch.object(self.artifacts.subprocess, "run", side_effect=compile_cpu) as run:
            for _ in range(2):
                result = self.artifacts.load_analysis(item, state)
                self.assertEqual(result["optimized_llvm"], "optimized module")
                self.assertNotIn("llvm", result)
        self.assertEqual(run.call_count, 2)
        self.assertIn("-emit-llvm", run.call_args_list[0].args[0])
        self.assertEqual(self.keys, ["candidate", "candidate"])

    def test_optimizer_failure_does_not_publish_a_cache_entry(self):
        item = dict(metadata={"hash": "failed"},
                    files={"compiler": {"path": str(self.compiler)}, "llvm": {"path":
                                                                              "candidate.ll"}}, meta_library='meta.bc')
        for error in (OSError("optimizer failed"), subprocess.TimeoutExpired("bisheng", 30)):
            with self.subTest(error=type(error).__name__), \
                 patch.object(self.artifacts.subprocess, "run", side_effect=error):
                result = self.artifacts.load_analysis(item, dict(vv=True, needs_llvm=True))
            self.assertEqual(result["status"], "STATIC_ANALYSIS_UNSUPPORTED")
            self.assertNotIn("optimized_llvm", result)
            self.assertFalse((self.root / "cache/failed/block_auto_optimized.ll").exists())

    def test_cv_and_a3_inputs_do_not_invoke_the_vv_optimizer(self):
        files = {}
        for role in ("llvm_aic", "llvm_aiv"):
            path = self.root / (role + ".ll")
            path.write_text(role + " body")
            files[role] = dict(path=str(path))
        with patch.object(self.artifacts.subprocess, "run") as run:
            cv = self.artifacts.load_analysis(dict(files=files), dict(vv=False, needs_llvm=False))
            a3 = self.artifacts.load_analysis(dict(files={}), dict(vv=True, needs_llvm=False, resource_policy=True))
            rejected = dict(factor=4, axis=1, status="COMPILE_REJECTED", error="unsupported rewrite")
            resources = self.artifacts.load_resources(rejected)
            failed = self.artifacts.load_analysis(resources, dict(needs_llvm=False))
        run.assert_not_called()
        self.assertEqual((cv["llvm_aic"], cv["llvm_aiv"]), ("llvm_aic body", "llvm_aiv body"))
        self.assertEqual(a3, dict(vv=True, needs_llvm=False, resource_policy=True))
        self.assertEqual(failed, dict(rejected, needs_llvm=False))

    def test_resource_loading_does_not_read_llvm_and_late_loss_rejects_analysis(self):
        kernel = self.collect("late-loss", "module")
        read_text = Path.read_text
        reads = []

        def read(path, *args, **kwargs):
            reads.append(path.suffix)
            return read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            resources = self.artifacts.load_resources(dict(factor=1, axis=-1, kernel=kernel))
        self.assertEqual(reads, [".mlir"])
        (self.root / "cache/late-loss/block_auto_llvm.ll").unlink()
        result = self.artifacts.load_analysis(resources, dict(vv=False, needs_llvm=False))
        self.assertEqual(result["status"], "STATIC_ANALYSIS_UNSUPPORTED")
        self.assertNotIn("llvm", result)

    def test_a3_requires_final_ir_but_not_unavailable_llvm(self):
        source = self.root / "a3-temps"
        source.mkdir()
        metadata = dict(hash="a3", target=dict(arch="Ascend910_9391"))
        collect = lambda: self.artifacts.collect(source, self.compiler, metadata, cache_factory=self.cache)
        self.assertEqual(collect()["status"], "UNAVAILABLE")
        (source / "module.hivm.opt.mlir").write_text("module {}")
        evidence = collect()
        self.assertEqual(evidence["status"], "COLLECTED")
        self.assertEqual(set(evidence["files"]), {"final_ir"})


if __name__ == "__main__":
    unittest.main()
