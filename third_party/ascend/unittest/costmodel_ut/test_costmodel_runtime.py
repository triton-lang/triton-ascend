import importlib.util
import builtins
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class DummyCacheManager:
    def __init__(self):
        self.storage = {}

    def get_file(self, name):
        return self.storage.get(name)

    def put(self, payload, name, binary=False):
        if binary:
            raise AssertionError("costmodel cache should be text json")
        self.storage[name] = payload


class CostmodelRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Stub triton runtime cache before importing module under test.
        triton_mod = types.ModuleType("triton")
        runtime_mod = types.ModuleType("triton.runtime")
        cache_mod = types.ModuleType("triton.runtime.cache")

        cls.bootstrap_cache = DummyCacheManager()
        cache_mod.get_cache_manager = lambda _ns: cls.bootstrap_cache
        cache_mod.triton_key = lambda: "bootstrap-key"

        runtime_mod.cache = cache_mod
        triton_mod.runtime = runtime_mod

        import sys

        sys.modules.setdefault("triton", triton_mod)
        sys.modules.setdefault("triton.runtime", runtime_mod)
        sys.modules.setdefault("triton.runtime.cache", cache_mod)

        repo_root = Path(__file__).resolve().parents[4]
        module_path = repo_root / "third_party" / "ascend" / "backend" / "runtime" / "costmodel_runtime.py"
        spec = importlib.util.spec_from_file_location("costmodel_runtime_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cls.cm = module

    def setUp(self):
        self.cm._COSTMODEL_MEM_CACHE.clear()

    def test_parse_latency_and_jobs(self):
        self.assertAlmostEqual(self.cm.parse_latency("Estimated Time: 3.25 us"), 3.25)
        self.assertEqual(self.cm.parse_latency("noise"), float("inf"))

        with patch.dict("os.environ", {"TRITON_COSTMODEL_WORKER_NUM": "2"}, clear=False):
            self.assertEqual(self.cm.get_costmodel_jobs(8), 2)

        with patch.dict("os.environ", {"TRITON_COSTMODEL_WORKER_NUM": "bad"}, clear=False), patch.object(
            self.cm.os, "cpu_count", return_value=6
        ):
            self.assertEqual(self.cm.get_costmodel_jobs(3), 3)
            self.assertEqual(self.cm.get_costmodel_jobs(0), 1)

    def test_cache_namespace_variants(self):
        with patch.object(self.cm, "_triton_key", None):
            key_none = self.cm._costmodel_cache_namespace()
            self.assertEqual(len(key_none), 64)

        with patch.object(self.cm, "_triton_key", lambda: "abc"):
            key_ok = self.cm._costmodel_cache_namespace()
            self.assertEqual(len(key_ok), 64)

        def _boom():
            raise RuntimeError("x")

        with patch.object(self.cm, "_triton_key", _boom):
            key_err = self.cm._costmodel_cache_namespace()
            self.assertEqual(len(key_err), 64)

    def test_store_and_load_costmodel_latency(self):
        mgr = DummyCacheManager()
        with patch.object(self.cm, "get_cache_manager", lambda _ns: mgr):
            cache_key = "k1"
            self.assertIsNone(self.cm.load_costmodel_latency(cache_key))

            self.cm.store_costmodel_latency(cache_key, 7.5)
            self.assertAlmostEqual(self.cm.load_costmodel_latency(cache_key), 7.5)

            self.cm._COSTMODEL_MEM_CACHE.clear()
            self.assertAlmostEqual(self.cm.load_costmodel_latency(cache_key), 7.5)

            payload = mgr.storage[f"{cache_key}.json"]
            self.assertAlmostEqual(float(json.loads(payload)["latency"]), 7.5)

            mgr.storage[f"{cache_key}.json"] = json.dumps({"latency": "bad-float"})
            self.cm._COSTMODEL_MEM_CACHE.clear()
            self.assertIsNone(self.cm.load_costmodel_latency(cache_key))

    def test_make_key_and_extra_args(self):
        k1 = self.cm.make_costmodel_cache_key("ttir_a", ["-ascend-perf-model"])
        k2 = self.cm.make_costmodel_cache_key("ttir_a", ["-ascend-perf-model", "arg-bindings=a=1"])
        self.assertNotEqual(k1, k2)

        with patch.object(self.cm, "_resolve_default_hardware_config", lambda: "/tmp/ascend_910b.json"):
            self.assertEqual(
                self.cm._build_costmodel_extra_args("arg1=3", ""),
                ["-ascend-perf-model", "arg-bindings=arg1=3"],
            )
            self.assertEqual(
                self.cm._build_costmodel_extra_args("", ""),
                ["-ascend-perf-model", "hardware-config=/tmp/ascend_910b.json"],
            )
        with patch.object(self.cm, "_resolve_default_hardware_config", lambda: ""):
            self.assertEqual(self.cm._build_costmodel_extra_args("", ""), ["-ascend-perf-model"])

    def test_run_costmodel_reads_file_and_adds_allow_unregistered_dialect(self):
        calls = []

        class AscendCapi:
            @staticmethod
            def run_costmodel_inproc(mlir_text, args):
                calls.append((mlir_text, tuple(args)))
                return "Estimated Time: 1.0 us"

        fake_libtriton = types.SimpleNamespace(ascend=AscendCapi)
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton._C.libtriton":
                return fake_libtriton
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            f.write("module-from-file")
            f.flush()
            with patch("builtins.__import__", fake_import):
                self.assertEqual(self.cm.run_costmodel(f.name, ["-ascend-perf-model"]), "Estimated Time: 1.0 us")

        self.assertEqual(calls[0][0], "module-from-file")
        self.assertIn("-allow-unregistered-dialect", calls[0][1])

    def test_run_costmodel_exception_paths(self):
        class GenericFailingCapi:
            @staticmethod
            def run_costmodel_inproc(_mlir_text, _args):
                raise RuntimeError("failed to parse input MLIR module")

        fake_libtriton = types.SimpleNamespace(ascend=GenericFailingCapi)
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton._C.libtriton":
                return fake_libtriton
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", fake_import):
            self.assertIsNone(self.cm.run_costmodel("module-text", ["-ascend-perf-model"]))

    def test_normalize_items_and_eval_item(self):
        cfg1 = object()
        cfg2 = object()
        cfg3 = object()
        items = [
            {"config": cfg1, "ttir": "ttir1", "arg_bindings": "a=1", "hardware_config": "h1"},
            {"config": cfg2, "ttir": ""},
            {"config": None, "ttir": "ignored"},
            {"config": cfg3, "ttir": "ttir3"},
            123,
        ]

        pending, lat = self.cm._normalize_costmodel_items(items)
        self.assertEqual(len(pending), 2)
        self.assertEqual(lat[cfg2], float("inf"))

        with patch.object(self.cm, "load_costmodel_latency", lambda _k: 1.23):
            cfg, t = self.cm._eval_one_costmodel_item(pending[0])
            self.assertIs(cfg, cfg1)
            self.assertAlmostEqual(t, 1.23)

    def test_eval_item_miss_and_pending_eval(self):
        cfg1, cfg2 = object(), object()
        pending = [
            (cfg1, "ttir1", "arg=1", ""),
            (cfg2, "ttir2", "", ""),
        ]
        lat = {}

        calls = []

        def fake_run(ttir_or_path, extra_args=None, dump_ir_on_error=False):
            calls.append((ttir_or_path, tuple(extra_args or [])))
            if "ttir1" in ttir_or_path:
                return "Estimated Time: 9.9 us"
            return None

        with patch.object(self.cm, "load_costmodel_latency", lambda _k: None), patch.object(
            self.cm, "store_costmodel_latency", lambda *_args, **_kwargs: None
        ), patch.object(self.cm, "run_costmodel", fake_run), patch.dict(
            "os.environ", {"TRITON_COSTMODEL_WORKER_NUM": "1"}, clear=False
        ):
            self.cm._evaluate_pending_items(pending, lat)

        self.assertAlmostEqual(lat[cfg1], 9.9)
        self.assertEqual(lat[cfg2], float("inf"))
        self.assertEqual(len(calls), 2)

    def test_evaluate_pending_empty(self):
        lat = {}
        self.cm._evaluate_pending_items([], lat)
        self.assertEqual(lat, {})

    def test_evaluate_pending_parallel_exception_tolerated(self):
        cfg1, cfg2 = object(), object()
        pending = [
            (cfg1, "ttir1", "", ""),
            (cfg2, "ttir2", "", ""),
        ]
        out = {}

        def fake_eval(item):
            if item[0] is cfg1:
                return cfg1, 0.5
            raise RuntimeError("bad worker")

        with patch.object(self.cm, "_eval_one_costmodel_item", fake_eval), patch.dict(
            "os.environ", {"TRITON_COSTMODEL_WORKER_NUM": "2"}, clear=False
        ):
            self.cm._evaluate_pending_items(pending, out)

        self.assertAlmostEqual(out[cfg1], 0.5)
        self.assertNotIn(cfg2, out)

    def test_costmodel_bench_paths(self):
        self.assertEqual(self.cm.costmodel_bench([]), {})

        class BadIter:
            def __iter__(self):
                raise RuntimeError("bad")

        self.assertEqual(self.cm.costmodel_bench(BadIter()), {})

        cfg1, cfg2 = object(), object()
        items = [{"config": cfg1, "ttir": "t1"}, {"config": cfg2, "ttir": ""}]

        with patch.object(self.cm, "_normalize_costmodel_items", lambda _x: ([(cfg1, "t1", "", "")], {cfg2: float("inf")})):
            def fake_eval(_pending, out):
                out[cfg1] = 0.88

            with patch.object(self.cm, "_evaluate_pending_items", fake_eval):
                result = self.cm.costmodel_bench(items)

        self.assertAlmostEqual(result[cfg1], 0.88)
        self.assertEqual(result[cfg2], float("inf"))

        def explode(*_args, **_kwargs):
            raise RuntimeError("oops")

        with patch.object(self.cm, "_normalize_costmodel_items", explode):
            fallback = self.cm.costmodel_bench(items)
        self.assertEqual(fallback[cfg1], float("inf"))
        self.assertEqual(fallback[cfg2], float("inf"))


if __name__ == "__main__":
    unittest.main()
