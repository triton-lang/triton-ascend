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

        with patch.dict("os.environ", {"TRITON_COSTMODEL_WORKER_NUM": "bad"},
                        clear=False), patch.object(self.cm.os, "cpu_count", return_value=6):
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

        with patch.object(self.cm, "_resolve_hardware_config", return_value="/tmp/ascend_910b.json"):
            self.assertEqual(
                self.cm._build_costmodel_extra_args("arg1=3", ""),
                [
                    "-ascend-perf-model",
                    "arg-bindings=arg1=3",
                    "hardware-config=/tmp/ascend_910b.json",
                ],
            )
            self.assertEqual(
                self.cm._build_costmodel_extra_args("", ""),
                ["-ascend-perf-model", "hardware-config=/tmp/ascend_910b.json"],
            )

    def test_soc_version_to_hardware_profile_mapping(self):
        for arch in ("Ascend910B1", "Ascend910B4", "Ascend910_9362", "Ascend910_9392"):
            self.assertEqual(self.cm._costmodel_config_filename_for_arch(arch), "ascend_910b.json")
        for arch in (
                "Ascend910_9579",
                "Ascend910_9589",
                "Ascend910_958B",
                "Ascend950",
                "Ascend950PR_9579",
                "davidV100",
        ):
            self.assertEqual(self.cm._costmodel_config_filename_for_arch(arch), "ascend_davidv100.json")
        with self.assertRaises(ValueError):
            self.cm._costmodel_config_filename_for_arch("Ascend310B4")

    def test_resolve_hardware_config_and_cache_key_use_file_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "ascend_910b.json"
            config.write_text('{"name": "test", "revision": 1}', encoding="utf-8")
            with patch.object(self.cm, "_costmodel_config_candidates", return_value=[str(config)]):
                resolved = self.cm._resolve_hardware_config(target_arch="Ascend910_9362")
            self.assertEqual(resolved, str(config.resolve()))

            key1 = self.cm.make_costmodel_cache_key("ttir", arg_bindings="pid_x=0,arg1=3", hardware_config=resolved,
                                                    target_arch="Ascend910_9362")
            key1_reordered = self.cm.make_costmodel_cache_key("ttir", arg_bindings="arg1=3,pidx=0",
                                                              hardware_config=resolved, target_arch="Ascend910_9362")
            self.assertEqual(key1, key1_reordered)

            config.write_text('{"name": "test", "revision": 2}', encoding="utf-8")
            key2 = self.cm.make_costmodel_cache_key("ttir", arg_bindings="arg1=3,pid_x=0", hardware_config=resolved,
                                                    target_arch="Ascend910_9362")
            self.assertNotEqual(key1, key2)

    def test_canonicalize_arg_bindings(self):
        self.assertEqual(
            self.cm._canonicalize_arg_bindings("program_id_x=0, 2=003, num_programsx=4, arg2=5"),
            "arg2=5,num_programs_x=4,pid_x=0",
        )
        self.assertEqual(self.cm._canonicalize_arg_bindings({"pidy": 0, "arg1": True}), "arg1=1,pid_y=0")
        with self.assertRaises(ValueError):
            self.cm._canonicalize_arg_bindings("arg1")

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

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as config:
            config.write("{}")
            config.flush()
            items[0]["hardware_config"] = config.name
            items[3]["hardware_config"] = config.name
            pending, lat = self.cm._normalize_costmodel_items(items)
            self.assertEqual(len(pending), 2)
            self.assertEqual(lat[cfg2], float("inf"))

            with patch.object(self.cm, "load_costmodel_latency", lambda _k: 1.23):
                cfg, t = self.cm._eval_one_costmodel_item(pending[0])
                self.assertIs(cfg, cfg1)
                self.assertAlmostEqual(t, 1.23)

    def test_eval_item_miss_and_pending_eval(self):
        cfg1, cfg2 = object(), object()
        lat = {}

        calls = []

        def fake_run(ttir_or_path, extra_args=None, dump_ir_on_error=False):
            calls.append((ttir_or_path, tuple(extra_args or [])))
            if "ttir1" in ttir_or_path:
                return "Estimated Time: 9.9 us"
            return None

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as config:
            config.write("{}")
            config.flush()
            pending = [
                (cfg1, "ttir1", "arg=1", config.name, "Ascend910_9362"),
                (cfg2, "ttir2", "", config.name, "Ascend910_9362"),
            ]
            with patch.object(self.cm, "load_costmodel_latency", lambda _k: None), patch.object(
                    self.cm, "store_costmodel_latency",
                    lambda *_args, **_kwargs: None), patch.object(self.cm, "run_costmodel", fake_run), patch.dict(
                        "os.environ", {"TRITON_COSTMODEL_WORKER_NUM": "1"}, clear=False):
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
            (cfg1, "ttir1", "", "/tmp/hw.json", "Ascend910_9362"),
            (cfg2, "ttir2", "", "/tmp/hw.json", "Ascend910_9362"),
        ]
        out = {}

        def fake_eval(item):
            if item[0] is cfg1:
                return cfg1, 0.5
            raise RuntimeError("bad worker")

        with patch.object(self.cm, "_eval_one_costmodel_item",
                          fake_eval), patch.dict("os.environ", {"TRITON_COSTMODEL_WORKER_NUM": "2"}, clear=False):
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

        with patch.object(self.cm, "_normalize_costmodel_items", lambda _x:
                          ([(cfg1, "t1", "", "/tmp/hw.json", "Ascend910_9362")], {cfg2: float("inf")})):

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

    def test_build_arg_bindings_uses_visible_scalars_and_grid(self):
        ttir = """
        tt.func public @kernel(%arg0: !tt.ptr<f16>, %arg1: i32) attributes {noinline = false} {
          %pid = tt.get_program_id x : i32
          %n = tt.get_num_programs x : i32
          tt.return
        }
        """
        bindings = self.cm._build_costmodel_arg_bindings(
            ttir,
            {"ptr": object(), "n": 257, "BLOCK": 128},
            {"ptr": "*fp16", "n": "i32", "BLOCK": "constexpr"},
            (3, ),
        )
        self.assertEqual(bindings, "arg1=257,num_programs_x=3,pid_x=0")

    def test_build_arg_bindings_accepts_named_ttir_arguments(self):
        ttir = """
        tt.func public @kernel(%ptr: !tt.ptr<f16>, %n_rows: i32, %n_cols: i32) attributes {noinline = false} {
          %pid = tt.get_program_id x : i32
          tt.return
        }
        """
        bindings = self.cm._build_costmodel_arg_bindings(
            ttir,
            {"ptr": object(), "n_rows": 1823, "n_cols": 781, "BLOCK": 1024},
            {"ptr": "*fp16", "n_rows": "i32", "n_cols": "i32", "BLOCK": "constexpr"},
            (32, ),
        )
        self.assertEqual(bindings, "arg1=1823,arg2=781,pid_x=0")

    def test_build_arg_bindings_skips_specialized_runtime_constants(self):
        ttir = """
        tt.func public @kernel(%arg0: !tt.ptr<f16>, %arg1: i32) attributes {noinline = false} {
          tt.return
        }
        """
        bindings = self.cm._build_costmodel_arg_bindings(
            ttir,
            {"ptr": object(), "stride": 1, "n": 257},
            {"ptr": "*fp16", "stride": "constexpr", "n": "i32"},
            (1, ),
        )
        self.assertEqual(bindings, "arg1=257")

    def test_select_costmodel_configs_preserves_equivalent_group(self):
        cfg1, cfg2, cfg3, cfg4 = object(), object(), object(), object()
        configs = [cfg1, cfg2, cfg3, cfg4]
        latency = {cfg1: 1.0, cfg2: 1.0, cfg3: 2.0, cfg4: float("inf")}
        equivalence = {cfg1: "same", cfg2: "same", cfg3: "other"}

        primary, fallback = self.cm._select_costmodel_configs(configs, latency, equivalence, top_k=1)

        self.assertEqual(primary, [cfg1, cfg2])
        self.assertEqual(fallback, [cfg3, cfg4])

    def test_select_costmodel_configs_fails_open(self):
        cfg1, cfg2 = object(), object()
        configs = [cfg1, cfg2]
        primary, fallback = self.cm._select_costmodel_configs(
            configs,
            {cfg1: float("inf"), cfg2: float("inf")},
            {},
            top_k=0.25,
        )
        self.assertEqual(primary, configs)
        self.assertEqual(fallback, [])

    def test_keep_count_uses_ceil_and_validates_input(self):
        self.assertEqual(self.cm._resolve_keep_count(5, 0.25), 2)
        self.assertEqual(self.cm._resolve_keep_count(5, 3), 3)
        self.assertEqual(self.cm._resolve_keep_count(5, 8), 5)
        with self.assertRaises(ValueError):
            self.cm._resolve_keep_count(5, 0.0)
        for value in (1.5, 0.0):
            with self.subTest(top_k=value), self.assertRaises(ValueError):
                self.cm._resolve_keep_count(5, value)
        for value in (True, "2", None):
            with self.subTest(top_k=value), self.assertRaises(TypeError):
                self.cm._resolve_keep_count(5, value)
        for value in (0, -1):
            with self.subTest(top_k=value), self.assertRaises(ValueError):
                self.cm._resolve_keep_count(5, value)

    def test_prune_validates_options_before_preparing_costmodel(self):
        backend = types.SimpleNamespace(target=types.SimpleNamespace(arch="Ascend910_9362"))
        configs = [object(), object()]
        for value in (1.5, True, "2", None, 0, -1):
            is_value_error = isinstance(value, (int, float)) and not isinstance(value, bool)
            expected = ValueError if is_value_error else TypeError
            with self.subTest(top_k=value), patch.object(
                    self.cm, "_resolve_hardware_config") as resolve_profile, self.assertRaises(expected):
                self.cm.prune_configs_by_costmodel(
                    backend=backend,
                    fn=object(),
                    configs=configs,
                    named_args={},
                    runtime_kwargs={},
                    options={"top_k": value},
                )
            resolve_profile.assert_not_called()

    def test_prune_configs_by_costmodel(self):
        cfg1, cfg2, cfg3 = object(), object(), object()
        configs = [cfg1, cfg2, cfg3]
        items = [{"config": cfg, "ttir": f"ttir-{idx}"} for idx, cfg in enumerate(configs)]
        equivalence = {cfg: f"key-{idx}" for idx, cfg in enumerate(configs)}

        backend = types.SimpleNamespace(target=types.SimpleNamespace(arch="Ascend910_9362"))
        with patch.object(self.cm, "_resolve_hardware_config", return_value="/tmp/ascend_910b.json"), patch.object(
                self.cm, "_build_costmodel_items",
                return_value=(items, equivalence)), patch.object(self.cm, "costmodel_bench",
                                                                 return_value={cfg1: 3.0, cfg2: 1.0, cfg3: 2.0}):
            primary, fallback = self.cm.prune_configs_by_costmodel(
                backend=backend,
                fn=object(),
                configs=configs,
                named_args={},
                runtime_kwargs={},
                options={"top_k": 1},
            )

        self.assertEqual(primary, [cfg2])
        self.assertEqual(fallback, [cfg3, cfg1])


if __name__ == "__main__":
    unittest.main()
