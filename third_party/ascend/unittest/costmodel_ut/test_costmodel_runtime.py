import importlib.util
import builtins
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


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

        # Other compiler-contract tests install narrower Triton stubs first;
        # replace them so discovery order does not affect this unit test.
        sys.modules["triton"] = triton_mod
        sys.modules["triton.runtime"] = runtime_mod
        sys.modules["triton.runtime.cache"] = cache_mod

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
        self.assertAlmostEqual(self.cm.parse_total_cycles("Total Cycles: 325"), 325)
        self.assertAlmostEqual(self.cm.parse_latency("ascend.scheduled_cycles = 64"), 64)
        self.assertAlmostEqual(self.cm.parse_latency("Roofline model total: 128"), 128)
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

    def test_dynamic_cv_segment_dag_switch_partitions_costmodel_cache(self):
        with patch.dict(self.cm.os.environ, {}, clear=True):
            legacy_key = self.cm.make_costmodel_cache_key("ttir", ["-ascend-perf-model"])
        with patch.dict(
                self.cm.os.environ,
            {"ASCEND_COSTMODEL_DYNAMIC_CV_SEGMENT_DAG_MODEL": "1"},
                clear=True,
        ):
            segment_dag_key = self.cm.make_costmodel_cache_key("ttir", ["-ascend-perf-model"])

        self.assertNotEqual(legacy_key, segment_dag_key)

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
            parsed = json.loads(payload)
            self.assertEqual(parsed["metric"], self.cm._COSTMODEL_CACHE_METRIC_VERSION)
            self.assertAlmostEqual(float(parsed["cycles"]), 7.5)

            mgr.storage[f"{cache_key}.json"] = json.dumps({"latency": "bad-float"})
            self.cm._COSTMODEL_MEM_CACHE.clear()
            self.assertIsNone(self.cm.load_costmodel_latency(cache_key))

            mgr.storage[f"{cache_key}.json"] = json.dumps({"cycles": float("inf")})
            self.cm._COSTMODEL_MEM_CACHE.clear()
            self.assertIsNone(self.cm.load_costmodel_latency(cache_key))

    def test_make_key_and_extra_args(self):
        k1 = self.cm.make_costmodel_cache_key("ttir_a", ["-ascend-perf-model"])
        k2 = self.cm.make_costmodel_cache_key("ttir_a", ["-ascend-perf-model", "arg-bindings=a=1"])
        self.assertNotEqual(k1, k2)

        with patch.object(self.cm, "_resolve_default_hardware_config", lambda: "/tmp/ascend_910b.json"):
            self.assertEqual(
                self.cm._build_costmodel_extra_args("arg1=3", ""),
                ["-ascend-perf-model=hardware-config=/tmp/ascend_910b.json arg-bindings=arg1=3"],
            )
            self.assertEqual(
                self.cm._build_costmodel_extra_args("", ""),
                ["-ascend-perf-model=hardware-config=/tmp/ascend_910b.json"],
            )
            self.assertEqual(
                self.cm._build_costmodel_extra_args(
                    "",
                    "",
                    {"tile_mix_vector_loop": 4, "tile_mix_cube_loop": 2, "BLOCK_M": 128},
                ),
                [
                    "-ascend-perf-model=hardware-config=/tmp/ascend_910b.json "
                    "compile-params=tile_mix_vector_loop=4,tile_mix_cube_loop=2"
                ],
            )
        with patch.object(self.cm, "_resolve_default_hardware_config", lambda: ""):
            self.assertEqual(self.cm._build_costmodel_extra_args("", ""), ["-ascend-perf-model"])

    def test_dynamic_cv_and_multibuffer_compile_params_are_forwarded(self):
        params = {
            "enable_dynamic_cv_pipeline": True,
            "compile_on_910_95": True,
            "intra_cache_num": 2,
            "inter_cache_num": 1,
            "load_cache_num": 1,
            "enable_buffer_insert_optimization": True,
            "enable_ub_refine_opt": True,
            "enable_cube_block_merge": False,
            "multibuffer": True,
            "num_stages": 2,
            "limit_auto_multi_buffer_only_for_local_buffer": False,
            "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
            "limit_auto_multi_buffer_buffer": "workspace0",
            "set_workspace_multibuffer": 4,
            "BLOCK_M": 128,
        }

        self.assertEqual(
            self.cm.build_ascend_perf_model_arg(params),
            "-ascend-perf-model=compile-params="
            "enable_dynamic_cv_pipeline=True,compile_on_910_95=True,"
            "intra_cache_num=2,inter_cache_num=1,load_cache_num=1,"
            "enable_buffer_insert_optimization=True,enable_ub_refine_opt=True,"
            "enable_cube_block_merge=False,multibuffer=True,num_stages=2,"
            "limit_auto_multi_buffer_only_for_local_buffer=False,"
            "limit_auto_multi_buffer_of_local_buffer=no-l0c,"
            "limit_auto_multi_buffer_buffer=workspace0,"
            "set_workspace_multibuffer=4",
        )

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

        with tempfile.TemporaryDirectory() as tmpdir:
            ttir_path = Path(tmpdir) / "module.ttir"
            ttir_path.write_text("module-from-file", encoding="utf-8")
            with patch("builtins.__import__", fake_import):
                self.assertEqual(self.cm.run_costmodel(ttir_path, ["-ascend-perf-model"]), "Estimated Time: 1.0 us")

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

    def test_run_costmodel_falls_back_to_tritonsim_opt_when_bridge_is_missing(self):
        fake_libtriton = types.SimpleNamespace(ascend=types.SimpleNamespace())
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton._C.libtriton":
                return fake_libtriton
            return real_import(name, globals, locals, fromlist, level)

        completed = Mock(stdout="Total Cycles: 321", stderr="")
        with patch("builtins.__import__", fake_import), patch.object(self.cm, "resolve_tritonsim_opt",
                                                                     return_value="/opt/tritonsim-opt"), patch.object(
                                                                         self.cm.subprocess, "run",
                                                                         return_value=completed) as run:
            result = self.cm.run_costmodel("module-text", ["-ascend-perf-model"])

        self.assertEqual(result, "Total Cycles: 321")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/opt/tritonsim-opt")
        self.assertIn("-allow-unregistered-dialect", command)
        self.assertEqual(command[-1], "-")

    def test_feature_compile_params_bypass_inproc_bridge(self):

        class InprocMustNotRun:

            @staticmethod
            def run_costmodel_inproc(_mlir_text, _args):
                raise AssertionError("in-process bridge cannot consume TileMix/Multibuffer compile-params")

        fake_libtriton = types.SimpleNamespace(ascend=InprocMustNotRun)
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton._C.libtriton":
                return fake_libtriton
            return real_import(name, globals, locals, fromlist, level)

        extra_args = [
            "-ascend-perf-model=hardware-config=/tmp/ascend_910b.json "
            "compile-params=set_workspace_multibuffer=2,"
            "tile_mix_vector_loop=4,tile_mix_cube_loop=2"
        ]
        with patch("builtins.__import__", fake_import), patch.object(
                self.cm,
                "_run_costmodel_subprocess",
                return_value="Total Cycles: 456",
        ) as fallback:
            result = self.cm.run_costmodel("module-text", extra_args)

        self.assertEqual(result, "Total Cycles: 456")
        fallback.assert_called_once()
        forwarded_args = fallback.call_args.args[1]
        self.assertIn("compile-params=", forwarded_args[0])
        self.assertIn("-allow-unregistered-dialect", forwarded_args)

    def test_normalize_items_and_eval_item(self):
        cfg1 = object()
        cfg2 = object()
        cfg3 = object()
        items = [
            {"config": cfg1, "ttir": "ttir1", "arg_bindings": "a=1", "hardware_config": "h1"},
            {"config": cfg2, "ttir": ""},
            {"config": None, "ttir": "ignored"},
            {"config": cfg3, "ttir": "ttir3", "compile_params": {"tile_mix_vector_loop": 4}},
            123,
        ]

        pending, lat = self.cm._normalize_costmodel_items(items)
        self.assertEqual(len(pending), 2)
        self.assertEqual(lat[cfg2], float("inf"))
        self.assertEqual(pending[1][4], {"tile_mix_vector_loop": 4})

        with patch.object(self.cm, "load_costmodel_latency", lambda _k: 1.23):
            cfg, t = self.cm._eval_one_costmodel_item(pending[0])
            self.assertIs(cfg, cfg1)
            self.assertAlmostEqual(t, 1.23)

    def test_normalize_items_infers_feature_params_from_autotune_config(self):

        class Config:

            def all_kwargs(self):
                return {
                    "set_workspace_multibuffer": 3,
                    "tile_mix_cube_loop": 2,
                    "tile_mix_vector_loop": 5,
                    "BLOCK_M": 128,
                }

        cfg = Config()
        pending, latencies = self.cm._normalize_costmodel_items([{"config": cfg, "ttir": "ttir"}])

        self.assertEqual(latencies, {})
        self.assertEqual(
            pending[0][4],
            {
                "set_workspace_multibuffer": 3,
                "tile_mix_cube_loop": 2,
                "tile_mix_vector_loop": 5,
            },
        )

    def test_dynamic_cv_cache_slot_names_are_normalized_for_costmodel(self):

        class Config:

            def all_kwargs(self):
                return {
                    "buf_slot_num_of_veccore": 2,
                    "buf_slot_num_of_crosscore": 3,
                    "buf_slot_num_of_gm": 4,
                }

        params = self.cm._extract_compile_params({"config": Config()})

        self.assertEqual(
            params,
            {
                "intra_cache_num": 2,
                "inter_cache_num": 3,
                "load_cache_num": 4,
            },
        )

    def test_dynamic_cv_cache_slot_aliases_preserve_source_precedence(self):

        class Config:

            def all_kwargs(self):
                return {"buf_slot_num_of_veccore": 2}

        params = self.cm._extract_compile_params({
            "config": Config(),
            "runtime_compile_params": {"intra_cache_num": 3},
            "compile_params": {
                "intra_cache_num": 5,
                "buf_slot_num_of_veccore": 4,
            },
        })

        self.assertEqual(params["intra_cache_num"], 4)

    def test_compile_param_sources_have_explicit_override_precedence(self):

        class Config:

            def all_kwargs(self):
                return {
                    "set_workspace_multibuffer": 2,
                    "tile_mix_cube_loop": 2,
                    "tile_mix_vector_loop": 2,
                }

        params = self.cm._extract_compile_params({
            "config": Config(),
            "runtime_compile_params": {
                "set_workspace_multibuffer": 3,
                "tile_mix_vector_loop": 4,
            },
            "compile_params": {
                "tile_mix_vector_loop": 5,
                "tile_mix_cube_loop": 6,
            },
        })

        self.assertEqual(
            params,
            {
                "set_workspace_multibuffer": 3,
                "tile_mix_cube_loop": 6,
                "tile_mix_vector_loop": 5,
            },
        )

    def test_dynamic_cv_compiler_final_status_is_forwarded(self):
        params = self.cm._extract_compile_params({
            "config": object(),
            "compile_params": {
                "enable_dynamic_cv_pipeline": True,
                "dynamic_cv_applied": False,
                "dynamic_cv_skip_reason": "compiler_ignored",
                "dynamic_cv_status_source": "compiler_final",
            },
        })

        self.assertEqual(params["dynamic_cv_applied"], False)
        self.assertEqual(params["dynamic_cv_skip_reason"], "compiler_ignored")
        self.assertEqual(params["dynamic_cv_status_source"], "compiler_final")
        model_arg = self.cm.build_ascend_perf_model_arg(params)
        self.assertIn("dynamic_cv_applied=False", model_arg)
        self.assertIn("dynamic_cv_skip_reason=compiler_ignored", model_arg)

    def test_dynamic_cv_status_accepts_packed_compiler_metadata(self):
        params = self.cm._extract_compile_params({
            "config": object(),
            "compiler_metadata": {
                "dynamic_cv_applied": True,
                "dynamic_cv_skip_reason": "none",
                "dynamic_cv_status_source": "compiler_final",
            },
            "compile_params": {
                "dynamic_cv_applied": False,
                "dynamic_cv_skip_reason": "validation_override",
            },
        })

        self.assertFalse(params["dynamic_cv_applied"])
        self.assertEqual(params["dynamic_cv_skip_reason"], "validation_override")
        self.assertEqual(params["dynamic_cv_status_source"], "compiler_final")

    def test_optional_tilemix_pass_summary_is_flattened(self):
        params = self.cm._extract_compile_params({
            "config": object(),
            "compile_params": {
                "tile_mix_cube_loop": 4,
                "tile_mix_vector_loop": 2,
            },
            "tile_mix_transform_summary": {
                "source": "tile_cube_vector_loop_ir_diff",
                "valid": True,
                "cube_applied": False,
                "vector_applied": True,
                "cube_segments": 1,
                "vector_segments": 2,
                "cube_skip_reason": "pass_rejected_suboptimal",
                "vector_skip_reason": "none",
                "sync_ops_before": 8,
                "sync_ops_after": 6,
            },
        })
        self.assertEqual(params["tile_mix_summary_valid"], 1)
        self.assertEqual(params["tile_mix_vector_applied"], 1)
        self.assertEqual(params["tile_mix_cube_skip_reason"], "pass_rejected_suboptimal")
        model_arg = self.cm.build_ascend_perf_model_arg(params)
        self.assertIn("tile_mix_summary_source=tile_cube_vector_loop_ir_diff", model_arg)
        self.assertIn("tile_mix_sync_ops_after=6", model_arg)

    def test_workspace_multibuffer_is_forwarded_to_ttir_model(self):
        model_arg = self.cm.build_ascend_perf_model_arg({
            "set_workspace_multibuffer": 4,
            "tile_mix_cube_loop": 2,
            "tile_mix_vector_loop": 4,
            "BLOCK_M": 128,
        })
        self.assertIn("set_workspace_multibuffer=4", model_arg)
        self.assertIn("tile_mix_cube_loop=2", model_arg)
        self.assertNotIn("BLOCK_M", model_arg)

    def test_eval_item_miss_and_pending_eval(self):
        cfg1, cfg2 = object(), object()
        pending = [
            (cfg1, "ttir1", "arg=1", "", {}),
            (cfg2, "ttir2", "", "", {}),
        ]
        lat = {}

        calls = []

        def fake_run(ttir_or_path, extra_args=None, dump_ir_on_error=False):
            calls.append((ttir_or_path, tuple(extra_args or [])))
            if "ttir1" in ttir_or_path:
                return "Total Cycles: 99"
            return None

        with patch.object(self.cm, "load_costmodel_latency",
                          lambda _k: None), patch.object(self.cm, "store_costmodel_latency",
                                                         lambda *_args, **_kwargs: None), patch.object(
                                                             self.cm, "run_costmodel",
                                                             fake_run), patch.dict("os.environ",
                                                                                   {"TRITON_COSTMODEL_WORKER_NUM": "1"},
                                                                                   clear=False):
            self.cm._evaluate_pending_items(pending, lat)

        self.assertAlmostEqual(lat[cfg1], 99)
        self.assertEqual(lat[cfg2], float("inf"))
        self.assertEqual(len(calls), 2)

    def test_evaluate_pending_empty(self):
        lat = {}
        self.cm._evaluate_pending_items([], lat)
        self.assertEqual(lat, {})

    def test_evaluate_pending_parallel_exception_tolerated(self):
        cfg1, cfg2 = object(), object()
        pending = [
            (cfg1, "ttir1", "", "", {}),
            (cfg2, "ttir2", "", "", {}),
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

        with patch.object(
                self.cm,
                "_normalize_costmodel_items",
                lambda _x: ([(cfg1, "t1", "", "", {})], {cfg2: float("inf")}),
        ):

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
