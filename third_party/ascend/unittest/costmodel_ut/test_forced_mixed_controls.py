"""Exercise source-loaded Mixed controls without importing NPU dependencies."""
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import types
from typing import Optional
import unittest
from unittest.mock import patch


class ForcedMixedControlsTest(unittest.TestCase):

    @staticmethod
    def _load_compiler_module():
        source = Path(__file__).resolve().parents[2] / "backend/compiler.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        names = {
            "_resolve_forced_mixed_options", "_verify_forced_mixed_result", "_publish_route_transform_capability",
            "_normalize_auto_simt_scope_mode"
        }
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(functions) == len(names)
        module = types.ModuleType("forced_mixed_controls_under_test")
        module.__dict__.update(os=os, json=json, re=re, hashlib=hashlib, Optional=Optional)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), module.__dict__)
        return module

    def test_forced_mixed_environment_and_explicit_override(self):
        cmplr = self._load_compiler_module()
        with patch.dict(os.environ, {"TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE": "mixed_simd_simt"}, clear=True):
            options = cmplr._resolve_forced_mixed_options({"num_warps": 4})
            self.assertEqual(options["auto_simt_force_route"], "mixed_simd_simt")
            self.assertEqual(options["superblock_factor"], 1)
            self.assertEqual(options["compile_mode"], "simd_simt")
            self.assertEqual(options["auto_simt_scope_mode"], "auto")
            self.assertEqual(cmplr._resolve_forced_mixed_options({"superblock_factor": 4})["superblock_factor"], 4)
            self.assertEqual(cmplr._resolve_forced_mixed_options({"auto_simt_force_route": ""}),
                             {"auto_simt_force_route": ""})

    def test_forcing_accepts_unspecified_modes_and_existing_scope_aliases(self):
        cmplr = self._load_compiler_module()
        with patch.dict(os.environ, {"TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE": "mixed_simd_simt"}, clear=True):
            resolved = cmplr._resolve_forced_mixed_options({"compile_mode": None, "auto_simt_scope_mode": ""})
            self.assertEqual(resolved["compile_mode"], "simd_simt")
            self.assertEqual(resolved["auto_simt_scope_mode"], "auto")
            resolved = cmplr._resolve_forced_mixed_options({"auto_simt_scope_mode": "ON"})
            self.assertEqual(resolved["auto_simt_scope_mode"], "auto")
            resolved = cmplr._resolve_forced_mixed_options({"compile_mode": types.SimpleNamespace(value="simd_simt")})
            self.assertEqual(resolved["compile_mode"], "simd_simt")

    def test_forcing_honors_existing_process_wide_compile_mode_precedence(self):
        cmplr = self._load_compiler_module()
        with patch.dict(
                os.environ,
            {"TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE": "mixed_simd_simt", "TRITON_ASCEND_COMPILE_MODE": "simd_simt"},
                clear=True):
            resolved = cmplr._resolve_forced_mixed_options({"compile_mode": "simd"})
            self.assertEqual(resolved["compile_mode"], "simd_simt")
        with patch.dict(
                os.environ,
            {"TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE": "mixed_simd_simt", "TRITON_ASCEND_COMPILE_MODE": "simd"},
                clear=True):
            with self.assertRaises(ValueError):
                cmplr._resolve_forced_mixed_options({"compile_mode": "simd_simt"})

    def test_forced_mixed_rejects_invalid_or_conflicting_controls(self):
        cmplr = self._load_compiler_module()
        with patch.dict(os.environ, {}, clear=True):
            for extra in ({"auto_simt_force_route": "all_simd"}, {"superblock_factor": 3}, {"superblock_factor": True},
                          {"superblock_factor": "2"}, {"compile_mode":
                                                       "simd"}, {"auto_simt_scope_mode":
                                                                 "off"}, {"auto_simt_scope_mode": "report"}):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    cmplr._resolve_forced_mixed_options({"auto_simt_force_route": "mixed_simd_simt", **extra})

    def test_forced_mixed_factor_restricts_capability_without_changing_legality(self):
        cmplr = self._load_compiler_module()
        cmplr._get_modeled_superblock_factors = lambda: (1, 2, 4, 8, 16, 32, 64)
        cmplr._get_current_scope_superblock_factors = lambda: (1, 2, 4)
        options = types.SimpleNamespace(compile_on_910_95=True, num_warps=8, physical_vector_core_count_hint=56,
                                        logical_program_count_hint=56, auto_simt_force_route="mixed_simd_simt",
                                        superblock_factor=2)
        metadata = {"auto_blockify_v1_enabled": True}
        capability = json.loads(cmplr._publish_route_transform_capability(metadata, options))
        self.assertEqual(capability["scope_superblock_factors"], [2])
        self.assertEqual(capability["backend_forced_route_candidate"], "mixed_simd_simt")
        self.assertTrue(capability["auto_blockify_v1_materializable"])
        options.superblock_factor = 8
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cmplr._publish_route_transform_capability({}, options)

    def test_forcing_preserves_target_warp_and_v1_factor_limits(self):
        cmplr = self._load_compiler_module()
        cmplr._get_modeled_superblock_factors = lambda: (1, 2, 4, 8, 16, 32, 64)
        cmplr._get_current_scope_superblock_factors = lambda: (1, 2, 4)
        options = types.SimpleNamespace(compile_on_910_95=True, num_warps=32, physical_vector_core_count_hint=56,
                                        physical_aicore_count_hint=28, logical_program_count_hint=56,
                                        auto_simt_force_route="mixed_simd_simt", superblock_factor=4)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cmplr._publish_route_transform_capability({"auto_blockify_v1_enabled": True}, options)
        options.num_warps = 4
        options.superblock_factor = 2
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cmplr._publish_route_transform_capability({"auto_blockify_v1_enabled": False}, options)
        options.compile_on_910_95 = False
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cmplr._publish_route_transform_capability({"auto_blockify_v1_enabled": True}, options)

    def test_route_and_factor_participate_in_the_real_options_hash(self):
        source = Path(__file__).resolve().parents[2] / "backend/compiler.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        options_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUOptions")
        hash_method = next(node for node in options_class.body
                           if isinstance(node, ast.FunctionDef) and node.name == "hash")
        namespace = {"hashlib": hashlib, "get_cann_version_file_hash": lambda: "same-cann"}
        exec(compile(ast.Module(body=[hash_method], type_ignores=[]), str(source), "exec"), namespace)
        hashes = []
        for route, factor in (("", 1), ("mixed_simd_simt", 1), ("mixed_simd_simt", 2)):
            options = types.SimpleNamespace(num_warps=4, auto_simt_force_route=route, superblock_factor=factor)
            hashes.append(namespace["hash"](options))
            self.assertEqual(namespace["hash"](options), hashes[-1])
        self.assertEqual(len(set(hashes)), 3)

    def test_forced_mixed_verification_refuses_a_fallback_or_wrong_factor(self):
        cmplr = self._load_compiler_module()
        options = types.SimpleNamespace(auto_simt_force_route="mixed_simd_simt", superblock_factor=2)
        report = json.dumps({"action_supported": True, "materialized_simt_anchor_count": 1})
        module = 'scope.scope {vector_mode = "simt", ascend.scope_superblock.factor = 2 : i64}'
        metadata = {}
        cmplr._verify_forced_mixed_result(module, metadata, options, "mixed_simd_simt", 2, report)
        self.assertEqual(metadata["auto_simt_forced_factor"], 2)
        self.assertEqual(len(metadata["auto_simt_forced_ttir_sha256"]), 64)
        for rejected in ({"action_supported": False, "materialized_simt_anchor_count":
                          1}, {"action_supported": True, "materialized_simt_anchor_count": 0}):
            with self.subTest(report=rejected), self.assertRaisesRegex(RuntimeError, "refusing fallback"):
                cmplr._verify_forced_mixed_result(module, {}, options, "mixed_simd_simt", 2, json.dumps(rejected))
        for effective, factor, text in (("all_simd", 2, module), ("mixed_simd_simt", 1, module), ("mixed_simd_simt", 2,
                                                                                                  "module {}")):
            with self.assertRaisesRegex(RuntimeError, "refusing fallback"):
                cmplr._verify_forced_mixed_result(text, {}, options, effective, factor, report)

    def test_parse_options_consumes_legacy_stream_from_the_callers_options(self):
        cmplr = self._load_compiler_module()
        source = Path(__file__).resolve().parents[2] / "backend/compiler.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendBackend")
        parse = next(node for node in backend.body
                     if isinstance(node, ast.FunctionDef) and node.name == "parse_options")

        @dataclass
        class Options:
            arch: str
            num_warps: int = 4
            auto_simt_force_route: str = ""
            enable_dynamic_cv_pipeline: object = None
            compile_on_910_95: bool = True

        namespace = {
            "Any": object,
            "NPUOptions": Options,
            "_warn_deprecated_ascend_env_vars": lambda: None,
            "_is_internal_npu_options": lambda options, arch: False,
            "_remove_deprecated_npu_options": dict,
            "_normalize_bishengir_simt_optimization_for_context": lambda options, raw: None,
        }
        exec(compile(ast.Module(body=[parse], type_ignores=[]), str(source), "exec"), namespace)
        backend = types.SimpleNamespace(target=types.SimpleNamespace(backend="npu", arch="Ascend950PR_9579"),
                                        resolve_launch_options=cmplr._resolve_forced_mixed_options)
        options = {"num_warps": 8, "stream": 123}
        with patch.dict(os.environ, {}, clear=True):
            parsed = namespace["parse_options"](backend, options)
        self.assertEqual(options, {"num_warps": 8})
        self.assertEqual(parsed.num_warps, 8)
        self.assertTrue(parsed.enable_dynamic_cv_pipeline)

    def test_disabled_forcing_keeps_capability_and_verification_inactive(self):
        cmplr = self._load_compiler_module()
        original = {"compile_mode": "simd", "auto_simt_scope_mode": "off", "superblock_factor": 3}
        with patch.dict(os.environ, {}, clear=True):
            resolved = cmplr._resolve_forced_mixed_options(original)
        self.assertEqual(original, {"compile_mode": "simd", "auto_simt_scope_mode": "off", "superblock_factor": 3})
        self.assertEqual(resolved, {**original, "auto_simt_force_route": ""})
        metadata = {"compile_mode": "simd"}
        cmplr._verify_forced_mixed_result(None, metadata, types.SimpleNamespace(), "all_simd", 1, "")
        self.assertEqual(metadata, {"compile_mode": "simd"})

    def test_environment_changes_are_resolved_before_real_jit_cache_lookup(self):
        cmplr = self._load_compiler_module()
        source = Path(__file__).resolve().parents[4] / "python/triton/runtime/jit.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        compute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_cache_key")
        jit = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JITFunction")
        run = next(n for n in jit.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        namespace = {
            "is_namedtuple":
            lambda value: False,
            "JITCallable":
            type("JITCallable", (), {}),
            "driver":
            types.SimpleNamespace(
                active=types.SimpleNamespace(get_current_device=lambda: 0, get_current_stream=lambda device: None)),
            "knobs":
            types.SimpleNamespace(runtime=types.SimpleNamespace(debug=False),
                                  compilation=types.SimpleNamespace(instrumentation_mode=None)),
        }
        exec(compile(ast.Module(body=[compute, run], type_ignores=[]), str(source), "exec"), namespace)
        cache, compiled = {}, []
        backend = types.SimpleNamespace(resolve_launch_options=cmplr._resolve_forced_mixed_options)
        instance = types.SimpleNamespace(
            debug=False, pre_run_hooks=[], used_global_vals={},
            device_caches={0: (cache, {}, None, backend, lambda **kwargs: ({}, [], kwargs))})
        instance._pack_args = lambda backend, kwargs, bound, specialization, options: (options, {}, {}, {})

        def compile_kernel(key, signature, device, constexprs, options, attrs, warmup):
            result = types.SimpleNamespace(options=options)
            compiled.append(result)
            cache[key] = result
            return result

        instance._do_compile = compile_kernel

        def launch(**kwargs):
            return namespace["run"](instance, grid=(1, ), warmup=True, compile_mode="simd_simt",
                                    auto_simt_scope_mode="auto", **kwargs)

        with patch.dict(os.environ, {}, clear=True):
            automatic = launch(superblock_factor=1)
            os.environ["TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE"] = "mixed_simd_simt"
            f2 = launch(superblock_factor=2)
            f4 = launch(superblock_factor=4)
            self.assertIs(launch(superblock_factor=2), f2)
            self.assertIs(launch(auto_simt_force_route="", superblock_factor=1), automatic)
        self.assertIsNot(f2, f4)
        self.assertEqual(f2.options["superblock_factor"], 2)
        self.assertEqual(f4.options["superblock_factor"], 4)
        self.assertEqual(len(compiled), 3)
        # Exercise the shared JIT entry point with a backend without this hook.
        plain_cache = {}
        cache = plain_cache
        binder = instance.device_caches[0][-1]
        instance.device_caches[0] = (plain_cache, {}, None, types.SimpleNamespace(), binder)
        with patch.dict(os.environ, {}, clear=True):
            plain = launch(superblock_factor=1)
            self.assertNotIn("auto_simt_force_route", plain.options)
            self.assertIs(launch(superblock_factor=1), plain)


if __name__ == "__main__":
    unittest.main()
