"""Exercise the real frontend boundary without loading an NPU runtime.

Cross-option attention legality is checked by the native frontend and its lit
suite. These tests cover the Python boundary's types, retained option surface,
CLI capability checks and protected query-buffer policy.
"""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ASCEND = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("CVP_FRONTEND_SOURCE", str(ASCEND / "backend/compiler.py")))
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
NAMES = {
    "_validate_native_compiler_options",
    "_reject_retired_cvp_options",
    "_uses_native_compiler_options",
    "_resolve_persistent_buffer_policy",
    "_probe_compiler_help",
    "_advertises_compiler_option",
    "_native_compiler_compile_options",
    "_is_a5_target_arch",
}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
nodes += [
    n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"_NATIVE_COMPILER_OPTION_SPECS", "_RETIRED_CVP_OPTIONS"}
        for t in n.targets)
]
assert len(nodes) == len(NAMES) + 2
namespace = {"re": re, "subprocess": subprocess, "is_cann_version_at_least": lambda *_args: False}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
COMMON = json.loads((ASCEND / "benchmarks/cvp_attention/recipes.json").read_text())["presets"]["common"]
# These controls are forwarded by the surrounding host/compiler route, rather
# than by the explicit native-control boundary exercised here.
HOST_CONTROLS = {
    "enable_dynamic_cv_pipeline",
    "enable_mixed_cv",
    "enable_preload",
    "unit_flag",
    "disable_auto_inject_block_sync",
    "disable_tightly_coupled_buffer_reuse",
}
FORWARD_CONTROLS = (
    "enable_cvp_state_copy_forward",
    "enable_cvp_final_output_forward",
    "enable_cvp_output_copy_forward",
    "enable_cvp_scalar_input_forward",
)
REMOVED = (
    "cvp_q_cache_mode",
    "cvp_q_cache_consider_overlapping",
    "enable_cvp_attention_rows2",
    "enable_cvp_attention_first_output",
    "enable_cvp_attention_last_output",
    "enable_cvp_attention_first_state",
    "enable_cvp_attention_half_p_stream",
)


def expected_flags(values):
    return {"--" + k.replace("_", "-"): v for k, v in values.items() if k not in HOST_CONTROLS}


def advertised(flags):
    return "\n".join("  " + flag + "=<value>" for flag in flags)


class FrontendWithoutStage6Test(unittest.TestCase):

    def compile(self, values, help_text=None):
        help_text = advertised(expected_flags(values)) if help_text is None else help_text
        with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=help_text)) as probe:
            result = namespace["_native_compiler_compile_options"](values, "/compiler", {"SELECTED": "yes"})
        return result, probe

    def test_common_recipe_forwards_every_native_control_with_one_probe(self):
        actual, probe = self.compile(dict(COMMON))
        self.assertEqual(set(actual), {f"{k}={v}" for k, v in expected_flags(COMMON).items()})
        self.assertEqual(len(actual), len(expected_flags(COMMON)))
        probe.assert_called_once()
        self.assertEqual(probe.call_args.args[0], ["/compiler", "--help"])
        self.assertEqual(probe.call_args.kwargs["env"], {"SELECTED": "yes"})
        for name in FORWARD_CONTROLS:
            self.assertIn("--" + name.replace("_", "-") + "=True", actual)

    def test_forwarding_controls_have_independent_native_capabilities(self):
        for name in FORWARD_CONTROLS:
            missing = "--" + name.replace("_", "-")
            help_text = advertised(k for k in expected_flags(COMMON) if k != missing)
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, re.escape(missing)):
                self.compile(dict(COMMON), help_text)

    def test_false_controls_are_forwarded_explicitly(self):
        values = {"enable_cvp_persistent_pipeline": False, "enable_fused_multiply_add": False}
        actual, _ = self.compile(values)
        self.assertEqual(set(actual), {f"{k}={v}" for k, v in expected_flags(values).items()})

    def test_default_request_only_probes_optional_metaop(self):
        actual, probe = self.compile({})
        self.assertEqual(actual, [])
        probe.assert_called_once()

    def test_optional_metaop_respects_selected_sdk(self):
        with patch.dict(namespace, {"is_cann_version_at_least": lambda *_args: True}):
            actual, _ = self.compile({}, "  --enable-optimized-metaop=<bool>")
        self.assertEqual(actual, ["--enable-optimized-metaop=True"])

    def test_failed_optional_probe_preserves_defaults(self):
        for failure in (OSError("missing"), subprocess.TimeoutExpired("compiler", 10)):
            with self.subTest(error=type(failure).__name__), patch.object(subprocess, "run", side_effect=failure):
                self.assertEqual(namespace["_native_compiler_compile_options"]({}, "/compiler", {}), [])

    def test_failed_probe_rejects_explicit_controls(self):
        with patch.object(subprocess, "run", side_effect=OSError("missing")):
            with self.assertRaisesRegex(RuntimeError, "cannot verify"):
                namespace["_native_compiler_compile_options"](dict(COMMON), "/compiler", {})
        with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")):
            with self.assertRaisesRegex(RuntimeError, "successful --help"):
                namespace["_native_compiler_compile_options"](dict(COMMON), "/compiler", {})

    def test_prefix_match_is_not_a_capability(self):
        with self.assertRaisesRegex(RuntimeError, "enable-cvp-persistent-pipeline"):
            self.compile({"enable_cvp_persistent_pipeline": True}, "--enable-cvp-persistent-pipeline-other=<bool>")

    def test_all_retained_forwarding_values_are_strict_booleans(self):
        for name in FORWARD_CONTROLS:
            for value in (0, 1, "true", [], {}):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    namespace["_validate_native_compiler_options"]({name: value})

    def test_removed_modes_are_rejected(self):
        for name, value in (("cvp_invariant_cache_mode", "neutral"), ("cvp_stage_fusion", "p-state"),
                            ("cvp_reduction_chains", 4), ("cvp_persistent_pipeline_mode",
                                                          "unroll"), ("cvp_inner_unroll", 2), ("cvp_inner_unroll", 4)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                namespace["_validate_native_compiler_options"]({name: value})

    def test_numeric_controls_reject_boolean_values(self):
        for name in ("cvp_reduction_chains", "cvp_inner_unroll"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                namespace["_validate_native_compiler_options"]({name: True})

    def test_fma_conflict_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "conflicts with disable_fma"):
            namespace["_validate_native_compiler_options"]({"enable_fused_multiply_add": True, "disable_fma": True})

    def test_vf_safety_disable_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "safety disable"):
            self.compile({"enable_vf_operand_substitution": True, "disable_vf_operand_substitution": True})

    def test_implicit_dynamic_vf_respects_safety_disable(self):
        for disabled in (False, True):
            for help_text in ("", "--enable-vf-operand-substitution=<bool>"):
                with self.subTest(disabled=disabled, help_text=help_text):
                    values = {
                        "target_arch": "Ascend950PR", "enable_dynamic_cv_pipeline": True,
                        "disable_vf_operand_substitution": disabled
                    }
                    actual, _ = self.compile(values, help_text)
                    self.assertEqual(actual, [] if disabled else ["--enable-vf-operand-substitution=True"])

    def test_query_buffer_policy_is_protected(self):
        resolve = namespace["_resolve_persistent_buffer_policy"]
        for value in (None, True):
            self.assertIs(
                resolve({"enable_cvp_persistent_pipeline": True, "disable_tightly_coupled_buffer_reuse": value}), True)
        with self.assertRaisesRegex(ValueError, "requires disable_tightly_coupled_buffer_reuse=True"):
            resolve({"enable_cvp_persistent_pipeline": True, "disable_tightly_coupled_buffer_reuse": False})
        for value in (0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                resolve({"disable_tightly_coupled_buffer_reuse": value})

    def test_non_query_buffer_policy_preserves_choices(self):
        resolve = namespace["_resolve_persistent_buffer_policy"]
        for value, expected in ((None, False), (False, False), (True, True)):
            self.assertIs(resolve({"disable_tightly_coupled_buffer_reuse": value}), expected)

    def test_common_fields_remain_declared_and_obsolete_fields_absent(self):
        options = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUOptions")
        fields = {
            n.target.id: n.value
            for n in options.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }
        self.assertTrue(set(COMMON).issubset(fields))
        for name in FORWARD_CONTROLS:
            self.assertIsNone(ast.literal_eval(fields[name]))
        for name in (*REMOVED, "enable_cvp_async_ownership", "enable_physical_memory_footprints"):
            self.assertNotIn(name, fields)
        self.assertNotIn("_read_cvp_async_launch_callback",
                         {n.name
                          for n in tree.body
                          if isinstance(n, ast.FunctionDef)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
