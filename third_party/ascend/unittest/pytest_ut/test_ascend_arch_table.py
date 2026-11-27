"""Offline tests for the centralized Ascend arch classification table.

The table lives in ``backend/utils.py``; that module pulls in pybind11 and
triton internals, so this test execs just the classification block with
``ast`` instead of importing it.  Runnable without an installed Triton wheel
or an NPU runtime.
"""
import ast
import builtins
import sys
import types
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BACKEND = Path(__file__).resolve().parents[2] / "backend"

_ARCH_FUNCTIONS = {
    "_match_arch_entry",
    "arch_generation",
    "is_register_based_arch",
    "ub_size_in_kbytes_for_arch",
    "rf_size_in_kbytes_for_arch",
    "is_ffts_supported",
}

# Module-level constants the extracted functions close over.
_ARCH_CONSTANTS = {"_ARCH_TABLE", "_FFTS_UNSUPPORTED_ARCHS"}


def load_arch_table():
    """Extract the arch-classification block from ``utils.py``."""
    path = BACKEND / "utils.py"
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AscendArchGen":
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in _ARCH_CONSTANTS for target in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in _ARCH_FUNCTIONS:
            nodes.append(node)
    namespace = {"IntEnum": IntEnum, "auto": auto}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class AscendArchTableTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mod = load_arch_table()

    def test_register_based_generations(self):
        gen5 = self.mod["AscendArchGen"].GEN5
        gen6 = self.mod["AscendArchGen"].GEN6
        for arch in ("Ascend910_95", "Ascend950", "Ascend910_958b", "Ascend910_96", "Ascend960", "Ascend960DT_9691"):
            with self.subTest(arch=arch):
                gen = self.mod["arch_generation"](arch)
                self.assertIn(gen, (gen5, gen6))
                self.assertTrue(self.mod["is_register_based_arch"](arch))

    def test_arch_helpers_query_runtime_soc(self):
        generations = self.mod["AscendArchGen"]
        for soc_name, gen, register_based, rf_kb in (
            ("Ascend910_958b", generations.GEN5, True, 128),
            ("ascend950DT_9571", generations.GEN5, True, 128),
            ("Ascend910_9691", generations.GEN6, True, 128),
            ("ascend960DT_9691", generations.GEN6, True, 128),
            ("Ascend910_9391", generations.LEGACY, False, None),
            ("ascend910b1", generations.LEGACY, False, None),
            ("Ascend910_9791", generations.LEGACY, False, None),
        ):
            for args in ((), (None, ), ("", )):
                for helper, expected in (
                    ("arch_generation", gen),
                    ("is_register_based_arch", register_based),
                    ("rf_size_in_kbytes_for_arch", rf_kb),
                ):
                    with self.subTest(soc_name=soc_name, args=args, helper=helper):
                        get_current_target = Mock(return_value=SimpleNamespace(arch=soc_name))
                        triton, runtime = self._mock_driver(get_current_target)
                        with patch.dict(sys.modules, {"triton": triton, "triton.runtime": runtime}):
                            self.assertEqual(self.mod[helper](*args), expected)
                        get_current_target.assert_called_once_with()

    def test_explicit_arch_helpers_do_not_query_runtime(self):
        generations = self.mod["AscendArchGen"]
        get_current_target = Mock(return_value=SimpleNamespace(arch="Ascend960"))
        triton, runtime = self._mock_driver(get_current_target)
        with patch.dict(sys.modules, {"triton": triton, "triton.runtime": runtime}):
            for arch, gen, register_based, rf_kb in (
                ("Ascend950", generations.GEN5, True, 128),
                ("ascend950DT_9571", generations.GEN5, True, 128),
                ("Ascend960", generations.GEN6, True, 128),
                ("Ascend910_9391", generations.LEGACY, False, None),
                ("Ascend910_9791", generations.LEGACY, False, None),
            ):
                with self.subTest(arch=arch):
                    self.assertEqual(self.mod["arch_generation"](arch), gen)
                    self.assertEqual(self.mod["is_register_based_arch"](arch), register_based)
                    self.assertEqual(self.mod["rf_size_in_kbytes_for_arch"](arch), rf_kb)
        get_current_target.assert_not_called()

    def test_arch_helpers_runtime_failures_fall_back(self):
        legacy = self.mod["AscendArchGen"].LEGACY
        helpers = (
            ("arch_generation", legacy),
            ("is_register_based_arch", False),
            ("rf_size_in_kbytes_for_arch", None),
        )
        real_import = builtins.__import__

        def fail_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton.runtime":
                raise ImportError("runtime unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fail_runtime_import):
            for helper, expected in helpers:
                with self.subTest(failure="import", helper=helper):
                    self.assertEqual(self.mod[helper](), expected)

        for label, target in (
            ("missing-target", None),
            ("missing-arch", SimpleNamespace()),
            ("none-arch", SimpleNamespace(arch=None)),
            ("empty-arch", SimpleNamespace(arch="")),
            ("non-string-arch", SimpleNamespace(arch=123)),
        ):
            for helper, expected in helpers:
                with self.subTest(target=label, helper=helper):
                    get_current_target = Mock(return_value=target)
                    triton, runtime = self._mock_driver(get_current_target)
                    with patch.dict(sys.modules, {"triton": triton, "triton.runtime": runtime}):
                        self.assertEqual(self.mod[helper](), expected)
                    get_current_target.assert_called_once_with()

        for helper, expected in helpers:
            with self.subTest(failure="target-query", helper=helper):
                get_current_target = Mock(side_effect=RuntimeError("target query failed"))
                triton, runtime = self._mock_driver(get_current_target)
                with patch.dict(sys.modules, {"triton": triton, "triton.runtime": runtime}):
                    self.assertEqual(self.mod[helper](), expected)
                get_current_target.assert_called_once_with()

    @staticmethod
    def _mock_driver(get_current_target):
        triton = types.ModuleType("triton")
        triton.__path__ = []
        runtime = types.ModuleType("triton.runtime")
        runtime.__path__ = []
        runtime.driver = SimpleNamespace(active=SimpleNamespace(get_current_target=get_current_target))
        triton.runtime = runtime
        return triton, runtime

    def test_legacy_not_register_based(self):
        for arch in ("Ascend910A", "Ascend910B1", "Ascend910_93", "Ascend310B", "Ascend910_97"):
            with self.subTest(arch=arch):
                self.assertFalse(self.mod["is_register_based_arch"](arch))

    def test_ub_and_rf_capacity(self):
        cases = {
            "Ascend960": (320, 128),
            "Ascend910_96": (320, 128),
            "Ascend950": (256, 128),
            "Ascend910_95": (256, 128),
            "Ascend910B1": (192, None),
            "Ascend910A": (192, None),
            "Ascend910_97": (0, None),
        }
        for arch, (ub, rf) in cases.items():
            with self.subTest(arch=arch):
                self.assertEqual(self.mod["ub_size_in_kbytes_for_arch"](arch), ub)
                self.assertEqual(self.mod["rf_size_in_kbytes_for_arch"](arch), rf)
        self.assertEqual(self.mod["ub_size_in_kbytes_for_arch"](""), 0)

    def test_ffts_policy(self):
        # Register-based targets (A5 / A6) do not support FFTS.
        for arch in ("Ascend960", "Ascend910_96", "Ascend910_95", "Ascend950"):
            with self.subTest(arch=arch):
                self.assertFalse(self.mod["is_ffts_supported"](arch))
        # Ascend910A and Ascend310B4 are legacy but have FFTS disabled.
        for arch in ("Ascend910A", "Ascend310B4"):
            with self.subTest(arch=arch):
                self.assertFalse(self.mod["is_ffts_supported"](arch))
        # Other legacy arches have FFTS enabled.
        for arch in ("Ascend910B1", "Ascend910_93", "Ascend310B", "Ascend910D"):
            with self.subTest(arch=arch):
                self.assertTrue(self.mod["is_ffts_supported"](arch))

    def test_non_string_inputs_do_not_query_runtime(self):
        legacy = self.mod["AscendArchGen"].LEGACY
        get_current_target = Mock(return_value=SimpleNamespace(arch="Ascend960"))
        triton, runtime = self._mock_driver(get_current_target)
        with patch.dict(sys.modules, {"triton": triton, "triton.runtime": runtime}):
            for bad in (123, 0, False, [], {}, ()):
                with self.subTest(bad=bad):
                    self.assertEqual(self.mod["arch_generation"](bad), legacy)
                    self.assertFalse(self.mod["is_register_based_arch"](bad))
                    self.assertEqual(self.mod["ub_size_in_kbytes_for_arch"](bad), 0)
                    self.assertIsNone(self.mod["rf_size_in_kbytes_for_arch"](bad))
                    self.assertFalse(self.mod["is_ffts_supported"](bad))
        get_current_target.assert_not_called()
        self.assertEqual(self.mod["ub_size_in_kbytes_for_arch"](None), 0)
        self.assertFalse(self.mod["is_ffts_supported"](None))


if __name__ == "__main__":
    unittest.main()
