"""Offline tests for the centralized Ascend arch classification table.

The table lives in ``backend/utils.py``; that module pulls in pybind11 and
triton internals, so this test execs just the classification block with
``ast`` instead of importing it.  Runnable without an installed Triton wheel
or an NPU runtime.
"""
import ast
import unittest
from enum import IntEnum, auto
from pathlib import Path

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

    def test_legacy_not_register_based(self):
        for arch in ("Ascend910A", "Ascend910B1", "Ascend910_93", "Ascend310B", "Ascend910_97", ""):
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
            "": (0, None),
        }
        for arch, (ub, rf) in cases.items():
            with self.subTest(arch=arch):
                self.assertEqual(self.mod["ub_size_in_kbytes_for_arch"](arch), ub)
                self.assertEqual(self.mod["rf_size_in_kbytes_for_arch"](arch), rf)

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

    def test_non_string_and_none_inputs(self):
        legacy = self.mod["AscendArchGen"].LEGACY
        for bad in (None, 123, [], {}):
            with self.subTest(bad=bad):
                self.assertFalse(self.mod["is_register_based_arch"](bad))
                self.assertEqual(self.mod["ub_size_in_kbytes_for_arch"](bad), 0)
                self.assertEqual(self.mod["rf_size_in_kbytes_for_arch"](bad), None)
                self.assertFalse(self.mod["is_ffts_supported"](bad))
                self.assertEqual(self.mod["arch_generation"](bad), legacy)


if __name__ == "__main__":
    unittest.main()
