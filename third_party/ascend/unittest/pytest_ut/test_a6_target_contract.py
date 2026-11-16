"""Offline target/launch policy checks; runnable with Python's unittest runner.

The backend modules pull in pybind11 and triton internals, so these checks
extract just the policy functions from this checkout with ``ast``.  That keeps
them runnable without an installed Triton wheel or an NPU runtime; device
execution is tested separately.
"""
import ast
import sys
import types
import unittest
from enum import IntEnum, auto
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[2] / "backend"

_ARCH_FUNCTIONS = {
    "_match_arch_entry",
    "arch_generation",
    "is_register_based_arch",
    "rf_size_in_kbytes_for_arch",
    "ub_size_in_kbytes_for_arch",
    "is_ffts_supported",
    "force_disable_ffts",
    "graph_ub_budget_bytes_for_arch",
    "is_compile_on_910_95",
}

# Module-level constants the extracted functions close over.
_ARCH_CONSTANTS = {"_ARCH_TABLE", "_FFTS_UNSUPPORTED_ARCHS"}


def _exec_nodes(path, nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_arch_utils():
    """Extract the arch-classification block from ``utils.py`` without importing it.

    Deprecated-helper notices are recorded instead of emitted so the policy
    assertions below do not print warnings.
    """
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
    deprecated_notices = []
    namespace = {
        "IntEnum": IntEnum,
        "auto": auto,
        "_is_compile_on_910_95": None,
        "_warn_deprecated_ascend_env_var": lambda name: None,
        "_warn_deprecated_arch_helper": lambda name, replacement: deprecated_notices.append((name, replacement)),
        "deprecated_notices": deprecated_notices,
    }
    return _exec_nodes(path, nodes, namespace)


def load_policies(path, names, extra=None):
    """Exec just the named functions from *path* (no module import required).

    The compiler policy functions were authored to ``import`` their own
    dependency lazily, so we inject a lightweight ``triton.backends.ascend``
    package shim into ``sys.modules`` that re-exports the arch helpers already
    extracted from ``utils.py``.  This keeps the offline test importable
    without a Trtion wheel while exercising the real function bodies.
    """
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(extra or {})
    namespace.setdefault("_warn_deprecated_ascend_env_var", lambda name: None)
    # Inject a triton shim so lazy ``from triton.backends.ascend.utils import …``
    # inside the policy bodies resolves to the helpers we already extracted.
    if not _triton_shim_installed():
        _install_triton_shim(namespace.get("is_register_based_arch"))
    return _exec_nodes(path, nodes, namespace)


def _triton_shim_installed():
    try:
        import triton.backends.ascend.utils as u  # noqa: F401
    except ImportError:
        return False
    return hasattr(u, "is_register_based_arch")


def _install_triton_shim(is_register_based_ref):
    import sys
    import types

    root = types.ModuleType("triton")
    root.__path__ = []
    sys.modules["triton"] = root
    backends = types.ModuleType("triton.backends")
    backends.__path__ = []
    root.backends = backends
    sys.modules["triton.backends"] = backends
    ascend_pkg = types.ModuleType("triton.backends.ascend")
    ascend_pkg.__path__ = []
    ascend_pkg.utils = types.ModuleType("triton.backends.ascend.utils")
    ascend_pkg.utils.is_register_based_arch = is_register_based_ref
    sys.modules["triton.backends.ascend"] = ascend_pkg
    sys.modules["triton.backends.ascend.utils"] = ascend_pkg.utils
    backends.ascend = ascend_pkg


class A6TargetContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.utils = load_arch_utils()
        cls.compiler = load_policies(BACKEND / "compiler.py",
                                     {"_is_register_based_target_arch", "_get_compile_on_register_based"},
                                     extra={"is_register_based_arch": cls.utils["is_register_based_arch"]})

    def test_target_policy(self):
        for arch, register_based, ub in [
            ("Ascend910_9691", True, 320),
            ("Ascend910_9699", True, 320),
            ("Ascend960", True, 320),
            ("Ascend960DT_9691", True, 320),
            ("Ascend910_958b", True, 256),
            ("Ascend950DT_9571", True, 256),
            ("Ascend910B1", False, 192),
            ("Ascend910_9391", False, 192),
            ("Ascend910_9791", False, 0),
            ("", False, 0),
            (123, False, 0),
        ]:
            with self.subTest(arch=arch):
                self.assertEqual(self.utils["is_compile_on_910_95"](arch), register_based)
                self.assertEqual(self.compiler["_is_register_based_target_arch"](arch), register_based)
                self.assertEqual(self.utils["ub_size_in_kbytes_for_arch"](arch), ub)
                self.assertEqual(self.utils["graph_ub_budget_bytes_for_arch"](arch), ub * 1024 * 80 // 100)

    def test_launch_and_compiler_ffts_policy(self):
        for arch in ("Ascend910_9691", "Ascend960", "Ascend910_958b", "Ascend950"):
            with self.subTest(arch=arch):
                self.assertTrue(self.utils["force_disable_ffts"](arch))
                self.assertFalse(self.utils["is_ffts_supported"](arch))
        self.assertFalse(self.utils["force_disable_ffts"]("Ascend910B1"))
        self.assertTrue(self.utils["is_ffts_supported"]("Ascend910B1"))

    def test_legacy_ffts_targets_are_not_force_disabled(self):
        """Ascend910A / Ascend310B4 lack FFTS support but must not force the flag.

        ``force_disable_ffts`` drives an explicit ``--disable-ffts`` compile
        option, so widening it to every FFTS-unsupported target would silently
        change the legacy A2 compile command line.
        """
        for arch in ("Ascend910A", "Ascend310B4"):
            with self.subTest(arch=arch):
                self.assertFalse(self.utils["force_disable_ffts"](arch))
                self.assertFalse(self.utils["is_ffts_supported"](arch))

    def test_compile_on_register_based_accessor(self):
        """The NPUOptions accessor must resolve, not leak the property object."""
        accessor = self.compiler["_get_compile_on_register_based"]

        class _Options:
            target_arch = "Ascend950"

        self.assertTrue(accessor(_Options()))
        _Options.target_arch = "Ascend910_96"
        self.assertTrue(accessor(_Options()))
        _Options.target_arch = "Ascend910B1"
        self.assertFalse(accessor(_Options()))

    def test_soc_detection_cache_is_initialized(self):
        """utils.py must bind the SOC-detection cache at module scope.

        ``is_compile_on_910_95`` declares it ``global`` and reads it before
        assigning, so a missing module-level binding raises NameError.
        """
        tree = ast.parse((BACKEND / "utils.py").read_text())
        assigned = {
            target.id
            for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        self.assertIn("_is_compile_on_910_95", assigned)

    def test_runtime_soc_detection(self):
        for soc_name, detected in (
            ("Ascend910_9691", True),
            ("Ascend910_9699", True),
            ("Ascend960", True),
            ("Ascend960DT", True),
            ("Ascend910_9791", False),
            ("Ascend910B1", False),
        ):
            with self.subTest(soc_name=soc_name):
                acl = types.ModuleType("acl")
                acl.get_soc_name = lambda: soc_name
                with patch.dict(sys.modules, {"acl": acl}):
                    self.utils["_is_compile_on_910_95"] = None
                    self.assertEqual(self.utils["is_compile_on_910_95"](), detected)
                self.utils["_is_compile_on_910_95"] = None

    def test_deprecated_helper_points_at_replacement(self):
        """is_compile_on_910_95 must advertise is_register_based_arch."""
        notices = self.utils["deprecated_notices"]
        notices.clear()
        self.utils["is_compile_on_910_95"]("Ascend950")
        self.assertIn(("is_compile_on_910_95", "is_register_based_arch"), notices)


if __name__ == "__main__":
    unittest.main()
