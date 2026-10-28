"""Offline target/launch policy checks; runnable with Python's unittest runner.

Extract the policy functions from this checkout so these checks need neither
an installed Triton wheel nor an NPU runtime. Device execution is separate.
"""
import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_policies(path, names):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'_is_compile_on_910_95': None, '_warn_deprecated_ascend_env_var': lambda name: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class A6TargetContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        backend = Path(__file__).resolve().parents[2] / 'backend'
        cls.utils = load_policies(
            backend / 'utils.py', {
                'is_compile_on_910_95',
                'is_ffts_supported',
                'force_disable_ffts',
                'ub_size_in_kbytes_for_arch',
                'graph_ub_budget_bytes_for_arch',
            })
        cls.compiler = load_policies(backend / 'compiler.py', {'_is_register_based_target_arch'})

    def test_target_policy(self):
        for arch, register_based, ub in [
            ('Ascend910_9691', True, 376),
            ('Ascend910_9699', True, 376),
            ('Ascend960', True, 376),
            ('Ascend960DT_9691', True, 376),
            ('Ascend910_958b', True, 256),
            ('Ascend950DT_9571', True, 256),
            ('Ascend910B1', False, 192),
            ('Ascend910_9391', False, 192),
            ('Ascend910_9791', False, 0),
            ('', False, 0),
            (123, False, 0),
        ]:
            with self.subTest(arch=arch):
                self.assertEqual(self.utils['is_compile_on_910_95'](arch), register_based)
                self.assertEqual(self.compiler['_is_register_based_target_arch'](arch), register_based)
                self.assertEqual(self.utils['ub_size_in_kbytes_for_arch'](arch), ub)
                self.assertEqual(self.utils['graph_ub_budget_bytes_for_arch'](arch), ub * 1024 * 80 // 100)

    def test_launch_and_compiler_ffts_policy(self):
        for arch in ('Ascend910_9691', 'Ascend960', 'Ascend910_958b', 'Ascend950'):
            with self.subTest(arch=arch):
                self.assertTrue(self.utils['force_disable_ffts'](arch))
                self.assertFalse(self.utils['is_ffts_supported'](arch))
        self.assertFalse(self.utils['force_disable_ffts']('Ascend910B1'))
        self.assertTrue(self.utils['is_ffts_supported']('Ascend910B1'))

    def test_runtime_soc_detection(self):
        for soc_name, detected in (
            ('Ascend910_9691', True),
            ('Ascend910_9699', True),
            ('Ascend960', True),
            ('Ascend960DT', True),
            ('Ascend910_9791', False),
            ('Ascend910B1', False),
        ):
            with self.subTest(soc_name=soc_name):
                acl = types.ModuleType('acl')
                acl.get_soc_name = lambda: soc_name
                with patch.dict(sys.modules, {'acl': acl}):
                    self.utils['_is_compile_on_910_95'] = None
                    self.assertEqual(self.utils['is_compile_on_910_95'](), detected)
                self.utils['_is_compile_on_910_95'] = None


if __name__ == '__main__':
    unittest.main()
