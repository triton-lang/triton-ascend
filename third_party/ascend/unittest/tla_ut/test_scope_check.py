"""Compile-time cube/vector name scope checks. No device and no Triton build."""

import ast
import importlib.util
import unittest
from pathlib import Path

_CHECK_PATH = Path(__file__).resolve().parents[2] / "language" / "tla" / "scope_check.py"
_spec = importlib.util.spec_from_file_location("tla_scope_check", _CHECK_PATH)
_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_check)
bind_name = _check.bind_name
check_scope_body = _check.check_scope_body
install_scope_tracking = _check.install_scope_tracking


def _fn_body(src):
    tree = ast.parse(src)
    return tree.body[0].body


def _check_kernel(src):
    """Walk a kernel the way sequential ``with`` regions do, sharing one binding map."""
    defs = {}
    check_scope_body(_fn_body(src), None, defs)
    return defs


class ScopeCheckTest(unittest.TestCase):

    def test_cube_name_rejected_in_vector_scope(self):
        src = """
def kernel():
    with tla.scope.cube():
        acc = tla.cube.mmad(a, b, c)
    with tla.scope.simd():
        out = acc
"""
        with self.assertRaises(ValueError) as caught:
            _check_kernel(src)
        self.assertIn("scope failure", str(caught.exception))
        self.assertIn("'acc'", str(caught.exception))
        self.assertIn("cube scope", str(caught.exception))
        self.assertIn("vector scope", str(caught.exception))

    def test_vector_name_rejected_in_cube_scope(self):
        src = """
def kernel():
    with tla.scope.simd():
        bias = tla.simd.vfull(1.0, tl.float32, 64)
    with tla.scope.cube():
        tla.cube.mmad(a, b, bias)
"""
        with self.assertRaises(ValueError) as caught:
            _check_kernel(src)
        self.assertIn("'bias'", str(caught.exception))
        self.assertIn("vector scope", str(caught.exception))
        self.assertIn("cube scope", str(caught.exception))

    def test_nested_vector_name_rejected_in_cube_scope(self):
        src = """
def kernel():
    with tla.scope.simd():
        v = tla.simd.vlds(buf[0])
        with tla.scope.cube():
            tla.cube.mmad(a, b, v)
"""
        with self.assertRaises(ValueError) as caught:
            _check_kernel(src)
        self.assertIn("'v'", str(caught.exception))

    def test_name_defined_in_nested_cube_rejected_in_outer_vector(self):
        src = """
def kernel():
    with tla.scope.simd():
        with tla.scope.cube():
            acc = tla.cube.mmad(a, b, c)
        out = acc
"""
        with self.assertRaises(ValueError) as caught:
            _check_kernel(src)
        self.assertIn("'acc'", str(caught.exception))
        self.assertIn("cube scope", str(caught.exception))

    def test_same_scope_use_is_legal(self):
        src = """
def kernel():
    with tla.scope.simd():
        v = tla.simd.vlds(buf[0])
        w = v
"""
        defs = _check_kernel(src)
        self.assertEqual(defs["v"], "vector")
        self.assertEqual(defs["w"], "vector")

    def test_outside_buffers_may_be_used_in_both_scopes(self):
        src = """
def kernel():
    a_l1 = tla.alloc_buffer((M, K), tl.float16, space="l1")
    c_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")
    with tla.scope.cube():
        tla.copy(a_blk, a_l1)
        tla.cube.mmad(a_l1, b_l1, c_l0c, clear_accum=True)
        tla.copy(c_l0c, c_ub)
    with tla.scope.simd():
        bias = tla.simd.vfull(BIAS, tl.float32, 64)
        for i in tl.static_range((M * N) // 64):
            v = tla.simd.vlds(c_ub[i, 0])
            tla.simd.vsts(c_ub[i, 0], tla.simd.vadd(v, bias))
"""
        defs = _check_kernel(src)
        self.assertNotIn("a_l1", defs)
        self.assertNotIn("c_ub", defs)
        self.assertEqual(defs["bias"], "vector")
        self.assertEqual(defs["i"], "vector")
        self.assertEqual(defs["v"], "vector")

    def test_reassign_outside_clears_core_binding(self):
        defs = {}
        cube = _fn_body("""
def kernel():
    with tla.scope.cube():
        x = acc
""")[0]
        check_scope_body(cube.body, "cube", defs)
        self.assertEqual(defs["x"], "cube")
        bind_name(defs, None, "x")
        simd = _fn_body("""
def kernel():
    with tla.scope.simd():
        y = x
""")[0]
        check_scope_body(simd.body, "vector", defs)
        self.assertEqual(defs["y"], "vector")
        self.assertNotIn("x", defs)

    def test_reassign_inside_the_using_scope_is_legal(self):
        src = """
def kernel():
    with tla.scope.cube():
        x = acc
    with tla.scope.simd():
        x = tla.simd.vlds(buf[0])
        y = x
"""
        defs = _check_kernel(src)
        self.assertEqual(defs["x"], "vector")
        self.assertEqual(defs["y"], "vector")

    def test_augassign_of_other_scope_is_rejected(self):
        src = """
def kernel():
    with tla.scope.simd():
        v = tla.simd.vlds(buf[0])
    with tla.scope.cube():
        v += one
"""
        with self.assertRaises(ValueError) as caught:
            _check_kernel(src)
        self.assertIn("'v'", str(caught.exception))

    def test_assignment_hook_clears_binding_outside_a_scope(self):

        class Generator:

            def set_value(self, name, value):
                return value

            def dereference_name(self, name):
                return name

        generator = Generator()
        install_scope_tracking(generator)
        generator._tla_scope_stack.append("cube")
        generator.set_value("x", "acc")
        self.assertEqual(generator._tla_scope_defs["x"], "cube")
        generator._tla_scope_stack.pop()
        generator.set_value("x", "realloc")
        self.assertNotIn("x", generator._tla_scope_defs)

    def test_read_hook_rejects_the_other_scope(self):

        class Generator:

            def set_value(self, name, value):
                return value

            def dereference_name(self, name):
                return name

        generator = Generator()
        install_scope_tracking(generator)
        generator._tla_scope_defs["acc"] = "cube"
        generator._tla_scope_stack.append("vector")
        with self.assertRaises(ValueError) as caught:
            generator.dereference_name("acc")
        self.assertIn("'acc'", str(caught.exception))
        self.assertEqual(generator.dereference_name("buf"), "buf")

    def test_existing_kernels_do_not_cross_scopes(self):
        root = Path(__file__).resolve().parent
        for path in list((root / "vector").glob("*.py")) + list((root / "cube").glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    check_scope_body(node.body, None, {})

    def test_loop_index_from_vector_rejected_in_cube(self):
        src = """
def kernel():
    with tla.scope.simd():
        for i in tl.static_range(4):
            v = tla.simd.vlds(buf[i])
    with tla.scope.cube():
        tla.cube.mmad(a, i, c)
"""
        with self.assertRaises(ValueError) as caught:
            _check_kernel(src)
        self.assertIn("'i'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
